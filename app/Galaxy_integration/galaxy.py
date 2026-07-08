import asyncio
import logging
import sys
import traceback
from app.Galaxy_integration.galaxy_content_clean import HTMLProcessor
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import subprocess
import json
import tempfile
import socket
from urllib.parse import urlparse

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GALAXY_MCP_SERVER = os.getenv("GALAXY_MCP_SERVER")
advanced_llm_provider = os.getenv("ADVANCED_LLM_PROVIDER", "gemini")  # gemini | openai | local_model
_GALAXY_MCP_ERROR = "Configuration error: Galaxy MCP server is unavailable."

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _check_mcp_reachable() -> bool:
    """Quick TCP probe to see if the MCP server is reachable. Runs once at startup."""
    if not GALAXY_MCP_SERVER:
        return False
    try:
        parsed = urlparse(GALAXY_MCP_SERVER)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False

_MCP_AVAILABLE = _check_mcp_reachable()
if not _MCP_AVAILABLE:
    logging.getLogger(__name__).warning(
        f"Galaxy MCP server unreachable at startup ({GALAXY_MCP_SERVER}) — will use LLM fallback for all Galaxy queries"
    )


class GalaxyHandler:
    def __init__(self, llm, qdrant_client=None, embedding_model=None):
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = "1_AI_ASSISTANT_GALAXY_DATASETS"
        logger.info(f"GalaxyHandler initialized with provider='{advanced_llm_provider}', mcp_available={_MCP_AVAILABLE}")

    def get_galaxy_info(self, query, user_id, token, urls=None):
        """Main entry point: returns text only for Flask"""
        logger.info(f"get_galaxy_info called with query='{query}', user_id='{user_id}', urls={urls}")
        try:
            if urls and query:
                return self._handle_files(query=query, urls=urls)
            else:
                logger.info("No urls provided, routing to MCP handler")
                return self._handle_mcp(query, token)
        except Exception as e:
            logger.error(f"Galaxy handler failed: {e}")
            traceback.print_exc()
            return {"text": f"Error processing Galaxy request: {e}"}

    def _collection_exists(self):
        try:
            collection_info = self.qdrant_client.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
            return True
        except Exception:
            logger.info(f"Collection '{self.collection_name}' does not exist yet")
            return False

    def _check_url_stored(self, query, url):
        try:
            stored = self.qdrant_client.retrieve_similar_content(
                collection_name=self.collection_name,
                query=query,
                content_ids=[url],
                filter=True
            )
            logger.info(f"Stored chunks for {url}: {len(stored) if stored else 0}")
            return bool(stored)
        except Exception as e:
            logger.warning(f"Error checking URL {url}: {e}")
            return False

    def _find_urls_to_process(self, query, urls):
        if not self._collection_exists():
            logger.info("Collection doesn't exist, will process all URLs")
            return urls.copy()
        urls_to_process = []
        for url in urls:
            logger.info(f"Checking if URL exists: {url}")
            if self._check_url_stored(query, url):
                logger.info(f"URL already in collection: {url}")
            else:
                logger.info(f"URL not found in collection, will process: {url}")
                urls_to_process.append(url)
        return urls_to_process

    def _store_new_urls(self, processor, urls_to_process):
        if not urls_to_process:
            logger.info("All URLs already exist in collection")
            return None
        logger.info(f"Processing {len(urls_to_process)} new URLs")
        storage_results = processor.store_embedded(
            urls=urls_to_process,
            collection_name=self.collection_name
        )
        for url, result in storage_results.items():
            logger.info(f"Storage result for {url}: {result}")
        successful_urls = [url for url, result in storage_results.items()
                           if "Successfully processed" in result]
        if not successful_urls:
            logger.error("Failed to process any URLs")
            failed_reasons = [f"{url}: {result}" for url, result in storage_results.items()]
            return {"text": "Failed to extract content from provided documents:\n" + "\n".join(failed_reasons)}
        return None

    def _retrieve_similar_content(self, query, urls):
        logger.info(f"Retrieving similar content for query: '{query}' from {len(urls)} URLs")
        try:
            results = self.qdrant_client.retrieve_similar_content(
                collection_name=self.collection_name,
                query=query,
                content_ids=urls,
                top_k=10
            )
            return results, None
        except Exception as e:
            logger.error(f"Error retrieving similar content: {e}")
            return None, {"text": f"Error retrieving content: {e}"}

    def _build_response(self, query, urls, similar_results):
        if not similar_results:
            logger.warning("No relevant chunks found in collection")
            return f"I could not find any relevant information in the provided {len(urls)} document(s) to answer your query."
        logger.info(f"Found {len(similar_results)} relevant chunks")
        results_by_url = {}
        for chunk in similar_results:
            url = chunk.get("content_id", "Unknown")
            if url not in results_by_url:
                results_by_url[url] = []
            results_by_url[url].append(chunk)
        context_parts = []
        for url, chunks in results_by_url.items():
            url_context = f"\n--- From {url} ---\n"
            url_context += "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
            context_parts.append(url_context)
        context_text = "\n\n".join(context_parts)
        llm_prompt = f"""
You are an expert AI assistant. You are given the following context extracted from {len(urls)} document(s):

{context_text}

User's query: "{query}"

Please provide a clear, professional, and concise answer to the user's query based solely on the context above.
- If information comes from multiple documents, you may synthesize it
- If the answer is not directly available, politely inform the user
- Do not hallucinate information
- Keep the response clear and concise
- If relevant, you can mention which document(s) the information comes from
"""
        return self.llm.generate(llm_prompt)

    def _handle_files(self, query, urls):
        """Handle file-based queries using RAG - supports multiple URLs"""
        logger.info(f"_handle_files called with urls={urls}")
        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            return {"text": "No urls provided for analysis."}
        try:
            processor = HTMLProcessor(self.qdrant_client, self.llm)
            urls_to_process = self._find_urls_to_process(query, urls)
            error = self._store_new_urls(processor, urls_to_process)
            if error:
                return error
            similar_results, error = self._retrieve_similar_content(query, urls)
            if error:
                return error
            return {"text": self._build_response(query, urls, similar_results)}
        except Exception as e:
            logger.error(f"Galaxy file analyzer failed: {e}")
            traceback.print_exc()
            return {"text": f"Error analyzing urls: {e}"}

    # ── FIX: renamed from `handle_mcp` → `_handle_mcp` (missing underscore caused AttributeError) ──
    def _handle_mcp(self, query, token):
        if not _MCP_AVAILABLE:
            logger.info("MCP server unavailable (checked at startup) — returning config error for aggregator LLM fallback")
            return {"text": _GALAXY_MCP_ERROR}
        if advanced_llm_provider == "openai":
            logger.info("Using async MCP path (OpenAI)")
            return self._handle_mcp_async(query, token)
        elif advanced_llm_provider in ("gemini", "local_model"):
            logger.info(f"Using subprocess MCP path ({advanced_llm_provider} — avoids async conflicts)")
            return self._handle_mcp_subprocess(query, token)
        else:
            logger.warning(f"Unknown provider '{advanced_llm_provider}' — Galaxy MCP unavailable")
            return {"text": _GALAXY_MCP_ERROR}

    def _handle_mcp_subprocess(self, query, token):
        """Runs MCP agent in a subprocess — works for both Gemini and local model."""
        logger.info(f"_handle_mcp_subprocess called | provider='{advanced_llm_provider}' | query='{query}'")

        try:
            query_escaped = query.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            if advanced_llm_provider == "gemini":
                google_api_key = os.getenv("GOOGLE_API_KEY", "")
                script_content = f'''
import asyncio, json
import google.generativeai as genai
from langchain_mcp_adapters.client import MultiServerMCPClient

async def run_mcp():
    mcp = MultiServerMCPClient({{
        "galaxyTools": {{
            "transport": "streamable_http",
            "url": "{GALAXY_MCP_SERVER}",
            "headers": {{"Authorization": "Bearer {token}"}}
        }}
    }})
    lc_tools = await mcp.get_tools()

    fn_decls = []
    for t in lc_tools:
        try:
            schema = t.args_schema.model_json_schema()
        except Exception:
            try:
                schema = t.args_schema.schema()
            except Exception:
                schema = {{"type": "object", "properties": {{}}}}
        fn_decls.append({{"name": t.name, "description": t.description or "", "parameters": schema}})

    genai.configure(api_key="{google_api_key}")
    model = genai.GenerativeModel("gemini-2.5-flash", tools=[{{"function_declarations": fn_decls}}])
    chat = model.start_chat()
    response = chat.send_message("{query_escaped}")

    for _ in range(10):
        fc_parts = []
        for cand in response.candidates:
            for part in cand.content.parts:
                if hasattr(part, "function_call") and part.function_call.name:
                    fc_parts.append(part)
        if not fc_parts:
            print(json.dumps({{"text": response.text}}))
            break
        tool_responses = []
        for part in fc_parts:
            fc = part.function_call
            tool = next((t for t in lc_tools if t.name == fc.name), None)
            if tool:
                result = await tool.ainvoke(dict(fc.args))
                tool_responses.append({{
                    "function_response": {{"name": fc.name, "response": {{"result": str(result)}}}}
                }})
        response = chat.send_message(tool_responses)

asyncio.run(run_mcp())
'''

            elif advanced_llm_provider == "local_model":
                local_model_host = os.getenv("LOCAL_MODEL_HOST", "http://localhost:8002")
                local_model_name = os.getenv("LOCAL_MODEL", "gemma4")
                local_model_api_key = os.getenv("LOCAL_MODEL_API_KEY", "sk-na")
                script_content = f'''
import asyncio, json
import openai
from langchain_mcp_adapters.client import MultiServerMCPClient

async def run_mcp():
    mcp = MultiServerMCPClient({{
        "galaxyTools": {{
            "transport": "streamable_http",
            "url": "{GALAXY_MCP_SERVER}",
            "headers": {{"Authorization": "Bearer {token}"}}
        }}
    }})
    lc_tools = await mcp.get_tools()

    openai_tools = []
    for t in lc_tools:
        try:
            schema = t.args_schema.model_json_schema()
        except Exception:
            try:
                schema = t.args_schema.schema()
            except Exception:
                schema = {{"type": "object", "properties": {{}}}}
        openai_tools.append({{"type": "function", "function": {{"name": t.name, "description": t.description or "", "parameters": schema}}}})

    client = openai.OpenAI(base_url="{local_model_host}/v1", api_key="{local_model_api_key}")
    messages = [{{"role": "user", "content": "{query_escaped}"}}]

    for _ in range(10):
        resp = client.chat.completions.create(
            model="{local_model_name}",
            messages=messages,
            tools=openai_tools if openai_tools else None
        )
        choice = resp.choices[0]
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                tool = next((t for t in lc_tools if t.name == tc.function.name), None)
                if tool:
                    result = await tool.ainvoke(json.loads(tc.function.arguments))
                    messages.append({{"role": "tool", "tool_call_id": tc.id, "content": str(result)}})
        else:
            print(json.dumps({{"text": choice.message.content or ""}}))
            break

asyncio.run(run_mcp())
'''

            else:
                return {"text": f"Unsupported provider for subprocess path: {advanced_llm_provider}"}

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                logger.info(f"Running MCP in subprocess: {script_path}")
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    raw_output = json.loads(result.stdout.strip())
                    logger.info(f"Subprocess completed: {raw_output}")
                    response = self.llm.generate(
                        f"Clean and summarize the following response concisely for the user query: {query}\n\n{raw_output}"
                    )
                    return {"text": response}
                else:
                    logger.error(f"Subprocess failed: {result.stderr}")
                    raise RuntimeError(f"Subprocess error: {result.stderr}")
            finally:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            logger.error("MCP subprocess timed out")
            return {"text": "Request timed out after 120 seconds. Please try again."}

        except Exception as e:
            logger.error(f"MCP subprocess failed: {e}")
            traceback.print_exc()
            return {"text": _GALAXY_MCP_ERROR}

    def _handle_mcp_async(self, query: str, token: str) -> dict:
        """Run MCP directly via asyncio — works fine with OpenAI (no eventlet)."""
        try:
            return asyncio.run(self._run_mcp_openai(query, token))
        except Exception as e:
            logger.error(f"Async MCP failed: {e}")
            traceback.print_exc()
            return {"text": _GALAXY_MCP_ERROR}

    async def _run_mcp_openai(self, query: str, token: str) -> dict:
        from langgraph.prebuilt import create_react_agent

        client = MultiServerMCPClient({
            "galaxyTools": {
                "transport": "streamable_http",
                "url": GALAXY_MCP_SERVER,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        })

        tools = await client.get_tools()
        agent = create_react_agent("openai:gpt-4o", tools)
        response = await agent.ainvoke({"messages": query})
        logger.info(f"OpenAI MCP raw response: {response}")

        messages = response.get("messages", [])
        output = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or msg.get("content", "")
            if content and isinstance(content, str):
                output = content
                break
        return {"text": output or str(response)}

