import asyncio
import logging
import traceback
from app.Galaxy_integration.galaxy_content_clean import HTMLProcessor
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import subprocess
import json
import tempfile

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GALAXY_MCP_SERVER = os.getenv("GALAXY_MCP_SERVER")
advanced_llm_provider = os.getenv("ADVANCED_LLM_PROVIDER", "gemini")  # gemini | openai | ollama

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GalaxyHandler:
    def __init__(self, llm, qdrant_client=None, embedding_model=None):
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = "1_AI_ASSISTANT_GALAXY_DATASETS"
        logger.info(f"GalaxyHandler initialized with provider='{advanced_llm_provider}'")

    def get_galaxy_info(self, query, user_id, token, urls=None):
        """Main entry point: returns text only for Flask"""
        logger.info(f"get_galaxy_info called with query='{query}', user_id='{user_id}', urls={urls}")
        try:
            if urls and query:
                return self._handle_files(query=query, user_id=user_id, urls=urls)
            else:
                logger.info("No urls provided, routing to MCP handler")
                return self._handle_mcp(query, token)  # ← was: self.handle_mcp (missing underscore)
        except Exception as e:
            logger.error(f"Galaxy handler failed: {e}")
            traceback.print_exc()
            return {"text": f"Error processing Galaxy request: {e}"}

    def _handle_files(self, query, user_id, urls):
        """Handle file-based queries using RAG - supports multiple URLs"""
        logger.info(f"_handle_files called with urls={urls}")

        if isinstance(urls, str):
            urls = [urls]

        if not urls:
            return {"text": "No urls provided for analysis."}

        try:
            processor = HTMLProcessor(self.qdrant_client, self.llm)

            collection_exists = False
            try:
                collection_info = self.qdrant_client.client.get_collection(self.collection_name)
                collection_exists = True
                logger.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
            except Exception:
                logger.info(f"Collection '{self.collection_name}' does not exist yet")

            urls_to_process = []
            if collection_exists:
                for url in urls:
                    logger.info(f"Checking if URL exists: {url}")
                    try:
                        stored = self.qdrant_client.retrieve_similar_content(
                            collection_name=self.collection_name,
                            query=query,
                            content_ids=[url],
                            filter=True
                        )
                        logger.info(f"Stored chunks for {url}: {len(stored) if stored else 0}")
                        if not stored:
                            logger.info(f"URL not found in collection, will process: {url}")
                            urls_to_process.append(url)
                        else:
                            logger.info(f"URL already in collection: {url}")
                    except Exception as e:
                        logger.warning(f"Error checking URL {url}: {e}")
                        urls_to_process.append(url)
            else:
                logger.info("Collection doesn't exist, will process all URLs")
                urls_to_process = urls.copy()

            if urls_to_process:
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
            else:
                logger.info("All URLs already exist in collection")

            logger.info(f"Retrieving similar content for query: '{query}' from {len(urls)} URLs")
            try:
                similar_results = self.qdrant_client.retrieve_similar_content(
                    collection_name=self.collection_name,
                    query=query,
                    content_ids=urls,
                    top_k=10
                )
            except Exception as e:
                logger.error(f"Error retrieving similar content: {e}")
                return {"text": f"Error retrieving content: {e}"}

            if similar_results:
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
                response_text = self.llm.generate(llm_prompt)
            else:
                logger.warning("No relevant chunks found in collection")
                response_text = f"I could not find any relevant information in the provided {len(urls)} document(s) to answer your query."

            return {"text": response_text}

        except Exception as e:
            logger.error(f"Galaxy file analyzer failed: {e}")
            traceback.print_exc()
            return {"text": f"Error analyzing urls: {e}"}

    # ── FIX: renamed from `handle_mcp` → `_handle_mcp` (missing underscore caused AttributeError) ──
    def _handle_mcp(self, query, token):
        if advanced_llm_provider == "openai":
            logger.info("Using async MCP path (OpenAI)")
            return self._handle_mcp_async(query, token)
        elif advanced_llm_provider in ("gemini", "ollama"):
            logger.info(f"Using subprocess MCP path ({advanced_llm_provider} — avoids async conflicts)")
            return self._handle_mcp_subprocess(query, token)
        else:
            logger.warning(f"Unknown provider '{advanced_llm_provider}', falling back to RAG")
            return self._rag_fallback(query)

    def _handle_mcp_subprocess(self, query, token):
        """Runs MCP agent in a subprocess — works for both Gemini and Ollama."""
        logger.info(f"_handle_mcp_subprocess called | provider='{advanced_llm_provider}' | query='{query}'")

        try:
            query_escaped = query.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            # Build provider-specific model setup to inject into the subprocess script
            if advanced_llm_provider == "gemini":
                model_setup = (
                    "from langchain_google_genai import ChatGoogleGenerativeAI\n"
                    "model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')"
                )
            elif advanced_llm_provider == "ollama":
                ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
                model_setup = (
                    f"from langchain_openai import ChatOpenAI\n"
                    f"model = ChatOpenAI(model='{ollama_model}', base_url='{ollama_host}/v1', api_key='ollama', temperature=0)"
                )
            else:
                return {"text": f"Unsupported provider for subprocess path: {advanced_llm_provider}"}

            script_content = f'''
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
{model_setup}

async def run_mcp():
    client = MultiServerMCPClient({{
        "galaxyTools": {{
            "transport": "streamable_http",
            "url": "{GALAXY_MCP_SERVER}",
            "headers": {{"Authorization": "Bearer {token}"}}
        }}
    }})
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    response = await agent.ainvoke({{"messages": "{query_escaped}"}})

    messages = response.get("messages", [])
    output = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or msg.get("content", "")
        if content and isinstance(content, str):
            output = content
            break
    print(json.dumps({{"text": output or str(response)}}))

asyncio.run(run_mcp())
'''

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                logger.info(f"Running MCP in subprocess: {script_path}")
                result = subprocess.run(
                    ['python', script_path],
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
                    raise Exception(f"Subprocess error: {result.stderr}")
            finally:
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            logger.error("MCP subprocess timed out")
            return {"text": "Request timed out after 120 seconds. Please try again."}

        except Exception as e:
            logger.error(f"MCP subprocess failed, falling back to RAG: {e}")
            traceback.print_exc()
            return self._rag_fallback(query)

    def _handle_mcp_async(self, query: str, token: str) -> dict:
        """Run MCP directly via asyncio — works fine with OpenAI (no eventlet)."""
        try:
            return asyncio.run(self._run_mcp_openai(query, token))
        except Exception as e:
            logger.error(f"Async MCP failed: {e}")
            traceback.print_exc()
            return self._rag_fallback(query)

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

    def _rag_fallback(self, query: str) -> dict:
        """Falls back to Qdrant vector search when MCP is unavailable."""
        logger.info("Attempting RAG fallback")
        try:
            from app.prompts.rag_prompts import RETRIEVE_PROMPT
            from qdrant_client import QdrantClient

            GALAXY_TOOLS_RECOMMEND_COLLECTION = os.getenv("GALAXY_TOOLS_RECOMMEND_COLLECTION")
            qdrant_url = os.getenv("galaxy_QDRANT_CLIENT")

            if not qdrant_url or not GALAXY_TOOLS_RECOMMEND_COLLECTION:
                logger.error("Missing environment variables for RAG fallback")
                return {"text": "Configuration error: Unable to process request"}

            client = QdrantClient(url=qdrant_url, port=6333, grpc_port=6334, prefer_grpc=False)

            if not self.embedding_model:
                logger.error("Embedding model not initialized")
                return {"text": "Configuration error: Embedding model not available"}

            embedded_text = self.embedding_model(query)
            result = client.search(
                collection_name=GALAXY_TOOLS_RECOMMEND_COLLECTION,
                query_vector=embedded_text,
                with_payload=True,
                score_threshold=0.5,
                limit=5
            )
            response = RETRIEVE_PROMPT.format(query=query, retrieved_content=result)
            return {"text": response}

        except Exception as e:
            logger.error(f"RAG fallback also failed: {e}")
            traceback.print_exc()
            return {"text": "Error: Unable to process request. Please try again later."}