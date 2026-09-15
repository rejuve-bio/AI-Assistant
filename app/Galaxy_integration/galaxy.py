import asyncio
import logging
import sys
import traceback
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import subprocess
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GALAXY_MCP_SERVER = os.getenv("GALAXY_MCP_SERVER")
advanced_llm_provider = os.getenv("ADVANCED_LLM_PROVIDER", "gemini")  # gemini | openai | local_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GalaxyHandler:
    def __init__(self, llm, qdrant_client=None, embedding_model=None):
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = "1_AI_ASSISTANT_GALAXY_DATASETS"
        logger.info(f"GalaxyHandler initialized with provider='{advanced_llm_provider}'")

    def get_galaxy_info(self, query, user_id, token):
        """Main entry point. Answering from client-supplied URLs is RAG's job --
        this handler only talks to the Galaxy MCP server."""
        logger.info(f"get_galaxy_info called with query='{query}', user_id='{user_id}'")
        try:
            return self._handle_mcp(query, token)
        except Exception as e:
            logger.error(f"Galaxy handler failed: {e}")
            traceback.print_exc()
            return self._unavailable_response(query, str(e))

    def _handle_mcp(self, query, token):
        if advanced_llm_provider == "openai":
            logger.info("Using async MCP path (OpenAI)")
            return self._handle_mcp_async(query, token)
        elif advanced_llm_provider in ("gemini", "local_model"):
            logger.info(f"Using subprocess MCP path ({advanced_llm_provider} — avoids async conflicts)")
            return self._handle_mcp_subprocess(query, token)
        else:
            logger.warning(f"Unknown provider '{advanced_llm_provider}'")
            return self._unavailable_response(query, f"unsupported LLM provider '{advanced_llm_provider}'")

    def _handle_mcp_subprocess(self, query, token):
        """Run the MCP session in a subprocess.

        The Gemini and local-model clients start their own event loop, which
        clashes with the server's. The job goes over stdin rather than into a
        generated script, so the token and API key never touch disk.
        """
        logger.info(f"_handle_mcp_subprocess | provider='{advanced_llm_provider}' | query='{query}'")

        job = {
            "provider": advanced_llm_provider,
            "mcp_url": GALAXY_MCP_SERVER,
            "token": token,
            "query": query,
        }
        if advanced_llm_provider == "gemini":
            job["api_key"] = os.getenv("GOOGLE_API_KEY", "")
            job["model"] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        elif advanced_llm_provider == "local_model":
            job["host"] = os.getenv("LOCAL_MODEL_HOST", "http://localhost:8002")
            job["model"] = os.getenv("LOCAL_MODEL", "gemma4")
            job["api_key"] = os.getenv("LOCAL_MODEL_API_KEY", "sk-na")
        else:
            return {"text": f"Unsupported provider for subprocess path: {advanced_llm_provider}"}

        try:
            result = subprocess.run(
                [sys.executable, "-m", "app.Galaxy_integration.mcp_worker"],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.error("MCP subprocess timed out")
            return self._unavailable_response(query, "the Galaxy tool timed out after 120 seconds")

        if result.returncode != 0:
            logger.error(f"MCP subprocess failed: {result.stderr.strip()}")
            return self._unavailable_response(query, self._worker_error(result.stdout))

        try:
            payload = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            logger.error(f"Unparseable worker output: {result.stdout[:500]}")
            return self._unavailable_response(query, "the Galaxy tool returned malformed output")

        logger.info(f"Subprocess completed: {payload}")
        summary = self.llm.generate(
            f"Clean and summarize the following response concisely for the user query: {query}"
            f"\n\n{payload}"
        )
        return {"text": summary}

    @staticmethod
    def _worker_error(stdout):
        """The worker reports failures as JSON on stdout; fall back to a generic
        reason when it died before it could."""
        try:
            return json.loads(stdout.strip()).get("error", "the Galaxy tool failed")
        except Exception:
            return "the Galaxy tool failed"


    def _handle_mcp_async(self, query: str, token: str) -> dict:
        """Run MCP directly via asyncio — works fine with OpenAI (no eventlet)."""
        try:
            return asyncio.run(self._run_mcp_openai(query, token))
        except Exception as e:
            logger.error(f"Async MCP failed: {e}")
            traceback.print_exc()
            return self._unavailable_response(query, str(e))

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

    def _unavailable_response(self, query: str, reason: str) -> dict:
        """Generates a clear, user-facing message when the Galaxy tool cannot service the query.

        Produced here (not by the aggregator) so the aggregator just receives a normal
        agent answer and doesn't need special-case disclaimer logic for tool failures.
        """
        prompt = f"""
The Galaxy platform tool needed to answer the user's request is currently unavailable.
Internal reason (do not repeat verbatim or expose technical details): {reason}

User's request: "{query}"

Write a short, clear, friendly message (2-3 sentences) telling the user that this specific
capability isn't available right now, and suggest a concrete next step (e.g. try again in a
few minutes, or rephrase as a general question). Do not mention error codes, exceptions, or
internal system names.
"""
        try:
            return {"text": self.llm.generate(prompt)}
        except Exception as e:
            logger.error(f"Failed to generate unavailable-tool response: {e}")
            return {"text": "The Galaxy tool is currently unavailable. Please try again in a few minutes."}