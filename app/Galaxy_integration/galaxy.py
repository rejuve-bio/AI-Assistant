import asyncio
import logging
import traceback
from app.Galaxy_integration.galaxy_content_clean import HTMLProcessor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import subprocess
import json
import tempfile

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GALAXY_MCP_SERVER = os.getenv("GALAXY_MCP_SERVER")
advanced_llm_provider = os.getenv("ADVANCED_LLM_PROVIDER")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GalaxyHandler:
    def __init__(self, llm, qdrant_client=None, embedding_model=None):
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = "1_AI_ASSISTANT_GALAXY_DATASETS"
        logger.info("GalaxyHandler initialized")

    def get_galaxy_info(self, query, user_id, token, urls=None):
        """Main entry point: returns text only for Flask"""
        logger.info(f"get_galaxy_info called with query='{query}', user_id='{user_id}', urls={urls}")
        try:
            if urls and query:
                return self._handle_files(query=query, user_id=user_id, urls=urls)
            else:
                logger.info("No urls provided, routing to MCP handler")
                return self._handle_mcp(query, token)
        except Exception as e:
            logger.error(f"Galaxy handler failed: {e}")
            traceback.print_exc()
            return {"text": f"Error processing Galaxy request: {e}"}

    def _handle_files(self, query, user_id, urls):
        """Handle file-based queries using RAG - supports multiple URLs"""
        logger.info(f"_handle_files called with urls={urls}")
        
        # Normalize to list
        if isinstance(urls, str):
            urls = [urls]
        
        if not urls:
            return {"text": "No urls provided for analysis."}

        try:
            processor = HTMLProcessor(self.qdrant_client, self.llm)
            
            # Check if collection exists first
            collection_exists = False
            try:
                # Try to get collection info
                collection_info = self.qdrant_client.client.get_collection(self.collection_name)
                collection_exists = True
                logger.info(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points")
            except Exception as e:
                logger.info(f"Collection '{self.collection_name}' does not exist yet")
                collection_exists = False
            
            # Check which URLs need to be stored
            urls_to_process = []
            
            if collection_exists:
                # Collection exists, check each URL
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
                # Collection doesn't exist, process all URLs
                logger.info("Collection doesn't exist, will process all URLs")
                urls_to_process = urls.copy()
            
            # Process all new URLs in batch
            if urls_to_process:
                logger.info(f"Processing {len(urls_to_process)} new URLs")
                storage_results = processor.store_embedded(
                    urls=urls_to_process, 
                    collection_name=self.collection_name
                )
                
                # Log results for each URL
                for url, result in storage_results.items():
                    logger.info(f"Storage result for {url}: {result}")
                    
                # Check if any URLs were successfully processed
                successful_urls = [url for url, result in storage_results.items() 
                                if "Successfully processed" in result]
                
                if not successful_urls:
                    logger.error("Failed to process any URLs")
                    failed_reasons = [f"{url}: {result}" for url, result in storage_results.items()]
                    return {"text": f"Failed to extract content from provided documents:\n" + "\n".join(failed_reasons)}
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
                
                # Group results by URL for better context
                results_by_url = {}
                for chunk in similar_results:
                    url = chunk.get("content_id", "Unknown")
                    if url not in results_by_url:
                        results_by_url[url] = []
                    results_by_url[url].append(chunk)
                
                context_parts = []
                for url, chunks in results_by_url.items():
                    url_context = f"\n--- From {url} ---\n"
                    url_context += "\n\n".join(
                        str(chunk.get("text", ""))  # convert dict or any type to string
                        for chunk in chunks
                    )
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
            import traceback
            traceback.print_exc()
            return {"text": f"Error analyzing urls: {e}"}
            
    def handle_mcp(self, query, token):
        if advanced_llm_provider == "openai":
            logger.info("Using async MCP path (OpenAI)")
            return self._handle_mcp_async(query, token)
        elif advanced_llm_provider == "gemini":
            logger.info("Using subprocess MCP path (Gemini — avoids async conflicts)")
            return self._handle_mcp_sync(query, token)
        else:
            logger.warning(f"Unknown llm_name '{self.llm}', falling back to subprocess path")
            return self._handle_mcp_subprocess(query, token)


    def _handle_mcp_sync(self, query, token):
        """Sync wrapper - runs MCP in subprocess to avoid eventlet conflicts"""
        logger.info(f"_handle_mcp_sync called with query='{query}'")

        try:
            # Escape the query properly for the script
            query_escaped = query.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            # Create a temporary Python script that runs WITHOUT eventlet
            script_content = f'''
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

async def run_mcp():
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    client = MultiServerMCPClient({{
        "galaxyTools": {{
            "transport": "streamable_http",
            "url": "{GALAXY_MCP_SERVER}",
            "headers": {{"Authorization": "Bearer {token}"}}
        }}
    }})
    query = "{query_escaped}"
    tools = await client.get_tools()
    agent = create_agent(model, tools)
    response = await agent.ainvoke({{"messages": query}})
    if isinstance(response, dict):
        output = response.get("output", str(response))
    else:
        output = str(response)
    return {{"text": output}}

if __name__ == "__main__":
    result = asyncio.run(run_mcp())
    print(json.dumps(result))
'''

            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                # Run the script in a separate process (no eventlet!)
                logger.info(f"Running MCP in subprocess: {script_path}")
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    output = json.loads(result.stdout.strip())
                    logger.info(f"Subprocess completed with response {output}")
                    resonse = self.llm.generate(f"clean it and Summarize the following response concisely:\n\n{output} for the user query {query}")
                    return resonse
                else:
                    logger.error(f"Subprocess failed: {result.stderr}")
                    raise Exception(f"Subprocess error: {result.stderr}")

            finally:
                try:
                    os.unlink(script_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            logger.error("MCP subprocess timed out")
            return {"text": "Request timed out after 120 seconds. Please try again."}

        except Exception as e:
            logger.error(f"MCP execution failed, falling back to RAG: {e}")
            traceback.print_exc()

            # Fallback to RAG-based approach
            try:
                from app.prompts.rag_prompts import RETRIEVE_PROMPT
                from qdrant_client import QdrantClient

                GALAXY_TOOLS_RECOMMEND_COLLECTION = os.getenv("GALAXY_TOOLS_RECOMMEND_COLLECTION")
                qdrant_url = os.getenv("galaxy_QDRANT_CLIENT")

                if not qdrant_url or not GALAXY_TOOLS_RECOMMEND_COLLECTION:
                    logger.error("Missing environment variables for RAG fallback")
                    return {"text": "Configuration error: Unable to process request"}

                client = QdrantClient(
                    url=qdrant_url,
                    port=6333,
                    grpc_port=6334,
                    prefer_grpc=False,
                )

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

                response = RETRIEVE_PROMPT.format(
                    query=query,
                    retrieved_content=result
                )
                return {"text": response}

            except Exception as fallback_error:
                logger.error(f"RAG fallback also failed: {fallback_error}")
                traceback.print_exc()
                return {"text": f"Error: Unable to process request. Please try again later."}


    def _handle_mcp_async(self, query: str, token: str) -> dict:
        """Run MCP directly via asyncio — works fine with OpenAI."""
        try:
            return asyncio.run(self._run_mcp_openai(query, token))
        except Exception as e:
            logger.error(f"Async MCP failed: {e}")
            traceback.print_exc()
            return self._rag_fallback(query)

    async def _run_mcp_openai(self, query: str, token: str) -> dict:
            token = os.getenv("GALAXY_DEFAULT_TOKEN")
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
            return {"text": response}

            