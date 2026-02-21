from .llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from .annotation_graph.annotated_graph import Graph
from .annotation_graph.schema_handler import SchemaHandler
from .rag.rag import RAG
from .rag.utils.web_search import SimpleWebSearch
from .prompts.conversation_handler import conversation_prompt
from .prompts.classifier_prompt import classifier_prompt, answer_from_graph
from .summarizer import Graph_Summarizer
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .storage.history_manager import HistoryManager
from .storage.mongo_storage import mongo_db_manager
from .socket_manager import emit_to_user
from .Galaxy_integration.galaxy import GalaxyHandler
from .orchestrator.handler import Orchestrator, CodeExecOptions
import asyncio
import logging.handlers as loghandlers
from dotenv import load_dotenv
import traceback
import json
import os
from flask_socketio import emit
from typing import List, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)
log_dir = "/AI-Assistant/logfiles"
log_file = os.path.join(log_dir, "Assistant.log")
os.makedirs(log_dir, exist_ok=True)
logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
    filename=log_file,
    when="D",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
loghandle.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)
logger = logging.getLogger(__name__)
load_dotenv()





class AiAssistance:

    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        qdrant_client=None,
        embedding_model=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(
            llm=advanced_llm,
            schema_handler=schema_handler,
            annotation_service_url=os.getenv("ANNOTATION_SERVICE_URL"),
            use_external_api=bool(os.getenv("ANNOTATION_SERVICE_URL"))
        )
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.history = HistoryManager()
        self.store = mongo_db_manager
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm)
        self.embedding_model = embedding_model
        # Initialize BioGPT agent
        from app.biogpt import BioGPTAgent
        self.biogpt = BioGPTAgent(llm=advanced_llm)
        logger.info("BioGPT agent initialized")

        # Initialize Orchestrator as the central brain with access to all tools
        self.orchestrator = Orchestrator(
            llm=advanced_llm,
            rag=self.rag,
            annotation_graph=self.annotation_graph,
            hypothesis_generation=self.hypothesis_generation,
            galaxy_handler=self.galaxy_handler,
            biogpt=self.biogpt
        )





    def get_content_summaries(self, user_id, content_ids=None):
        # Get summaries for all content types (PDF and web)
        content_summaries = []

        # Get all content files for the user
        all_content = self.store.get_user_content_files(user_id)

        if content_ids:
            # Filter by specific content IDs
            filtered_content = [
                content
                for content in all_content
                if content.get("content_id") in content_ids
            ]
        else:
            # Get all content
            filtered_content = all_content

        for content in filtered_content:
            if content.get("content_type") == "pdf":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "pdf",
                        "filename": content.get("filename"),
                        "summary": content.get("summary") or "",
                    }
                )
            elif content.get("content_type") == "web":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "web",
                        "url": content.get("url"),
                        "title": content.get("title"),
                        "summary": content.get("summary") or "",
                    }
                )

        return content_summaries

    async def assistant(
        self,
        query,
        user_id: str,
        token: str,
        resource=None,
        graph_id=None,
        content_ids: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        urls: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        logger.info(
            f"Assistant called with query: {query}, user_id: {user_id}, resource: {resource}, graph_id: {graph_id}, content_ids: {content_ids}"
        )

        try:
            user_information = self.store.get_context_and_memory(user_id)
            history = []
            memory = []
            for item in user_information:
                q = item["QUESTION"]["question"]
                c = item["QUESTION"]["context"]
                m = item["MEMORIES"]
                history.append({"question": q, "context": c})
                memory.append(m)
            content_summaries = self.get_content_summaries(user_id, content_ids)
        except Exception as e:
            history = ""
            memory = ""
            content_summaries = []

        logger.info(f"Histories of the user are : {history} and memories are {memory}")
        graph_context = None
        if graph_id:
            logger.info(
                f"Graph id has been passed to the given query {query} answering based on the graph"
            )
            graph_context = self.answer_from_graph_summaries(
                query, user_id, resource, token, graph_id
            )
            return graph_context


        # NEW FLOW: Route ALL requests directly to Orchestrator (the central brain)
        # The Orchestrator will decide which specialized tool to use based on the query
        logger.info(f"Routing query to Orchestrator (central brain): {query}")
        
        # Convert options dict to CodeExecOptions if provided
        options_dict = options or {}
        if isinstance(options_dict, dict):
            exec_options = CodeExecOptions(
                timeout_seconds=options_dict.get("timeout_seconds", 120),
                max_memory_mb=options_dict.get("max_memory_mb", 2048),
                output_formats=options_dict.get("output_formats"),
                allow_network=options_dict.get("allow_network", False),
                max_iterations=options_dict.get("max_iterations", 20),
            )
        else:
            exec_options = CodeExecOptions()
        
        # Call Orchestrator directly
        agent_response = self.orchestrator.execute(
            instructions=query,
            files=files,
            urls=urls,
            options=exec_options,
            user_id=user_id,
            token=token
        )
        
        # Ensure response is in dict format
        if isinstance(agent_response, str):
            agent_response = {"text": agent_response}
        elif not isinstance(agent_response, dict):
            agent_response = {"text": str(agent_response)}
        
        emit_to_user(user=user_id, message=agent_response, status="completed")
        
        # Extract text from agent_response for history storage
        assistant_answer = (
            agent_response.get("text", str(agent_response))
            if isinstance(agent_response, dict)
            else str(agent_response)
        )
        self.history.create_history(
            user_id, query, assistant_answer, graph_id_referenced=graph_id
        )
        return agent_response



    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        logger.info(
            f"Answer from graph summaries called with query: {query}, user_id: {user_id}, resource: {resource}, graph_id: {graph_id}"
        )
        if query:
            logger.debug("Query provided with graph_id")
            summary = None
            if resource == "annotation":
                # Process summary with query
                summary = self.graph_summarizer.summary(token=token, graph_id=graph_id)
                emit_to_user(user=user_id, message="Analyzing...")

            elif resource == "hypothesis":
                summary = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                emit_to_user(user=user_id, message="Analyzing...")
                logger.info(f"Summaries of the graph id {graph_id} is {summary}")

            prompt = classifier_prompt.format(query=query, graph_summary=summary)
            response = self.advanced_llm.generate(prompt)
            if response.startswith("related:"):
                logger.info("question is related with the graph")
                query_response = response[len("related:") :].strip()
                # creating users history
                self.history.create_history(
                    user_id, query, query_response, graph_id_referenced=graph_id
                )
                logger.info(f"user query is {query} response is {query_response}")
                return {"text": query_response}
            elif "not" in response:
                return None

        logger.info("Only Graphid is provided")
        if resource == "annotation":
            # Process summary without query
            summary = self.graph_summarizer.summary(
                token=token, graph_id=graph_id, user_query=None
            )
            return summary
        elif resource == "hypothesis":
            logger.info("Hypothesis resource, no query provided")
            summary = self.hypothesis_generation.get_by_hypothesis_id(
                token, graph_id, query
            )
            return {"text": summary}
        else:
            logger.error(f"Unsupported resource type: '{resource}'")
            return {"text": f"Unsupported resource type: '{resource}'"}

    def assistant_response(
        self,
        query,
        user_id,
        token,
        graph=None,
        graph_id=None,
        file=None,
        resource="annotation",
        json_query=None,
        content_ids=None,
        files=None,
        urls=None,
        options=None,
    ):
        logger.info(
            f"Assistant response called with query: {query}, user_id: {user_id}, resource: {resource}, graph_id: {graph_id}, content_ids: {content_ids}"
        )
        try:
            logger.info(
                f"passes parameters are query = {query}, user_id= {user_id}, graphid={graph_id}, graph = {graph}, resource = {resource}, content_ids = {content_ids}"
            )
            logger.info(
                f"agent being called for a given query {query} from resource {resource} with content_ids: {content_ids}"
            )
            response = asyncio.run(
                self.assistant(
                    query=query,
                    user_id=user_id,
                    token=token,
                    resource=resource,
                    graph_id=graph_id,
                    content_ids=content_ids,
                    files=files,
                    urls=urls,
                    options=options,
                )
            )
            return response

            # if query and graph:
            #     summary = self.graph_summarizer.summary(user_query=query,graph=graph)
            #     self.history.create_history(user_id, query, response)
            #     return summary

            # if graph:
            #     summary = self.graph_summarizer.summary(user_query=query,graph=graph)
            #     self.history.create_history(user_id, query, response)
            #     return summary

            # if json_query:
            #     logger.info(f"Executing a json query {json_query} to the annotation service")
            #     try:
            #         logger.info(f"Generating graph with arguments: {json_query}")  # Add this line to log the arguments
            #         response = self.annotation_graph.generate_graph(f"",json_query,token)
            #         return response
            #     except Exception as e:
            #         logger.error("Error in generating graph", exc_info=True)
            #         return f"I couldn't generate a graph for the given format would you please try again."

        except:
            traceback.print_exc()
