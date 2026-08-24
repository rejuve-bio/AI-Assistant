"""AiAssistance — thin composition root and HTTP-facing façade.

All agent logic lives in ``app.orchestration.agents``.
All aggregation lives in ``app.orchestration.composer``.
This module only wires services together and handles the request/response boundary.
"""

from .llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from .prompts.classifier_prompt import (
    hypothesis_aggregator_prompt,
    classifier_prompt,
    main_classifier_prompt,
    aggregator_prompt,
)
from .annotation_graph.annotated_graph import Graph
from .annotation_graph.schema_handler import SchemaHandler
from .rag.rag import RAG
from .rag.utils.web_search import SimpleWebSearch
from .prompts.conversation_handler import conversation_prompt
from .summarizer import GraphSummarizer
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .socket_manager import emit_to_user
from .Galaxy_integration.galaxy import GalaxyHandler
from .biogpt_agent.biogpt import BioGPTAgent
from .orchestration.contracts import AgentName, AgentState, AssistantRequest
from .orchestration.planner import QueryPlanner
from .orchestration.registry import AgentDefinition, AgentRegistry
from .orchestration.workflow import AssistantWorkflow
from .orchestration.composer import ResponseComposer
from .orchestration.agents import (
    AnnotationAgent,
    BioGPTQueryAgent,
    ClinicalTrialsAgent,
    ContentRetrievalAgent,
    GalaxyAgent,
    HypothesisAgent,
    PubMedAgent,
    RagQueryAgent,
)
from .orchestration.agents.dependencies import AgentDependencies
from typing import List, Any, Dict, Optional
from flask_socketio import emit
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import asyncio
import traceback
import json
import os
import logging


logger = logging.getLogger(__name__)

load_dotenv()


ANALYZING_MSG = "Analyzing..."


class AiAssistance:

    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        fly_schema_handler=None,
        qdrant_client=None,
        embedding_model=None,
        mongo_db_manager=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler, fly_schema_handler=fly_schema_handler)
        self.graph_summarizer = GraphSummarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.store = mongo_db_manager
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm, qdrant_client, embedding_model)
        self.embedding_model = embedding_model
        self.biogpt = BioGPTAgent(basic_llm=basic_llm, advanced_llm=advanced_llm)

        logger.info(
            f"AiAssistance initialized with advanced_llm: {type(self.advanced_llm).__name__}"
        )
        logger.info(f"Galaxy handler initialized: {type(self.galaxy_handler).__name__}")

        # ----- Dependency Injection -----
        agent_dependencies = AgentDependencies(
            rag=self.rag,
            biogpt=self.biogpt,
            basic_llm=self.basic_llm,
            advanced_llm=self.advanced_llm,
            annotation_graph=self.annotation_graph,
            hypothesis_generation=self.hypothesis_generation,
            galaxy_handler=self.galaxy_handler,
            graph_summarizer=self.graph_summarizer,
            store=self.store,
            emit_status=emit_to_user,
        )

        # ----- Agent Instantiation -----
        rag_agent = RagQueryAgent(agent_dependencies)
        biogpt_agent = BioGPTQueryAgent(agent_dependencies)
        pubmed_agent = PubMedAgent(agent_dependencies)
        clinical_trials_agent = ClinicalTrialsAgent(agent_dependencies)
        annotation_agent = AnnotationAgent(agent_dependencies)
        hypothesis_agent = HypothesisAgent(agent_dependencies)
        galaxy_agent = GalaxyAgent(agent_dependencies)
        content_retrieval_agent = ContentRetrievalAgent(agent_dependencies)

        # ----- Response Composer -----
        composer = ResponseComposer(
            advanced_llm=self.advanced_llm,
            aggregator_prompt=aggregator_prompt,
            hypothesis_aggregator_prompt=hypothesis_aggregator_prompt,
            emit_status=emit_to_user,
        )

        # ----- Agent Registry -----
        self.agent_registry = AgentRegistry([
            AgentDefinition(AgentName.ANNOTATION, annotation_agent.execute, "annotation_response"),
            AgentDefinition(AgentName.HYPOTHESIS, hypothesis_agent.execute, "hypothesis_response"),
            AgentDefinition(AgentName.RAG, rag_agent.execute, "rag_response"),
            AgentDefinition(AgentName.GALAXY, galaxy_agent.execute, "galaxy_response"),
            AgentDefinition(AgentName.CONTENT_RETRIEVAL, content_retrieval_agent.execute, "content_retrieval_response"),
            AgentDefinition(AgentName.BIOGPT, biogpt_agent.execute, "biogpt_response"),
            AgentDefinition(AgentName.PUBMED, pubmed_agent.execute, "pubmed_response"),
            AgentDefinition(AgentName.CLINICAL_TRIALS, clinical_trials_agent.execute, "clinical_trials_response"),
        ])

        # ----- Query Planner -----
        self.query_planner = QueryPlanner(
            llm=self.advanced_llm,
            classifier_prompt=main_classifier_prompt,
            content_summaries=self.get_content_summaries,
            registry=self.agent_registry,
        )

        # ----- Orchestrator -----
        self.orchestrator = AssistantWorkflow(
            planner=self.query_planner,
            registry=self.agent_registry,
            aggregate=composer.aggregate,
            finalize=composer.finalize,
        )

    # ------------------------------------------------------------------
    # Content summaries (used by the planner for classification context)
    # ------------------------------------------------------------------

    def get_content_summaries(self, user_id, content_ids=None):
        """Get summaries for all content types (PDF and web)"""
        content_summaries = []
        all_content = self.store.get_user_content_files(user_id)

        if content_ids:
            filtered_content = [
                content
                for content in all_content
                if content.get("content_id") in content_ids
            ]
        else:
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

    # ------------------------------------------------------------------
    # Public entry points (HTTP / SocketIO boundary)
    # ------------------------------------------------------------------

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        resource: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Main entry point for processing queries with parallel agent execution"""
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, "
            f"content_ids: {content_ids}, graph_id: {graph_id}, urls: {urls}"
        )
        try:
            request = AssistantRequest(
                message=message,
                user_id=user_id,
                token=token,
                content_ids=content_ids,
                graph_id=graph_id,
                urls=urls,
                resource=resource,
                conversation_history=conversation_history,
            )
            response = self.orchestrator.invoke(request).model_dump()

            logger.info(f"Agent completed successfully for user: {user_id}")
            return response

        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_format": None,
                "agents_completed": []
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response

    def _route_to_agent(self, response: str, query: str, user_id: str, token: str,
                        graph_id, content_ids, urls, resource, conversation_history=None) -> Dict[str, Any]:
        if "response:" in response:
            result = response.split("response:")[1].strip()
            final_response = result.strip('"')
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=final_response,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=[],
            )
            emit_to_user(user=user_id, message=final_response, status="completed")
            return {"text": final_response}

        if "question:" in response:
            refactored_question = response.split("question:")[1].strip()
            agent_response = self.agent(
                refactored_question,
                user_id,
                token,
                content_ids=content_ids,
                graph_id=graph_id,
                urls=urls,
                resource=resource,
                conversation_history=conversation_history,
            )
            if isinstance(agent_response, str):
                agent_response = {"text": agent_response, "agents_completed": []}
            elif not isinstance(agent_response, dict):
                agent_response = {"text": str(agent_response), "agents_completed": []}
            resource_data = agent_response.get("resource")
            if isinstance(resource_data, dict):
                resource_type = resource_data.get("type")
                if resource_type:
                    logger.info(f"Resource successfully created: {resource_type}")
            assistant_answer = agent_response.get("text", str(agent_response))
            agents_used = agent_response.get("agents_completed", [])
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=assistant_answer,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=agents_used,
            )
            emit_to_user(user=user_id, message=agent_response, status="completed")
            return agent_response

        logger.error("No response generated from LLM")
        error_msg = "I apologize, but I encountered an error while processing your request."
        self.store.create_history(
            user_id=user_id,
            user_message=query,
            assistant_answer=error_msg,
            graph_id_referenced=graph_id,
            content_ids=content_ids,
            urls=urls,
            agents_used=[],
        )
        emit_to_user(user=user_id, message={"text": error_msg}, status="completed")
        return {"text": error_msg}

    def assistant_response(
        self,
        query: str,
        user_id: str,
        token: str,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        resource: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for assistant responses.
        Routes to parallel agent execution system.
        """
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, "
                f"graph_id={graph_id}, content_ids={content_ids}, urls={urls}, resource={resource}"
            )

            # Delegate to annotation_graph if a confirmation is pending for this user
            if self.annotation_graph.has_pending_for(user_id):
                resp = self.annotation_graph.handle_confirmation_response(user_id, query)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", []),
                    )
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp
                # else: new unrelated query — annotation_graph cleared the pending state, continue normally

            # Get conversation history and memory
            try:
                user_information = self.store.get_context_and_memory(user_id)
                history = []
                memory = []
                for item in user_information:
                    q = item["question"]
                    c = item["context"]
                    history.append({"question": q, "asked": item.get("asked", "unknown time ago"), "context": c})
                    memory.append(c["memory"])
            except Exception:
                history = []
                memory = []

            logger.info(f"Histories of the user are: {history} and memories are {memory}")

            prompt = conversation_prompt.format(
                memory=memory,
                query=query,
                conversation_history=history,
                graph_id=graph_id or "",
            )
            logger.info("Advanced llm response")
            response = self.advanced_llm.generate(prompt)
            logger.info(f"Response from the advanced LLM: {response}")
            emit_to_user(user=user_id, message=ANALYZING_MSG)

            return self._route_to_agent(
                response or "",
                query, user_id, token, graph_id, content_ids, urls, resource,
                conversation_history=history,
            )

        except Exception as e:
            logger.error(f"Error in assistant_response: {e}", exc_info=True)
            error_msg = "I apologize, but I encountered an error while processing your request."
            try:
                self.store.create_history(
                    user_id=user_id,
                    user_message=query,
                    assistant_answer=error_msg,
                    graph_id_referenced=graph_id,
                    content_ids=content_ids,
                    urls=urls,
                    agents_used=[],
                )
            except Exception as save_error:
                logger.error(f"Failed to save error history: {save_error}")
            return {
                "text": error_msg,
                "json_format": None
            }

    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        """Legacy method for backward compatibility"""
        logger.info(
            f"Answer from graph summaries called with query: {query}, user_id: {user_id}, "
            f"resource: {resource}, graph_id: {graph_id}"
        )

        try:
            entity_found = None
            if resource == "annotation":
                summary_result = self.graph_summarizer.summary(
                    token=token, graph_id=graph_id, user_query=query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                if isinstance(summary_result, dict):
                    entity_found = summary_result.get('entity_found')
                emit_to_user(user=user_id, message=ANALYZING_MSG)

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message=ANALYZING_MSG)
            else:
                return "Invalid resource type specified."

            # Return summary as dict for consistency
            return {"text": summary_text, "json_format": None, "entity_found": entity_found}

        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {
                "text": f"Error processing query: {str(e)}",
                "json_format": None
            }
