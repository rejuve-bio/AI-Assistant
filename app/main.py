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
    aggregator_prompt
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
from typing import TypedDict, List, Annotated, Any, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import asyncio
import traceback
import json
import os
import operator
import logging


logger = logging.getLogger(__name__)

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_types: List[str]
    response: Dict[str, Any]
    error: str
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[Any]
    pipeline_details: Dict[str, Any]
    # Agent-specific responses with source attribution
    annotation_response: Optional[Dict[str, Any]]
    # Generic pause/resume payload for ANY agent's human-in-the-loop confirmation,
    pending_confirmation: Optional[Dict[str, Any]]
    confirmation_outcome: Optional[str]
    rag_response: Optional[Dict[str, Any]]
    galaxy_response: Optional[Dict[str, Any]]
    content_retrieval_response: Optional[Dict[str, Any]]
    biogpt_response:Optional[Dict[str, Any]]
    hypothesis_response: Optional[Dict[str, Any]]
    pubmed_response: Optional[Dict[str, Any]]
    clinical_trials_response: Optional[Dict[str, Any]]
    # Parallel execution control
    agents_to_run: List[str]
    agents_completed: Annotated[List[str], operator.add]
    stop_pipeline: Optional[bool]


ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"
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
        checkpointer=None,
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

        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=checkpointer)

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with proper parallel agent execution"""
        logger.info("Creating LangGraph workflow with parallel agent execution")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("router", self._router)
        workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("annotation_agent", self._annotation_agent)
        workflow.add_node("annotation_confirmation_agent", self._annotation_confirmation_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("galaxy_agent", self._galaxy_agent)
        workflow.add_node("content_retrieval_agent", self._content_retrieval_agent)
        workflow.add_node("biogpt_agent", self._biogpt_agent)
        workflow.add_node("aggregator", self._aggregate_responses)
        workflow.add_node("finalizer", self._finalize_response)
        workflow.add_node("pubmed_agent", self._pubmed_agent)
        workflow.add_node("clinical_trials_agent", self._clinical_trials_agent)

        # Define edges
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "router")

        # Router decides which agents to invoke
        workflow.add_conditional_edges(
            "router",
            self._should_run_agent,
            {
                "annotation_agent": "annotation_agent",
                "annotation_confirmation_agent": "annotation_confirmation_agent",
                "hypothesis_agent": "hypothesis_agent",
                "rag_agent": "rag_agent",
                "galaxy_agent": "galaxy_agent",
                "content_retrieval_agent": "content_retrieval_agent",
                "biogpt_agent": "biogpt_agent",
                "pubmed_agent": "pubmed_agent",
                "clinical_trials_agent": "clinical_trials_agent",
                "aggregator": "aggregator",
                "error" : "finalizer"
            },
        )

        workflow.add_edge("annotation_agent", "router")
        workflow.add_conditional_edges(
            "annotation_confirmation_agent",
            self._confirmation_agent_next,
            {"finalizer": "finalizer", "router": "router"},
        )
        workflow.add_edge("rag_agent", "router")
        workflow.add_edge("galaxy_agent", "router")
        workflow.add_edge("content_retrieval_agent", "router")
        workflow.add_edge("biogpt_agent", "router")
        workflow.add_edge("hypothesis_agent", "router")
        workflow.add_edge("pubmed_agent", "router")
        workflow.add_edge("clinical_trials_agent", "router")
        # Aggregator flows to finalizer
        workflow.add_edge("aggregator", "finalizer")
        workflow.add_edge("finalizer", END)
        return workflow

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

    def _classify_query_types(self, qtype: str, query_types: list) -> None:
        if ("annotation_biological" in qtype or "annotation biological" in qtype) and "annotation_biological" not in query_types:
            query_types.append("annotation_biological")
        if ("annotation_general" in qtype or "annotation general" in qtype) and "annotation_general" not in query_types:
            query_types.append("annotation_general")
        if "galaxy" in qtype and "galaxy" not in query_types:
            query_types.append("galaxy")
        if "rag" in qtype and "rag" not in query_types:
            query_types.append("rag")
        if "hypothesis" in qtype and "hypothesis_generation" not in query_types:
            query_types.append("hypothesis_generation")
        if "biogpt" in qtype and "biogpt" not in query_types:
            query_types.append("biogpt")
        if "literature" in qtype and "literature" not in query_types:
            query_types.append("literature")

    def _build_agent_list(self, query_types: list, content_ids, urls, graph_id) -> list:
        agents_to_run = []

        # content_retrieval_agent always runs first when a graph_id is present,
        # so subsequent agents have the graph context available in state.
        if content_ids or urls or graph_id:
            agents_to_run.append("content_retrieval_agent")

        type_to_agent = {
            "annotation_biological": "annotation_agent",
            "hypothesis_generation": "hypothesis_agent",
            "annotation_general": "annotation_agent",
            "galaxy": "galaxy_agent",
            "rag": "rag_agent",
            "biogpt": "biogpt_agent",
        }
        for qtype in query_types:
            if qtype == "literature":
                for agent in ("rag_agent", "pubmed_agent", "clinical_trials_agent"):
                    if agent not in agents_to_run:
                        agents_to_run.append(agent)
                continue
            agent = type_to_agent.get(qtype)
            if agent and agent not in agents_to_run:
                agents_to_run.append(agent)

        if not agents_to_run:
            agents_to_run.append("rag_agent")
        return agents_to_run

    def _classify_query(self, state: AgentState) -> Dict[str, Any]:
        """Classify query and determine which agents to invoke (can be multiple)"""
        query = state["user_query"]
        user_id = state["user_id"]
        content_ids = state.get("content_ids")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        resource = state.get("resource")

        # If the client explicitly set resource="hypothesis", skip LLM classification.
        # - graph_id present: query an existing hypothesis via content_retrieval_agent → get_by_hypothesis_id
        # - no graph_id: generate a new hypothesis via hypothesis_agent
        if resource == "hypothesis":
            if graph_id:
                logger.info("Resource='hypothesis' + graph_id — routing only to content_retrieval_agent")
                return {
                    "query_types": ["hypothesis_generation"],
                    "agents_to_run": ["content_retrieval_agent"],
                    "agents_completed": [],
                    "messages": [HumanMessage(content="Query classified as: hypothesis_generation")],
                }
            else:
                logger.info("Resource='hypothesis' + no graph_id — routing to hypothesis_agent")
                return {
                    "query_types": ["hypothesis_generation"],
                    "agents_to_run": ["hypothesis_agent"],
                    "agents_completed": [],
                    "messages": [HumanMessage(content="Query classified as: hypothesis_generation")],
                }

        content_summaries = self.get_content_summaries(user_id, content_ids)
        logger.info(f"Classifying query: {query}")

        classifier_prompt_text = main_classifier_prompt.format(
            query=query,
            content_summaries=content_summaries,
        )
        response = self.advanced_llm.generate(classifier_prompt_text).lower()
        logger.info(f"question classified as {response}")

        query_types = []
        cleaned_response = response.replace("and", ",").replace("\n", ",")
        potential_types = [t.strip() for t in cleaned_response.split(",")]

        for qtype in potential_types:
            self._classify_query_types(qtype, query_types)

        if not query_types:
            query_types = ["rag"]

        logger.info(f"Query classified as: {query_types}")

        agents_to_run = self._build_agent_list(query_types, content_ids, urls, graph_id)

        logger.info(f"Agents to run: {agents_to_run}")

        return {
            "query_types": query_types,
            "agents_to_run": agents_to_run,
            "agents_completed": [],
            "messages": [HumanMessage(content=f"Query classified as: {', '.join(query_types)}")],
        }

    def _router(self, state: AgentState) -> Dict[str, Any]:
        """Router node that doesn't change state, just passes through"""
        return {}

    def _should_run_agent(self, state: AgentState) -> Any:
        """
        Determine which agent(s) to run next.
        Returns a single agent name, a list of agent names to run concurrently
        (only when they're independent of each other's output), or 'aggregator'
        once all agents have completed.
        """
        # Short-circuit: an agent signalled that no further processing is needed
        if state.get("stop_pipeline"):
            logger.info("stop_pipeline flag set — skipping remaining agents, going to aggregator")
            return "aggregator"

        pending_confirmation = state.get("pending_confirmation")
        if pending_confirmation:
            return f"{pending_confirmation['agent']}_confirmation_agent"

        agents_to_run = state.get("agents_to_run", [])
        agents_completed = state.get("agents_completed", [])
        remaining = [a for a in agents_to_run if a not in agents_completed]

        if not remaining:
            logger.info("All agents completed, moving to aggregator")
            return "aggregator"

        if "content_retrieval_agent" in remaining and "annotation_agent" in remaining and state.get("graph_id"):
            concurrent = ["content_retrieval_agent"]
            if "biogpt_agent" in remaining:
                concurrent.append("biogpt_agent")
            logger.info(f"annotation_agent gated on content_retrieval_agent's result (graph_id present) — running {concurrent} now")
            return concurrent if len(concurrent) > 1 else concurrent[0]

        if "annotation_agent" in remaining and "biogpt_agent" in remaining:
            logger.info("Running annotation_agent and biogpt_agent concurrently")
            return ["annotation_agent", "biogpt_agent"]

        next_agent = remaining[0]
        logger.info(f"Running next agent: {next_agent}")
        return next_agent

    def _annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle annotation-related queries"""
        query_types = state.get("query_types", [])
        query_type = next((qt for qt in query_types if "annotation" in qt), "annotation_biological")
        
        logger.info(
            f"Annotation agent processing query: {state['user_query']} for user: {state['user_id']}, type: {query_type}"
        )
        
        try:
            if query_type == "annotation_biological":
                emit_to_user(
                    user=state["user_id"], 
                    message="Processing your biological query..."
                )
            elif query_type == "annotation_general":
                emit_to_user(
                    user=state["user_id"], 
                    message="Analyzing database information..."
                )

            pipeline_response = self.annotation_graph.process_annotation_query(
                query=state["user_query"],
                user_id=state["user_id"],
                query_type=query_type,
            )

            logger.info(f"Pipeline response: {pipeline_response}")

            if pipeline_response.get("needs_confirmation"):
                return {
                    "pending_confirmation": {
                        "agent": "annotation",
                        "confirmation_text": pipeline_response.get("confirmation_text", ""),
                        "data": pipeline_response.get("pending") or {},
                    },                    
                    "agents_completed": [],
                    "messages": [AIMessage(content="Annotation needs user confirmation")],
                }

            if pipeline_response.get("success", False):
                summary = pipeline_response.get("summary", "")
                json_format = pipeline_response.get("json_format", None)
                validation_report = pipeline_response.get("validation_report", {})
                organism = pipeline_response.get("organism", "human")

                response_dict = {
                    "text": summary if summary else "",
                    "json_format": json_format,
                    "validation_report": validation_report,
                    "organism": organism,
                    "source": ANNOTATION_DB
                }

                return {
                    "annotation_response": response_dict,
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation processing completed")]
                }

            else:
                error_msg = pipeline_response.get("error", "Unknown error")
                logger.error(f"Annotation pipeline failed: {error_msg}")
                return {
                    "annotation_response": {
                        "text": f"Error: {error_msg}", 
                        "json_format": None,
                        "source": ANNOTATION_DB
                    },
                    "agents_completed": ["annotation_agent"],
                    "error": error_msg,
                }

        except Exception as e:
            logger.error("Unexpected error in annotation agent", exc_info=True)
            return {
                "annotation_response": {
                    "text": f"Error: {str(e)}",
                    "json_format": None,
                    "source": ANNOTATION_DB
                },
                "agents_completed": ["annotation_agent"],
                "error": str(e),
            }

    def _annotation_confirmation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Pauses on interrupt() to ask the user about an unrecognised annotation
        entity, then resumes with their reply.

        Deliberately its own node (not folded into _annotation_agent): interrupt()
        replays the whole node function on resume
        """
        pending_confirmation = state.get("pending_confirmation") or {}
        data = pending_confirmation.get("data", {})
        user_reply = interrupt(pending_confirmation.get("confirmation_text", ""))

        override_value = None
        if isinstance(user_reply, dict) and user_reply.get("type") == "direct":
            value = user_reply.get("value")
            if value in ("confirm", "reject", "show_alternatives"):
                verdict = value
            else:
                verdict = "confirm"
                override_value = value
        else:
            verdict = self.annotation_graph._classify_confirmation(user_reply)
            if verdict == "confirm":
                override_value = self._named_candidate_in_reply(user_reply, data)

        if verdict in ("confirm", "reject"):
            resolved = self.annotation_graph._apply_pending_substitutions(
                data.get("json", {}), apply=(verdict == "confirm"), override_value=override_value
            )
            text = self.annotation_graph._describe_annotation_result(state["user_query"], resolved)
            return {
                "annotation_response": {
                    "text": text,
                    "json_format": resolved,
                    "organism": data.get("organism", "human"),
                    "source": ANNOTATION_DB,
                },
                "pending_confirmation": None,
                "confirmation_outcome": "resolved",
                "agents_completed": ["annotation_agent"],
            }

        if verdict == "show_alternatives":
            alt_text = self.annotation_graph._build_alternatives_text(data)
            return {
                "pending_confirmation": {
                    **pending_confirmation,
                    "confirmation_text": alt_text,
                    "data": {**data, "alternatives_shown": True},
                },
                "confirmation_outcome": "pending",
                "agents_completed": [],
            }

        logger.info("Annotation confirmation abandoned for an unrelated new query")
        return {
            "pending_confirmation": None,
            "confirmation_outcome": "abandoned",
            "agents_completed": [],
        }

    def _named_candidate_in_reply(self, user_reply: str, data: Dict[str, Any]) -> Optional[str]:
        if not isinstance(user_reply, str) or not data:
            return None

        candidates = set(self.annotation_graph._alternative_candidate_values(data) or [])
        for entry in data.get("unconfirmed", []) or []:
            if entry.get("suggestion"):
                candidates.add(entry["suggestion"])
        if not candidates:
            return None

        tokens = {t.strip(" \t\n.,;:!?'\"()[]").lower() for t in user_reply.split()}
        named = [c for c in candidates if c.lower() in tokens]
        # Only act when it's unambiguous — if they somehow named several, fall
        # back to the plain confirm rather than guessing which one they meant.
        if len(named) == 1:
            logger.info(f"Free-text reply named candidate '{named[0]}' — applying it instead of the default suggestion")
            return named[0]
        return None

    def _confirmation_agent_next(self, state: AgentState) -> str:
        if state.get("confirmation_outcome") == "abandoned":
            return "finalizer"
        return "router"

    def _hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle hypothesis generation queries"""
        logger.info(
            f"Hypothesis agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        try:
            emit_to_user(user=state["user_id"], message="Generating hypothesis...")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"],
                user_query=state["user_query"],
                user_id=state["user_id"],
            )

            hypothesis_text = response.get("text", "")
            # A real hypothesis always returns resource: {id, type, graph} — all fallback/failure paths omit it
            succeeded = isinstance(response.get("resource"), dict) and response["resource"].get("type") == "hypothesis"

            state_update = {
                "hypothesis_response": response,
                "messages": [AIMessage(content=f"Hypothesis generated: {hypothesis_text}")],
                "agents_completed": ["hypothesis_agent"],
            }

            if succeeded:
                current_agents = state.get("agents_to_run", [])
                extra = [a for a in ("clinical_trials_agent", "pubmed_agent") if a not in current_agents]
                if extra:
                    logger.info(f"Hypothesis succeeded — injecting literature agents: {extra}")
                    state_update["agents_to_run"] = current_agents + extra

            return state_update

        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "hypothesis_response": {
                    "text": "The hypothesis service is not returning any results at the moment. There is nothing I can help with for this request.",
                    "resource": None,
                    "status": "failed",
                    "reason": str(e),
                },
                "stop_pipeline": True,
                "error": str(e),
                "messages": [AIMessage(content=f"Error in hypothesis generation: {str(e)}")],
                "agents_completed": ["hypothesis_agent"],
            }

    def _rag_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle general information queries"""
        logger.info(
            f"RAG agent processing query: {state['user_query']} for user: {state['user_id']}"
        )

        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            
            response = self.rag.get_result_from_rag(
                state["user_query"],
                state["user_id"],
                content_ids=state.get("content_ids"),
            )

            # Normalize response to dict with text key
            if response and isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else ""
            logger.debug(f"RAG response: {response_text}")

            # No useful results → inject PubMed as fallback
            if self._rag_has_no_results(response_text):
                current_agents = state.get("agents_to_run", [])
                if "pubmed_agent" not in current_agents:
                    logger.info("RAG found no results — injecting pubmed_agent as fallback")
                    emit_to_user(user=state["user_id"], message="Nothing found in knowledge base, searching PubMed...")
                    return {
                        "rag_response": {"text": response_text, "json_format": None, "source": KNOWLEDGE_BASE},
                        "agents_to_run": current_agents + ["pubmed_agent"],
                        "agents_completed": ["rag_agent"],
                        "messages": [AIMessage(content="RAG found no results — triggering PubMed fallback")],
                    }

            return {
                "rag_response": {
                    "text": response_text,
                    "json_format": None,
                    "source": KNOWLEDGE_BASE
                },
                "agents_completed": ["rag_agent"],
                "messages": [AIMessage(content="RAG query processed")],
            }
            
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "rag_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": KNOWLEDGE_BASE
                },
                "agents_completed": ["rag_agent"],
                "error": str(e),
            }

    def _galaxy_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle Galaxy tools and workflows queries"""
        logger.info(
            f"Galaxy agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        
        try:
            emit_to_user(
                user=state["user_id"], 
                message="Retrieving Galaxy tools information..."
            )
            
            response = self.galaxy_handler.get_galaxy_info(
                state["user_query"], 
                state["user_id"], 
                state["token"]
            )

            # Normalize response
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No Galaxy information found"
            logger.debug(f"Galaxy response: {response_text}")
            return {
                "galaxy_response": {
                    "text": response_text, 
                    "json_format": None,
                    "source": GALAXY_PLATFORM
                },
                "agents_completed": ["galaxy_agent"],
                "messages": [AIMessage(content="Galaxy query processed")],
            }
            
        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": GALAXY_PLATFORM
                },
                "agents_completed": ["galaxy_agent"],
                "error": str(e),
            }

    def _retrieve_from_graph(self, query, user_id, graph_id, token, resource, content_parts, sources):
        """Returns (early_return, entity_found) — early_return is a state update dict to
        short-circuit the pipeline on graph-fetch failure, or None to continue normally.
        entity_found is True/False/None (unknown) per whether the graph actually covered
        the query, used to decide whether a fresh annotation_agent run is still needed."""
        logger.info(f"Retrieving graph summary for graph_id: {graph_id}")
        graph_summary = self.answer_from_graph_summaries(
            query=query,
            user_id=user_id,
            graph_id=graph_id,
            token=token,
            resource=resource
        )
        if not graph_summary:
            return None, None
        entity_found = graph_summary.get("entity_found") if isinstance(graph_summary, dict) else None
        graph_text = graph_summary.get("text", str(graph_summary)) if isinstance(graph_summary, dict) else str(graph_summary)
        if graph_text and not graph_text.startswith("Failed to contact") and not graph_text.startswith("Error"):
            content_parts.append({"source": f"graph:{graph_id}", "content": graph_text})
            sources.append(f"graph:{graph_id}")
            return None, entity_found
        if graph_text:
            logger.warning(f"Graph fetch failed for {graph_id}: {graph_text}")
            last_topic = None
            try:
                history = self.store.get_context_and_memory(user_id)
                for item in reversed(history):
                    agents_used = item.get("context", {}).get("agents_used", [])
                    if "annotation_agent" in agents_used:
                        last_topic = item.get("question")
                        break
            except Exception:
                pass
            if last_topic:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Did you mean to ask about your previous annotation: *\"{last_topic}\"*? "
                    f"Or would you like to ask a different question?"
                )
            else:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Please check that the graph exists, or let me know what you'd like to explore."
                )
            return {
                "content_retrieval_response": {
                    "text": confirmation_text,
                    "json_format": None,
                    "sources": [],
                    "status": "needs_input",
                    "reason": "graph_not_found",
                },
                "agents_completed": ["content_retrieval_agent"],
                "stop_pipeline": True,
            }, None
        return None, entity_found

    def _retrieve_from_galaxy(self, query, user_id, token, urls, content_parts, sources):
        logger.info(f"Retrieving Galaxy urls for user: {user_id}")
        urls_response = self.galaxy_handler.get_galaxy_info(
            query=query, user_id=user_id, token=token, urls=urls
        )
        if urls_response:
            urls_text = urls_response.get("text", str(urls_response)) if isinstance(urls_response, dict) else str(urls_response)
            for file in (urls if isinstance(urls, list) else [urls]):
                content_parts.append({"source": f"file:{file}", "content": urls_text})
                sources.append(f"file:{file}")

    def _retrieve_from_rag(self, query, user_id, content_ids, content_parts, sources):
        logger.info(f"Retrieving RAG content for content_ids: {content_ids}")
        rag_content = self.rag.get_result_from_rag(query, user_id, content_ids)
        if rag_content:
            rag_text = rag_content.get("text", str(rag_content)) if isinstance(rag_content, dict) else str(rag_content)
            resources = rag_content.get("resource", {})
            content_parts.append({
                "source": f"content IDs: {', '.join(content_ids)}",
                "content": rag_text,
                "resource": resources
            })
            sources.append(f"content IDs: {', '.join(content_ids)}")

    def _content_retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant content from multiple sources with source attribution
        """
        query = state.get("user_query")
        user_id = state.get("user_id")
        token = state.get("token")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        logger.info(f"ContentRetrievalAgent called for user: {user_id}")
        emit_to_user(user=user_id, message="Retrieving relevant content...")

        content_parts = []
        sources = []
        graph_covers_query = False

        try:
            if graph_id:
                early_return, entity_found = self._retrieve_from_graph(query, user_id, graph_id, token, resource, content_parts, sources)
                if early_return is not None:
                    return early_return

                if entity_found is True and "annotation_agent" in state.get("agents_to_run", []):
                    graph_covers_query = True

            if urls:
                self._retrieve_from_galaxy(query, user_id, token, urls, content_parts, sources)

            if content_ids:
                self._retrieve_from_rag(query, user_id, content_ids, content_parts, sources)

            response_dict = {
                "text": content_parts,
                "json_format": None,
                "sources": sources
            }
            logger.info(f"Content retrieval response prepared with {len(content_parts)} parts. response is {response_dict}")
            state_update = {
                "content_retrieval_response": response_dict,
                "agents_completed": ["content_retrieval_agent"],
                "messages": [AIMessage(content="Content retrieval completed")]
            }
            if graph_covers_query:
                logger.info("Existing attached graph already covers the query — skipping redundant annotation_agent run")
                current_agents = state.get("agents_to_run", [])
                state_update["agents_to_run"] = [a for a in current_agents if a != "annotation_agent"]
            return state_update

        except Exception as e:
            logger.error(f"Error in ContentRetrievalAgent: {str(e)}", exc_info=True)
            return {
                "content_retrieval_response": {
                    "text": [],
                    "json_format": None,
                    "sources": []
                },
                "agents_completed": ["content_retrieval_agent"],
                "error": str(e),
            }

    def _biogpt_agent(self, state: AgentState) -> dict:
        try:
            emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
            response = self.biogpt.generate_answer(state["user_query"])
            logger.info(f"BioGPT response: {response}")
            return {
                "biogpt_response": {
                    "text": response,
                    "source": "BioGPT"
                },
                "agents_completed": ["biogpt_agent"],
                "messages": [AIMessage(content="BioGPT query processed")]
            }
        except Exception as e:
            logger.error(f"Error in biogpt agent: {str(e)}", exc_info=True)
            return {
                "biogpt_response": {
                    "text": None,
                    "json_format": None,
                    "source": "BioGPT"
                },
                "agents_completed": ["biogpt_agent"],
                "error": str(e)
            }

    _NO_RESULT_PHRASES = (
        "couldn't find", "could not find", "no relevant", "no information",
        "no results", "not found", "no documents", "unable to find",
        "no data", "i don't have information", "i do not have",
        "no specific", "no details",
    )

    def _rag_has_no_results(self, text: str) -> bool:
        t = text.lower().strip()
        return len(t) < 120 or any(p in t for p in self._NO_RESULT_PHRASES)

    def _extract_search_term(self, user_query: str, context: str = "") -> str:
        """Distil a question (and optional context) into a concise API-friendly search term."""
        context_line = f"\nAdditional context: {context[:500]}" if context else ""
        prompt = (
            "Extract a short, keyword-based search term (3-7 words) suitable for searching "
            "PubMed or ClinicalTrials.gov. Focus on the biological topic, gene, drug, or condition. "
            "Do NOT include words like: clinical trials, studies, papers, literature, search, find, pubmed, research. "
            "Do NOT use only a variant rs number — expand to the gene name and condition it is associated with. "
            "Return ONLY the search term, no explanation, no punctuation.\n\n"
            f"User question: {user_query}{context_line}\n\nSearch term:"
        )
        try:
            term = self.basic_llm.generate(prompt).strip().strip('"').strip("'")
            logger.info(f"Extracted search term: '{term}'")
            return term if term else user_query
        except Exception:
            return user_query

    def _pubmed_agent(self, state: AgentState) -> Dict[str, Any]:
        from app.rag.literature import search_pubmed
        user_id = state["user_id"]
        hypothesis = state.get("hypothesis_response") or {}
        context = hypothesis.get("text", "")
        search_term = self._extract_search_term(state["user_query"], context=context)
        logger.info(f"PubMed agent searching for: {search_term}")
        try:
            emit_to_user(user=user_id, message="Searching PubMed literature...")
            result = search_pubmed(search_term, max_results=8)
            papers = result.get("papers", [])
            if not papers:
                text = "No relevant publications found in PubMed for this query."
            else:
                lines = [f"Found {len(papers)} relevant paper(s) from PubMed:\n"]
                for p in papers:
                    authors = ", ".join(p.get("authors", [])) or "Unknown authors"
                    lines.append(
                        f"- **{p.get('title', 'No title')}** ({p.get('year', '')}) — {authors}\n"
                        f"  {p.get('abstract', '')}\n"
                        f"  URL: {p.get('url', '')}"
                    )
                text = "\n".join(lines)
            return {
                "pubmed_response": {"text": text, "source": "PubMed", "items": papers},
                "agents_completed": ["pubmed_agent"],
                "messages": [AIMessage(content="PubMed search completed")],
            }
        except Exception as e:
            logger.error(f"PubMed agent error: {e}", exc_info=True)
            return {
                "pubmed_response": {"text": f"PubMed search unavailable: {str(e)}", "source": "PubMed", "items": []},
                "agents_completed": ["pubmed_agent"],
            }

    def _clinical_trials_agent(self, state: AgentState) -> Dict[str, Any]:
        from app.rag.literature import search_clinical_trials
        user_id = state["user_id"]
        hypothesis = state.get("hypothesis_response") or {}
        context = hypothesis.get("text", "")
        search_term = self._extract_search_term(state["user_query"], context=context)
        logger.info(f"ClinicalTrials agent searching for: {search_term}")
        try:
            emit_to_user(user=user_id, message="Searching ClinicalTrials.gov...")
            result = search_clinical_trials(search_term, status="RECRUITING", max_results=5)
            trials = result.get("trials", [])
            if not trials:
                result = search_clinical_trials(search_term, status="", max_results=5)
                trials = result.get("trials", [])
            if not trials:
                text = "No clinical trials found for this query on ClinicalTrials.gov."
            else:
                lines = [f"Found {len(trials)} clinical trial(s) on ClinicalTrials.gov:\n"]
                for t in trials:
                    phase = ", ".join(t.get("phase", [])) or "N/A"
                    conditions = ", ".join(t.get("conditions", [])) or "N/A"
                    interventions = ", ".join(t.get("interventions", [])) or "N/A"
                    lines.append(
                        f"- **{t.get('title', 'No title')}** ({t.get('nct_id', '')})\n"
                        f"  Phase: {phase} | Status: {t.get('status', '')} | Started: {t.get('start_date', 'N/A')}\n"
                        f"  Conditions: {conditions}\n"
                        f"  Interventions: {interventions}\n"
                        f"  URL: {t.get('url', '')}"
                    )
                text = "\n".join(lines)
            return {
                "clinical_trials_response": {"text": text, "source": "ClinicalTrials.gov", "items": trials},
                "agents_completed": ["clinical_trials_agent"],
                "messages": [AIMessage(content="ClinicalTrials search completed")],
            }
        except Exception as e:
            logger.error(f"ClinicalTrials agent error: {e}", exc_info=True)
            return {
                "clinical_trials_response": {"text": f"ClinicalTrials search unavailable: {str(e)}", "source": "ClinicalTrials.gov",  "items": []},
                "agents_completed": ["clinical_trials_agent"],
            }

    def _format_annotation_section(self, failed: list) -> str:
        missing_parts = []
        for n in failed:
            not_validated = n.get("not_validated")
            if not_validated:
                items = not_validated if isinstance(not_validated, list) else [not_validated]
                for item in items:
                    missing_parts.append(f'"{item}"')
            else:
                props = n.get("properties", {})
                name = next(iter(props.values()), n.get("type", "unknown"))
                missing_parts.append(f'"{name}"')
        verb = "was" if len(missing_parts) == 1 else "were"
        joined = ", ".join(missing_parts)
        return f" Note: {joined} {verb} not found in the database."

    def _build_annotation_text(self, json_format: dict) -> str:
        """Build human-readable text from annotation validation results for truly-failed nodes."""
        nodes = json_format.get("nodes", [])
        # Ignore nodes pending confirmation — those are handled by _build_confirmation_text
        failed = [n for n in nodes if n.get("status") is False and not n.get("needs_confirmation")]
        text = "The annotation structure was created successfully (see structured data)."
        if failed:
            text += self._format_annotation_section(failed)
        return text

    def _build_sources_footer(self, state: dict) -> str:
        """Build a markdown Sources section with clickable links from PubMed and ClinicalTrials."""
        sections = []

        pubmed_resp = state.get("pubmed_response")
        if pubmed_resp:
            papers = pubmed_resp.get("items", [])
            links = [
                f"- [{p.get('title', p.get('pmid', 'Article'))}]({p['url']})"
                for p in papers if p.get("url")
            ]
            if links:
                sections.append("**PubMed Sources:**\n" + "\n".join(links))

        clinical_resp = state.get("clinical_trials_response")
        if clinical_resp:
            trials = clinical_resp.get("items", [])
            links = [
                f"- [{t.get('title', t.get('nct_id', 'Trial'))} ({t.get('nct_id', '')})]({t['url']})"
                for t in trials if t.get("url")
            ]
            if links:
                sections.append("**ClinicalTrials.gov Sources:**\n" + "\n".join(links))

        return "\n\n".join(sections)

    def _aggregate_annotation_response(self, annotation_resp: dict, agent_outputs: list) -> tuple:
        if annotation_resp.get("needs_confirmation"):
            return None, None, True
        text_content = annotation_resp.get("text") or annotation_resp.get("summary") or ""
        json_format = annotation_resp.get("json_format")
        organism = annotation_resp.get("organism") if json_format else None
        if not text_content and json_format:
            text_content = self._build_annotation_text(json_format)
        if text_content:
            agent_outputs.append({
                "agent": "annotation_agent",
                "source": annotation_resp.get("source", ANNOTATION_DB),
                "content": text_content
            })
        return json_format, organism, False

    def _aggregate_content_responses(self, state: dict, agent_outputs: list) -> Any:
        rag_resp = state.get("rag_response")
        if rag_resp:
            text_content = rag_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "rag_agent", "source": rag_resp.get("source", KNOWLEDGE_BASE), "content": text_content})

        galaxy_resp = state.get("galaxy_response")
        if galaxy_resp:
            text_content = galaxy_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "galaxy_agent", "source": galaxy_resp.get("source", GALAXY_PLATFORM), "content": text_content})

        biogpt_resp = state.get("biogpt_response")
        if biogpt_resp:
            text_content = biogpt_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "biogpt_agent", "source": biogpt_resp.get("source", "biogpt"), "content": text_content})

        resource_to_save = state.get("resource")

        content_resp = state.get("content_retrieval_response")
        if content_resp:
            content_parts = content_resp.get("text", [])
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict) and part.get("content"):
                        agent_outputs.append({"agent": "content_retrieval_agent", "source": part.get("source", "external content"), "content": part["content"]})
            elif isinstance(content_parts, str) and content_parts:
                sources = content_resp.get("sources", ["external content"])
                agent_outputs.append({"agent": "content_retrieval_agent", "source": ", ".join(sources), "content": content_parts})

        pubmed_resp = state.get("pubmed_response")
        if pubmed_resp:
            text_content = pubmed_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "pubmed_agent", "source": pubmed_resp.get("source", "PubMed"), "content": text_content})

        clinical_trials_resp = state.get("clinical_trials_response")
        if clinical_trials_resp:
            text_content = clinical_trials_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "clinical_trials_agent", "source": clinical_trials_resp.get("source", "ClinicalTrials.gov"), "content": text_content})

        return resource_to_save

    def _aggregate_responses(self, state: AgentState) -> Dict[str, Any]:
        """
        Aggregate responses from all agents with source attribution.
        Ensures that text content is combined coherently and structured JSON data (json_format)
        is always included when available.
        """
        # If an agent stopped the pipeline (e.g. content_retrieval_agent couldn't find
        # the referenced graph), surface its response directly with a structured
        # status/reason instead of letting it get paraphrased by LLM aggregation below.
        if state.get("stop_pipeline"):
            fallback_resp = state.get("content_retrieval_response") or state.get("response") or {}
            fallback_text = fallback_resp.get("text", "")
            if fallback_text:
                logger.info("stop_pipeline — returning structured fallback response")
                return {
                    "response": {
                        "text": fallback_text,
                        "json_format": fallback_resp.get("json_format"),
                        "status": fallback_resp.get("status", "needs_input"),
                        "reason": fallback_resp.get("reason"),
                    }
                }

        # Hypothesis fast-path — bypass LLM aggregation entirely
        hyp_resp = state.get("hypothesis_response") or {}
        if hyp_resp:
            hyp_succeeded = isinstance(hyp_resp.get("resource"), dict) and hyp_resp["resource"].get("type") == "hypothesis"
            if hyp_succeeded:
                # Success: return hypothesis text + supporting literature links
                hyp_text = hyp_resp.get("text", "")
                sources_footer = self._build_sources_footer(state)
                final_text = hyp_text.rstrip()
                if sources_footer:
                    final_text += "\n\n" + sources_footer
                return {
                    "response": {"text": final_text, "json_format": None, "organism": None},
                    "resource": hyp_resp.get("resource"),
                }
            else:
                # Soft failure (no project, validation failed) — return module message as-is
                # Hard failure (exception) routes through literature fallback instead, so hyp_text here is always a user-facing message
                hyp_text = hyp_resp.get("text") or "The hypothesis service is not returning any results at the moment. There is nothing I can help with directly, but I can search for similar clinical trials and published research — please try asking about the topic directly."
                return {
                    "response": {
                        "text": hyp_text,
                        "json_format": None,
                        "organism": None,
                        "status": hyp_resp.get("status", "needs_input"),
                        "reason": hyp_resp.get("reason"),
                    },
                }

        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs = []
        json_format = None
        organism = None

        # ---------------- Annotation Agent ----------------
        annotation_resp = state.get("annotation_response")
        if annotation_resp:
            json_format, organism, needs_confirm = self._aggregate_annotation_response(annotation_resp, agent_outputs)
            if needs_confirm:
                return {
                    "response": {
                        "text": annotation_resp.get("text", ""),
                        "json_format": None,
                    }
                }

        resource_to_save = self._aggregate_content_responses(state, agent_outputs)

        # ---------------- Handle JSON-only case ----------------
        if json_format and not agent_outputs:
            nodes = json_format.get("nodes", [])
            logger.info(f"[note-check] node statuses: { {n.get('node_id'): n.get('status') for n in nodes} }")
            failed = [n for n in nodes if n.get("status") is False]
            logger.info(f"[note-check] failed nodes: {[n.get('node_id') for n in failed]}")
            return {
                "response": {
                    "text": self._build_annotation_text(json_format),
                    "json_format": json_format,
                    "organism": organism
                }
            }

        if not agent_outputs:
            return {
                "response": {
                    "text": "I couldn't find any relevant information to answer your query.",
                    "json_format": None
                }
            }

        # ---------------- LLM Aggregation ----------------
        try:
            sources_info = []
            for output in agent_outputs:
                logger.info("=== [%s] source=%s ===\n%s",
                output['agent'],
                output.get('source', 'unknown'),
                str(output.get('content', ''))[:300])

                content = output.get("content", "")
                # Handle if content is a dict (convert to string)
                if isinstance(content, dict):
                    content = str(content)
                content = content.strip() if isinstance(content, str) else ""
                if content:
                    sources_info.append(f"From {output.get('source', 'unknown')}: {content}")

            combined_text = "\n\n".join(sources_info)

            json_note = ""
            if json_format:
                nodes = json_format.get("nodes", [])
                predicates = json_format.get("predicates", [])
                if nodes or predicates:
                    failed = [n for n in nodes if n.get("status") is False]
                    node_ids = [n.get("id", "") for n in nodes if n.get("id")]
                    if failed:
                        missing = ", ".join(
                            f"'{(n.get('properties') or {}).get('gene_name') or n.get('id') or n.get('node_id')}'"
                            for n in failed
                        )
                        json_note = (
                            f"\n\nNote: Only a partial annotation structure was built — these entities were "
                            f"NOT found in the database and remain unresolved: {missing}. Do NOT describe the "
                            f"annotation as successful or complete; state plainly that they were not found."
                        )
                    else:
                        json_note = (
                            f"\n\nNote: An annotation query was successfully built for "
                            f"{', '.join(node_ids) if node_ids else 'the requested entities'} "
                            f"and will be displayed on the graph."
                        )
                else:
                    json_note = "\n\nNote: Structured annotation data is also available for this query."

            query_types = state.get("query_types", [])
            if "hypothesis_generation" in query_types and state.get("graph_id"):
                prompt = hypothesis_aggregator_prompt.format(user_query=user_query, combined_text=combined_text)
            else:
                prompt = aggregator_prompt.format(user_query=user_query, combined_text=combined_text, json_note=json_note)

            aggregated_text = self.advanced_llm.generate(prompt)
            logger.info(f"Successfully aggregated response: {aggregated_text[:100]}...")

            sources_footer = self._build_sources_footer(state)
            if sources_footer:
                aggregated_text = aggregated_text.rstrip() + "\n\n" + sources_footer

            return {
                "response": {
                    "text": aggregated_text,
                    "json_format": json_format,
                    "organism": organism
                },
                "resource": resource_to_save
            }

        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}", exc_info=True)
            fallback_parts = []
            for output in agent_outputs:
                content = output.get('content', '')
                if isinstance(content, dict):
                    content = str(content)
                if content:
                    content_str = content.strip() if isinstance(content, str) else str(content)
                    fallback_parts.append(f"**From {output.get('source', 'unknown')}:**\n{content_str}")

            fallback_text = "\n\n".join(fallback_parts) if fallback_parts else "Annotation data retrieved."

            return {
                "response": {
                    "text": fallback_text,
                    "json_format": json_format,
                    "organism": organism
                },
                "resource": resource_to_save
            }

    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response"""
        response = state.get("response", {})
        user_id = state.get("user_id")
        
        logger.info(f"Finalizing response for user: {user_id}")
        logger.info(f"here is the response : {response}")
        
        # Ensure response has correct structure
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
         # Include the resource (hypothesis graph) if available
        if state.get("resource"):
            response["resource"] = state.get("resource")

        emit_to_user(user=user_id, message=response, status="completed")

        return {"response": response}

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        resource: Optional[Any] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point for processing queries with parallel agent execution.

        `thread_id` scopes the checkpointer — defaults to `user_id` (today's
        behavior: one thread per user) when the caller doesn't pick one, so
        clients that haven't adopted real per-conversation thread_ids yet keep
        working unchanged.
        """
        thread_id = thread_id or user_id
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, thread_id: {thread_id}, "
            f"content_ids: {content_ids}, graph_id: {graph_id}, urls: {urls}"
        )
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "token": token,
                "query_types": [],
                "response": {"text": "", "json_format": None},
                "error": "",
                "content_ids": content_ids,
                "graph_id": graph_id,
                "urls": urls,
                "resource": resource,
                "pipeline_details": {},
                "annotation_response": None,
                "pending_confirmation": None,
                "confirmation_outcome": None,
                "rag_response": None,
                "galaxy_response": None,
                "biogpt_response": None,
                "content_retrieval_response": None,
                "hypothesis_response": None,
                "pubmed_response": None,
                "clinical_trials_response": None,
                "stop_pipeline": False,
                "agents_to_run": [],
                "agents_completed": [],
            }

            # Run the workflow, scoped to this thread so the checkpointer can
            # persist/resume this conversation's graph state.
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config=config)

            response = self._extract_response_or_interrupt(result, config)
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

    def _confirmation_options(self, pending: dict) -> List[Dict[str, str]]:
        
        if pending.get("alternatives_shown"):
            options = [
                {"label": f"Use {v}", "value": v}
                for v in self.annotation_graph._alternative_candidate_values(pending)
            ]
            options.append({"label": "No, skip it", "value": "reject"})
            return options
        return [
            {"label": "Yes, use the suggested match", "value": "confirm"},
            {"label": "No, skip it", "value": "reject"},
            {"label": "Show other matches", "value": "show_alternatives"},
        ]

    def _extract_response_or_interrupt(self, result: Dict[str, Any], config: dict) -> Dict[str, Any]:
        """Build the response dict from an invoke() result, whether the graph
        finished normally or paused on any agent's confirmation interrupt().
        """
        state = self.app.get_state(config)
        if state.next:
            # Paused — the graph never reached the aggregator, so state["response"]
            # is still the empty placeholder from initial_state. Surface the
            # confirmation question instead.
            pending_confirmation = state.values.get("pending_confirmation") or {}
            data = pending_confirmation.get("data", {})
            return {
                "text": pending_confirmation.get("confirmation_text", ""),
                "json_format": None,
                "needs_confirmation": True,
                "confirmation": {
                    "options": self._confirmation_options(data),
                    "allow_free_text": True,
                },
                "agents_completed": result.get("agents_completed", []),
            }

        # pause/resume window, not the whole conversation.
        self._reset_thread(config)

        response = result.get("response", {"text": ""})
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        else:
            response.setdefault("text", "")
            response.setdefault("json_format", None)

        # ✅ Add agents_completed to the response so assistant_response can save it
        response["agents_completed"] = result.get("agents_completed", [])
        return response

    # Threads grow unbounded before compaction kicks in — summarizing every
    # single message would be wasteful. Only compact once there's real bulk to
    # fold in, and always leave this many recent messages verbatim afterward.
    _THREAD_COMPACT_THRESHOLD = 20
    _THREAD_KEEP_RECENT = 10

    def _record_thread_turn(self, thread_id: str, user_id: str, query: str, resp: Dict[str, Any],
                            graph_id: Optional[str] = None, resource: Optional[Any] = None) -> None:
        try:
            # graph_id on the USER turn = what they pointed at, typed by whatever
            # resource kind the client said it was (context.resource).
            self.store.append_message(
                thread_id, user_id, "user", query,
                graph_id=graph_id,
                resource_type=resource if isinstance(resource, str) else None,
            )

            resource = resp.get("resource")
            answered_about = graph_id
            # Always keep the type next to the id — an untyped id can't be
            # disambiguated later ("the annotation <id> or the hypothesis <id>?").
            resource_type = resource if isinstance(resource, str) else None
            if isinstance(resource, dict) and resource.get("id"):
                answered_about = resource["id"]
                resource_type = resource.get("type") or resource_type
            if answered_about and not resource_type:
                agents = resp.get("agents_completed") or []
                if "hypothesis_agent" in agents:
                    resource_type = "hypothesis"
                elif "annotation_agent" in agents:
                    resource_type = "annotation"

            thread_doc = self.store.append_message(
                thread_id,
                user_id,
                "assistant",
                resp.get("text", ""),
                json_format=resp.get("json_format"),
                agents=resp.get("agents_completed"),
                graph_id=answered_about,
                resource_type=resource_type,
            )

            label = (resp.get("text") or "")[:80]
            if isinstance(resource, dict) and resource.get("id"):
                self.store.record_tool_call(thread_id, resource.get("type", "unknown"), resource["id"], label)
            elif graph_id and resp.get("text"):
                # e.g. content_retrieval_agent summarizing an existing graph — no
                # new resource was created, but this turn is still a reference to
                # that graph and belongs in the thread's resource trail.
                agents = resp.get("agents_completed") or []
                agent = "content_retrieval" if "content_retrieval_agent" in agents else (agents[0] if agents else "unknown")
                self.store.record_tool_call(thread_id, agent, graph_id, label)

            if thread_doc.get("message_count", 0) >= self._THREAD_COMPACT_THRESHOLD:
                self._compact_thread(thread_id, thread_doc)
        except Exception as e:
            logger.warning(f"Failed to record thread turn for {thread_id}: {e}")

    def _compact_thread(self, thread_id: str, thread_doc: Dict[str, Any]) -> None:

        messages = thread_doc.get("messages", [])
        keep = self._THREAD_KEEP_RECENT
        if len(messages) <= keep:
            return
        older, recent = messages[:-keep], messages[-keep:]
        transcript = "\n".join(f"{m.get('role', '?')}: {m.get('text', '')}" for m in older)
        prior_summary = thread_doc.get("running_summary", "")
        prompt = (
            "Summarize this conversation excerpt into a concise running summary an "
            "assistant can use as context later. Merge it with the prior summary "
            "below rather than replacing it — keep anything from the prior summary "
            "still relevant.\n\n"
            f"Prior summary: {prior_summary or '(none yet)'}\n\n"
            f"Conversation excerpt:\n{transcript}\n\n"
            "Write the merged summary, 3-6 sentences, factual, no meta-commentary."
        )
        try:
            new_summary = self.advanced_llm.generate(prompt)
            new_summary = new_summary.strip() if isinstance(new_summary, str) else str(new_summary)
        except Exception as e:
            logger.warning(f"Thread compaction summarization failed for {thread_id}: {e}")
            return
        self.store.compact_thread(thread_id, new_summary, recent)

    def _reset_thread(self, config: dict) -> None:

        thread_id = config.get("configurable", {}).get("thread_id")
        checkpointer = getattr(self.app, "checkpointer", None)
        if not thread_id or not checkpointer:
            return
        try:
            checkpointer.delete_thread(thread_id)
        except Exception as e:
            logger.warning(f"Failed to reset checkpoint thread {thread_id}: {e}")

    def resume_confirmation_with_value(self, resume_value: str, user_id: str, thread_id: Optional[str] = None) -> Dict[str, Any]:

        thread_id = thread_id or user_id
        config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(config)
        if not state.next:
            return {
                "text": "There's nothing pending to confirm right now.",
                "json_format": None,
                "error": "no_pending_confirmation",
            }

        data = (state.values.get("pending_confirmation") or {}).get("data", {})
        valid_values = {opt["value"] for opt in self._confirmation_options(data)}
        if resume_value not in valid_values:
            logger.warning(f"Rejected stale/invalid resume value '{resume_value}' — valid: {valid_values}")
            return {
                "text": "That option isn't available anymore. Here's what's currently available:",
                "json_format": None,
                "needs_confirmation": True,
                "confirmation": {"options": self._confirmation_options(data), "allow_free_text": True},
                "error": "invalid_resume_value",
            }

        try:
            result = self.app.invoke(Command(resume={"type": "direct", "value": resume_value}), config=config)
            return self._extract_response_or_interrupt(result, config)
        except Exception as e:
            logger.error("Error resuming annotation confirmation with a direct value", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_format": None,
                "agents_completed": [],
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response

    def resume_pending_confirmation(self, query: str, user_id: str, thread_id: Optional[str] = None) -> Optional[Dict[str, Any]]:

        thread_id = thread_id or user_id
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = self.app.invoke(Command(resume=query), config=config)
            if result.get("confirmation_outcome") == "abandoned":
                logger.info("Resumed confirmation was abandoned for an unrelated query — falling back to normal flow")
                self._reset_thread(config)
                return None
            return self._extract_response_or_interrupt(result, config)
        except Exception as e:
            logger.error("Error resuming annotation confirmation", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_format": None,
                "agents_completed": [],
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response


    def _route_to_agent(self, response: str, query: str, user_id: str, token: str,
                        graph_id, content_ids, urls, resource, thread_id: Optional[str] = None) -> Dict[str, Any]:
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
            self._record_thread_turn(thread_id, user_id, query, {"text": final_response}, graph_id=graph_id, resource=resource)
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
                thread_id=thread_id,
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
            self._record_thread_turn(thread_id, user_id, query, agent_response, graph_id=graph_id, resource=resource)
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
        resume: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        thread_id = thread_id or user_id
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, thread_id={thread_id}, "
                f"graph_id={graph_id}, content_ids={content_ids}, urls={urls}, resource={resource}, resume={resume}"
            )

            if resume is not None:
                resp = self.resume_confirmation_with_value(resume, user_id, thread_id=thread_id)
                self.store.create_history(
                    user_id=user_id,
                    user_message=f"[resume:{resume}]",
                    assistant_answer=resp.get("text", ""),
                    graph_id_referenced=graph_id,
                    content_ids=content_ids,
                    urls=urls,
                    agents_used=resp.get("agents_completed", []),
                )
                self._record_thread_turn(thread_id, user_id, f"[resume:{resume}]", resp, graph_id=graph_id, resource=resource)
                emit_to_user(user=user_id, message=resp, status="completed")
                return resp
            try:
                is_pending = bool(self.app.get_state({"configurable": {"thread_id": thread_id}}).next)
            except Exception as e:
                logger.warning(f"Could not check for a pending annotation confirmation: {e}")
                is_pending = False

            if is_pending:
                resp = self.resume_pending_confirmation(query, user_id, thread_id=thread_id)
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
                    self._record_thread_turn(thread_id, user_id, query, resp, graph_id=graph_id, resource=resource)
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp

            try:
                thread_doc = self.store.get_thread(thread_id)
                running_summary = thread_doc.get("running_summary", "")
                messages = thread_doc.get("messages", [])
                history = []
                i = 0
                while i < len(messages):
                    if messages[i].get("role") == "user":
                        # Capture the question BEFORE advancing past the paired
                        # assistant message — reading messages[i] after the skip
                        # picked up the assistant's own text as the "question",
                        # feeding the LLM a history where every entry was the
                        # assistant talking to itself.
                        question = messages[i].get("text", "")
                        answer = ""
                        if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                            answer = messages[i + 1].get("text", "")
                            i += 1
                        history.append({"question": question, "context": {"answer": answer}})
                    i += 1
                memory = [running_summary] if running_summary else []
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
                thread_id=thread_id,
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