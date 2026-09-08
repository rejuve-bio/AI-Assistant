
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.prompts.classifier_prompt import main_classifier_prompt

logger = logging.getLogger(__name__)


class WorkflowMixin:
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


    def _classify_query_types(self, qtype: str, query_types: list) -> None:
        if ("annotation_biological" in qtype or "annotation biological" in qtype) and "annotation_biological" not in query_types:
            query_types.append("annotation_biological")
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

        content_summaries = self.store.get_content_summaries(user_id, content_ids)
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

