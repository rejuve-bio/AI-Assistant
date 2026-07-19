"""
LangGraph workflow: state, agent nodes, and routing.
AgentPipeline owns the DAG and all agent node implementations.
Aggregation and finalization are injected from AiAssistance in main.py.
"""

import logging
import operator
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .prompts.classifier_prompt import main_classifier_prompt
from .socket_manager import emit_to_user

logger = logging.getLogger(__name__)


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
    annotation_response: Optional[Dict[str, Any]]
    rag_response: Optional[Dict[str, Any]]
    galaxy_response: Optional[Dict[str, Any]]
    content_retrieval_response: Optional[Dict[str, Any]]
    biogpt_response: Optional[Dict[str, Any]]
    hypothesis_response: Optional[Dict[str, Any]]
    pubmed_response: Optional[Dict[str, Any]]
    clinical_trials_response: Optional[Dict[str, Any]]
    agents_to_run: List[str]
    agents_completed: Annotated[List[str], operator.add]
    stop_pipeline: Optional[bool]


_PARALLELIZABLE_AGENTS = frozenset({
    "rag_agent", "pubmed_agent", "clinical_trials_agent",
    "biogpt_agent", "annotation_agent", "galaxy_agent",
})

_QUERY_TYPE_KEYWORD_MAP = (
    (("annotation_biological", "annotation biological"), "annotation_biological"),
    (("annotation_general", "annotation general"), "annotation_general"),
    (("galaxy",), "galaxy"),
    (("rag",), "rag"),
    (("hypothesis",), "hypothesis_generation"),
    (("biogpt",), "biogpt"),
    (("literature",), "literature"),
    (("general_conversation", "greeting"), "general_conversation"),
)

_TYPE_TO_AGENTS = {
    "annotation_biological": ["annotation_agent"],
    "hypothesis_generation": ["hypothesis_agent"],
    "annotation_general": ["annotation_agent"],
    "galaxy": ["galaxy_agent"],
    "rag": ["rag_agent"],
    "biogpt": ["biogpt_agent"],
    "general_conversation": ["conversational_agent"],
    "literature": ["rag_agent", "pubmed_agent", "clinical_trials_agent"],
}

ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"
ANALYZING_MSG = "Analyzing..."

_HYPOTHESIS_WORD_RE = re.compile(r"hypothes(is|es)", re.IGNORECASE)
_VARIANT_RE = re.compile(r"\brs\d+\b", re.IGNORECASE)
_HYPOTHESIS_DOMAIN_SIGNAL_RE = re.compile(r"\b(graph|generated|do i have|have i|which|list)\b", re.IGNORECASE)


class AgentPipeline:
    """
    Self-contained LangGraph pipeline.
    Receives pre-initialized components; owns the full workflow DAG.
    """

    def __init__(
        self,
        *,
        annotation_graph,
        rag,
        hypothesis_generation,
        galaxy_handler,
        biogpt,
        basic_llm,
        advanced_llm,
        store,
        graph_summarizer,
        aggregator,
        finalizer,
    ) -> None:
        self.annotation_graph = annotation_graph
        self.rag = rag
        self.hypothesis_generation = hypothesis_generation
        self.galaxy_handler = galaxy_handler
        self.biogpt = biogpt
        self.basic_llm = basic_llm
        self.advanced_llm = advanced_llm
        self.store = store
        self.graph_summarizer = graph_summarizer
        self._aggregator = aggregator
        self._finalizer = finalizer
        self.app = self._create_workflow().compile()

    def run(self, initial_state: AgentState) -> Dict[str, Any]:
        return self.app.invoke(initial_state)

    # ------------------------------------------------------------------ #
    #  Content helpers                                                     #
    # ------------------------------------------------------------------ #

    def get_content_summaries(self, user_id, content_ids=None):
        content_summaries = []
        all_content = self.store.get_user_content_files(user_id)
        filtered_content = (
            [c for c in all_content if c.get("content_id") in content_ids]
            if content_ids
            else all_content
        )
        for content in filtered_content:
            if content.get("content_type") == "pdf":
                content_summaries.append({
                    "content_id": content.get("content_id"),
                    "content_type": "pdf",
                    "filename": content.get("filename"),
                    "summary": content.get("summary") or "",
                })
            elif content.get("content_type") == "web":
                content_summaries.append({
                    "content_id": content.get("content_id"),
                    "content_type": "web",
                    "url": content.get("url"),
                    "title": content.get("title"),
                    "summary": content.get("summary") or "",
                })
        return content_summaries

    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        logger.info(
            "Answer from graph summaries: query=%s user_id=%s resource=%s graph_id=%s",
            query, user_id, resource, graph_id,
        )
        try:
            if resource == "annotation":
                result = self.graph_summarizer.summarize(token=token, graph_id=graph_id, query=query)
                emit_to_user(user=user_id, message=ANALYZING_MSG)
            elif resource == "hypothesis":
                result = self.hypothesis_generation.get_by_hypothesis_id(token, graph_id, user_id)
                emit_to_user(user=user_id, message=ANALYZING_MSG)
            else:
                return "Invalid resource type specified."
            summary_text = result.get("text", "") if isinstance(result, dict) else result
            return {"text": summary_text, "json_format": None}
        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {"text": f"Error processing query: {str(e)}", "json_format": None}

    # ------------------------------------------------------------------ #
    #  Workflow creation                                                   #
    # ------------------------------------------------------------------ #

    def _create_workflow(self) -> StateGraph:
        logger.info("Creating LangGraph workflow with parallel agent execution")
        workflow = StateGraph(AgentState)

        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("router", self._router)
        workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("annotation_agent", self._annotation_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("galaxy_agent", self._galaxy_agent)
        workflow.add_node("content_retrieval_agent", self._content_retrieval_agent)
        workflow.add_node("biogpt_agent", self._biogpt_agent)
        workflow.add_node("aggregator", self._aggregator)
        workflow.add_node("finalizer", self._finalizer)
        workflow.add_node("pubmed_agent", self._pubmed_agent)
        workflow.add_node("clinical_trials_agent", self._clinical_trials_agent)
        workflow.add_node("parallel_runner", self._parallel_runner)
        workflow.add_node("conversational_agent", self._conversational_agent)

        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "router")

        workflow.add_conditional_edges(
            "router",
            self._should_run_agent,
            {
                "annotation_agent": "annotation_agent",
                "hypothesis_agent": "hypothesis_agent",
                "rag_agent": "rag_agent",
                "galaxy_agent": "galaxy_agent",
                "content_retrieval_agent": "content_retrieval_agent",
                "biogpt_agent": "biogpt_agent",
                "pubmed_agent": "pubmed_agent",
                "clinical_trials_agent": "clinical_trials_agent",
                "parallel_runner": "parallel_runner",
                "conversational_agent": "conversational_agent",
                "aggregator": "aggregator",
                "error": "finalizer",
            },
        )

        for node in (
            "annotation_agent", "rag_agent", "galaxy_agent", "content_retrieval_agent",
            "biogpt_agent", "hypothesis_agent", "pubmed_agent", "clinical_trials_agent",
            "parallel_runner", "conversational_agent",
        ):
            workflow.add_edge(node, "router")

        workflow.add_edge("aggregator", "finalizer")
        workflow.add_edge("finalizer", END)
        return workflow

    # ------------------------------------------------------------------ #
    #  Classification                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_hypothesis_ambiguity(self, query: str, query_types: list) -> Optional[Dict[str, Any]]:
        """
        Guard against the classifier LLM inconsistently labeling a bare variant-lookup
        query as hypothesis generation (it's nondeterministic run-to-run). Real hypothesis
        requests always say "hypothesis" — if that word never appears in the user's own
        text, this either isn't a hypothesis request at all, or it's genuinely ambiguous
        with a competing intent (e.g. annotation_biological) and the user should be asked
        rather than have us silently guess.
        """
        if "hypothesis_generation" not in query_types or _HYPOTHESIS_WORD_RE.search(query):
            return None

        other_types = [qt for qt in query_types if qt != "hypothesis_generation"]
        variant_match = _VARIANT_RE.search(query)
        variant_text = variant_match.group(0) if variant_match else "this variant"

        if not other_types:
            logger.info(
                "hypothesis_generation classified with no explicit 'hypothesis' wording and no "
                "competing intent — defaulting to annotation_biological instead of guessing"
            )
            query_types[:] = ["annotation_biological"]
            return None

        logger.info(
            "Ambiguous classification for '%s': hypothesis_generation + %s with no explicit "
            "'hypothesis' wording — asking user to clarify instead of guessing",
            query, other_types,
        )
        clarifying_text = (
            f"Just to make sure I get this right — for **{variant_text}**, do you want me to:\n\n"
            f"- **look it up** in the annotation database (its gene, and what it's linked to), or\n"
            f"- **generate or view a genetic hypothesis** for it (tissue/pathway enrichment analysis)?\n\n"
            f"Let me know which one."
        )
        return {
            "query_types": query_types,
            "agents_to_run": [],
            "agents_completed": [],
            "stop_pipeline": True,
            "response": {"text": clarifying_text, "json_format": None},
            "messages": [HumanMessage(
                content=f"Ambiguous classification ({', '.join(query_types)}) — asked user to clarify"
            )],
        }

    def _classify_query_types(self, qtype: str, query_types: list) -> None:
        for keywords, tag in _QUERY_TYPE_KEYWORD_MAP:
            if any(kw in qtype for kw in keywords) and tag not in query_types:
                query_types.append(tag)

    def _build_agent_list(self, query_types: list, content_ids, urls, graph_id) -> list:
        agents_to_run = []
        if content_ids or urls or graph_id:
            agents_to_run.append("content_retrieval_agent")

        for qtype in query_types:
            for agent in _TYPE_TO_AGENTS.get(qtype, []):
                if agent not in agents_to_run:
                    agents_to_run.append(agent)

        if not agents_to_run:
            agents_to_run.append("rag_agent")
        return agents_to_run

    def _classify_query(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        user_id = state["user_id"]
        content_ids = state.get("content_ids")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        resource = state.get("resource")

        content_summaries = self.get_content_summaries(user_id, content_ids)
        logger.info("Classifying query: %s", query)

        response = self.advanced_llm.generate(
            main_classifier_prompt.format(query=query, content_summaries=content_summaries)
        ).lower()
        logger.info("Question classified as: %s", response)

        query_types: List[str] = []
        for qtype in response.replace("and", ",").replace("\n", ",").split(","):
            self._classify_query_types(qtype.strip(), query_types)

        if not query_types:
            query_types = ["general_conversation"]

        if "hypothesis_generation" not in query_types and _HYPOTHESIS_WORD_RE.search(query) and _HYPOTHESIS_DOMAIN_SIGNAL_RE.search(query):
            logger.info(
                "Classifier missed an explicit 'hypothesis' + domain-signal query (%s) — "
                "forcing hypothesis_generation instead of trusting %s", query, query_types,
            )
            query_types = ["hypothesis_generation"]

        logger.info("Query classified as: %s", query_types)

        ambiguity_response = self._resolve_hypothesis_ambiguity(query, query_types)
        if ambiguity_response is not None:
            return ambiguity_response

        agents_to_run = self._build_agent_list(query_types, content_ids, urls, graph_id)

        # graph_id present: only use it as context for generic queries (rag/general/literature).
        # Anything with a clear agent intent (annotation, new hypothesis, galaxy, biogpt) runs as-is.
        if graph_id and resource == "hypothesis":
            GRAPH_CONTEXT_TYPES = {"rag", "general_conversation", "literature"}
            if any(qt in GRAPH_CONTEXT_TYPES for qt in query_types):
                logger.info("Resource='hypothesis' + graph_id + context query — routing to content_retrieval_agent")
                agents_to_run = ["content_retrieval_agent"]
                query_types = ["hypothesis_generation"]
            else:
                logger.info("Resource='hypothesis' + graph_id but explicit query (%s) — ignoring graph context", query_types)

        logger.info("Agents to run: %s", agents_to_run)

        return {
            "query_types": query_types,
            "agents_to_run": agents_to_run,
            "agents_completed": [],
            "messages": [HumanMessage(content=f"Query classified as: {', '.join(query_types)}")],
        }

    # ------------------------------------------------------------------ #
    #  Routing                                                             #
    # ------------------------------------------------------------------ #

    def _router(self, state: AgentState) -> Dict[str, Any]:
        return {}

    def _should_run_agent(self, state: AgentState) -> str:
        if state.get("stop_pipeline"):
            logger.info("stop_pipeline — going to aggregator")
            return "aggregator"

        agents_to_run = state.get("agents_to_run", [])
        agents_completed = state.get("agents_completed", [])
        remaining = [a for a in agents_to_run if a not in agents_completed]

        if not remaining:
            logger.info("All agents completed, moving to aggregator")
            return "aggregator"

        next_agent = remaining[0]

        if next_agent in ("content_retrieval_agent", "hypothesis_agent"):
            logger.info("Running sequential agent: %s", next_agent)
            return next_agent

        parallelizable = [a for a in remaining if a in _PARALLELIZABLE_AGENTS]
        if len(parallelizable) > 1:
            logger.info("Parallel execution: %s", parallelizable)
            return "parallel_runner"

        logger.info("Running agent: %s", next_agent)
        return next_agent

    def _merge_result_key(self, merged: dict, agents_to_run: list, key: str, value: Any) -> None:
        if key == "agents_completed":
            return
        if key == "messages":
            merged["messages"].extend(value if isinstance(value, list) else [value])
        elif key == "agents_to_run":
            base = merged.get("agents_to_run", list(agents_to_run))
            new_agents = [a for a in value if a not in base]
            if new_agents:
                merged["agents_to_run"] = base + new_agents
        elif key == "stop_pipeline" and value:
            merged["stop_pipeline"] = True
        elif value is not None:
            merged[key] = value

    def _parallel_runner(self, state: AgentState) -> Dict[str, Any]:
        agents_to_run = state.get("agents_to_run", [])
        agents_completed = state.get("agents_completed", [])
        to_run = [
            a for a in agents_to_run
            if a not in agents_completed and a in _PARALLELIZABLE_AGENTS
        ]

        agent_map = {
            "annotation_agent": self._annotation_agent,
            "rag_agent": self._rag_agent,
            "galaxy_agent": self._galaxy_agent,
            "biogpt_agent": self._biogpt_agent,
            "pubmed_agent": self._pubmed_agent,
            "clinical_trials_agent": self._clinical_trials_agent,
        }

        def run_one(name):
            return name, agent_map[name](state)

        with ThreadPoolExecutor(max_workers=min(len(to_run), 6)) as executor:
            futures = [executor.submit(run_one, a) for a in to_run]
            raw_results = []
            for f in futures:
                try:
                    raw_results.append(f.result())
                except Exception as e:
                    logger.error("Parallel agent failed: %s", e, exc_info=True)

        merged: Dict[str, Any] = {"agents_completed": to_run, "messages": []}
        for _name, result in raw_results:
            for k, v in result.items():
                self._merge_result_key(merged, agents_to_run, k, v)
        return merged

    # ------------------------------------------------------------------ #
    #  Agent nodes                                                         #
    # ------------------------------------------------------------------ #

    def _annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        query_types = state.get("query_types", [])
        query_type = next((qt for qt in query_types if "annotation" in qt), "annotation_biological")
        logger.info(
            "Annotation agent: query=%s user=%s type=%s",
            state["user_query"], state["user_id"], query_type,
        )
        try:
            msg = (
                "Processing your biological query..."
                if query_type == "annotation_biological"
                else "Analyzing database information..."
            )
            emit_to_user(user=state["user_id"], message=msg)

            pipeline_response = self.annotation_graph.process_annotation_query(
                query=state["user_query"],
                user_id=state["user_id"],
                query_type=query_type,
            )
            logger.info("Pipeline response: %s", pipeline_response)

            if pipeline_response.get("needs_confirmation"):
                validation_report = pipeline_response.get("validation_report", {})
                # Suppress confirmation only when EVERY node failed — i.e. the query had no
                # specific named entities at all (generic phrases like "genetic variants",
                # "aging"). When some nodes passed and some failed, the confirmation is real
                # and should be shown so the user can approve the substitution.
                total_nodes = validation_report.get("total_nodes", 0)
                failed_count = len(validation_report.get("failed_nodes", []))
                all_failed = total_nodes > 0 and failed_count >= total_nodes
                if all_failed:
                    logger.info(
                        "Annotation: all nodes failed validation — returning not-found instead of substitution prompt"
                    )
                    return {
                        "annotation_response": {
                            "text": "No specific entities were found in the annotation knowledge graph for this query.",
                            "json_format": None,
                            "source": ANNOTATION_DB,
                        },
                        "agents_completed": ["annotation_agent"],
                        "messages": [AIMessage(content="Annotation found no specific entities")],
                    }
                # Partial failure: some entities passed (e.g. FOXO3), some failed (e.g. "longevity").
                # Auto-skip the failed entities — do not ask the user about nonsensical substitutions.
                logger.info(
                    "Annotation: partial failure (%d/%d nodes failed) — auto-skipping failed entities",
                    failed_count, total_nodes,
                )
                auto = self.annotation_graph.handle_confirmation_response(
                    state["user_id"], "no"
                )
                if auto and auto.get("json_format") is not None:
                    return {
                        "annotation_response": {
                            "text": auto.get("text", "") or auto.get("summary", ""),
                            "json_format": auto.get("json_format"),
                            "organism": auto.get("organism", "human"),
                            "source": ANNOTATION_DB,
                        },
                        "agents_completed": ["annotation_agent"],
                        "messages": [AIMessage(content="Annotation completed (failed entities skipped)")],
                    }
                # auto-confirm failed — fall through to not-found
                return {
                    "annotation_response": {
                        "text": "No specific entities were found in the annotation knowledge graph for this query.",
                        "json_format": None,
                        "source": ANNOTATION_DB,
                    },
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation found no specific entities")],
                }

            if pipeline_response.get("success", False):
                return {
                    "annotation_response": {
                        "text": pipeline_response.get("summary", ""),
                        "json_format": pipeline_response.get("json_format"),
                        "validation_report": pipeline_response.get("validation_report", {}),
                        "organism": pipeline_response.get("organism", "human"),
                        "source": ANNOTATION_DB,
                    },
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation processing completed")],
                }

            error_msg = pipeline_response.get("error", "Unknown error")
            logger.error("Annotation pipeline failed: %s", error_msg)
            return {
                "annotation_response": {"text": f"Error: {error_msg}", "json_format": None, "source": ANNOTATION_DB},
                "agents_completed": ["annotation_agent"],
                "error": error_msg,
            }

        except Exception as e:
            logger.error("Unexpected error in annotation agent", exc_info=True)
            return {
                "annotation_response": {"text": f"Error: {str(e)}", "json_format": None, "source": ANNOTATION_DB},
                "agents_completed": ["annotation_agent"],
                "error": str(e),
            }

    def _hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Hypothesis agent: query=%s user=%s", state["user_query"], state["user_id"])
        try:
            emit_to_user(user=state["user_id"], message="Generating hypothesis...")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"],
                user_query=state["user_query"],
                user_id=state["user_id"],
            )
            hypothesis_text = response.get("text", "")
            succeeded = (
                isinstance(response.get("resource"), dict)
                and response["resource"].get("type") == "hypothesis"
                and not response.get("is_existing_hypothesis")
            )
            state_update = {
                "hypothesis_response": response,
                "messages": [AIMessage(content=f"Hypothesis generated: {hypothesis_text}")],
                "agents_completed": ["hypothesis_agent"],
            }
            if succeeded:
                current_agents = state.get("agents_to_run", [])
                extra = [a for a in ("clinical_trials_agent", "pubmed_agent") if a not in current_agents]
                if extra:
                    logger.info("Hypothesis succeeded — injecting literature agents: %s", extra)
                    state_update["agents_to_run"] = current_agents + extra
            return state_update

        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "hypothesis_response": {
                    "text": "The hypothesis service is not returning any results at the moment. There is nothing I can help with for this request.",
                    "resource": None,
                },
                "stop_pipeline": True,
                "error": str(e),
                "messages": [AIMessage(content=f"Error in hypothesis generation: {str(e)}")],
                "agents_completed": ["hypothesis_agent"],
            }

    def _rag_agent(self, state: AgentState) -> Dict[str, Any]:
        logger.info("RAG agent: query=%s user=%s", state["user_query"], state["user_id"])
        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            response = self.rag.get_result_from_rag(
                state["user_query"],
                state["user_id"],
                content_ids=state.get("content_ids"),
            )
            if response and isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            elif response:
                response_text = str(response)
            else:
                response_text = ""
            logger.debug("RAG response: %s", response_text)

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
                "rag_response": {"text": response_text, "json_format": None, "source": KNOWLEDGE_BASE},
                "agents_completed": ["rag_agent"],
                "messages": [AIMessage(content="RAG query processed")],
            }

        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "rag_response": {"text": f"Error: {str(e)}", "json_format": None, "source": KNOWLEDGE_BASE},
                "agents_completed": ["rag_agent"],
                "error": str(e),
            }

    def _galaxy_agent(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Galaxy agent: query=%s user=%s", state["user_query"], state["user_id"])
        try:
            emit_to_user(user=state["user_id"], message="Retrieving Galaxy tools information...")
            response = self.galaxy_handler.get_galaxy_info(
                state["user_query"], state["user_id"], state["token"]
            )
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            elif response:
                response_text = str(response)
            else:
                response_text = "No Galaxy information found"
            logger.debug("Galaxy response: %s", response_text)
            return {
                "galaxy_response": {"text": response_text, "json_format": None, "source": GALAXY_PLATFORM},
                "agents_completed": ["galaxy_agent"],
                "messages": [AIMessage(content="Galaxy query processed")],
            }

        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {"text": f"Error: {str(e)}", "json_format": None, "source": GALAXY_PLATFORM},
                "agents_completed": ["galaxy_agent"],
                "error": str(e),
            }

    def _biogpt_agent(self, state: AgentState) -> Dict[str, Any]:
        try:
            emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
            response = self.biogpt.generate_answer(state["user_query"])
            logger.info("BioGPT response: %s", response)
            return {
                "biogpt_response": {"text": response, "source": "BioGPT"},
                "agents_completed": ["biogpt_agent"],
                "messages": [AIMessage(content="BioGPT query processed")],
            }
        except Exception as e:
            logger.error("Error in biogpt agent: %s", str(e), exc_info=True)
            return {
                "biogpt_response": {"text": None, "json_format": None, "source": "BioGPT"},
                "agents_completed": ["biogpt_agent"],
                "error": str(e),
            }

    _NO_RESULT_PHRASES = (
        "couldn't find", "could not find", "no relevant", "no information",
        "no results", "not found", "no documents", "unable to find",
        "no data", "i don't have information", "i do not have",
        "no specific", "no details", "can't help", "cannot help",
        "i don't have", "not in my knowledge",
    )

    def _rag_has_no_results(self, text: str) -> bool:
        t = text.lower().strip()
        return len(t) < 120 or any(p in t for p in self._NO_RESULT_PHRASES)

    def _extract_search_term(self, user_query: str, context: str = "") -> str:
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
            logger.info("Extracted search term: '%s'", term)
            return term if term else user_query
        except Exception:
            return user_query

    def _conversational_agent(self, state: AgentState) -> Dict[str, Any]:
        query = state["user_query"]
        try:
            response = self.basic_llm.generate(
                f"You are a helpful biomedical research assistant. Respond naturally and briefly to this message: {query}"
            )
            return {
                "rag_response": {"text": response, "json_format": None, "source": None},
                "agents_completed": ["conversational_agent"],
                "messages": [AIMessage(content="Conversational response")],
            }
        except Exception as e:
            logger.error("Conversational agent error: %s", e)
            return {
                "rag_response": {"text": "Hello! How can I help you?", "json_format": None, "source": None},
                "agents_completed": ["conversational_agent"],
            }

    def _pubmed_agent(self, state: AgentState) -> Dict[str, Any]:
        from app.rag.literature import search_pubmed
        user_id = state["user_id"]
        context = (state.get("hypothesis_response") or {}).get("text", "")
        search_term = self._extract_search_term(state["user_query"], context=context)
        logger.info("PubMed agent searching for: %s", search_term)
        try:
            emit_to_user(user=user_id, message="Searching PubMed literature...")
            result = search_pubmed(search_term, max_results=5)
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
            logger.error("PubMed agent error: %s", e, exc_info=True)
            return {
                "pubmed_response": {"text": f"PubMed search unavailable: {str(e)}", "source": "PubMed", "items": []},
                "agents_completed": ["pubmed_agent"],
            }

    def _clinical_trials_agent(self, state: AgentState) -> Dict[str, Any]:
        from app.rag.literature import search_clinical_trials
        user_id = state["user_id"]
        context = (state.get("hypothesis_response") or {}).get("text", "")
        search_term = self._extract_search_term(state["user_query"], context=context)
        logger.info("ClinicalTrials agent searching for: %s", search_term)
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
                    lines.append(
                        f"- **{t.get('title', 'No title')}** ({t.get('nct_id', '')})\n"
                        f"  Phase: {', '.join(t.get('phase', [])) or 'N/A'} | "
                        f"Status: {t.get('status', '')} | Started: {t.get('start_date', 'N/A')}\n"
                        f"  Conditions: {', '.join(t.get('conditions', [])) or 'N/A'}\n"
                        f"  Interventions: {', '.join(t.get('interventions', [])) or 'N/A'}\n"
                        f"  URL: {t.get('url', '')}"
                    )
                text = "\n".join(lines)
            return {
                "clinical_trials_response": {"text": text, "source": "ClinicalTrials.gov", "items": trials},
                "agents_completed": ["clinical_trials_agent"],
                "messages": [AIMessage(content="ClinicalTrials search completed")],
            }
        except Exception as e:
            logger.error("ClinicalTrials agent error: %s", e, exc_info=True)
            return {
                "clinical_trials_response": {"text": f"ClinicalTrials search unavailable: {str(e)}", "source": "ClinicalTrials.gov", "items": []},
                "agents_completed": ["clinical_trials_agent"],
            }

    # ------------------------------------------------------------------ #
    #  Content retrieval                                                   #
    # ------------------------------------------------------------------ #

    def _retrieve_from_graph(self, query, user_id, graph_id, token, resource, content_parts, sources):
        logger.info("Retrieving graph summary for graph_id: %s", graph_id)
        graph_summary = self.answer_from_graph_summaries(
            query=query, user_id=user_id, graph_id=graph_id, token=token, resource=resource
        )
        if not graph_summary:
            return None
        graph_text = (
            graph_summary.get("text", str(graph_summary))
            if isinstance(graph_summary, dict)
            else str(graph_summary)
        )
        if graph_text and not graph_text.startswith("Failed to contact") and not graph_text.startswith("Error"):
            content_parts.append({"source": f"graph:{graph_id}", "content": graph_text})
            sources.append(f"graph:{graph_id}")
            return None
        if graph_text:
            logger.warning("Graph fetch failed for %s: %s", graph_id, graph_text)
            last_topic = None
            try:
                history = self.store.get_context_and_memory(user_id)
                for item in reversed(history):
                    if "annotation_agent" in item.get("context", {}).get("agents_used", []):
                        last_topic = item.get("question")
                        break
            except Exception:
                pass
            confirmation_text = (
                f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                f"Did you mean to ask about your previous annotation: *\"{last_topic}\"*? "
                f"Or would you like to ask a different question?"
                if last_topic else
                f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                f"Please check that the graph exists, or let me know what you'd like to explore."
            )
            return {
                "content_retrieval_response": {"text": confirmation_text, "json_format": None, "sources": []},
                "agents_completed": ["content_retrieval_agent"],
                "stop_pipeline": True,
            }
        return None

    def _retrieve_from_galaxy(self, query, user_id, token, urls, content_parts, sources):
        logger.info("Retrieving Galaxy urls for user: %s", user_id)
        urls_response = self.galaxy_handler.get_galaxy_info(query=query, user_id=user_id, token=token, urls=urls)
        if urls_response:
            urls_text = (
                urls_response.get("text", str(urls_response))
                if isinstance(urls_response, dict)
                else str(urls_response)
            )
            for file in (urls if isinstance(urls, list) else [urls]):
                content_parts.append({"source": f"file:{file}", "content": urls_text})
                sources.append(f"file:{file}")

    def _retrieve_from_rag(self, query, user_id, content_ids, content_parts, sources):
        logger.info("Retrieving RAG content for content_ids: %s", content_ids)
        rag_content = self.rag.get_result_from_rag(query, user_id, content_ids)
        if rag_content:
            rag_text = (
                rag_content.get("text", str(rag_content))
                if isinstance(rag_content, dict)
                else str(rag_content)
            )
            content_parts.append({
                "source": f"content IDs: {', '.join(content_ids)}",
                "content": rag_text,
                "resource": rag_content.get("resource", {}),
            })
            sources.append(f"content IDs: {', '.join(content_ids)}")

    def _content_retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("user_query")
        user_id = state.get("user_id")
        token = state.get("token")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        logger.info("ContentRetrievalAgent called for user: %s", user_id)
        emit_to_user(user=user_id, message="Retrieving relevant content...")

        content_parts: List = []
        sources: List = []

        try:
            if graph_id:
                early_return = self._retrieve_from_graph(
                    query, user_id, graph_id, token, resource, content_parts, sources
                )
                if early_return is not None:
                    return early_return

            if urls:
                self._retrieve_from_galaxy(query, user_id, token, urls, content_parts, sources)

            if content_ids:
                self._retrieve_from_rag(query, user_id, content_ids, content_parts, sources)

            response_dict = {"text": content_parts, "json_format": None, "sources": sources}
            logger.info(
                "Content retrieval prepared %d parts. response=%s", len(content_parts), response_dict
            )
            return {
                "content_retrieval_response": response_dict,
                "agents_completed": ["content_retrieval_agent"],
                "messages": [AIMessage(content="Content retrieval completed")],
            }

        except Exception as e:
            logger.error("Error in ContentRetrievalAgent: %s", str(e), exc_info=True)
            return {
                "content_retrieval_response": {"text": [], "json_format": None, "sources": []},
                "agents_completed": ["content_retrieval_agent"],
                "error": str(e),
            }
