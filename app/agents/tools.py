
import logging
import re
from typing import Any, Dict, Optional

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from app.agents.state import AgentState, ANALYZING_MSG
from app.agents.subgraphs.annotation import ABANDONED, build_annotation_subgraph
from app.agents.subgraphs.biogpt import build_biogpt_subgraph
from app.agents.subgraphs.content_retrieval import build_content_retrieval_subgraph
from app.agents.subgraphs.galaxy import build_galaxy_subgraph
from app.agents.subgraphs.hypothesis import (
    ABANDONED as HYPOTHESIS_ABANDONED,
    UNAVAILABLE_TEXT,
    build_hypothesis_subgraph,
)
from app.agents.subgraphs.literature import CLINICAL_TRIALS, PUBMED, RAG, build_literature_subgraph
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


TOOL_SPECS = [
    {
        "name": "annotate_gene",
        "description": (
            "Build a structured annotation graph for a specific gene or other "
            "biological entity already named in the conversation -- either by "
            "the user directly, or turned up by a prior tool's result. Only "
            "call this once you have a real entity name; never guess or use a "
            "placeholder."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the exact gene symbol or entity to annotate, e.g. 'BRCA1'"},
        }, "required": ["query"]},
        "requires_grounded_query": True,
    },
    {
        "name": "search_literature",
        "description": "Search the knowledge base, PubMed, and ClinicalTrials.gov for papers and trials matching a topic.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the search topic"},
        }, "required": ["query"]},
    },
    {
        "name": "generate_hypothesis",
        "description": "Generate a biological hypothesis from a genetic variant and tissue (GWAS-style enrichment).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the hypothesis request, naming the variant/tissue if known"},
        }, "required": ["query"]},
    },
    {
        "name": "run_galaxy",
        "description": "Look up or run a Galaxy platform bioinformatics tool.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what Galaxy tool/info is needed"},
        }, "required": ["query"]},
    },
    {
        "name": "run_biogpt",
        "description": "Ask the fine-tuned BioGPT model a direct biomedical question.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the biomedical question"},
        }, "required": ["query"]},
    },
    {
        "name": "retrieve_content",
        "description": "Retrieve content already attached to this conversation -- an existing annotation/hypothesis graph, uploaded files, or URLs.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what's being asked about the attached content"},
        }, "required": ["query"]},
    },
    {
        "name": "ask_human",
        "description": (
            "Ask the user a clarifying question. Use this whenever several "
            "options are equally plausible and nothing so far clearly picks "
            "one -- never guess in that situation, and never ask a question "
            "as your own plain-text answer; always call this tool instead."
        ),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
        }, "required": ["question"]},
    },
]

TOOL_RUNNERS = {
    "annotate_gene": "run_annotate_gene",
    "search_literature": "run_search_literature",
    "generate_hypothesis": "run_generate_hypothesis",
    "run_galaxy": "run_galaxy_tool",
    "run_biogpt": "run_biogpt_tool",
    "retrieve_content": "run_retrieve_content",
    "ask_human": "run_ask_human",
}

_SPEC_BY_NAME = {spec["name"]: spec for spec in TOOL_SPECS}


def openai_tool_defs() -> list:
    return [
        {"type": "function", "function": {
            "name": spec["name"], "description": spec["description"],
            "parameters": spec["parameters"],
        }}
        for spec in TOOL_SPECS
    ]


_PLACEHOLDER_WORDS = {"tbd", "placeholder", "unknown", "n/a", "none", "xxx", "todo"}
_TEMPLATE_TOKEN = re.compile(r"^[A-Z][A-Z_]{2,}$")


def _looks_like_placeholder(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    if v.lower() in _PLACEHOLDER_WORDS:
        return True
    return bool(_TEMPLATE_TOKEN.match(v) and "_" in v)


def grounding_violation(tool_name: str, arguments: Dict[str, Any], state: AgentState) -> Optional[str]:
    """None if the call is fine; otherwise a human-readable reason it was
    rejected, to hand back to the orchestrator as a corrective ToolMessage."""
    spec = _SPEC_BY_NAME.get(tool_name)
    query = arguments.get("query")
    if not spec or not spec.get("requires_grounded_query") or not isinstance(query, str):
        return None

    if _looks_like_placeholder(query):
        return (
            f"'{query}' looks like a placeholder, not a real value. "
            f"{tool_name} needs an actual, already-established entity name -- "
            "call whatever tool would discover it first, then use its real result."
        )

    haystack = (state.get("user_query") or "") + "\n" + "\n".join(
        (m.content if isinstance(getattr(m, "content", None), str) else str(getattr(m, "content", "")))
        for m in state.get("messages", [])
    )
    if query.lower() not in haystack.lower():
        return (
            f"'{query}' hasn't actually been established anywhere in this "
            f"conversation yet -- {tool_name} needs a value that's either "
            "named by the user or came back from a prior tool's real result."
        )
    return None


class ToolsMixin:
    def run_annotate_gene(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_annotation_subgraph", None) is None:
            self._annotation_subgraph = build_annotation_subgraph(self.annotation_graph)

        effective_query = arguments.get("query") or state["user_query"]
        logger.info(f"annotate_gene: {effective_query}")
        emit_to_user(user=state["user_id"], message="Processing your biological query...")

        try:
            result = self._annotation_subgraph.invoke({
                "user_query": f"annotate the gene {effective_query}" if effective_query else state["user_query"],
                "user_id": state["user_id"],
                "query_type": "annotation_biological",
            })
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.error("Unexpected error in annotate_gene", exc_info=True)
            return {"content": f"annotate_gene failed: {e}", "error": str(e)}

        if result.get("outcome") == ABANDONED:
            logger.info("Annotation confirmation abandoned — the user said something else")
            new_text = result.get("handoff_query") or "(no reply captured)"
            return {"content": f"The pending confirmation wasn't answered — the user said instead: {new_text!r}"}

        response = result.get("annotation_response") or {}
        return {
            "content": response.get("text", "") or "Annotation completed.",
            "state_update": {"annotation_response": response},
            "agent_name": "annotation_agent",
        }

    def run_search_literature(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_literature_graph", None) is None:
            self._literature_graph = build_literature_subgraph(self.rag, self._extract_search_term)

        query = arguments.get("query") or state["user_query"]
        context = (state.get("hypothesis_response") or {}).get("text", "")
        result = self._literature_graph.invoke({
            "user_query": query,
            "user_id": state["user_id"],
            "content_ids": state.get("content_ids"),
            "sources": [RAG, PUBMED, CLINICAL_TRIALS],
            "search_context": context,
        })

        state_update, parts = {}, []
        for key in ("rag_response", "pubmed_response", "clinical_trials_response"):
            if result.get(key):
                state_update[key] = result[key]
                text = result[key].get("text", "")
                if text:
                    parts.append(text)

        return {
            "content": "\n\n".join(parts) if parts else "No relevant literature found.",
            "state_update": state_update,
            "agent_name": "literature",
        }

    def run_generate_hypothesis(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_hypothesis_subgraph", None) is None:
            self._hypothesis_subgraph = build_hypothesis_subgraph(self.hypothesis_generation)

        query = arguments.get("query") or state["user_query"]
        emit_to_user(user=state["user_id"], message="Generating hypothesis...")

        try:
            result = self._hypothesis_subgraph.invoke({
                "user_query": query, "user_id": state["user_id"], "token": state.get("token"),
            })
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.error("Unexpected error in generate_hypothesis", exc_info=True)
            return {"content": UNAVAILABLE_TEXT, "error": str(e)}

        if result.get("outcome") == HYPOTHESIS_ABANDONED:
            new_text = result.get("handoff_query") or "(no reply captured)"
            return {"content": f"The pending GO-term confirmation wasn't answered — the user said instead: {new_text!r}"}

        response = result.get("hypothesis_response") or {}
        if result.get("failed"):
            return {
                "content": response.get("text", UNAVAILABLE_TEXT),
                "error": result.get("error"),
                "state_update": {"hypothesis_response": response},
            }

        return {
            "content": response.get("text", ""),
            "state_update": {"hypothesis_response": response},
            "agent_name": "hypothesis_agent",
        }

    def run_galaxy_tool(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_galaxy_subgraph", None) is None:
            self._galaxy_subgraph = build_galaxy_subgraph(self.galaxy_handler)

        query = arguments.get("query") or state["user_query"]
        emit_to_user(user=state["user_id"], message="Retrieving Galaxy tools information...")

        result = self._galaxy_subgraph.invoke({
            "user_query": query, "user_id": state["user_id"], "token": state.get("token"),
        })
        response = result.get("galaxy_response") or {}
        return {
            "content": response.get("text", ""),
            "state_update": {"galaxy_response": response},
            "agent_name": "galaxy_agent",
        }

    def run_biogpt_tool(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_biogpt_subgraph", None) is None:
            self._biogpt_subgraph = build_biogpt_subgraph(self.biogpt)

        emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
        query = arguments.get("query") or state["user_query"]
        result = self._biogpt_subgraph.invoke({"user_query": query})
        response = result.get("biogpt_response") or {}
        return {
            "content": response.get("text", ""),
            "state_update": {"biogpt_response": response},
            "agent_name": "biogpt_agent",
        }

    def run_retrieve_content(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_content_graph", None) is None:
            self._content_graph = build_content_retrieval_subgraph(
                self.rag, self.answer_from_graph_summaries, self.store
            )

        user_id = state["user_id"]
        emit_to_user(user=user_id, message="Retrieving relevant content...")
        try:
            result = self._content_graph.invoke({
                "user_query": arguments.get("query") or state.get("user_query"),
                "user_id": user_id,
                "token": state.get("token"),
                "graph_id": state.get("graph_id"),
                "urls": state.get("urls"),
                "content_ids": state.get("content_ids"),
                "resource": state.get("resource"),
            })
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.error(f"Error in retrieve_content: {e}", exc_info=True)
            return {"content": "Content retrieval failed.", "error": str(e)}

        if result.get("stop_response"):
            stop = result["stop_response"]
            return {
                "content": stop.get("text", ""),
                "state_update": {"content_retrieval_response": stop},
                "agent_name": "content_retrieval_agent",
            }

        parts = result.get("content_parts", [])
        response = {"text": parts, "json_format": None, "sources": result.get("sources", [])}
        combined_text = "\n\n".join(
            p.get("content", "") for p in parts if isinstance(p, dict) and p.get("content")
        )
        if result.get("graph_covers_query"):
            combined_text = f"(this existing graph already answers the question)\n{combined_text}"

        return {
            "content": combined_text or "No relevant content found.",
            "state_update": {"content_retrieval_response": response},
            "agent_name": "content_retrieval_agent",
        }

    def run_ask_human(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        question = arguments.get("question") or "Could you clarify what you'd like me to do?"
        answer = interrupt({"confirmation_text": question, "options": [], "allow_free_text": True})
        return {"content": str(answer)}


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
            logger.info(f"Extracted search term: '{term}'")
            return term if term else user_query
        except Exception:
            return user_query

    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
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

            return {"text": summary_text, "json_format": None, "entity_found": entity_found}

        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {"text": f"Error processing query: {str(e)}", "json_format": None}
