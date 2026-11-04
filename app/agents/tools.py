
import logging
import re
from typing import Any, Dict, Optional

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from app.agents.state import AgentState, ANALYZING_MSG, UNRESOLVED_MARKER
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
            "placeholder. If the user names a gene TOGETHER WITH related items "
            "that belong in the SAME connected graph (its transcripts, variants, "
            "proteins -- e.g. 'annotate FTO with its transcripts X, Y, Z'), that "
            "is ONE call with all of them in a single query string, not one call "
            "per item -- splitting it produces several disconnected graphs "
            "instead of one graph showing how they relate. Only use separate "
            "calls for entities the user asked about as genuinely independent "
            "(no stated relationship between them)."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the entity (or entities) to annotate together as one connected graph -- e.g. 'BRCA1', or 'FTO with its transcripts ENST00000538872, ENST00000546113' when they're related. Name only, no organism/schema wording here."},
            "organism": {
                "type": "string", "enum": ["human", "fly"],
                "description": (
                    "Only set this if the user explicitly names an organism or schema "
                    "(e.g. 'fly schema', 'drosophila', 'human'). Omit entirely otherwise -- "
                    "it will be auto-detected from the gene name."
                ),
            },
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
    def _append_proposal_summary(self, state: AgentState, line: str) -> None:
        """One short, deterministic line (real ids/title/kind, never
        LLM-paraphrased) recording a proposal just made, so a much later
        turn can be told "a hypothesis/annotation like this existed" without
        the orchestrator needing the full, potentially-stale proposal object
        -- see mongo_storage.py's append_proposal_summary."""
        if not line or not state.get("thread_id") or not state.get("user_id"):
            return
        try:
            self.store.append_proposal_summary(state["thread_id"], state["user_id"], line)
        except Exception as e:
            logger.warning(f"Could not append proposal summary: {e}")

    def run_annotate_gene(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_annotation_subgraph", None) is None:
            self._annotation_subgraph = build_annotation_subgraph(self.annotation_graph)

        effective_query = arguments.get("query") or state["user_query"]
        organism_override = arguments.get("organism")
        logger.info(f"annotate_gene: {effective_query} (organism_override={organism_override})")
        emit_to_user(user=state["user_id"], message="Processing your biological query...", thread_id=state.get("thread_id"))

        try:
            result = self._annotation_subgraph.invoke({
                "user_query": f"annotate the gene {effective_query}" if effective_query else state["user_query"],
                "user_id": state["user_id"],
                "query_type": "annotation_biological",
                "organism_override": organism_override,
                "organism_hint_text": state["user_query"],
                "thread_id": state.get("thread_id"),
            })
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.error("Unexpected error in annotate_gene", exc_info=True)
            return {"content": f"annotate_gene failed: {e}", "error": str(e)}

        if result.get("outcome") == ABANDONED:
            logger.info("Annotation confirmation abandoned — the user said something else")
            new_text = result.get("handoff_query") or "(no reply captured)"
            return {"content": f"{UNRESOLVED_MARKER}The pending confirmation wasn't answered — the user said instead: {new_text!r}"}

        response = result.get("annotation_response") or {}
        state_update = {"annotation_response": response}
        if response.get("json_format"):
            state_update["proposals"] = [{"kind": "annotation", "data": response["json_format"]}]
            data = response["json_format"]
            self._append_proposal_summary(
                state,
                f'[annotation] query="{effective_query}" organism={response.get("organism", "human")} '
                f'nodes={len(data.get("nodes") or [])}'
            )
        return {
            "content": response.get("text", "") or "Annotation completed.",
            "state_update": state_update,
            "agent_name": "annotation_agent",
        }

    def run_search_literature(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_literature_graph", None) is None:
            self._literature_graph = build_literature_subgraph(self.rag, self._extract_search_term)

        query = arguments.get("query") or state["user_query"]
        context = (state.get("hypothesis_response") or {}).get("text", "")
        min_year = self._extract_min_year(state["user_query"])
        result = self._literature_graph.invoke({
            "user_query": query,
            "user_id": state["user_id"],
            "thread_id": state.get("thread_id"),
            "content_ids": state.get("content_ids"),
            "sources": [RAG, PUBMED, CLINICAL_TRIALS],
            "search_context": context,
            "min_year": min_year,
            "attached_files": state.get("attached_files"),
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
        emit_to_user(user=state["user_id"], message="Generating hypothesis...", thread_id=state.get("thread_id"))

        try:
            result = self._hypothesis_subgraph.invoke({
                "user_query": query, "user_id": state["user_id"], "token": state.get("token"),
                "thread_id": state.get("thread_id"),
            })
        except GraphBubbleUp:
            raise
        except Exception as e:
            logger.error("Unexpected error in generate_hypothesis", exc_info=True)
            return {"content": UNAVAILABLE_TEXT, "error": str(e)}

        if result.get("outcome") == HYPOTHESIS_ABANDONED:
            new_text = result.get("handoff_query") or "(no reply captured)"
            return {"content": f"{UNRESOLVED_MARKER}The pending GO-term confirmation wasn't answered — the user said instead: {new_text!r}"}

        response = result.get("hypothesis_response") or {}
        if result.get("failed"):
            return {
                "content": response.get("text", UNAVAILABLE_TEXT),
                "error": result.get("error"),
                "state_update": {"hypothesis_response": response},
            }

        state_update = {"hypothesis_response": response}
        if response.get("json_format"):
            state_update["proposals"] = [{"kind": "hypothesis", "data": response["json_format"]}]
            data = response["json_format"]
            self._append_proposal_summary(
                state,
                f'[hypothesis] project="{data.get("project_name")}" (project_id={data.get("project_id")}) '
                f'phenotype={data.get("phenotype")} credible_set={data.get("credible_set_id") or "none"} '
                f'variant={data.get("variant") or "none"} tissue={data.get("tissue") or "none"}'
            )
        return {
            "content": response.get("text", ""),
            "state_update": state_update,
            "agent_name": "hypothesis_agent",
        }

    def run_galaxy_tool(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_galaxy_subgraph", None) is None:
            self._galaxy_subgraph = build_galaxy_subgraph(self.galaxy_handler)

        query = arguments.get("query") or state["user_query"]
        emit_to_user(user=state["user_id"], message="Retrieving Galaxy tools information...", thread_id=state.get("thread_id"))

        result = self._galaxy_subgraph.invoke({
            "user_query": query, "user_id": state["user_id"], "token": state.get("token"),
            "attached_files": state.get("attached_files"),
        })
        response = result.get("galaxy_response") or {}
        return {
            "content": response.get("text", ""),
            "state_update": {"galaxy_response": response},
            "agent_name": "galaxy_agent",
        }

    @staticmethod
    def _galaxy_api_error(response: Dict[str, Any]) -> Optional[str]:
        return response.get("error") if isinstance(response, dict) and response.get("error") else None

    def run_galaxy_search(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        query = arguments.get("query") or state["user_query"]
        kind = arguments.get("kind")
        emit_to_user(user=state["user_id"], message="Searching Galaxy's installed tools...", thread_id=state.get("thread_id"))

        response = self.galaxy_runs_api.search(state.get("token"), query, kind)
        error = self._galaxy_api_error(response)
        if error:
            return {"content": f"Couldn't search Galaxy's tools: {error}"}

        items = response.get("items") if isinstance(response, dict) else response
        items = items or []
        if not items:
            return {"content": f"No installed Galaxy tool or workflow matched '{query}'."}
        lines = [
            f"- {item.get('name', item.get('id', '?'))} ({item.get('type', 'tool')}, id={item.get('id')}, version={item.get('version', '?')})"
            for item in items
        ]
        return {"content": "Matching Galaxy tools/workflows:\n" + "\n".join(lines)}

    def run_galaxy_describe(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        target = {"type": arguments.get("target_type"), "id": arguments.get("target_id")}
        if arguments.get("target_version"):
            target["version"] = arguments["target_version"]
        emit_to_user(user=state["user_id"], message="Reading the Galaxy tool's requirements...", thread_id=state.get("thread_id"))

        response = self.galaxy_runs_api.describe(state.get("token"), state.get("thread_id"), target)
        error = self._galaxy_api_error(response)
        if error:
            return {"content": f"Couldn't read {target.get('id')}'s requirements: {error}"}

        inputs = response.get("inputs") or []
        if not inputs:
            return {"content": f"{response.get('target', {}).get('name', target.get('id'))} has no inputs to report (or all are at Galaxy defaults)."}
        lines = []
        for inp in inputs:
            required = "" if inp.get("when") else " (required unless its branch doesn't apply)"
            match_note = ""
            matches = inp.get("matches") or []
            if matches:
                match_note = " -- already matched in this thread: " + ", ".join(m.get("name", m.get("id", "?")) for m in matches)
            lines.append(f"- {inp.get('label', inp.get('path'))} [{inp.get('type')}]{required}{match_note}")
        return {"content": f"{target.get('id')}'s real inputs:\n" + "\n".join(lines)}

    def run_galaxy_thread_files(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        emit_to_user(user=state["user_id"], message="Checking this thread's Galaxy files...", thread_id=state.get("thread_id"))

        response = self.galaxy_runs_api.thread_files(state.get("token"), state.get("thread_id"))
        error = self._galaxy_api_error(response)
        if error:
            return {"content": f"Couldn't read this thread's Galaxy files: {error}"}

        items = response.get("items") if isinstance(response, dict) else response
        items = items or []
        if not items:
            return {"content": "No files have been uploaded into this thread's Galaxy history yet."}
        lines = [f"- {f.get('name', f.get('id', '?'))} ({f.get('format', '?')}, id={f.get('id')})" for f in items]
        return {"content": "Files already in this thread's Galaxy history:\n" + "\n".join(lines)}

    def run_galaxy_validate(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        target = {"type": arguments.get("target_type"), "id": arguments.get("target_id")}
        if arguments.get("target_version"):
            target["version"] = arguments["target_version"]
        emit_to_user(user=state["user_id"], message="Checking these inputs with Galaxy...", thread_id=state.get("thread_id"))

        response = self.galaxy_runs_api.validate(
            state.get("token"), state.get("thread_id"), target,
            arguments.get("inputs") or {}, arguments.get("parameters"),
        )
        error = self._galaxy_api_error(response)
        if error:
            return {"content": f"Couldn't validate these inputs: {error}"}

        if response.get("valid"):
            return {"content": "Galaxy accepts these inputs -- ready to propose."}
        errors = response.get("errors") or {}
        lines = [f"- {path}: {msg}" for path, msg in errors.items()] if isinstance(errors, dict) else [str(errors)]
        return {"content": "Galaxy rejected these inputs:\n" + "\n".join(lines)}

    def run_biogpt_tool(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if getattr(self, "_biogpt_subgraph", None) is None:
            self._biogpt_subgraph = build_biogpt_subgraph(self.biogpt)

        emit_to_user(user=state["user_id"], message="Analyzing biomedical information...", thread_id=state.get("thread_id"))
        query = arguments.get("query") or state["user_query"]
        result = self._biogpt_subgraph.invoke({"user_query": query, "attached_files": state.get("attached_files")})
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
        emit_to_user(user=user_id, message="Retrieving relevant content...", thread_id=state.get("thread_id"))
        try:
            result = self._content_graph.invoke({
                "user_query": arguments.get("query") or state.get("user_query"),
                "user_id": user_id,
                "token": state.get("token"),
                "graph_id": state.get("graph_id"),
                "urls": state.get("urls"),
                "content_ids": state.get("content_ids"),
                "resource": state.get("resource"),
                "thread_id": state.get("thread_id"),
                "attached_files": state.get("attached_files"),
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


    @staticmethod
    def _extract_min_year(user_query: str) -> Optional[int]:
        q = user_query.lower()

        m = re.search(r"\b(?:since|from|after)\s+(\d{4})\b", q)
        if m:
            return int(m.group(1))

        m = re.search(r"\b(?:last|past|previous)\s+(\d{1,2})\s+years?\b", q)
        if m:
            from datetime import datetime
            return datetime.now().year - int(m.group(1)) + 1

        return None

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

    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id, thread_id=None):
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
                emit_to_user(user=user_id, message=ANALYZING_MSG, thread_id=thread_id)

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query, thread_id=thread_id
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message=ANALYZING_MSG, thread_id=thread_id)
            else:
                return "Invalid resource type specified."

            return {"text": summary_text, "json_format": None, "entity_found": entity_found}

        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {"text": f"Error processing query: {str(e)}", "json_format": None}
