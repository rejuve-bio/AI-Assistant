
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
from app.Galaxy_integration.galaxy_proposal import propose_galaxy_run
from app.prompts.rag_prompts import SEARCH_TERM_EXTRACTION_PROMPT
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
            "placeholder. The query string must carry the FULL scope of what "
            "the user asked for as ONE connected graph, whatever entity types "
            "that involves -- always driven by the actual question, never a "
            "fixed list. This covers both: (a) the user naming specific related "
            "items (its transcripts, variants, proteins -- e.g. 'annotate FTO "
            "with its transcripts X, Y, Z'), and (b) open-ended relationships "
            "the query should discover (e.g. 'map apolpp, then which group it "
            "belongs to and what human diseases it models' -> one query string "
            "mentioning apolpp, its gene group, AND the diseases it models -- "
            "not just 'apolpp' alone). Splitting a connected request into "
            "narrower calls, or dropping part of it, produces a disconnected "
            "or incomplete graph instead of the one graph the user asked for. "
            "Only use separate calls for entities the user asked about as "
            "genuinely independent (no stated relationship between them)."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the FULL request to annotate as one connected graph, in the user's own terms -- e.g. 'BRCA1', 'FTO with its transcripts ENST00000538872, ENST00000546113', or 'apolpp, the gene group it belongs to, and the human diseases it models'. Carry over every entity/relationship the question asked about, not just the first noun. Name only, no organism/schema wording here."},
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
        "description": (
            "Check the assistant's own existing GWAS analysis projects (by "
            "phenotype), and match a named lead variant/locus against the "
            "matched project's credible sets, to PROPOSE the parameters for "
            "a hypothesis -- project, credible set, tissue -- never "
            "generates or runs anything itself. Call this for ANY request "
            "to build/generate a hypothesis for a phenotype, even when NO "
            "variant is named yet (e.g. 'generate a hypothesis about "
            "obesity') -- there is always at least a sample GWAS project to "
            "check against, so the right response is grounded in what that "
            "check actually finds (which project matched, its tissue), "
            "with the tool's own text asking for the missing variant -- "
            "NEVER answer directly from a guess about whether a project "
            "might exist without calling this tool first. Also use this "
            "whenever the question is phrased as working from a GWAS locus "
            "toward a causal gene using the assistant's own sample data, "
            "e.g. 'rs123 is the lead variant in an X locus -- which gene is "
            "it acting through?', 'do the same for rsXXX in <phenotype>'. "
            "This is NOT a literature lookup or a known-biology question "
            "(that's search_literature/annotate_gene) -- it's specifically "
            "about whether the assistant's own stored GWAS projects contain "
            "a matching locus. When this phrasing pattern applies, call "
            "ONLY this tool -- don't also search literature or annotate a "
            "graph for the same request."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the full hypothesis request, in the user's own words, for context"},
            "phenotype": {"type": "string", "description": "the phenotype/condition this is about. If the CURRENT message names one (even if it's the same word used before, e.g. 'arthritis' again), ALWAYS use that exact word -- never substitute a different phenotype/file name recorded in an earlier turn's proposal-summary note just because it's sitting in context. Only carry one forward from an earlier turn when the current message itself doesn't name one at all (e.g. 'same phenotype', 'try again'). Required: never omit, never guess one that wasn't actually named or implied."},
            "variant": {"type": "string", "description": "the lead variant (rsID) named, if any. Omit entirely if none was named -- never invent one."},
            "tissue": {"type": "string", "description": "ONLY set this if the user explicitly asked for a specific tissue instead of letting the strongest one be auto-picked (e.g. 'use adipose tissue instead', 'in CD8 T cells'). Omit entirely otherwise -- the tool auto-selects the project's strongest tissue by default. If the named tissue wasn't actually analyzed for the matched project, the tool falls back to the strongest one and says so."},
        }, "required": ["query", "phenotype"]},
    },
    {
        "name": "list_lead_variants",
        "description": (
            "Get a GWAS project's REAL, analyzed credible-set lead variants "
            "-- the only ones that will actually match when generating a "
            "hypothesis from that project. Call this whenever the user asks "
            "for an example/candidate/real lead variant for a project "
            "already matched earlier in this thread (e.g. 'give me an "
            "example lead variant I can use', 'what variants are available "
            "for this project'). NEVER use search_literature or run_biogpt "
            "to guess a variant for an existing project -- a variant from "
            "general biomedical knowledge is very unlikely to be one of "
            "this specific project's own analyzed credible sets, so it "
            "would fail to match anyway. Only reach for search_literature/"
            "run_biogpt afterward, to describe one of THESE real variants "
            "in more depth once the user has picked one."
        ),
        "parameters": {"type": "object", "properties": {
            "project_id": {"type": "string", "description": "ONLY the project's real database id (e.g. a long hex/ObjectId string like '6a2ec66760333bb84216b4f6'), never a phenotype or project display name like 'obesity' -- that is NOT an id and will fail to look up. If you don't have that exact real id string memorized from an earlier tool result, OMIT this argument entirely; it then auto-resolves to whichever GWAS project was already matched earlier in this thread."},
        }, "required": []},
    },
    {
        "name": "run_galaxy",
        "description": "Look up or run a Galaxy platform bioinformatics tool.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what Galaxy tool/info is needed"},
        }, "required": ["query"]},
    },
    {
        "name": "galaxy_search",
        "description": (
            "Search Galaxy's installed, runnable tools and workflows by "
            "keyword. Call this ONLY when the user wants to run something "
            "in Galaxy but hasn't named an exact tool or workflow (e.g. "
            "'trim my reads', 'what can align sequencing reads for me'). "
            "Returns real candidate targets (id, version, name) to use "
            "with galaxy_describe next. Do NOT call this once a specific "
            "tool or workflow is already named or known -- call "
            "galaxy_describe directly instead. This never reads a target's "
            "inputs, checks files, or validates anything."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the keyword or topic to search for, in the user's own words (e.g. 'trim reads', 'variant calling')"},
            "kind": {"type": "string", "enum": ["tool", "workflow"], "description": "ONLY set this if the user explicitly said which kind they want. Omit entirely otherwise -- search both."},
        }, "required": ["query"]},
    },
    {
        "name": "galaxy_describe",
        "description": (
            "Get a Galaxy tool or workflow's REAL input requirements -- "
            "which inputs are required, their types, any conditional "
            "branches (e.g. choosing single-end vs. paired-end changes "
            "what else is asked), accepted file formats, and which of the "
            "thread's already-uploaded files already match a given input. "
            "Call this as soon as a specific target is known -- named "
            "directly by the user, or chosen from galaxy_search's results "
            "-- and BEFORE asking the user anything about what it needs. "
            "Guessing a tool's parameters instead of reading them here is "
            "never acceptable. Do NOT call this to find a tool by keyword "
            "(that's galaxy_search) or to check whether assembled inputs "
            "are acceptable (that's galaxy_validate)."
        ),
        "parameters": {"type": "object", "properties": {
            "target_type": {"type": "string", "enum": ["tool", "workflow"], "description": "whether this target is a tool or a workflow"},
            "target_id": {"type": "string", "description": "the target's real id -- a tool's full ToolShed id, or a workflow's id -- exactly as named by the user or returned by galaxy_search. Never invent one."},
            "target_version": {"type": "string", "description": "the target's version, if known (from galaxy_search or named by the user). Omit if not known."},
        }, "required": ["target_type", "target_id"]},
    },
    {
        "name": "galaxy_thread_files",
        "description": (
            "List every file already uploaded into THIS thread's Galaxy "
            "history (dataset ids, names, formats) -- Galaxy's own ground "
            "truth of what's available to use as an input. Call this "
            "before asking the user to specify a file for a Galaxy input, "
            "so an already-uploaded file can be matched automatically "
            "instead of asking for one that's already there. Takes no "
            "arguments -- the thread is already known from context."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "galaxy_validate",
        "description": (
            "Check with Galaxy itself whether a tentative set of inputs "
            "for a target tool/workflow would actually be accepted, BEFORE "
            "presenting it as a ready proposal. Call this once inputs have "
            "been assembled from the user's words, galaxy_thread_files' "
            "file matches, and galaxy_describe's defaults. If Galaxy "
            "rejects something, use ITS real error message to ask the "
            "user again -- never invent a reason of your own. Do NOT call "
            "this to discover what inputs exist in the first place (that's "
            "galaxy_describe)."
        ),
        "parameters": {"type": "object", "properties": {
            "target_type": {"type": "string", "enum": ["tool", "workflow"]},
            "target_id": {"type": "string", "description": "the target's real id, exactly as used with galaxy_describe"},
            "target_version": {"type": "string", "description": "the target's version, if known"},
            "inputs": {
                "type": "object",
                "description": (
                    "The tentative inputs to validate, keyed exactly as galaxy_describe named "
                    "them (a tool's keys are '|'-joined paths like 'single_paired|in1'; a "
                    "workflow's are its input labels). A file value is "
                    "{\"src\": \"hda\" or \"hdca\", \"id\": \"<dataset id from galaxy_thread_files>\"}; "
                    "any other value is used as-is."
                ),
            },
            "parameters": {"type": "object", "description": "a workflow's runtime settings by step, keyed as galaxy_describe named them. Omit for a tool."},
        }, "required": ["target_type", "target_id", "inputs"]},
    },
    {
        "name": "propose_galaxy_run",
        "description": (
            "Build the actual Galaxy proposal card for a specific tool/"
            "workflow -- a real, structured card (target, inputs, a plain-"
            "words summary, and anything still unresolved) the user can "
            "review and run, not just a text description. Call "
            "galaxy_describe on this exact target FIRST if its real input "
            "paths aren't already known from this conversation -- "
            "requested_values must be keyed by those exact real paths "
            "(e.g. 'filter_options|length_required'), never invented or "
            "guessed names. Only include a value the user explicitly "
            "named; leave everything else out so it either stays at "
            "Galaxy's own default or is honestly asked for in the reply. "
            "Call this once per turn, after enough is known to build a "
            "useful card -- it already calls describe and validate "
            "itself, so don't call those separately just to build this "
            "same card."
        ),
        "parameters": {"type": "object", "properties": {
            "target_type": {"type": "string", "enum": ["tool", "workflow"]},
            "target_id": {"type": "string", "description": "the target's real id, exactly as returned by galaxy_search or galaxy_describe"},
            "target_version": {"type": "string", "description": "the target's version, if known"},
            "requested_values": {
                "type": "object",
                "description": "settings the user explicitly named, keyed by their exact real path from galaxy_describe (e.g. {\"filter_options|length_required\": 30}). Omit anything not explicitly named -- never guess a value.",
            },
        }, "required": ["target_type", "target_id"]},
    },
    {
        "name": "run_biogpt",
        "description": (
            "Ask the fine-tuned BioGPT model a direct biomedical question, "
            "from its own general training -- never a source of the "
            "assistant's own stored GWAS project data. Do NOT use this to "
            "name a 'real'/'example' lead variant for one of the "
            "assistant's own GWAS projects (use list_lead_variants for "
            "that instead) -- a variant BioGPT names from general "
            "knowledge is very unlikely to be one of that project's own "
            "analyzed credible sets."
        ),
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
        "name": "kg_overview",
        "description": (
            "Get overall statistics for the WHOLE knowledge graph database -- total "
            "node/edge counts, node and edge types, and similar. This is a RARE, "
            "last-resort tool -- almost every question that mentions 'the graph' is "
            "actually about a specific annotation/hypothesis graph already attached "
            "to this conversation, and that case MUST use retrieve_content instead, "
            "never this tool. Only call kg_overview when the question is unmistakably "
            "about the database as a whole, with nothing specific attached -- e.g. "
            "literally 'how many nodes are in the knowledge base', asked with no "
            "annotation/hypothesis graph anywhere in this conversation's context. "
            "If in doubt, prefer retrieve_content."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "the question about the overall knowledge graph"},
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
    "list_lead_variants": "run_list_lead_variants",
    "run_galaxy": "run_galaxy_tool",
    "galaxy_search": "run_galaxy_search",
    "galaxy_describe": "run_galaxy_describe",
    "galaxy_thread_files": "run_galaxy_thread_files",
    "galaxy_validate": "run_galaxy_validate",
    "propose_galaxy_run": "run_propose_galaxy",
    "run_biogpt": "run_biogpt_tool",
    "retrieve_content": "run_retrieve_content",
    "kg_overview": "run_kg_overview",
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


_GROUNDING_STOPWORDS = {
    "the", "a", "an", "and", "or", "its", "it", "to", "of", "with", "in",
    "on", "for", "is", "are", "that", "which", "what", "this", "then",
    "gene", "genes", "protein", "proteins", "group", "groups", "belongs",
    "belong", "model", "models", "modeled", "disease", "diseases", "human",
    "humans", "association", "associated", "associations", "late",
    "onset", "related", "relationship", "relationships", "involved",
    "role", "roles", "map", "mapping", "annotate", "annotation",
}
_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def _entity_tokens(query: str) -> list:
    return [
        w for w in _ENTITY_TOKEN_RE.findall(query)
        if len(w) >= 3 and w.lower() not in _GROUNDING_STOPWORDS
    ]


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
    haystack_lower = haystack.lower()
    tokens = _entity_tokens(query)
    if tokens:
        ungrounded = [t for t in tokens if t.lower() not in haystack_lower]
        if ungrounded:
            return (
                f"{', '.join(repr(t) for t in ungrounded)} hasn't actually been "
                f"established anywhere in this conversation yet -- {tool_name} "
                "needs a value that's either named by the user or came back "
                "from a prior tool's real result."
            )
    elif query.lower() not in haystack_lower:
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
                "propose_only": state.get("propose_only", False),
                "thread_species": state.get("thread_species"),
                "thread_sources": state.get("thread_sources"),
                "attached_files": state.get("attached_files"),
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
            state_update["proposals"] = [{
                "kind": "annotation", "data": response["json_format"],
                "tool": "annotate_gene", "arguments": arguments,
            }]
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
                "propose_only": state.get("propose_only", False),
                "requested_tissue": arguments.get("tissue"),
                "requested_phenotype": arguments.get("phenotype"),
                "requested_variant": arguments.get("variant"),
                "attached_files": state.get("attached_files"),
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
            kind = response.get("kind", "hypothesis")
            state_update["proposals"] = [{
                "kind": kind, "data": response["json_format"],
                "tool": "generate_hypothesis", "arguments": arguments,
            }]
            data = response["json_format"]
            if kind == "hypothesis-project":
                rf = response.get("recommended_file") or {}
                self._append_proposal_summary(
                    state,
                    f'[hypothesis-project] recommended gwas_file={data.get("gwas_file")} '
                    f'project_name="{data.get("project_name")}" source={rf.get("source", "?")} '
                    f'sex={rf.get("sex", "?")} genome_build={rf.get("genome_build", "?")} '
                    f'(catalog file\'s own condition label, NOT a phenotype the user searched for: {data.get("phenotype")})'
                )
            else:
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

    def run_list_lead_variants(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        auto_project_id = None
        for entry in reversed(state.get("proposals") or []):
            data = entry.get("data") or {}
            if entry.get("kind") == "hypothesis" and data.get("project_id"):
                auto_project_id = data["project_id"]
                break

        project_id = arguments.get("project_id") or auto_project_id
        if not project_id:
            return {"content": "No GWAS project has been matched in this thread yet -- call generate_hypothesis with a phenotype first."}

        emit_to_user(user=state["user_id"], message="Reading the project's real credible sets...", thread_id=state.get("thread_id"))
        result = self.hypothesis_generation.list_credible_set_variants(state.get("token"), project_id)
        if result.get("error") and auto_project_id and auto_project_id != project_id:
            result = self.hypothesis_generation.list_credible_set_variants(state.get("token"), auto_project_id)
            project_id = auto_project_id
        if result.get("error"):
            return {"content": result["error"]}

        variants = result.get("variants") or []
        if not variants:
            return {"content": f"Project {project_id} doesn't have a lead variant recorded on any of its credible sets yet."}

        total = len(variants)
        top = variants[:10]
        lines = "\n".join(
            f"- {v['variant']} (credible_set_id={v['credible_set_id']}, "
            f"-log10(p)={v['log_pvalue']:.1f})" if v.get("log_pvalue") is not None
            else f"- {v['variant']} (credible_set_id={v['credible_set_id']})"
            for v in top
        )
        more_note = f" (showing the {len(top)} most significant of {total} total)" if total > len(top) else ""
        return {
            "content": (
                f"Project {project_id}'s own analyzed credible sets have these real lead "
                f"variants, most significant first{more_note} -- these are the only ones "
                f"that will actually match when generating a hypothesis from this "
                f"project:\n{lines}"
            ),
        }

    def run_galaxy_tool(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        return self.run_galaxy_search(arguments, state)

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

    def run_propose_galaxy(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        emit_to_user(user=state["user_id"], message="Building the Galaxy proposal...", thread_id=state.get("thread_id"))

        response = propose_galaxy_run(
            self.galaxy_runs_api, state.get("token"), state.get("thread_id"),
            arguments.get("target_type"), arguments.get("target_id"),
            arguments.get("target_version"), arguments.get("requested_values") or {},
        )

        state_update = {}
        if response.get("json_format"):
            state_update["proposals"] = [{
                "kind": "galaxy", "data": response["json_format"],
                "tool": "propose_galaxy_run",
                "arguments": {
                    "target_type": arguments.get("target_type"),
                    "target_id": arguments.get("target_id"),
                    "target_version": arguments.get("target_version"),
                    "requested_values": arguments.get("requested_values") or {},
                },
            }]
        return {"content": response.get("text", ""), "state_update": state_update}

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
        prompt = SEARCH_TERM_EXTRACTION_PROMPT.format(user_query=user_query, context_line=context_line)
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

    def run_kg_overview(self, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        query = arguments.get("query") or state["user_query"]
        logger.info(f"kg_overview: {query}")
        emit_to_user(user=state["user_id"], message="Looking up the knowledge graph's overall statistics...", thread_id=state.get("thread_id"))

        result = self.graph_summarizer.get_graph_overview(token=state.get("token"), query=query)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return {"content": text or "I couldn't retrieve the knowledge graph's statistics."}
