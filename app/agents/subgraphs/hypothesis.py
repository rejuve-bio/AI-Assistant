
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents.proposal_requirements import FILE_FORMAT_REQUIREMENTS

logger = logging.getLogger(__name__)

GENERATE = "generate"
CONFIRM = "confirm"
ABANDONED = "abandoned"
UNAVAILABLE_TEXT = (
    "The hypothesis service is not returning any results at the moment. "
    "There is nothing I can help with for this request."
)


class HypothesisState(TypedDict, total=False):
    user_query: str
    user_id: str
    thread_id: Optional[str]
    token: str
    pending: Optional[Dict[str, Any]]
    confirmation_text: str
    hypothesis_response: Optional[Dict[str, Any]]
    succeeded: bool
    failed: bool
    outcome: Optional[str]
    handoff_query: Optional[str]
    error: Optional[str]
    attached_files: Optional[List[Dict[str, str]]]
    propose_only: Optional[bool]
    requested_tissue: Optional[str]
    requested_phenotype: Optional[str]
    requested_variant: Optional[str]


def _go_term_options(go_terms: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    options = [
        {
            "label": f"{g.get('name')} (p={g.get('p', 'N/A')}, rank {g.get('rank', '?')})",
            "value": g.get("id"),
        }
        for g in go_terms
    ]
    options.append({"label": "Use the most significant one (lowest p-value)", "value": "auto"})
    return options


def _named_go_term_in_reply(user_reply: Any, go_terms: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(user_reply, str):
        return None
    reply = user_reply.strip().lower()
    reply_words = set(reply.split())

    scored = []
    for g in go_terms:
        gid, name = (g.get("id") or "").lower(), (g.get("name") or "").lower()
        if reply == gid or (name and name in reply):
            scored.append((1.0, g["id"]))
            continue
        name_words = set(name.split())
        if name_words:
            overlap = len(name_words & reply_words) / len(name_words)
            if overlap >= 0.5:
                scored.append((overlap, g["id"]))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _resolve_attached_gwas(attached_files: Optional[List[Dict[str, str]]]) -> Optional[str]:
    """Checks any attached file(s) against the hypothesis service's real,
    confirmed GWAS-file format requirement. Never silently drops a
    mismatched file and never guesses at using one -- actually generating a
    hypothesis from a user's own uploaded data isn't built yet regardless,
    so every attached file here is only ever acknowledged honestly. No
    per-file bookkeeping beyond that: attached_files is itself a
    checkpointed, accumulating field (state.py), so the orchestrator already
    sees it alongside the full conversation history and can tell from
    context what's already been addressed."""
    if not attached_files:
        return None

    accepted = FILE_FORMAT_REQUIREMENTS.get("hypothesis", {}).get("accepted_extensions") or []
    mismatched = [
        f.get("name", "") for f in attached_files
        if f.get("name") and accepted and not any(f["name"].lower().endswith(ext) for ext in accepted)
    ]

    if mismatched:
        formats_str = ", ".join(accepted)
        return (
            f"The attached file(s) ({', '.join(mismatched)}) aren't in a format the "
            f"hypothesis service accepts for a GWAS file ({formats_str}) -- matching "
            "against the existing sample project data below instead."
        )
    return (
        "The attached file is in a format the hypothesis service accepts for a GWAS "
        "file, but generating a hypothesis from your own uploaded data isn't built yet "
        "-- matching against the existing sample project data below instead."
    )


def build_hypothesis_subgraph(hypothesis_generation):

    def generate_node(state: HypothesisState) -> Dict[str, Any]:
        if state.get("propose_only"):
            attached_note = _resolve_attached_gwas(state.get("attached_files"))
            try:
                response = hypothesis_generation.propose_hypothesis(
                    token=state.get("token"),
                    user_query=state["user_query"],
                    user_id=state["user_id"],
                    thread_id=state.get("thread_id"),
                    requested_tissue=state.get("requested_tissue"),
                    requested_phenotype=state.get("requested_phenotype"),
                    requested_variant=state.get("requested_variant"),
                    attached_file_note=attached_note,
                )
            except Exception as e:
                logger.error("Error in hypothesis propose", exc_info=True)
                return {
                    "hypothesis_response": {
                        "text": UNAVAILABLE_TEXT,
                        "resource": None,
                        "status": "failed",
                        "reason": str(e),
                    },
                    "failed": True,
                    "error": str(e),
                }
            return {"hypothesis_response": response, "succeeded": True}

        '''
        try:
            response = hypothesis_generation.generate_hypothesis(
                token=state.get("token"),
                user_query=state["user_query"],
                user_id=state["user_id"],
                thread_id=state.get("thread_id"),
            )
        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "hypothesis_response": {
                    "text": UNAVAILABLE_TEXT,
                    "resource": None,
                    "status": "failed",
                    "reason": str(e),
                },
                "failed": True,
                "error": str(e),
            }

        if response.get("go_terms"):
            go_terms = response["go_terms"]
            names = ", ".join(g.get("name", g.get("id")) for g in go_terms)
            return {
                "pending": response,
                "confirmation_text": (
                    f"I found {len(go_terms)} possible mechanisms this variant "
                    f"could act through: {names}. Which one should I build the "
                    f"hypothesis from?"
                ),
            }

        succeeded = (
            isinstance(response.get("resource"), dict)
            and response["resource"].get("type") == "hypothesis"
        )
        return {"hypothesis_response": response, "succeeded": succeeded}
        '''
        return {
            "hypothesis_response": {"text": UNAVAILABLE_TEXT, "resource": None, "status": "failed"},
            "failed": True,
            "error": "non-propose_only hypothesis generation is disabled",
        }

    def confirm_node(state: HypothesisState) -> Dict[str, Any]:
        '''
        pending = state.get("pending") or {}
        go_terms = pending.get("go_terms", [])

        user_reply = interrupt({
            "confirmation_text": state.get("confirmation_text", ""),
            "options": _go_term_options(go_terms),
            "allow_free_text": True,
        })

        if isinstance(user_reply, dict) and user_reply.get("type") == "direct":
            go_id = user_reply.get("value")
        else:
            go_id = _named_go_term_in_reply(user_reply, go_terms)

        if go_id == "auto":
            go_id = min(go_terms, key=lambda g: g.get("p", float("inf"))).get("id")

        valid_ids = {g.get("id") for g in go_terms}
        if go_id not in valid_ids:
            logger.info("Hypothesis GO-term confirmation abandoned for an unrelated new query")
            return {
                "pending": None,
                "outcome": ABANDONED,
                "handoff_query": user_reply if isinstance(user_reply, str) else None,
            }

        response = hypothesis_generation._finish_enrichment(
            token=state.get("token"),
            hypothesis_id=pending.get("hypothesis_id"),
            enrich_id=pending.get("enrich_id"),
            go_id=go_id,
            user_id=state["user_id"],
            thread_id=state.get("thread_id"),
        )
        succeeded = isinstance(response.get("resource"), dict) and response["resource"].get("type") == "hypothesis"
        return {
            "hypothesis_response": response,
            "pending": None,
            "outcome": None,
            "succeeded": succeeded,
        }
        '''
        return {"pending": None, "outcome": ABANDONED, "handoff_query": None}

    def after_generate(state: HypothesisState) -> str:
        return CONFIRM if state.get("pending") else END

    graph = StateGraph(HypothesisState)
    graph.add_node(GENERATE, generate_node)
    graph.add_node(CONFIRM, confirm_node)
    graph.set_entry_point(GENERATE)
    graph.add_conditional_edges(GENERATE, after_generate, {CONFIRM: CONFIRM, END: END})
    graph.add_edge(CONFIRM, END)
    return graph.compile()
