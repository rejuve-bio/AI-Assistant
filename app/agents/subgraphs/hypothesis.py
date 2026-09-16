
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

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
    token: str
    pending: Optional[Dict[str, Any]]
    confirmation_text: str
    hypothesis_response: Optional[Dict[str, Any]]
    succeeded: bool
    failed: bool
    outcome: Optional[str]
    handoff_query: Optional[str]
    error: Optional[str]


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
        return None  # tied -- ambiguous, not a clear match
    return scored[0][1]


def build_hypothesis_subgraph(hypothesis_generation):

    def generate_node(state: HypothesisState) -> Dict[str, Any]:
        try:
            response = hypothesis_generation.generate_hypothesis(
                token=state.get("token"),
                user_query=state["user_query"],
                user_id=state["user_id"],
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

    def confirm_node(state: HypothesisState) -> Dict[str, Any]:
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
        )
        succeeded = isinstance(response.get("resource"), dict) and response["resource"].get("type") == "hypothesis"
        return {
            "hypothesis_response": response,
            "pending": None,
            "outcome": None,
            "succeeded": succeeded,
        }

    def after_generate(state: HypothesisState) -> str:
        return CONFIRM if state.get("pending") else END

    graph = StateGraph(HypothesisState)
    graph.add_node(GENERATE, generate_node)
    graph.add_node(CONFIRM, confirm_node)
    graph.set_entry_point(GENERATE)
    graph.add_conditional_edges(GENERATE, after_generate, {CONFIRM: CONFIRM, END: END})
    graph.add_edge(CONFIRM, END)
    return graph.compile()
