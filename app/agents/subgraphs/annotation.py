
import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents.state import ANNOTATION_DB

logger = logging.getLogger(__name__)

BUILD = "build"
CONFIRM = "confirm"

RESOLVED = "resolved"
ABANDONED = "abandoned"


class AnnotationState(TypedDict, total=False):
    user_query: str
    user_id: str
    query_type: str
    organism_override: Optional[str]
    organism_hint_text: Optional[str]
    pending: Optional[Dict[str, Any]]
    confirmation_text: str
    annotation_response: Optional[Dict[str, Any]]
    outcome: Optional[str]
    handoff_query: Optional[str]
    agents_completed: Annotated[List[str], operator.add]


def confirmation_options(pending: Dict[str, Any], alternative_values) -> List[Dict[str, str]]:

    if pending.get("alternatives_shown"):
        options = [{"label": f"Use {v}", "value": v} for v in alternative_values(pending)]
        options.append({"label": "No, skip it", "value": "reject"})
        return options
    return [
        {"label": "Yes, use the suggested match", "value": "confirm"},
        {"label": "No, skip it", "value": "reject"},
        {"label": "Show other matches", "value": "show_alternatives"},
    ]


def build_annotation_subgraph(annotation_graph):

    def named_candidate_in_reply(user_reply, data: Dict[str, Any]) -> Optional[str]:
        if not isinstance(user_reply, str) or not data:
            return None

        candidates = set(annotation_graph._alternative_candidate_values(data) or [])
        for entry in data.get("unconfirmed", []) or []:
            if entry.get("suggestion"):
                candidates.add(entry["suggestion"])
        if not candidates:
            return None

        tokens = {t.strip(" \t\n.,;:!?'\"()[]").lower() for t in user_reply.split()}
        named = [c for c in candidates if c.lower() in tokens]
        if len(named) == 1:
            logger.info(f"Reply named candidate '{named[0]}' — applying it")
            return named[0]
        return None

    def build_node(state: AnnotationState) -> Dict[str, Any]:
        result = annotation_graph.process_annotation_query(
            query=state["user_query"],
            user_id=state["user_id"],
            query_type=state.get("query_type", "annotation_biological"),
            organism_override=state.get("organism_override"),
            organism_hint_text=state.get("organism_hint_text"),
        )
        logger.info(f"Pipeline response: {result}")

        if result.get("needs_confirmation"):
            return {
                "pending": result.get("pending") or {},
                "confirmation_text": result.get("confirmation_text", ""),
            }

        if result.get("success"):
            return {
                "annotation_response": {
                    "text": result.get("summary") or "",
                    "json_format": result.get("json_format"),
                    "validation_report": result.get("validation_report", {}),
                    "organism": result.get("organism", "human"),
                    "source": ANNOTATION_DB,
                },
                "outcome": RESOLVED,
                "agents_completed": ["annotation_agent"],
            }

        error = result.get("error", "Unknown error")
        logger.error(f"Annotation pipeline failed: {error}")
        return {
            "annotation_response": {
                "text": f"Error: {error}", "json_format": None, "source": ANNOTATION_DB
            },
            "outcome": RESOLVED,
            "agents_completed": ["annotation_agent"],
        }

    def confirm_node(state: AnnotationState) -> Dict[str, Any]:
        pending = state.get("pending") or {}
        user_reply = interrupt({
            "confirmation_text": state.get("confirmation_text", ""),
            "options": confirmation_options(
                pending, annotation_graph._alternative_candidate_values
            ),
            "allow_free_text": True,
        })

        override_value = None
        if isinstance(user_reply, dict) and user_reply.get("type") == "direct":
            value = user_reply.get("value")
            if value in ("confirm", "reject", "show_alternatives"):
                verdict = value
            else:
                verdict, override_value = "confirm", value
        else:
            verdict = annotation_graph._classify_confirmation(user_reply)
            if verdict == "confirm":
                override_value = named_candidate_in_reply(user_reply, pending)

        if verdict in ("confirm", "reject"):
            resolved = annotation_graph._apply_pending_substitutions(
                pending.get("json", {}),
                apply=(verdict == "confirm"),
                override_value=override_value,
            )
            return {
                "annotation_response": {
                    "text": annotation_graph._describe_annotation_result(
                        state["user_query"], resolved
                    ),
                    "json_format": resolved,
                    "organism": pending.get("organism", "human"),
                    "source": ANNOTATION_DB,
                },
                "pending": None,
                "outcome": RESOLVED,
                "agents_completed": ["annotation_agent"],
            }

        if verdict == "show_alternatives":
            return {
                "confirmation_text": annotation_graph._build_alternatives_text(pending),
                "pending": {**pending, "alternatives_shown": True},
            }

        logger.info("Annotation confirmation abandoned for an unrelated new query")
        return {
            "pending": None,
            "outcome": ABANDONED,
            "handoff_query": user_reply if isinstance(user_reply, str) else None,
        }

    def after_build(state: AnnotationState) -> str:
        return CONFIRM if state.get("pending") else END

    def after_confirm(state: AnnotationState) -> str:
        return CONFIRM if state.get("pending") else END

    graph = StateGraph(AnnotationState)
    graph.add_node(BUILD, build_node)
    graph.add_node(CONFIRM, confirm_node)
    graph.set_entry_point(BUILD)
    graph.add_conditional_edges(BUILD, after_build, {CONFIRM: CONFIRM, END: END})
    graph.add_conditional_edges(CONFIRM, after_confirm, {CONFIRM: CONFIRM, END: END})

    return graph.compile()
