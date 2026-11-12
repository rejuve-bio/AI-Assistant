
import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.state import AgentState
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


class AggregationMixin:
    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        user_id = state.get("user_id")
        messages = state.get("messages", [])
        last = messages[-1] if messages else None

        if isinstance(last, AIMessage) and (last.content or "").strip():
            text = last.content
        else:
            recent = []
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    break
                if isinstance(m, ToolMessage):
                    recent.append(m.content)
            text = "\n\n".join(reversed(recent)) if recent else "I don't have a response for that."

        response = {"text": text, "json_format": None}
        baseline = state.get("proposals_baseline") or 0
        response["proposals"] = (state.get("proposals") or [])[baseline:]

        annotation_resp = state.get("annotation_response")
        if annotation_resp and annotation_resp.get("json_format"):
            response["json_format"] = annotation_resp["json_format"]
            response["organism"] = annotation_resp.get("organism")

        hypothesis_resp = state.get("hypothesis_response") or {}
        if hypothesis_resp.get("json_format") and not response["json_format"]:
            response["json_format"] = hypothesis_resp["json_format"]
        if isinstance(hypothesis_resp.get("resource"), dict):
            response["resource"] = hypothesis_resp["resource"]
        elif state.get("resource"):
            response["resource"] = state.get("resource")

        if state.get("error"):
            response.setdefault("status", "error")

        logger.info(
            f"[STEP] turn finished for thread {state.get('thread_id')}: "
            f"{len(response.get('proposals') or [])} new proposal(s), "
            f"text preview: {response.get('text', '')[:150]!r}"
        )
        logger.info(f"Finalizing response for user: {user_id}")
        emit_to_user(user=user_id, message=response, status="completed", thread_id=state.get("thread_id"))
        return {"response": response}
