
import logging
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt, Command

from app.agents.state import AgentState, ANNOTATION_DB
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


class ConfirmationMixin:
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


