
import logging
from typing import Any, Dict, Optional

from langgraph.types import Command

from app.agents.state import AgentState
from app.socket_manager import emit_to_user

logger = logging.getLogger(__name__)


class ConfirmationMixin:








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

        question = self._pending_question(state)
        options = question.get("options", [])
        valid_values = {opt["value"] for opt in options}
        if resume_value not in valid_values:
            logger.warning(f"Rejected stale/invalid resume value '{resume_value}' — valid: {valid_values}")
            return {
                "text": "That option isn't available anymore. Here's what's currently available:",
                "json_format": None,
                "needs_confirmation": True,
                "confirmation": {"options": options, "allow_free_text": True},
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


