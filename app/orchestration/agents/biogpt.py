"""Biomedical LLM agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)


class BioGPTQueryAgent:
    def __init__(self, dependencies: AgentDependencies) -> None:
        self._biogpt = dependencies.biogpt
        self._emit_status = dependencies.emit_status

    def execute(self, state: AgentState) -> dict[str, Any]:
        try:
            self._emit_status(user=state["user_id"], message="Analyzing biomedical information...")
            answer = self._biogpt.generate_answer(state["user_query"])
            return {
                "biogpt_response": {"text": answer, "source": "BioGPT"},
                "agents_completed": ["biogpt_agent"],
                "messages": [AIMessage(content="BioGPT query processed")],
            }
        except Exception as exc:
            logger.exception("BioGPT agent failed")
            return {
                "biogpt_response": {"text": None, "json_format": None, "source": "BioGPT"},
                "agents_completed": ["biogpt_agent"],
                "error": str(exc),
            }
