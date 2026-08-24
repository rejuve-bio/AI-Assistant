"""Knowledge-base retrieval agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE = "knowledge base"
_NO_RESULT_PHRASES = (
    "couldn't find", "could not find", "no relevant", "no information",
    "no results", "not found", "no documents", "unable to find",
    "no data", "i don't have information", "i do not have", "no specific", "no details",
)


class RagQueryAgent:
    """Retrieves from RAG and explicitly schedules PubMed on weak retrieval."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._rag = dependencies.rag
        self._emit_status = dependencies.emit_status

    def execute(self, state: AgentState) -> dict[str, Any]:
        try:
            self._emit_status(user=state["user_id"], message="Retrieving information...")
            response = self._rag.get_result_from_rag(
                state["user_query"], state["user_id"],
                content_ids=state.get("content_ids"),
                conversation_history=state.get("conversation_history"),
            )
            response_text = response.get("text", "") if isinstance(response, dict) else str(response or "")
            confidence = response.get("confidence", 0.5) if isinstance(response, dict) else 0.5

            if self._has_no_results(response_text):
                current_agents = state.get("agents_to_run", [])
                update: dict[str, Any] = {
                    "rag_response": {"text": response_text, "json_format": None, "source": KNOWLEDGE_BASE, "confidence": 0.0},
                    "agents_completed": ["rag_agent"],
                    "messages": [AIMessage(content="RAG found no results — triggering PubMed fallback")],
                }
                if "pubmed_agent" not in current_agents:
                    self._emit_status(user=state["user_id"], message="Nothing found in knowledge base, searching PubMed...")
                    update["agents_to_run"] = current_agents + ["pubmed_agent"]
                return update

            return {
                "rag_response": {"text": response_text, "json_format": None, "source": KNOWLEDGE_BASE, "confidence": confidence},
                "agents_completed": ["rag_agent"],
                "messages": [AIMessage(content="RAG query processed")],
            }
        except Exception as exc:
            logger.exception("RAG agent failed")
            return {
                "rag_response": {"text": f"Error: {exc}", "json_format": None, "source": KNOWLEDGE_BASE, "confidence": 0.0},
                "agents_completed": ["rag_agent"],
                "error": str(exc),
            }

    @staticmethod
    def _has_no_results(text: str) -> bool:
        normalized = text.lower().strip()
        return len(normalized) < 120 or any(phrase in normalized for phrase in _NO_RESULT_PHRASES)
