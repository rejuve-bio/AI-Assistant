"""Hypothesis generation agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)


class HypothesisAgent:
    """Generate biological hypotheses and optionally trigger literature follow-up."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._hypothesis = dependencies.hypothesis_generation
        self._emit_status = dependencies.emit_status

    def execute(self, state: AgentState) -> dict[str, Any]:
        logger.info(
            "Hypothesis agent processing query: %s for user: %s",
            state["user_query"], state["user_id"],
        )

        try:
            self._emit_status(user=state["user_id"], message="Generating hypothesis...")
            response = self._hypothesis.generate_hypothesis(
                token=state["token"],
                user_query=state["user_query"],
                user_id=state["user_id"],
            )

            hypothesis_text = response.get("text", "")
            # A real hypothesis always returns resource: {id, type, graph} — all fallback/failure paths omit it
            succeeded = (
                isinstance(response.get("resource"), dict)
                and response["resource"].get("type") == "hypothesis"
            )

            state_update: dict[str, Any] = {
                "hypothesis_response": response,
                "messages": [AIMessage(content=f"Hypothesis generated: {hypothesis_text}")],
                "agents_completed": ["hypothesis_agent"],
            }

            if succeeded:
                current_agents = state.get("agents_to_run", [])
                extra = [
                    a for a in ("clinical_trials_agent", "pubmed_agent")
                    if a not in current_agents
                ]
                if extra:
                    logger.info("Hypothesis succeeded — injecting literature agents: %s", extra)
                    state_update["agents_to_run"] = current_agents + extra

            return state_update

        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "hypothesis_response": {
                    "text": (
                        "The hypothesis service is not returning any results at the moment. "
                        "There is nothing I can help with for this request."
                    ),
                    "resource": None,
                },
                "stop_pipeline": True,
                "error": str(e),
                "messages": [AIMessage(content=f"Error in hypothesis generation: {str(e)}")],
                "agents_completed": ["hypothesis_agent"],
            }
