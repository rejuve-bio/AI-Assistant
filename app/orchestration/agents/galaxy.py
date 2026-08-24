"""Galaxy tools and workflows agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)

GALAXY_PLATFORM = "Galaxy platform"


class GalaxyAgent:
    """Retrieve Galaxy tools and workflow information."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._galaxy = dependencies.galaxy_handler
        self._emit_status = dependencies.emit_status

    def execute(self, state: AgentState) -> dict[str, Any]:
        logger.info(
            "Galaxy agent processing query: %s for user: %s",
            state["user_query"], state["user_id"],
        )

        try:
            self._emit_status(
                user=state["user_id"],
                message="Retrieving Galaxy tools information...",
            )

            response = self._galaxy.get_galaxy_info(
                state["user_query"],
                state["user_id"],
                state["token"],
            )

            # Normalize response
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No Galaxy information found"

            logger.debug("Galaxy response: %s", response_text)
            return {
                "galaxy_response": {
                    "text": response_text,
                    "json_format": None,
                    "source": GALAXY_PLATFORM,
                },
                "agents_completed": ["galaxy_agent"],
                "messages": [AIMessage(content="Galaxy query processed")],
            }

        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {
                    "text": f"Error: {str(e)}",
                    "json_format": None,
                    "source": GALAXY_PLATFORM,
                },
                "agents_completed": ["galaxy_agent"],
                "error": str(e),
            }
