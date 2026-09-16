
import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.state import GALAXY_PLATFORM

logger = logging.getLogger(__name__)

MCP = "mcp"


class GalaxyState(TypedDict, total=False):
    user_query: str
    user_id: str
    token: str
    galaxy_response: Optional[Dict[str, Any]]


def build_galaxy_subgraph(galaxy_handler):

    def mcp_node(state: GalaxyState) -> Dict[str, Any]:
        try:
            response = galaxy_handler.get_galaxy_info(
                state["user_query"], state["user_id"], state.get("token")
            )
            text = (
                response["text"]
                if isinstance(response, dict) and "text" in response
                else (str(response) if response else "No Galaxy information found")
            )
            logger.debug(f"Galaxy response: {text}")
        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {
                    "text": f"Error: {e}", "json_format": None, "source": GALAXY_PLATFORM
                },
            }

        return {
            "galaxy_response": {"text": text, "json_format": None, "source": GALAXY_PLATFORM},
        }

    graph = StateGraph(GalaxyState)
    graph.add_node(MCP, mcp_node)
    graph.set_entry_point(MCP)
    graph.add_edge(MCP, END)
    return graph.compile()
