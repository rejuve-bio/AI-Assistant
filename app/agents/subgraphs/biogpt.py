
import logging
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

ANSWER = "answer"


class BioGPTState(TypedDict, total=False):
    user_query: str
    biogpt_response: Optional[Dict[str, Any]]


def build_biogpt_subgraph(biogpt):

    def answer_node(state: BioGPTState) -> Dict[str, Any]:
        try:
            text = biogpt.generate_answer(state["user_query"])
            logger.info(f"BioGPT response: {text}")
        except Exception as e:
            logger.error(f"Error in biogpt agent: {e}", exc_info=True)
            return {"biogpt_response": {"text": None, "json_format": None, "source": "BioGPT"}}

        return {"biogpt_response": {"text": text, "source": "BioGPT"}}

    graph = StateGraph(BioGPTState)
    graph.add_node(ANSWER, answer_node)
    graph.set_entry_point(ANSWER)
    graph.add_edge(ANSWER, END)
    return graph.compile()
