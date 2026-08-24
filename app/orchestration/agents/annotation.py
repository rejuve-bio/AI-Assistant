"""Annotation graph agent — processes biological and general annotation queries."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)

ANNOTATION_DB = "annotation database"


class AnnotationAgent:
    """Handle annotation-related queries via the annotation graph pipeline."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._annotation_graph = dependencies.annotation_graph
        self._emit_status = dependencies.emit_status

    def execute(self, state: AgentState) -> dict[str, Any]:
        query_types = state.get("query_types", [])
        query_type = next(
            (qt for qt in query_types if "annotation" in qt),
            "annotation_biological",
        )

        logger.info(
            "Annotation agent processing query: %s for user: %s, type: %s",
            state["user_query"], state["user_id"], query_type,
        )

        try:
            if query_type == "annotation_biological":
                self._emit_status(
                    user=state["user_id"],
                    message="Processing your biological query...",
                )
            elif query_type == "annotation_general":
                self._emit_status(
                    user=state["user_id"],
                    message="Analyzing database information...",
                )

            pipeline_response = self._annotation_graph.process_annotation_query(
                query=state["user_query"],
                user_id=state["user_id"],
                query_type=query_type,
            )

            logger.info("Pipeline response: %s", pipeline_response)

            if pipeline_response.get("needs_confirmation"):
                return {
                    "annotation_response": {
                        "text": pipeline_response.get("confirmation_text", ""),
                        "json_format": None,
                        "needs_confirmation": True,
                        "source": ANNOTATION_DB,
                    },
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation needs user confirmation")],
                }

            if pipeline_response.get("success", False):
                summary = pipeline_response.get("summary", "")
                json_format = pipeline_response.get("json_format", None)
                validation_report = pipeline_response.get("validation_report", {})
                organism = pipeline_response.get("organism", "human")

                response_dict = {
                    "text": summary if summary else "",
                    "json_format": json_format,
                    "validation_report": validation_report,
                    "organism": organism,
                    "source": ANNOTATION_DB,
                }

                return {
                    "annotation_response": response_dict,
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation processing completed")],
                }

            error_msg = pipeline_response.get("error", "Unknown error")
            logger.error("Annotation pipeline failed: %s", error_msg)
            return {
                "annotation_response": {
                    "text": f"Error: {error_msg}",
                    "json_format": None,
                    "source": ANNOTATION_DB,
                },
                "agents_completed": ["annotation_agent"],
                "error": error_msg,
            }

        except Exception as e:
            logger.error("Unexpected error in annotation agent", exc_info=True)
            return {
                "annotation_response": {
                    "text": f"Error: {str(e)}",
                    "json_format": None,
                    "source": ANNOTATION_DB,
                },
                "agents_completed": ["annotation_agent"],
                "error": str(e),
            }
