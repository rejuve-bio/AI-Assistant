"""Multi-source content retrieval agent (graph, Galaxy URLs, RAG content)."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from ..contracts import AgentState
from .dependencies import AgentDependencies

logger = logging.getLogger(__name__)

ANALYZING_MSG = "Analyzing..."


class ContentRetrievalAgent:
    """Retrieve and aggregate content from graphs, Galaxy URLs, and RAG stores."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._rag = dependencies.rag
        self._galaxy = dependencies.galaxy_handler
        self._graph_summarizer = dependencies.graph_summarizer
        self._hypothesis = dependencies.hypothesis_generation
        self._store = dependencies.store
        self._emit_status = dependencies.emit_status

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant content from multiple sources with source attribution."""
        query = state.get("user_query")
        user_id = state.get("user_id")
        token = state.get("token")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        logger.info("ContentRetrievalAgent called for user: %s", user_id)
        self._emit_status(user=user_id, message="Retrieving relevant content...")

        content_parts: list[dict[str, Any]] = []
        sources: list[str] = []
        graph_covers_query = False

        try:
            if graph_id:
                early_return, entity_found = self._retrieve_from_graph(
                    query, user_id, graph_id, token, resource, content_parts, sources,
                )
                if early_return is not None:
                    return early_return

                if entity_found is True and "annotation_agent" in state.get("agents_to_run", []):
                    graph_covers_query = True

            if urls:
                self._retrieve_from_galaxy(query, user_id, token, urls, content_parts, sources)

            if content_ids:
                self._retrieve_from_rag(query, user_id, content_ids, content_parts, sources)

            response_dict = {
                "text": content_parts,
                "json_format": None,
                "sources": sources,
            }
            logger.info(
                "Content retrieval response prepared with %d parts. response is %s",
                len(content_parts), response_dict,
            )
            state_update: dict[str, Any] = {
                "content_retrieval_response": response_dict,
                "agents_completed": ["content_retrieval_agent"],
                "messages": [AIMessage(content="Content retrieval completed")],
            }
            if graph_covers_query:
                logger.info(
                    "Existing attached graph already covers the query — "
                    "skipping redundant annotation_agent run"
                )
                current_agents = state.get("agents_to_run", [])
                state_update["agents_to_run"] = [a for a in current_agents if a != "annotation_agent"]
            return state_update

        except Exception as e:
            logger.error("Error in ContentRetrievalAgent: %s", str(e), exc_info=True)
            return {
                "content_retrieval_response": {
                    "text": [],
                    "json_format": None,
                    "sources": [],
                },
                "agents_completed": ["content_retrieval_agent"],
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_from_graph(
        self, query, user_id, graph_id, token, resource, content_parts, sources,
    ):
        """Returns (early_return, entity_found) — early_return is a state update dict to
        short-circuit the pipeline on graph-fetch failure, or None to continue normally.
        entity_found is True/False/None (unknown) per whether the graph actually covered
        the query, used to decide whether a fresh annotation_agent run is still needed."""
        logger.info("Retrieving graph summary for graph_id: %s", graph_id)
        graph_summary = self._answer_from_graph(
            query=query, user_id=user_id, graph_id=graph_id,
            token=token, resource=resource,
        )
        if not graph_summary:
            return None, None
        entity_found = graph_summary.get("entity_found") if isinstance(graph_summary, dict) else None
        graph_text = (
            graph_summary.get("text", str(graph_summary))
            if isinstance(graph_summary, dict) else str(graph_summary)
        )
        if graph_text and not graph_text.startswith("Failed to contact") and not graph_text.startswith("Error"):
            content_parts.append({"source": f"graph:{graph_id}", "content": graph_text})
            sources.append(f"graph:{graph_id}")
            return None, entity_found
        if graph_text:
            logger.warning("Graph fetch failed for %s: %s", graph_id, graph_text)
            last_topic = None
            try:
                history = self._store.get_context_and_memory(user_id)
                for item in reversed(history):
                    agents_used = item.get("context", {}).get("agents_used", [])
                    if "annotation_agent" in agents_used:
                        last_topic = item.get("question")
                        break
            except Exception:
                pass
            if last_topic:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Did you mean to ask about your previous annotation: *\"{last_topic}\"*? "
                    f"Or would you like to ask a different question?"
                )
            else:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Please check that the graph exists, or let me know what you'd like to explore."
                )
            return {
                "content_retrieval_response": {
                    "text": confirmation_text,
                    "json_format": None,
                    "sources": [],
                },
                "agents_completed": ["content_retrieval_agent"],
                "stop_pipeline": True,
            }, None
        return None, entity_found

    def _answer_from_graph(self, query, user_id, resource, token, graph_id):
        """Fetch a graph summary from the appropriate service."""
        logger.info(
            "Answer from graph summaries called with query: %s, user_id: %s, "
            "resource: %s, graph_id: %s",
            query, user_id, resource, graph_id,
        )
        try:
            entity_found = None
            if resource == "annotation":
                summary_result = self._graph_summarizer.summary(
                    token=token, graph_id=graph_id, user_query=query,
                )
                summary_text = (
                    summary_result.get("text", "")
                    if isinstance(summary_result, dict) else summary_result
                )
                if isinstance(summary_result, dict):
                    entity_found = summary_result.get("entity_found")
                self._emit_status(user=user_id, message=ANALYZING_MSG)
            elif resource == "hypothesis":
                summary_result = self._hypothesis.get_by_hypothesis_id(
                    token, graph_id, user_id, query,
                )
                summary_text = (
                    summary_result.get("text", "")
                    if isinstance(summary_result, dict) else summary_result
                )
                self._emit_status(user=user_id, message=ANALYZING_MSG)
            else:
                return {"text": "Invalid resource type specified.", "json_format": None}

            return {"text": summary_text, "json_format": None, "entity_found": entity_found}
        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {"text": f"Error processing query: {str(e)}", "json_format": None}

    def _retrieve_from_galaxy(self, query, user_id, token, urls, content_parts, sources):
        logger.info("Retrieving Galaxy urls for user: %s", user_id)
        urls_response = self._galaxy.get_galaxy_info(
            query=query, user_id=user_id, token=token, urls=urls,
        )
        if urls_response:
            urls_text = (
                urls_response.get("text", str(urls_response))
                if isinstance(urls_response, dict) else str(urls_response)
            )
            for file in (urls if isinstance(urls, list) else [urls]):
                content_parts.append({"source": f"file:{file}", "content": urls_text})
                sources.append(f"file:{file}")

    def _retrieve_from_rag(self, query, user_id, content_ids, content_parts, sources):
        logger.info("Retrieving RAG content for content_ids: %s", content_ids)
        rag_content = self._rag.get_result_from_rag(query, user_id, content_ids)
        if rag_content:
            rag_text = (
                rag_content.get("text", str(rag_content))
                if isinstance(rag_content, dict) else str(rag_content)
            )
            resources = rag_content.get("resource", {})
            content_parts.append({
                "source": f"content IDs: {', '.join(content_ids)}",
                "content": rag_text,
                "resource": resources,
            })
            sources.append(f"content IDs: {', '.join(content_ids)}")
