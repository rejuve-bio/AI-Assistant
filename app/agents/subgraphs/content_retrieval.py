
import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.rag.utils.galaxy_content import GALAXY_COLLECTION, clean_galaxy_text

logger = logging.getLogger(__name__)

GRAPH = "graph"
URLS = "urls"
UPLOADED = "uploaded"


class ContentRetrievalState(TypedDict, total=False):
    user_query: str
    user_id: str
    token: str
    graph_id: Optional[str]
    urls: Optional[List[str]]
    content_ids: Optional[List[str]]
    resource: Optional[Any]
    content_parts: Annotated[List[Dict[str, Any]], operator.add]
    sources: Annotated[List[str], operator.add]
    graph_covers_query: bool
    stop_response: Optional[Dict[str, Any]]


def build_content_retrieval_subgraph(rag, answer_from_graph_summaries, store):

    def graph_node(state: ContentRetrievalState) -> Dict[str, Any]:
        graph_id = state["graph_id"]
        logger.info(f"Retrieving graph summary for graph_id: {graph_id}")
        summary = answer_from_graph_summaries(
            query=state["user_query"],
            user_id=state["user_id"],
            graph_id=graph_id,
            token=state.get("token"),
            resource=state.get("resource"),
        )
        if not summary:
            return {}

        entity_found = summary.get("entity_found") if isinstance(summary, dict) else None
        text = summary.get("text", str(summary)) if isinstance(summary, dict) else str(summary)

        if text and not text.startswith(("Failed to contact", "Error")):
            return {
                "content_parts": [{"source": f"graph:{graph_id}", "content": text}],
                "sources": [f"graph:{graph_id}"],
                "graph_covers_query": entity_found is True,
            }

        if not text:
            return {"graph_covers_query": entity_found is True}

        logger.warning(f"Graph fetch failed for {graph_id}: {text}")
        return {"stop_response": _graph_missing_response(graph_id, state["user_id"], store)}

    def urls_node(state: ContentRetrievalState) -> Dict[str, Any]:
        urls = state["urls"]
        logger.info(f"Retrieving URL content for: {urls}")
        rag.save_url_content(
            urls,
            collection_name=GALAXY_COLLECTION,
            cleaner=clean_galaxy_text,
            summarize=True,
            include_tables=True,
        )
        text = rag.query_url_content(state["user_query"], urls, GALAXY_COLLECTION)
        if not text:
            return {}

        listed = urls if isinstance(urls, list) else [urls]
        return {
            "content_parts": [{"source": f"file:{u}", "content": text} for u in listed],
            "sources": [f"file:{u}" for u in listed],
        }

    def uploaded_node(state: ContentRetrievalState) -> Dict[str, Any]:
        content_ids = state["content_ids"]
        logger.info(f"Retrieving uploaded content for content_ids: {content_ids}")
        found = rag.get_result_from_rag(state["user_query"], state["user_id"], content_ids)
        if not found:
            return {}

        text = found.get("text", str(found)) if isinstance(found, dict) else str(found)
        label = f"content IDs: {', '.join(content_ids)}"
        return {
            "content_parts": [{
                "source": label,
                "content": text,
                "resource": found.get("resource", {}) if isinstance(found, dict) else {},
            }],
            "sources": [label],
        }


    def entry(state: ContentRetrievalState) -> str:
        return _next_source(state, after=None)

    def after_graph(state: ContentRetrievalState) -> str:
        if state.get("stop_response"):
            return END
        return _next_source(state, after=GRAPH)

    def after_urls(state: ContentRetrievalState) -> str:
        return _next_source(state, after=URLS)

    graph = StateGraph(ContentRetrievalState)
    graph.add_node(GRAPH, graph_node)
    graph.add_node(URLS, urls_node)
    graph.add_node(UPLOADED, uploaded_node)

    routes = {GRAPH: GRAPH, URLS: URLS, UPLOADED: UPLOADED, END: END}
    graph.set_conditional_entry_point(entry, routes)
    graph.add_conditional_edges(GRAPH, after_graph, {URLS: URLS, UPLOADED: UPLOADED, END: END})
    graph.add_conditional_edges(URLS, after_urls, {UPLOADED: UPLOADED, END: END})
    graph.add_edge(UPLOADED, END)

    return graph.compile()


def _next_source(state: ContentRetrievalState, after: Optional[str]) -> str:
    available = [
        (GRAPH, state.get("graph_id")),
        (URLS, state.get("urls")),
        (UPLOADED, state.get("content_ids")),
    ]
    seen_previous = after is None
    for name, value in available:
        if not seen_previous:
            seen_previous = name == after
            continue
        if value:
            return name
    return END


def _graph_missing_response(graph_id: str, user_id: str, store) -> Dict[str, Any]:
    last_topic = None
    try:
        for item in reversed(store.get_context_and_memory(user_id)):
            if "annotation_agent" in item.get("context", {}).get("agents_used", []):
                last_topic = item.get("question")
                break
    except Exception:
        pass

    if last_topic:
        text = (
            f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
            f"Did you mean to ask about your previous annotation: *\"{last_topic}\"*? "
            f"Or would you like to ask a different question?"
        )
    else:
        text = (
            f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
            f"Please check that the graph exists, or let me know what you'd like to explore."
        )

    return {
        "text": text,
        "json_format": None,
        "sources": [],
        "status": "needs_input",
        "reason": "graph_not_found",
    }
