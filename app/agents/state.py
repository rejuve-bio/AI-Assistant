import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


def merge_errors(existing, new):
    if not new:
        return ""
    if existing and existing != new:
        return f"{existing}; {new}"
    return new


def keep_latest(existing, new):
    return new if new is not None else existing


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    response: Dict[str, Any]
    error: Annotated[str, merge_errors]
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[Any]
    annotation_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    rag_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    galaxy_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    content_retrieval_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    biogpt_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    hypothesis_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    pubmed_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    clinical_trials_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    agents_completed: Annotated[List[str], operator.add]
    loop_iterations: int
    one_round_only: Optional[bool]
    current_tool_call: Optional[Dict[str, Any]]


ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"
ANALYZING_MSG = "Analyzing..."
