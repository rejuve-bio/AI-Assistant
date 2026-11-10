import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage

UNRESOLVED_MARKER = "[not run] "


def merge_errors(existing, new):
    if not new:
        return ""
    if existing and existing != new:
        return f"{existing}; {new}"
    return new


RESET = "__agent_state_reset__"


def keep_latest(existing, new):
    if new == RESET:
        return None
    return new if new is not None else existing


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    thread_id: Optional[str]
    propose_only: Optional[bool]
    thread_species: Optional[str]
    thread_sources: Optional[list]
    thread_resources: Optional[list]
    token: str
    response: Dict[str, Any]
    error: Annotated[str, merge_errors]
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[Any]
    attached_files: Annotated[List[Dict[str, str]], operator.add]
    annotation_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    rag_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    galaxy_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    content_retrieval_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    biogpt_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    hypothesis_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    pubmed_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    clinical_trials_response: Annotated[Optional[Dict[str, Any]], keep_latest]
    proposals: Annotated[List[Dict[str, Any]], operator.add]
    proposals_baseline: Optional[int]
    agents_completed: Annotated[List[str], operator.add]
    loop_iterations: int
    one_round_only: Optional[bool]
    current_tool_call: Optional[Dict[str, Any]]


ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"
ANALYZING_MSG = "Analyzing..."
