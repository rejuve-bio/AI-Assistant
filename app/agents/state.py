import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_types: List[str]
    response: Dict[str, Any]
    error: str
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[Any]
    pipeline_details: Dict[str, Any]
    # Agent-specific responses with source attribution
    annotation_response: Optional[Dict[str, Any]]
    # Generic pause/resume payload for ANY agent's human-in-the-loop confirmation,
    pending_confirmation: Optional[Dict[str, Any]]
    confirmation_outcome: Optional[str]
    rag_response: Optional[Dict[str, Any]]
    galaxy_response: Optional[Dict[str, Any]]
    content_retrieval_response: Optional[Dict[str, Any]]
    biogpt_response:Optional[Dict[str, Any]]
    hypothesis_response: Optional[Dict[str, Any]]
    pubmed_response: Optional[Dict[str, Any]]
    clinical_trials_response: Optional[Dict[str, Any]]
    # Parallel execution control
    agents_to_run: List[str]
    agents_completed: Annotated[List[str], operator.add]
    stop_pipeline: Optional[bool]


ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"
ANALYZING_MSG = "Analyzing..."
