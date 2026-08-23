"""Stable contracts at the boundary of the assistant workflow.

LangGraph state deliberately remains a TypedDict because reducers are part of its
state schema. Public inputs and outputs use Pydantic models so HTTP, SocketIO,
and workflow callers share one validated contract.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field


class AgentName(str, Enum):
    ANNOTATION = "annotation_agent"
    HYPOTHESIS = "hypothesis_agent"
    RAG = "rag_agent"
    GALAXY = "galaxy_agent"
    CONTENT_RETRIEVAL = "content_retrieval_agent"
    BIOGPT = "biogpt_agent"
    PUBMED = "pubmed_agent"
    CLINICAL_TRIALS = "clinical_trials_agent"


class AssistantRequest(BaseModel):
    """Validated input for a single agent-workflow invocation."""

    model_config = ConfigDict(frozen=True)

    message: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    token: str = ""
    content_ids: Optional[list[str]] = None
    graph_id: Optional[str] = None
    urls: Optional[list[str]] = None
    resource: Any = None
    conversation_history: Optional[list[dict[str, Any]]] = None

    def initial_state(self) -> "AgentState":
        """Create the complete state required by LangGraph reducers and nodes."""
        return {
            "messages": [HumanMessage(content=self.message)],
            "user_query": self.message,
            "user_id": self.user_id,
            "token": self.token,
            "query_types": [],
            "response": {"text": "", "json_format": None},
            "error": "",
            "content_ids": self.content_ids,
            "graph_id": self.graph_id,
            "urls": self.urls,
            "resource": self.resource,
            "pipeline_details": {},
            "annotation_response": None,
            "rag_response": None,
            "galaxy_response": None,
            "content_retrieval_response": None,
            "biogpt_response": None,
            "hypothesis_response": None,
            "pubmed_response": None,
            "clinical_trials_response": None,
            "stop_pipeline": False,
            "agents_to_run": [],
            "agents_completed": [],
            "conversation_history": self.conversation_history,
        }


class AssistantResponse(BaseModel):
    """Normalized response returned by the workflow boundary."""

    model_config = ConfigDict(extra="allow")

    text: str = ""
    json_format: Optional[dict[str, Any]] = None
    agents_completed: list[str] = Field(default_factory=list)

    @classmethod
    def from_workflow(cls, response: Any, agents_completed: list[str]) -> "AssistantResponse":
        if isinstance(response, dict):
            payload = dict(response)
        else:
            payload = {"text": "" if response is None else str(response)}
        payload.setdefault("text", "")
        payload.setdefault("json_format", None)
        payload["agents_completed"] = agents_completed
        return cls.model_validate(payload)


class AgentState(TypedDict):
    """The internal state contract shared by all LangGraph nodes."""

    messages: Annotated[list[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_types: list[str]
    response: dict[str, Any]
    error: str
    content_ids: Optional[list[str]]
    graph_id: Optional[str]
    urls: Optional[list[str]]
    resource: Any
    pipeline_details: dict[str, Any]
    annotation_response: Optional[dict[str, Any]]
    rag_response: Optional[dict[str, Any]]
    galaxy_response: Optional[dict[str, Any]]
    content_retrieval_response: Optional[dict[str, Any]]
    biogpt_response: Optional[dict[str, Any]]
    hypothesis_response: Optional[dict[str, Any]]
    pubmed_response: Optional[dict[str, Any]]
    clinical_trials_response: Optional[dict[str, Any]]
    agents_to_run: list[str]
    agents_completed: Annotated[list[str], operator.add]
    stop_pipeline: Optional[bool]
    conversation_history: Optional[list[dict[str, Any]]]
