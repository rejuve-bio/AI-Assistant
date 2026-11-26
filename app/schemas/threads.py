from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ResourceRef, ThreadContext


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Message text content.")


class FilesPart(BaseModel):
    type: Literal["files"] = "files"
    files: list[dict[str, Any]] = Field(default_factory=list, description="Attached file descriptors.")


class AboutPart(BaseModel):
    type: Literal["about"] = "about"
    kind: str = Field(..., description="Referenced resource kind (e.g. 'annotation', 'hypothesis').")
    id: str = Field(..., description="Referenced resource identifier.")


class ProposalPart(BaseModel):
    type: Literal["proposal"] = "proposal"
    id: str = Field(..., description="Deterministic proposal identifier within the thread.")
    kind: str = Field(..., description="Proposal kind ('annotation', 'hypothesis', 'hypothesis-project', 'galaxy').")
    title: str = Field(..., description="Human-readable proposal title.")
    data: Any = Field(..., description="Structured proposal payload.")
    replaces: str | None = Field(None, description="Prior proposal ID superseded by this proposal.")


class QuickRepliesPart(BaseModel):
    type: Literal["quick_replies"] = "quick_replies"
    replies: list[str] = Field(default_factory=list, description="Suggested follow-up prompts.")


class ContextPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["context"] = "context"
    species: str = Field(..., description="Target species.")
    sources: list[str] = Field(default_factory=list, description="Active data sources.")
    resources: list[dict[str, Any]] | None = Field(None, description="Attached context resources.")


class RunPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["run"] = "run"
    kind: str | None = Field(None, description="Executed run kind ('annotation', 'hypothesis', 'galaxy', etc.).")
    id: str | None = Field(None, description="Created/executed resource identifier.")


V1MessagePart = Union[
    TextPart,
    FilesPart,
    AboutPart,
    ProposalPart,
    QuickRepliesPart,
    ContextPart,
    RunPart,
]


class V1Message(BaseModel):
    """Canonical Threads v1 message object."""

    id: str = Field(..., description="Stable message ID formatted as '{thread_id}_msg_{seq}'.")
    thread_id: str = Field(..., description="Parent thread UUID.")
    role: str = Field(..., description="Sender role ('user', 'assistant', or 'system').")
    created_at: str | None = Field(None, description="UTC ISO-8601 timestamp (YYYY-MM-DDTHH:MM:SSZ).")
    client_message_id: str | None = Field(None, description="Client-supplied idempotency key, if any.")
    parts: list[V1MessagePart] = Field(default_factory=list, description="Ordered structured content parts.")


class ThreadListItem(BaseModel):
    """Summary row for a conversation thread."""

    id: str = Field(..., description="Thread UUID.")
    title: str = Field(..., description="Custom title or derived preview from first message.")
    context: ThreadContext | None = Field(None, description="Current thread context, if configured.")
    created_at: str | None = Field(None, description="UTC ISO-8601 creation timestamp.")
    updated_at: str | None = Field(None, description="UTC ISO-8601 last-update timestamp.")
    message_count: int = Field(..., description="Number of stored messages in the thread.")


class ThreadDetail(ThreadListItem):
    """Single thread summary including resources generated during the conversation."""

    generated_resources: list[dict[str, Any]] = Field(
        default_factory=list, description="Resources produced by completed runs in this thread."
    )


class ThreadListResponse(BaseModel):
    items: list[ThreadListItem] = Field(default_factory=list)
    next_cursor: str | None = Field(None, description="Pagination cursor (currently null).")


class ThreadMessageListResponse(BaseModel):
    items: list[V1Message] = Field(default_factory=list)
    next_cursor: str | None = Field(None, description="Pagination cursor (currently null).")


class ThreadContextUpdateRequest(BaseModel):
    """Request body for PUT /v1/threads/{thread_id}/context."""

    species: str = Field(..., description="Target species (e.g. 'human', 'fly').")
    sources: list[str] | None = Field(
        None,
        description="Annotation data sources for the species; defaults to species defaults when omitted or empty.",
    )
    resources: list[ResourceRef] | None = Field(
        None, description="Optional replacement list of context resources."
    )


class RecordRunFilesPart(BaseModel):
    type: Literal["files"] = "files"
    files: list[dict[str, Any]] = Field(default_factory=list, description="Output files produced by the run.")


class RecordThreadRunRequest(BaseModel):
    """Request body for POST /v1/threads/{thread_id}/messages."""

    client_message_id: str | None = Field(None, description="Optional idempotency key.")
    parts: list[RunPart | RecordRunFilesPart] = Field(
        ..., description="Must contain exactly one 'run' part, plus an optional 'files' part."
    )


class RenameThreadRequest(BaseModel):
    """Request body for PATCH /v1/threads/{thread_id}."""

    title: str | None = Field(
        None, description="New custom thread title. Passing null or empty clears the custom title."
    )

