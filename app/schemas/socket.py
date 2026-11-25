from typing import Any, Optional
from pydantic import BaseModel, Field


class SocketJoinRoomPayload(BaseModel):
    """Client -> Server `join_room` event payload."""

    thread_id: Optional[str] = Field(
        None,
        description="Optional thread ID. Joins room '{user_id}:{thread_id}' if set, else '{user_id}'.",
    )


class SocketQuestionPayload(BaseModel):
    """Client -> Server `question` event payload."""

    question: str = Field(..., description="User question text (required, non-empty).")
    thread_id: Optional[str] = Field(
        None, description="Optional conversation thread ID used for room scoping and locking."
    )


class SocketUpdateEvent(BaseModel):
    """Server -> Client `update` event emitted during intermediate processing stages."""

    status: str = Field(..., description="Status tag (e.g. 'update', 'completed', 'error').")
    response: Any = Field(..., description="Progress message string or structured response payload.")


class SocketResponseEvent(BaseModel):
    """Server -> Client `response` event emitted on connect, join_room, or final answer."""

    status: Optional[str] = Field(None, description="'completed' for question responses.")
    response: Optional[Any] = Field(None, description="Final agent response or join confirmation string.")
    message: Optional[str] = Field(None, description="Connection acknowledgment message.")


class SocketErrorEvent(BaseModel):
    """Server -> Client `error` event emitted when validation or processing fails."""

    error: str = Field(..., description="Error description.")

