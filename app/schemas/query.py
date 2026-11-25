from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class V1QueryFileRef(BaseModel):
    """Reference either an already-uploaded file by `id` or a file attached to the multipart request by `name`."""

    id: Optional[str] = Field(None, description="Existing content_id or File Service file ID.")
    name: Optional[str] = Field(
        None,
        description="Filename matching an uploaded file part in a multipart/form-data request.",
    )


class V1QueryTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(..., description="User prompt text.")


class V1QueryFilesPart(BaseModel):
    type: Literal["files"] = "files"
    files: List[V1QueryFileRef] = Field(default_factory=list, description="Files referenced by this message.")


class V1QueryAboutPart(BaseModel):
    type: Literal["about"] = "about"
    kind: Optional[str] = Field(None, description="Resource kind (e.g. 'annotation', 'hypothesis').")
    id: Optional[str] = Field(None, description="Resource identifier.")


V1QueryInputPart = Union[V1QueryTextPart, V1QueryFilesPart, V1QueryAboutPart]


class V1QueryContext(BaseModel):
    """Initial thread context applied when `thread_id` is brand new."""

    species: Optional[str] = Field("human", description="Target species (defaults to 'human').")
    sources: Optional[List[str]] = Field(
        None, description="Annotation sources validated against species schemas."
    )


class V1QueryRequest(BaseModel):
    """JSON payload (or `body` field in multipart/form-data) for POST /v1/query."""

    thread_id: str = Field(..., description="Client-generated UUIDv4 identifying the conversation thread.")
    client_message_id: Optional[str] = Field(
        None, description="Optional idempotency key; repeated requests return 200 with the stored message."
    )
    context: Optional[V1QueryContext] = Field(
        None, description="Optional context applied on the first message of a new thread."
    )
    parts: List[V1QueryInputPart] = Field(
        ..., description="Message parts. Must include at least one non-empty 'text' part."
    )

