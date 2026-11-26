from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class TextErrorResponse(BaseModel):
    """Standard error payload returned by /v1/* endpoints and HTTPException handler."""

    text: str = Field(..., description="Human-readable error or status message.")


class LegacyErrorResponse(BaseModel):
    """Error payload returned by legacy endpoints."""

    error: str = Field(..., description="Error description.")
    field: Optional[str] = Field(None, description="Optional failing field name (e.g. 'thread_id').")


class MessageResponse(BaseModel):
    """Simple confirmation message returned by legacy delete operations."""

    message: str = Field(..., description="Operation outcome message.")


class ResourceRef(BaseModel):
    """Reference to an annotation, hypothesis, or file resource attached to a thread."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["annotation", "hypothesis", "file"] = Field(
        ..., description="Resource category."
    )
    id: str = Field(..., description="Resource identifier.")
    name: Optional[str] = Field(None, description="Optional display name (for files).")
    url: Optional[str] = Field(None, description="Optional URL (for files).")
    content_type: Optional[str] = Field(None, description="Optional MIME type (for files).")
    size: Optional[int] = Field(None, description="Optional file size in bytes.")
    summary: Optional[Dict[str, Any]] = Field(None, description="Optional file summary metadata.")
    run: Optional[str] = Field(None, description="Optional Galaxy run identifier that produced the file.")


class ThreadContext(BaseModel):
    """Active species, data sources, and attached resources for a conversation thread."""

    species: str = Field(..., description="Target organism/species (e.g. 'human', 'fly').")
    sources: List[str] = Field(default_factory=list, description="Enabled annotation data sources.")
    resources: List[ResourceRef] = Field(
        default_factory=list, description="Resources attached to this thread's context."
    )

