from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class ConfirmationOption(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: Optional[str] = Field(None, description="Display label for a confirmation choice.")
    value: str = Field(..., description="Exact value to echo back in `resume`.")


class ConfirmationPrompt(BaseModel):
    options: List[ConfirmationOption] = Field(default_factory=list)
    allow_free_text: bool = Field(True, description="Whether free-text `question` replies are also accepted.")


class LegacyProposalItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str = Field(..., description="Proposal kind ('annotation', 'hypothesis', 'hypothesis-project', 'galaxy').")
    data: Any = Field(..., description="Proposal data payload.")


class LegacyQueryResponse(BaseModel):
    """Response from POST /query for both conversational turns and upload-only calls."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="Assistant response text or upload summary.")
    json_format: Optional[Any] = Field(None, description="Structured annotation, hypothesis, or Galaxy JSON payload.")
    proposals: Optional[List[LegacyProposalItem]] = Field(None, description="Structured proposals produced in this turn.")
    needs_confirmation: Optional[bool] = Field(None, description="True when awaiting user confirmation.")
    confirmation: Optional[ConfirmationPrompt] = Field(None, description="Confirmation choices when paused.")
    consensus_result: Optional[Dict[str, Any]] = Field(None, description="Literature consensus summary, if evaluated.")
    organism: Optional[str] = Field(None, description="Organism for annotation results.")
    resource: Optional[Any] = Field(None, description="Associated resource metadata.")
    agents_completed: Optional[List[str]] = Field(None, description="Agents invoked during the turn.")
    content_ids: Optional[List[str]] = Field(None, description="Uploaded content IDs (upload-only responses).")
    suggested_questions: Optional[List[Any]] = Field(None, description="Suggested questions extracted from uploads.")


class LegacyBusyResponse(BaseModel):
    text: str = Field(..., description="Busy notice explaining a previous request is still running.")
    json_format: Optional[Any] = None
    status: Literal["busy"] = "busy"


class UserDocumentItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    content_id: Optional[str] = None
    content_type: Literal["pdf", "web"]
    filename: Optional[str] = Field(None, description="Present when content_type == 'pdf'.")
    num_pages: Optional[int] = Field(None, description="Present when content_type == 'pdf'.")
    url: Optional[str] = Field(None, description="Present when content_type == 'web'.")
    title: Optional[str] = Field(None, description="Present when content_type == 'web'.")
    author: Optional[str] = Field(None, description="Present when content_type == 'web'.")
    file_size: Optional[Any] = None
    upload_time: Optional[str] = Field(None, description="Formatted as 'YYYY-MM-DD HH:MM:SS'.")
    summary: Optional[str] = None


class UserStatusDocumentsResponse(BaseModel):
    user_id: str
    total_count: int
    pdf_count: int
    web_count: int
    files: List[UserDocumentItem] = Field(default_factory=list)


class LegacyHistoryEntry(BaseModel):
    query_id: Optional[str] = None
    user: Optional[str] = None
    assistant_answer: Optional[str] = None
    graph_id_referenced: Optional[str] = None
    time: Optional[str] = None


class FaqQuestionLink(BaseModel):
    id: str
    text: str
    link: str


class FaqIntroResponse(BaseModel):
    text: str
    questions: List[FaqQuestionLink] = Field(default_factory=list)


class HypothesisProjectSummary(BaseModel):
    project_id: str
    project_name: Optional[str] = None
    phenotype: Optional[str] = None
    causal_genes: List[str] = Field(default_factory=list)
    tissues: List[str] = Field(default_factory=list)


class HypothesisFaqResponse(BaseModel):
    text: str
    projects: List[HypothesisProjectSummary] = Field(default_factory=list)
    sample_questions: Any = Field(..., description="LLM-generated sample questions.")


class FaqAnswerResponse(BaseModel):
    question: str
    text: str
    json_format: Any


class FaqNotFoundResponse(BaseModel):
    error: str
    text: str

