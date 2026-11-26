from typing import List, Optional
from pydantic import BaseModel, Field


class UploadedFileItem(BaseModel):
    id: str = Field(..., description="Assigned content_id for referencing the file in /v1/query.")
    name: str = Field(..., description="Original filename.")


class FileUploadError(BaseModel):
    name: Optional[str] = Field(None, description="Filename that failed to upload.")
    error: str = Field(..., description="Failure reason.")


class V1FilesUploadResponse(BaseModel):
    items: List[UploadedFileItem] = Field(..., description="Successfully stored files.")
    errors: Optional[List[FileUploadError]] = Field(
        None, description="Per-file errors for any files in the batch that failed."
    )


class V1FilesUploadErrorResponse(BaseModel):
    text: str = Field(..., description="Summary error message.")
    errors: List[FileUploadError] = Field(default_factory=list, description="Per-file upload errors.")

