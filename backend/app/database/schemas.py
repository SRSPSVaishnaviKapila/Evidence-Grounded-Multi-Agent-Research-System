from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from datetime import datetime


class PaperCreate(BaseModel):
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None


class PaperResponse(PaperCreate):
    id: int
    pdf_path: Optional[str] = None
    text_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
class PDFUploadResponse(BaseModel):
    paper_id: int
    filename: str
    pdf_path: str
    text_path: str
    pages_processed: int
    status: str
class RAGQuestion(BaseModel):
    question: str
    top_k: int = 5


class CrossPaperRequest(BaseModel):
    paper_ids: list[int] = Field(
        min_length=2,
        max_length=10
    )


class ResearchGapRequest(BaseModel):
    paper_ids: list[int] = Field(
        min_length=2,
        max_length=10
    )


class LiteratureSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)
    from_year: Optional[int] = Field(default=None, ge=1000, le=3000)
    to_year: Optional[int] = Field(default=None, ge=1000, le=3000)


class LiteratureImportItem(BaseModel):
    result_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=2000)
    authors: list[str] = Field(default_factory=list, max_length=500)
    year: Optional[int] = Field(default=None, ge=1000, le=3000)
    abstract: Optional[str] = None
    doi: Optional[str] = Field(default=None, max_length=500)
    url: Optional[str] = Field(default=None, max_length=4000)
    source: str = Field(default="Crossref", min_length=1, max_length=200)
    venue: Optional[str] = Field(default=None, max_length=1000)
    publication_type: Optional[str] = Field(default=None, max_length=200)


class LiteratureImportRequest(BaseModel):
    papers: list[LiteratureImportItem] = Field(min_length=1, max_length=50)


class VerificationRequest(BaseModel):
    paper_ids: list[int] = Field(min_length=1, max_length=10)
    verify_agent2: bool = True
    verify_agent3: bool = True
    verify_agent4: bool = True
    verify_report: bool = True

    @field_validator("paper_ids")
    @classmethod
    def require_unique_paper_ids(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("paper_ids must be unique")
        return value

    @model_validator(mode="after")
    def require_a_verification_scope(self):
        if not (self.verify_agent2 or self.verify_agent3 or self.verify_agent4):
            raise ValueError("At least one agent verification scope is required")
        return self
