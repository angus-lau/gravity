from typing import Any

from pydantic import BaseModel, Field, field_validator


class UserContext(BaseModel):
    gender: str | None = None
    age: int | None = Field(default=None, ge=0, le=120)
    location: str | None = None
    interests: list[str] | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    context: UserContext | None = None

    @field_validator("query")
    @classmethod
    def query_not_empty_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class Campaign(BaseModel):
    campaign_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    advertiser: str | None = None
    title: str | None = None
    categories: list[str] | None = None

    model_config = {"extra": "allow"}


class TimingMetadata(BaseModel):
    eligibility_ms: float = Field(..., ge=0)
    embedding_ms: float = Field(..., ge=0)
    category_match_ms: float = Field(..., ge=0)
    faiss_search_ms: float = Field(..., ge=0)
    reranking_ms: float = Field(..., ge=0)
    total_ms: float = Field(..., ge=0)
    # Phase 1: Modular guardrails timing breakdown
    blocklist_ms: float = Field(default=0, ge=0)
    safety_ms: float = Field(default=0, ge=0)
    commercial_ms: float = Field(default=0, ge=0)
    # Phase 2: Query expansion
    expansion_ms: float = Field(default=0, ge=0)
    # Phase 3: Hybrid retrieval
    bm25_search_ms: float = Field(default=0, ge=0)
    fusion_ms: float = Field(default=0, ge=0)
    # Phase 4: Image search
    image_search_ms: float = Field(default=0, ge=0)


class ResponseMetadata(BaseModel):
    timing: TimingMetadata | None = None
    model_versions: dict[str, str] | None = None
    query_embedding_dim: int | None = None
    candidates_before_rerank: int | None = None
    expanded_query: str | None = None

    model_config = {"extra": "allow"}


class RetrievalResponse(BaseModel):
    ad_eligibility: float = Field(..., ge=0.0, le=1.0)
    extracted_categories: list[str] = Field(..., min_length=0, max_length=10)
    campaigns: list[Campaign]
    latency_ms: float = Field(..., ge=0)
    metadata: ResponseMetadata | dict[str, Any] = Field(default_factory=dict)

    @field_validator("campaigns")
    @classmethod
    def campaigns_sorted_descending(cls, v: list[Campaign]) -> list[Campaign]:
        for i in range(len(v) - 1):
            if v[i].relevance_score < v[i + 1].relevance_score:
                raise ValueError("Campaigns must be sorted by relevance_score in descending order")
        return v


class HealthResponse(BaseModel):
    status: str = "healthy"
    models_loaded: bool
    campaigns_indexed: int


class ImageRetrievalResponse(BaseModel):
    campaigns: list[Campaign]
    latency_ms: float = Field(..., ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] | None = None
