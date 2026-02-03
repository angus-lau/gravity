from typing import Any

from pydantic import BaseModel, Field, field_validator


class UserContext(BaseModel):
    """Optional user context for personalized ad retrieval."""
    gender: str | None = Field(default=None, description="User's gender")
    age: int | None = Field(default=None, ge=0, le=120, description="User's age")
    location: str | None = Field(default=None, description="User's location (e.g., 'San Francisco, CA')")
    interests: list[str] | None = Field(default=None, description="User's interests")


class RetrievalRequest(BaseModel):
    """Request body for the /api/retrieve endpoint."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's natural language query"
    )
    context: UserContext | None = Field(
        default=None,
        description="Optional user context for personalization"
    )
    
    @field_validator("query")
    @classmethod
    def query_not_empty_whitespace(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v


class Campaign(BaseModel):
    """A single campaign in the response.
    
    Contains the minimum required fields plus optional extras.
    """
    campaign_id: str = Field(..., description="Unique campaign identifier")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0 to 1.0)"
    )
    
    # Optional fields that may be included
    advertiser: str | None = Field(default=None, description="Advertiser name")
    title: str | None = Field(default=None, description="Campaign title")
    categories: list[str] | None = Field(default=None, description="Campaign categories")
    
    model_config = {"extra": "allow"}


class TimingMetadata(BaseModel):
    """Detailed timing breakdown for each component."""
    eligibility_ms: float = Field(..., ge=0, description="Time for eligibility scoring")
    embedding_ms: float = Field(..., ge=0, description="Time for query embedding")
    category_match_ms: float = Field(..., ge=0, description="Time for category matching")
    faiss_search_ms: float = Field(..., ge=0, description="Time for FAISS vector search")
    reranking_ms: float = Field(..., ge=0, description="Time for context-based reranking")
    total_ms: float = Field(..., ge=0, description="Total processing time")


class ResponseMetadata(BaseModel):
    """Metadata about the retrieval process."""
    timing: TimingMetadata | None = Field(default=None, description="Component timing breakdown")
    model_versions: dict[str, str] | None = Field(default=None, description="Model versions used")
    query_embedding_dim: int | None = Field(default=None, description="Dimension of query embedding")
    candidates_before_rerank: int | None = Field(default=None, description="Candidates retrieved before reranking")
    
    model_config = {"extra": "allow"}


class RetrievalResponse(BaseModel):
    """Response body for the /api/retrieve endpoint.
    
    This is the contract that the API must fulfill.
    """
    ad_eligibility: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score indicating how appropriate it is to show ads (0.0 = never, 1.0 = perfectly appropriate)"
    )
    extracted_categories: list[str] = Field(
        ...,
        min_length=0,
        max_length=10,
        description="Product/service categories relevant to the query (1-10 categories)"
    )
    campaigns: list[Campaign] = Field(
        ...,
        description="Top campaigns ordered by relevance (descending)"
    )
    latency_ms: float = Field(
        ...,
        ge=0,
        description="Actual processing time in milliseconds"
    )
    metadata: ResponseMetadata | dict[str, Any] = Field(
        default_factory=dict,
        description="Additional debugging/diagnostic information"
    )
    
    @field_validator("campaigns")
    @classmethod
    def campaigns_sorted_descending(cls, v: list[Campaign]) -> list[Campaign]:
        """Verify campaigns are sorted by relevance_score descending."""
        if len(v) > 1:
            scores = [c.relevance_score for c in v]
            if scores != sorted(scores, reverse=True):
                raise ValueError("Campaigns must be sorted by relevance_score in descending order")
        return v


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = Field(default="healthy", description="Service status")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    campaigns_indexed: int = Field(..., description="Number of campaigns in index")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
