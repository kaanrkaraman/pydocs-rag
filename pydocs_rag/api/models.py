"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to answer")
    include_trace: bool = Field(False, description="Include full trace in response")
    override_rewrite: bool | None = Field(None, description="Override rewrite setting")
    override_rerank: bool | None = Field(None, description="Override rerank setting")


class CitationResponse(BaseModel):
    chunk_id: str
    url: str
    title: str
    section: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    refusal_triggered: bool
    refusal_reason: str | None
    confidence: float
    warning: str | None
    timing_ms: dict[str, float] | None
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    corpus_version: str
    index_version: str
    config_hash: str


class IndexInfoResponse(BaseModel):
    index_version: str
    corpus_version: str
    num_vectors: int
    embedding_model: str
    index_type: str
    created_at: str


class IndexSwitchRequest(BaseModel):
    index_version: str


class IndexSwitchResponse(BaseModel):
    success: bool
    previous_version: str
    current_version: str
    message: str


class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    refusal_rate: float
    stage_timings: dict[str, dict[str, float]]
