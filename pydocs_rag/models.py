"""Data models for documents, chunks, and retrieval results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RefusalReason(Enum):
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS_QUERY = "ambiguous_query"
    NO_RELEVANT_CONTENT = "no_relevant_content"
    INSUFFICIENT_CITATIONS = "insufficient_citations"
    STALE_CONTENT = "stale_content"


@dataclass
class DocumentMetadata:
    url: str
    title: str
    source_name: str
    crawl_timestamp: datetime
    last_modified: str | None = None
    etag: str | None = None
    content_hash: str = ""
    section_path: list[str] = field(default_factory=list)
    anchor: str | None = None


@dataclass
class ParsedSection:
    heading: str
    heading_level: int
    content: str
    code_blocks: list[str] = field(default_factory=list)
    anchor: str | None = None
    parent_path: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    doc_id: str
    url: str
    title: str
    sections: list[ParsedSection]
    metadata: DocumentMetadata
    raw_text: str = ""

    def get_full_text(self) -> str:
        return "\n\n".join(
            f"{'#' * s.heading_level} {s.heading}\n{s.content}" for s in self.sections
        )


@dataclass
class ChunkMetadata:
    doc_id: str
    url: str
    title: str
    section_path: list[str]
    section_anchor: str | None
    heading: str
    heading_level: int
    chunk_index: int
    total_chunks_in_section: int
    has_code: bool
    crawl_timestamp: datetime
    source_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "section_path": self.section_path,
            "section_anchor": self.section_anchor,
            "heading": self.heading,
            "heading_level": self.heading_level,
            "chunk_index": self.chunk_index,
            "total_chunks_in_section": self.total_chunks_in_section,
            "has_code": self.has_code,
            "crawl_timestamp": self.crawl_timestamp.isoformat(),
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkMetadata":
        return cls(
            doc_id=data["doc_id"],
            url=data["url"],
            title=data["title"],
            section_path=data["section_path"],
            section_anchor=data.get("section_anchor"),
            heading=data["heading"],
            heading_level=data["heading_level"],
            chunk_index=data["chunk_index"],
            total_chunks_in_section=data["total_chunks_in_section"],
            has_code=data["has_code"],
            crawl_timestamp=datetime.fromisoformat(data["crawl_timestamp"]),
            source_name=data["source_name"],
        )


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    corpus_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "corpus_version": self.corpus_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            metadata=ChunkMetadata.from_dict(data["metadata"]),
            corpus_version=data["corpus_version"],
        )


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    rerank_score: float | None = None
    rerank_rank: int | None = None


@dataclass
class Citation:
    chunk_id: str
    url: str
    title: str
    section: str
    anchor: str | None
    snippet: str
    relevance_score: float

    def to_citation_string(self) -> str:
        url_with_anchor = self.url
        if self.anchor:
            url_with_anchor = f"{self.url}#{self.anchor}"
        return f"[{self.title} - {self.section}]({url_with_anchor})"


@dataclass
class QueryTrace:
    trace_id: str
    timestamp: datetime
    original_query: str
    rewritten_query: str | None = None
    rewrite_rationale: str | None = None
    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    reranked_results: list[RetrievalResult] = field(default_factory=list)
    selected_chunks: list[Chunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    refusal_triggered: bool = False
    refusal_reason: RefusalReason | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    corpus_version: str = ""
    index_version: str = ""
    is_conversational: bool = False


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]
    refusal_triggered: bool
    refusal_reason: RefusalReason | None
    fallback_response: str | None
    input_tokens: int
    output_tokens: int
    model: str
    prompt_version: str


@dataclass
class QueryResponse:
    query: str
    answer: str
    citations: list[Citation]
    refusal_triggered: bool
    refusal_reason: str | None
    warning: str | None
    trace: QueryTrace | None
