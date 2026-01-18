"""Structured logging for the RAG system."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from pydocs_rag.config import LoggingConfig
from pydocs_rag.models import QueryTrace


def configure_logging(config: LoggingConfig) -> None:
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


@dataclass
class QueryLogEntry:
    trace_id: str
    timestamp: datetime
    corpus_version: str
    index_version: str
    config_hash: str
    rewrite_enabled: bool
    reranker_enabled: bool
    retrieval_k: int
    original_query: str
    rewritten_query: str | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    rerank_scores: list[float] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    refusal_triggered: bool = False
    refusal_reason: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "corpus_version": self.corpus_version,
            "index_version": self.index_version,
            "config_hash": self.config_hash,
            "rewrite_enabled": self.rewrite_enabled,
            "reranker_enabled": self.reranker_enabled,
            "retrieval_k": self.retrieval_k,
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieval_scores": self.retrieval_scores,
            "rerank_scores": self.rerank_scores,
            "citations": self.citations,
            "refusal_triggered": self.refusal_triggered,
            "refusal_reason": self.refusal_reason,
            "timing_ms": self.timing_ms,
            "total_ms": self.total_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


class QueryLogger:
    def __init__(self, config: LoggingConfig, output_dir: Path | None = None) -> None:
        self.config = config
        self.output_dir = output_dir or Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / f"queries_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.logger = get_logger("query")

    def log_query(
        self,
        trace: QueryTrace,
        input_tokens: int = 0,
        output_tokens: int = 0,
        rewrite_enabled: bool = False,
        reranker_enabled: bool = False,
        retrieval_k: int = 20,
        error: str | None = None,
    ) -> None:
        entry = QueryLogEntry(
            trace_id=trace.trace_id,
            timestamp=trace.timestamp,
            corpus_version=trace.corpus_version,
            index_version=trace.index_version,
            config_hash=trace.config_hash,
            rewrite_enabled=rewrite_enabled,
            reranker_enabled=reranker_enabled,
            retrieval_k=retrieval_k,
            original_query=trace.original_query,
            rewritten_query=trace.rewritten_query,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in trace.retrieval_results],
            retrieval_scores=[r.score for r in trace.retrieval_results],
            rerank_scores=[r.rerank_score for r in trace.reranked_results if r.rerank_score]
            if trace.reranked_results
            else None,
            citations=[{"chunk_id": c.chunk_id, "url": c.url} for c in trace.citations],
            refusal_triggered=trace.refusal_triggered,
            refusal_reason=trace.refusal_reason.value if trace.refusal_reason else None,
            timing_ms=trace.timing_ms,
            total_ms=sum(trace.timing_ms.values()),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

        if self.config.include_traces:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

        self.logger.info(
            "query_processed",
            trace_id=entry.trace_id,
            total_ms=entry.total_ms,
            num_results=len(entry.retrieved_chunk_ids),
            refusal=entry.refusal_triggered,
        )

    def get_recent_queries(self, limit: int = 100) -> list[QueryLogEntry]:
        entries = []
        if not self.log_file.exists():
            return entries

        with open(self.log_file) as f:
            lines = f.readlines()[-limit:]
            for line in lines:
                data = json.loads(line)
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                entries.append(QueryLogEntry(**data))

        return entries
