"""Failure mode detection and handling strategies."""

from dataclasses import dataclass
from enum import Enum

from pydocs_rag.config import FailureModesConfig
from pydocs_rag.models import Chunk, QueryTrace, RetrievalResult


class FailureMode(Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    HALLUCINATION_RISK = "hallucination_risk"
    STALE_DOCS = "stale_docs"
    AMBIGUOUS_QUERY = "ambiguous_query"
    CONTEXT_OVERFLOW = "context_overflow"


@dataclass
class FailureModeDetection:
    mode: FailureMode
    detected: bool
    severity: float
    details: str
    suggested_action: str


class FailureModeHandler:
    def __init__(self, config: FailureModesConfig) -> None:
        self.config = config

    def detect_retrieval_miss(
        self,
        results: list[RetrievalResult],
        score_threshold: float = 0.3,
    ) -> FailureModeDetection:
        if not results:
            return FailureModeDetection(
                mode=FailureMode.RETRIEVAL_MISS,
                detected=True,
                severity=1.0,
                details="No retrieval results found",
                suggested_action="expand_k_and_rewrite",
            )

        top_score = results[0].rerank_score if results[0].rerank_score else results[0].score

        if top_score < score_threshold:
            return FailureModeDetection(
                mode=FailureMode.RETRIEVAL_MISS,
                detected=True,
                severity=1.0 - top_score,
                details=f"Top score {top_score:.3f} below threshold {score_threshold}",
                suggested_action="expand_k_and_rewrite"
                if self.config.retrieval_miss.try_rewrite_if_off
                else "expand_k",
            )

        return FailureModeDetection(
            mode=FailureMode.RETRIEVAL_MISS,
            detected=False,
            severity=0.0,
            details="Retrieval scores acceptable",
            suggested_action="none",
        )

    def detect_ambiguous_query(
        self,
        results: list[RetrievalResult],
        score_margin_threshold: float = 0.05,
    ) -> FailureModeDetection:
        if len(results) < 2:
            return FailureModeDetection(
                mode=FailureMode.AMBIGUOUS_QUERY,
                detected=False,
                severity=0.0,
                details="Not enough results to determine ambiguity",
                suggested_action="none",
            )

        top_score = results[0].rerank_score if results[0].rerank_score else results[0].score
        second_score = results[1].rerank_score if results[1].rerank_score else results[1].score
        margin = top_score - second_score

        top_topics = set()
        for r in results[:5]:
            if r.chunk.metadata.section_path:
                top_topics.add(tuple(r.chunk.metadata.section_path[:2]))

        if margin < score_margin_threshold and len(top_topics) > 2:
            return FailureModeDetection(
                mode=FailureMode.AMBIGUOUS_QUERY,
                detected=True,
                severity=1.0 - margin,
                details=f"Score margin {margin:.3f} small, {len(top_topics)} different topics in top results",
                suggested_action="ask_clarification"
                if self.config.ambiguous_query.ask_clarification
                else "proceed_with_warning",
            )

        return FailureModeDetection(
            mode=FailureMode.AMBIGUOUS_QUERY,
            detected=False,
            severity=0.0,
            details="Query specificity acceptable",
            suggested_action="none",
        )

    def detect_stale_docs(
        self,
        chunks: list[Chunk],
    ) -> FailureModeDetection:
        from datetime import datetime, timedelta

        if not chunks:
            return FailureModeDetection(
                mode=FailureMode.STALE_DOCS,
                detected=False,
                severity=0.0,
                details="No chunks to check",
                suggested_action="none",
            )

        max_age = timedelta(days=self.config.stale_docs.max_age_days)
        cutoff = datetime.now() - max_age

        stale_count = sum(1 for c in chunks if c.metadata.crawl_timestamp < cutoff)
        stale_ratio = stale_count / len(chunks)

        if stale_ratio > 0.5:
            oldest = min(c.metadata.crawl_timestamp for c in chunks)
            age_days = (datetime.now() - oldest).days

            return FailureModeDetection(
                mode=FailureMode.STALE_DOCS,
                detected=True,
                severity=stale_ratio,
                details=f"{stale_count}/{len(chunks)} chunks are older than {self.config.stale_docs.max_age_days} days (oldest: {age_days} days)",
                suggested_action="show_warning"
                if self.config.stale_docs.show_crawl_timestamp
                else "none",
            )

        return FailureModeDetection(
            mode=FailureMode.STALE_DOCS,
            detected=False,
            severity=stale_ratio,
            details="Document freshness acceptable",
            suggested_action="none",
        )

    def detect_context_overflow(
        self,
        chunks: list[Chunk],
        max_tokens: int,
        current_tokens: int,
    ) -> FailureModeDetection:
        if current_tokens > max_tokens:
            overflow_ratio = current_tokens / max_tokens

            return FailureModeDetection(
                mode=FailureMode.CONTEXT_OVERFLOW,
                detected=True,
                severity=min(1.0, overflow_ratio - 1.0),
                details=f"Context has {current_tokens} tokens, max is {max_tokens}",
                suggested_action=self.config.context_overflow.strategy,
            )

        return FailureModeDetection(
            mode=FailureMode.CONTEXT_OVERFLOW,
            detected=False,
            severity=0.0,
            details="Context size within limits",
            suggested_action="none",
        )

    def detect_all(
        self,
        trace: QueryTrace,
        max_context_tokens: int = 4000,
    ) -> list[FailureModeDetection]:
        detections = []

        results = trace.reranked_results if trace.reranked_results else trace.retrieval_results
        detections.append(self.detect_retrieval_miss(results))
        detections.append(self.detect_ambiguous_query(results))
        detections.append(self.detect_stale_docs(trace.selected_chunks))

        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(tokenizer.encode(c.text)) for c in trace.selected_chunks)
        detections.append(
            self.detect_context_overflow(trace.selected_chunks, max_context_tokens, total_tokens)
        )

        return detections

    def get_recommended_actions(
        self,
        detections: list[FailureModeDetection],
    ) -> list[str]:
        actions = []

        for detection in detections:
            if detection.detected and detection.suggested_action != "none":
                actions.append(detection.suggested_action)

        return list(set(actions))
