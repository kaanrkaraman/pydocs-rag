"""Refusal policy for handling low-confidence and insufficient context."""

from dataclasses import dataclass

from pydocs_rag.config import RefusalConfig
from pydocs_rag.models import Chunk, RefusalReason, RetrievalResult


@dataclass
class RefusalDecision:
    should_refuse: bool
    reason: RefusalReason | None
    confidence_score: float
    details: str


def check_retrieval_confidence(
    results: list[RetrievalResult],
    config: RefusalConfig,
) -> RefusalDecision:
    if not results:
        return RefusalDecision(
            should_refuse=True,
            reason=RefusalReason.NO_RELEVANT_CONTENT,
            confidence_score=0.0,
            details="No retrieval results found",
        )

    top_score = results[0].rerank_score if results[0].rerank_score is not None else results[0].score

    if top_score < config.min_top_score:
        return RefusalDecision(
            should_refuse=True,
            reason=RefusalReason.LOW_CONFIDENCE,
            confidence_score=top_score,
            details=f"Top score {top_score:.3f} below threshold {config.min_top_score}",
        )

    if len(results) >= 2:
        second_score = (
            results[1].rerank_score if results[1].rerank_score is not None else results[1].score
        )
        margin = top_score - second_score

        if margin < config.min_score_margin and top_score < config.min_top_score * 1.5:
            return RefusalDecision(
                should_refuse=True,
                reason=RefusalReason.AMBIGUOUS_QUERY,
                confidence_score=top_score,
                details=f"Score margin {margin:.3f} too small, results may be ambiguous",
            )

    return RefusalDecision(
        should_refuse=False,
        reason=None,
        confidence_score=top_score,
        details="Retrieval confidence acceptable",
    )


def check_citation_coverage(
    num_cited: int,
    num_chunks: int,
    config: RefusalConfig,
) -> RefusalDecision:
    if num_cited < config.min_citations:
        return RefusalDecision(
            should_refuse=True,
            reason=RefusalReason.INSUFFICIENT_CITATIONS,
            confidence_score=num_cited / max(config.min_citations, 1),
            details=f"Only {num_cited} citations, minimum required: {config.min_citations}",
        )

    return RefusalDecision(
        should_refuse=False,
        reason=None,
        confidence_score=1.0,
        details="Citation coverage acceptable",
    )


def generate_refusal_response(
    decision: RefusalDecision,
    selected_chunks: list[Chunk],
    fallback_strategy: str,
) -> str:
    if decision.reason == RefusalReason.NO_RELEVANT_CONTENT:
        return (
            "I couldn't find relevant information in the Python documentation to answer "
            "your question. Could you please rephrase or provide more context?"
        )

    if decision.reason == RefusalReason.LOW_CONFIDENCE:
        if fallback_strategy == "extractive" and selected_chunks:
            excerpts = []
            for i, chunk in enumerate(selected_chunks[:3], 1):
                excerpt = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
                excerpts.append(f'[{i}] From {chunk.metadata.title}:\n"{excerpt}"')
            return (
                "⚠️ **Low Confidence Response**\n\n"
                "I found some potentially relevant content, but I'm not confident it fully "
                "answers your question. Here are the most relevant excerpts:\n\n"
                + "\n\n".join(excerpts)
            )
        return (
            "I found some information that might be related, but I'm not confident enough "
            "to provide a definitive answer. Could you be more specific or try rephrasing?"
        )

    if decision.reason == RefusalReason.AMBIGUOUS_QUERY:
        if selected_chunks:
            topics = set()
            for chunk in selected_chunks[:5]:
                if chunk.metadata.section_path:
                    topics.add(chunk.metadata.section_path[0])
                elif chunk.metadata.title:
                    topics.add(chunk.metadata.title)

            if len(topics) > 1:
                topic_list = ", ".join(list(topics)[:4])
                return (
                    f"Your query could relate to several topics: {topic_list}.\n\n"
                    "Could you clarify which aspect you're interested in?"
                )

        return (
            "Your question seems ambiguous. Could you provide more context or be more "
            "specific about what you're looking for?"
        )

    if decision.reason == RefusalReason.INSUFFICIENT_CITATIONS:
        return (
            "I couldn't find enough supporting documentation to provide a well-cited answer. "
            "The available information may be incomplete or the topic might not be covered "
            "in the indexed documentation."
        )

    return "I'm unable to provide a reliable answer at this time."


def generate_staleness_warning(chunks: list[Chunk], max_age_days: int = 90) -> str | None:
    from datetime import datetime, timedelta

    if not chunks:
        return None

    cutoff = datetime.now() - timedelta(days=max_age_days)
    stale_chunks = [c for c in chunks if c.metadata.crawl_timestamp < cutoff]

    if stale_chunks:
        oldest = min(c.metadata.crawl_timestamp for c in stale_chunks)
        return (
            f"⚠️ Some sources were last updated {(datetime.now() - oldest).days} days ago. "
            "Consider checking the official documentation for the latest information."
        )

    return None
