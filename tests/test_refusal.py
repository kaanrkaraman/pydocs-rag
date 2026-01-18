"""Unit tests for refusal policy."""

from datetime import datetime

import pytest

from pydocs_rag.config import RefusalConfig
from pydocs_rag.generation.refusal import (
    RefusalDecision,
    check_citation_coverage,
    check_retrieval_confidence,
    generate_refusal_response,
)
from pydocs_rag.models import Chunk, ChunkMetadata, RefusalReason, RetrievalResult


@pytest.fixture
def refusal_config():
    return RefusalConfig(
        enabled=True,
        min_top_score=0.4,
        min_score_margin=0.1,
        min_citations=1,
    )


@pytest.fixture
def sample_chunk():
    return Chunk(
        chunk_id="test:chunk:1",
        text="Test content",
        metadata=ChunkMetadata(
            doc_id="doc1",
            url="http://example.com",
            title="Test Doc",
            section_path=["Section"],
            section_anchor="section",
            heading="Section",
            heading_level=1,
            chunk_index=0,
            total_chunks_in_section=1,
            has_code=False,
            crawl_timestamp=datetime.now(),
            source_name="test",
        ),
        corpus_version="v1",
    )


class TestRetrievalConfidence:
    def test_empty_results(self, refusal_config):
        decision = check_retrieval_confidence([], refusal_config)
        assert decision.should_refuse
        assert decision.reason == RefusalReason.NO_RELEVANT_CONTENT

    def test_low_score(self, refusal_config, sample_chunk):
        results = [RetrievalResult(chunk=sample_chunk, score=0.2, rank=0)]
        decision = check_retrieval_confidence(results, refusal_config)
        assert decision.should_refuse
        assert decision.reason == RefusalReason.LOW_CONFIDENCE

    def test_high_score(self, refusal_config, sample_chunk):
        results = [RetrievalResult(chunk=sample_chunk, score=0.8, rank=0)]
        decision = check_retrieval_confidence(results, refusal_config)
        assert not decision.should_refuse


class TestCitationCoverage:
    def test_insufficient_citations(self, refusal_config):
        decision = check_citation_coverage(0, 5, refusal_config)
        assert decision.should_refuse
        assert decision.reason == RefusalReason.INSUFFICIENT_CITATIONS

    def test_sufficient_citations(self, refusal_config):
        decision = check_citation_coverage(2, 5, refusal_config)
        assert not decision.should_refuse


class TestRefusalResponse:
    def test_no_content_response(self, sample_chunk):
        decision = RefusalDecision(
            should_refuse=True,
            reason=RefusalReason.NO_RELEVANT_CONTENT,
            confidence_score=0.0,
            details="test",
        )
        response = generate_refusal_response(decision, [], "extractive")
        assert "couldn't find" in response.lower()

    def test_low_confidence_extractive(self, sample_chunk):
        decision = RefusalDecision(
            should_refuse=True,
            reason=RefusalReason.LOW_CONFIDENCE,
            confidence_score=0.3,
            details="test",
        )
        response = generate_refusal_response(decision, [sample_chunk], "extractive")
        assert "Low Confidence" in response or "potentially relevant" in response
