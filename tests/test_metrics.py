"""Unit tests for evaluation metrics."""

import pytest

from pydocs_rag.evaluation.metrics import (
    aggregate_retrieval_metrics,
    compute_hit_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall,
)


class TestHitAtK:
    def test_hit_found(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"c"}
        result = compute_hit_at_k(retrieved, relevant, [1, 3, 5])
        assert result[1] == 0.0
        assert result[3] == 1.0
        assert result[5] == 1.0

    def test_no_hit(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x"}
        result = compute_hit_at_k(retrieved, relevant, [1, 3])
        assert result[1] == 0.0
        assert result[3] == 0.0


class TestMRR:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert compute_mrr(retrieved, relevant) == 1.0

    def test_third_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"c"}
        assert compute_mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_not_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"x"}
        assert compute_mrr(retrieved, relevant) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
        result = compute_ndcg_at_k(retrieved, relevance, [3])
        assert result[3] == pytest.approx(1.0)

    def test_empty_retrieval(self):
        retrieved = []
        relevance = {"a": 1.0}
        result = compute_ndcg_at_k(retrieved, relevance, [3])
        assert result[3] == 0.0


class TestPrecision:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = compute_precision_at_k(retrieved, relevant, [3])
        assert result[3] == 1.0

    def test_partial_relevant(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        result = compute_precision_at_k(retrieved, relevant, [4])
        assert result[4] == 0.5


class TestRecall:
    def test_full_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert compute_recall(retrieved, relevant) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c", "d"}
        assert compute_recall(retrieved, relevant) == 0.5


class TestAggregation:
    def test_aggregate_metrics(self):
        from pydocs_rag.evaluation.metrics import RetrievalMetrics

        metrics = [
            RetrievalMetrics(
                hit_at_k={5: 1.0},
                mrr=1.0,
                ndcg_at_k={5: 1.0},
                precision_at_k={5: 0.2},
                recall=1.0,
            ),
            RetrievalMetrics(
                hit_at_k={5: 0.0},
                mrr=0.0,
                ndcg_at_k={5: 0.0},
                precision_at_k={5: 0.0},
                recall=0.0,
            ),
        ]

        agg = aggregate_retrieval_metrics(metrics)
        assert agg.mrr == 0.5
        assert agg.hit_at_k[5] == 0.5
