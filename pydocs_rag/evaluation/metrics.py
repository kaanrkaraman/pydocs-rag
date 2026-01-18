"""Retrieval and answer quality metrics."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalMetrics:
    hit_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "precision_at_k": self.precision_at_k,
            "recall": self.recall,
        }


@dataclass
class AnswerMetrics:
    correctness: float
    groundedness: float
    completeness: float
    weighted_score: float
    citation_count: int
    has_hallucination: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "completeness": self.completeness,
            "weighted_score": self.weighted_score,
            "citation_count": self.citation_count,
            "has_hallucination": self.has_hallucination,
        }


def compute_hit_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> dict[int, float]:
    results = {}
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        hit = 1.0 if top_k & relevant_ids else 0.0
        results[k] = hit
    return results


def compute_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg(relevances: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k_values: list[int],
) -> dict[int, float]:
    relevances = [relevance_scores.get(doc_id, 0.0) for doc_id in retrieved_ids]

    ideal_relevances = sorted(relevance_scores.values(), reverse=True)

    results = {}
    for k in k_values:
        dcg = compute_dcg(relevances, k)
        idcg = compute_dcg(ideal_relevances, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        results[k] = ndcg

    return results


def compute_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> dict[int, float]:
    results = {}
    for k in k_values:
        top_k = retrieved_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        precision = relevant_in_top_k / k if k > 0 else 0.0
        results[k] = precision
    return results


def compute_recall(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_relevant = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
    return retrieved_relevant / len(relevant_ids)


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int] = [1, 3, 5, 10],
    relevance_scores: dict[str, float] | None = None,
) -> RetrievalMetrics:
    if relevance_scores is None:
        relevance_scores = dict.fromkeys(relevant_ids, 1.0)

    return RetrievalMetrics(
        hit_at_k=compute_hit_at_k(retrieved_ids, relevant_ids, k_values),
        mrr=compute_mrr(retrieved_ids, relevant_ids),
        ndcg_at_k=compute_ndcg_at_k(retrieved_ids, relevance_scores, k_values),
        precision_at_k=compute_precision_at_k(retrieved_ids, relevant_ids, k_values),
        recall=compute_recall(retrieved_ids, relevant_ids),
    )


def aggregate_retrieval_metrics(
    all_metrics: list[RetrievalMetrics],
) -> RetrievalMetrics:
    if not all_metrics:
        return RetrievalMetrics(hit_at_k={}, mrr=0.0, ndcg_at_k={}, precision_at_k={}, recall=0.0)

    k_values = list(all_metrics[0].hit_at_k.keys())

    avg_hit_at_k = {}
    avg_ndcg_at_k = {}
    avg_precision_at_k = {}

    for k in k_values:
        avg_hit_at_k[k] = sum(m.hit_at_k.get(k, 0) for m in all_metrics) / len(all_metrics)
        avg_ndcg_at_k[k] = sum(m.ndcg_at_k.get(k, 0) for m in all_metrics) / len(all_metrics)
        avg_precision_at_k[k] = sum(m.precision_at_k.get(k, 0) for m in all_metrics) / len(
            all_metrics
        )

    avg_mrr = sum(m.mrr for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m.recall for m in all_metrics) / len(all_metrics)

    return RetrievalMetrics(
        hit_at_k=avg_hit_at_k,
        mrr=avg_mrr,
        ndcg_at_k=avg_ndcg_at_k,
        precision_at_k=avg_precision_at_k,
        recall=avg_recall,
    )
