"""Regression comparison between evaluation runs."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FailureBucket(Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    RERANKER_MISS = "reranker_miss"
    REWRITE_HARM = "rewrite_harm"
    TRUNCATION_HARM = "truncation_harm"
    GENERATOR_UNSUPPORTED = "generator_unsupported"
    IMPROVED = "improved"
    STABLE = "stable"
    REGRESSED = "regressed"


@dataclass
class QueryComparison:
    query_id: str
    query: str
    bucket: FailureBucket
    baseline_score: float
    current_score: float
    delta: float
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "bucket": self.bucket.value,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "delta": self.delta,
            "details": self.details,
        }


@dataclass
class RegressionReport:
    baseline_run_id: str
    current_run_id: str
    total_queries: int
    improved_count: int
    regressed_count: int
    stable_count: int
    bucket_counts: dict[str, int]
    comparisons: list[QueryComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "total_queries": self.total_queries,
            "improved_count": self.improved_count,
            "regressed_count": self.regressed_count,
            "stable_count": self.stable_count,
            "bucket_counts": self.bucket_counts,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def generate_markdown(self) -> str:
        lines = [
            "# Regression Report",
            "",
            f"**Baseline Run**: {self.baseline_run_id}",
            f"**Current Run**: {self.current_run_id}",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Queries | {self.total_queries} |",
            f"| Improved | {self.improved_count} ({100 * self.improved_count / self.total_queries:.1f}%) |",
            f"| Regressed | {self.regressed_count} ({100 * self.regressed_count / self.total_queries:.1f}%) |",
            f"| Stable | {self.stable_count} ({100 * self.stable_count / self.total_queries:.1f}%) |",
            "",
            "## Failure Buckets",
            "",
        ]

        for bucket, count in sorted(self.bucket_counts.items()):
            if count > 0:
                lines.append(f"- **{bucket}**: {count}")

        if any(c.bucket == FailureBucket.REGRESSED for c in self.comparisons):
            lines.extend(
                [
                    "",
                    "## Regressed Queries",
                    "",
                ]
            )
            for comp in self.comparisons:
                if comp.bucket == FailureBucket.REGRESSED:
                    lines.append(f"### {comp.query_id}")
                    lines.append(f"Query: {comp.query}")
                    lines.append(
                        f"Score: {comp.baseline_score:.3f} → {comp.current_score:.3f} ({comp.delta:+.3f})"
                    )
                    lines.append(f"Details: {comp.details}")
                    lines.append("")

        return "\n".join(lines)


class RegressionAnalyzer:
    def __init__(
        self, improvement_threshold: float = 0.05, regression_threshold: float = -0.05
    ) -> None:
        self.improvement_threshold = improvement_threshold
        self.regression_threshold = regression_threshold

    def load_report(self, path: Path) -> dict[str, Any]:
        with open(path) as f:
            return json.load(f)

    def _get_primary_score(self, result: dict[str, Any]) -> float:
        if result.get("retrieval_metrics"):
            return result["retrieval_metrics"].get("mrr", 0.0)
        if result.get("rubric_score"):
            return result["rubric_score"].get("weighted_score", 0.0)
        return 0.0

    def _classify_change(
        self,
        baseline_result: dict[str, Any],
        current_result: dict[str, Any],
    ) -> tuple[FailureBucket, str]:
        baseline_score = self._get_primary_score(baseline_result)
        current_score = self._get_primary_score(current_result)
        delta = current_score - baseline_score

        if delta >= self.improvement_threshold:
            return FailureBucket.IMPROVED, "Score improved significantly"

        if delta <= self.regression_threshold:
            baseline_retrieved = set(baseline_result.get("retrieved_chunk_ids", [])[:10])
            current_retrieved = set(current_result.get("retrieved_chunk_ids", [])[:10])

            if baseline_retrieved and not (baseline_retrieved & current_retrieved):
                return FailureBucket.RETRIEVAL_MISS, "Top results completely different"

            if baseline_result.get("reranked") and current_result.get("reranked"):
                return FailureBucket.RERANKER_MISS, "Reranker ordering changed"

            if current_result.get("refusal_triggered") and not baseline_result.get(
                "refusal_triggered"
            ):
                return FailureBucket.GENERATOR_UNSUPPORTED, "Started refusing to answer"

            return FailureBucket.REGRESSED, "General regression"

        return FailureBucket.STABLE, "No significant change"

    def compare(
        self,
        baseline_path: Path,
        current_path: Path,
    ) -> RegressionReport:
        baseline_report = self.load_report(baseline_path)
        current_report = self.load_report(current_path)

        baseline_by_id = {r["query_id"]: r for r in baseline_report["results"]}
        current_by_id = {r["query_id"]: r for r in current_report["results"]}

        common_ids = set(baseline_by_id.keys()) & set(current_by_id.keys())

        comparisons = []
        bucket_counts: dict[str, int] = {b.value: 0 for b in FailureBucket}
        improved = regressed = stable = 0

        for query_id in sorted(common_ids):
            baseline_result = baseline_by_id[query_id]
            current_result = current_by_id[query_id]

            baseline_score = self._get_primary_score(baseline_result)
            current_score = self._get_primary_score(current_result)
            delta = current_score - baseline_score

            bucket, details = self._classify_change(baseline_result, current_result)
            bucket_counts[bucket.value] += 1

            if bucket == FailureBucket.IMPROVED:
                improved += 1
            elif bucket in (
                FailureBucket.REGRESSED,
                FailureBucket.RETRIEVAL_MISS,
                FailureBucket.RERANKER_MISS,
                FailureBucket.REWRITE_HARM,
                FailureBucket.TRUNCATION_HARM,
                FailureBucket.GENERATOR_UNSUPPORTED,
            ):
                regressed += 1
            else:
                stable += 1

            comparisons.append(
                QueryComparison(
                    query_id=query_id,
                    query=baseline_result.get("query", ""),
                    bucket=bucket,
                    baseline_score=baseline_score,
                    current_score=current_score,
                    delta=delta,
                    details=details,
                )
            )

        return RegressionReport(
            baseline_run_id=baseline_report["run_id"],
            current_run_id=current_report["run_id"],
            total_queries=len(common_ids),
            improved_count=improved,
            regressed_count=regressed,
            stable_count=stable,
            bucket_counts=bucket_counts,
            comparisons=comparisons,
        )
