"""Evaluation harness for running pipeline on test datasets."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydocs_rag.config import RAGConfig
from pydocs_rag.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
)
from pydocs_rag.evaluation.rubric import RubricScore, RubricScorer
from pydocs_rag.generation.generator import Generator
from pydocs_rag.models import QueryTrace
from pydocs_rag.query.pipeline import QueryPipeline


@dataclass
class EvalSample:
    query_id: str
    query: str
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSample":
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            expected_doc_ids=data.get("expected_doc_ids", []),
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            reference_answer=data.get("reference_answer"),
            tags=data.get("tags", []),
        )


@dataclass
class EvalResult:
    sample: EvalSample
    trace: QueryTrace
    answer: str
    retrieval_metrics: RetrievalMetrics | None
    rubric_score: RubricScore | None
    latency_ms: float
    refusal_triggered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.sample.query_id,
            "query": self.sample.query,
            "answer": self.answer,
            "retrieval_metrics": self.retrieval_metrics.to_dict()
            if self.retrieval_metrics
            else None,
            "rubric_score": self.rubric_score.to_dict() if self.rubric_score else None,
            "latency_ms": self.latency_ms,
            "refusal_triggered": self.refusal_triggered,
            "retrieved_chunk_ids": [r.chunk.chunk_id for r in self.trace.retrieval_results],
            "timing": self.trace.timing_ms,
        }


@dataclass
class EvalReport:
    run_id: str
    timestamp: datetime
    config_hash: str
    corpus_version: str
    index_version: str
    num_samples: int
    aggregate_metrics: RetrievalMetrics | None
    avg_rubric_score: float
    avg_latency_ms: float
    refusal_rate: float
    results: list[EvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "config_hash": self.config_hash,
            "corpus_version": self.corpus_version,
            "index_version": self.index_version,
            "num_samples": self.num_samples,
            "aggregate_metrics": self.aggregate_metrics.to_dict()
            if self.aggregate_metrics
            else None,
            "avg_rubric_score": self.avg_rubric_score,
            "avg_latency_ms": self.avg_latency_ms,
            "refusal_rate": self.refusal_rate,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class EvaluationHarness:
    def __init__(
        self,
        pipeline: QueryPipeline,
        generator: Generator,
        config: RAGConfig,
        rubric_scorer: RubricScorer | None = None,
        corpus_version: str = "",
        index_version: str = "",
    ) -> None:
        self.pipeline = pipeline
        self.generator = generator
        self.config = config
        self.rubric_scorer = rubric_scorer
        self.corpus_version = corpus_version
        self.index_version = index_version

    def load_dataset(self, path: Path) -> list[EvalSample]:
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    samples.append(EvalSample.from_dict(data))
        return samples

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        start_time = time.perf_counter()

        trace = self.pipeline.run(
            sample.query,
            corpus_version=self.corpus_version,
            index_version=self.index_version,
        )

        results = trace.reranked_results if trace.reranked_results else trace.retrieval_results
        gen_result = self.generator.generate(
            sample.query,
            trace.selected_chunks,
            results,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        retrieval_metrics = None
        if sample.expected_chunk_ids or sample.expected_doc_ids:
            retrieved_ids = [r.chunk.chunk_id for r in trace.retrieval_results]

            if sample.expected_chunk_ids:
                relevant_ids = set(sample.expected_chunk_ids)
            else:
                relevant_ids = set()
                for r in trace.retrieval_results:
                    if r.chunk.metadata.doc_id in sample.expected_doc_ids:
                        relevant_ids.add(r.chunk.chunk_id)

            retrieval_metrics = compute_retrieval_metrics(
                retrieved_ids,
                relevant_ids,
                self.config.evaluation.k_values,
            )

        rubric_score = None
        if self.rubric_scorer and not gen_result.refusal_triggered:
            context = "\n\n".join(c.text for c in trace.selected_chunks)
            rubric_score = self.rubric_scorer.score(
                question=sample.query,
                answer=gen_result.answer,
                context=context,
                reference=sample.reference_answer,
            )

        answer = gen_result.fallback_response if gen_result.refusal_triggered else gen_result.answer

        return EvalResult(
            sample=sample,
            trace=trace,
            answer=answer or "",
            retrieval_metrics=retrieval_metrics,
            rubric_score=rubric_score,
            latency_ms=latency_ms,
            refusal_triggered=gen_result.refusal_triggered,
        )

    def run(
        self,
        dataset_path: Path,
        run_id: str | None = None,
    ) -> EvalReport:
        import uuid

        samples = self.load_dataset(dataset_path)
        run_id = run_id or str(uuid.uuid4())[:8]

        results = []
        for sample in samples:
            result = self.evaluate_sample(sample)
            results.append(result)
            print(f"  [{len(results)}/{len(samples)}] {sample.query_id}: {result.latency_ms:.0f}ms")

        retrieval_metrics_list = [r.retrieval_metrics for r in results if r.retrieval_metrics]
        aggregate_metrics = (
            aggregate_retrieval_metrics(retrieval_metrics_list) if retrieval_metrics_list else None
        )

        rubric_scores = [r.rubric_score.weighted_score for r in results if r.rubric_score]
        avg_rubric = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 0.0

        avg_latency = sum(r.latency_ms for r in results) / len(results)
        refusal_rate = sum(1 for r in results if r.refusal_triggered) / len(results)

        return EvalReport(
            run_id=run_id,
            timestamp=datetime.now(),
            config_hash=self.config.get_config_hash(),
            corpus_version=self.corpus_version,
            index_version=self.index_version,
            num_samples=len(samples),
            aggregate_metrics=aggregate_metrics,
            avg_rubric_score=avg_rubric,
            avg_latency_ms=avg_latency,
            refusal_rate=refusal_rate,
            results=results,
        )
