"""Cross-encoder reranker for improving retrieval precision."""

from sentence_transformers import CrossEncoder

from pydocs_rag.config import RerankerConfig
from pydocs_rag.models import RetrievalResult


class CrossEncoderReranker:
    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.model = CrossEncoder(config.model_name, max_length=512)

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not self.config.enabled or not results:
            return results

        pairs = [(query, result.chunk.text) for result in results]

        scores = self.model.predict(
            pairs, batch_size=self.config.batch_size, show_progress_bar=False
        )

        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        reranked = sorted(results, key=lambda x: x.rerank_score or 0, reverse=True)

        for new_rank, result in enumerate(reranked):
            result.rerank_rank = new_rank

        return reranked[: self.config.top_k]

    def score_single(self, query: str, text: str) -> float:
        return float(self.model.predict([(query, text)])[0])
