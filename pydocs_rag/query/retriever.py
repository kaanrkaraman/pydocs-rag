"""FAISS-based retriever for finding relevant chunks."""

import json
from pathlib import Path

import faiss
import numpy as np

from pydocs_rag.config import RetrievalConfig
from pydocs_rag.embedding.embedder import Embedder
from pydocs_rag.models import Chunk, RetrievalResult


class FAISSRetriever:
    def __init__(
        self,
        index: faiss.Index,
        chunk_id_mapping: list[str],
        chunks_by_id: dict[str, Chunk],
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> None:
        self.index = index
        self.chunk_id_mapping = chunk_id_mapping
        self.chunks_by_id = chunks_by_id
        self.embedder = embedder
        self.config = config

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.config.top_k

        query_embedding = self.embedder.embed_query(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Debug: log dimensions
        index_dim = self.index.d
        query_dim = query_embedding.shape[1]
        if index_dim != query_dim:
            raise ValueError(
                f"Dimension mismatch: index expects {index_dim}D vectors, "
                f"but query embedding is {query_dim}D. "
                f"This likely means the index was built with a different embedding model or configuration. "
                f"Current model: {self.embedder.config.model_name}"
            )

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunk_id_mapping):
                continue

            if score < self.config.score_threshold:
                continue

            chunk_id = self.chunk_id_mapping[idx]
            chunk = self.chunks_by_id.get(chunk_id)

            if chunk is None:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def retrieve_with_expanded_k(self, query: str, k_multiplier: int = 2) -> list[RetrievalResult]:
        expanded_k = self.config.top_k * k_multiplier
        return self.retrieve(query, top_k=expanded_k)

    @classmethod
    def from_index_store(
        cls,
        index_store_path: Path,
        index_version: str,
        chunks_path: Path,
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> "FAISSRetriever":
        index_dir = index_store_path / index_version

        index = faiss.read_index(str(index_dir / "index.faiss"))

        with open(index_dir / "id_mapping.json") as f:
            mapping = json.load(f)
        chunk_id_mapping = mapping["vector_id_to_chunk_id"]

        chunks_by_id = {}
        with open(chunks_path) as f:
            for line in f:
                chunk_data = json.loads(line)
                chunk = Chunk.from_dict(chunk_data)
                chunks_by_id[chunk.chunk_id] = chunk

        return cls(
            index=index,
            chunk_id_mapping=chunk_id_mapping,
            chunks_by_id=chunks_by_id,
            embedder=embedder,
            config=config,
        )
