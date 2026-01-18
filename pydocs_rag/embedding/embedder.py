"""Embedding computation using sentence transformers."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from pydocs_rag.config import EmbeddingConfig
from pydocs_rag.models import Chunk


class Embedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model = SentenceTransformer(config.model_name, trust_remote_code=True)
        if config.max_seq_length:
            self.model.max_seq_length = config.max_seq_length

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )
        return embedding[0]

    def get_dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def get_model_config(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "normalize": self.config.normalize,
            "max_seq_length": self.config.max_seq_length,
            "dimension": self.get_dimension(),
        }


class EmbeddingStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        version_id: str,
        metadata: dict[str, Any],
    ) -> Path:
        version_dir = self.base_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = version_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        chunk_ids_path = version_dir / "chunk_ids.json"
        with open(chunk_ids_path, "w") as f:
            json.dump(chunk_ids, f)

        metadata_path = version_dir / "metadata.json"
        full_metadata = {
            **metadata,
            "num_embeddings": len(chunk_ids),
            "dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
        }
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        return version_dir

    def load(self, version_id: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        version_dir = self.base_dir / version_id

        embeddings_path = version_dir / "embeddings.npy"
        embeddings = np.load(embeddings_path)

        chunk_ids_path = version_dir / "chunk_ids.json"
        with open(chunk_ids_path) as f:
            chunk_ids = json.load(f)

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        return embeddings, chunk_ids, metadata

    def exists(self, version_id: str) -> bool:
        version_dir = self.base_dir / version_id
        return (
            version_dir.exists()
            and (version_dir / "embeddings.npy").exists()
            and (version_dir / "chunk_ids.json").exists()
        )

    def list_versions(self) -> list[str]:
        versions = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "embeddings.npy").exists():
                versions.append(path.name)
        return sorted(versions)
