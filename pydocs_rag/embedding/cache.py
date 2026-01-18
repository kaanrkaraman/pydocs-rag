"""Embedding cache for query embeddings."""

import hashlib
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: Path, model_name: str) -> None:
        self.cache_dir = cache_dir / "query_embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:8]

    def _get_cache_key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{self.model_hash}_{text_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.npy"

    def get(self, text: str) -> np.ndarray | None:
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                return None
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)

        try:
            np.save(cache_path, embedding)
        except Exception:
            pass

    def clear(self) -> None:
        for path in self.cache_dir.glob("*.npy"):
            path.unlink()

    def get_stats(self) -> dict:
        cache_files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "num_entries": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "model_name": self.model_name,
        }
