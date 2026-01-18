"""FAISS index builder supporting multiple index types."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from pydocs_rag.config import FAISSConfig


class FAISSIndexBuilder:
    def __init__(self, config: FAISSConfig) -> None:
        self.config = config

    def _create_flat_index(self, dimension: int) -> faiss.Index:
        if self.config.metric == "inner_product":
            return faiss.IndexFlatIP(dimension)
        return faiss.IndexFlatL2(dimension)

    def _create_ivf_index(self, dimension: int, embeddings: np.ndarray) -> faiss.Index:
        nlist = min(self.config.ivf.nlist, len(embeddings) // 10)
        nlist = max(1, nlist)

        if self.config.metric == "inner_product":
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

        index.train(embeddings)
        index.nprobe = min(self.config.ivf.nprobe, nlist)

        return index

    def _create_hnsw_index(self, dimension: int) -> faiss.Index:
        if self.config.metric == "inner_product":
            index = faiss.IndexHNSWFlat(dimension, self.config.hnsw.M, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexHNSWFlat(dimension, self.config.hnsw.M)

        index.hnsw.efConstruction = self.config.hnsw.ef_construction
        index.hnsw.efSearch = self.config.hnsw.ef_search

        return index

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        dimension = embeddings.shape[1]

        if self.config.index_type == "Flat":
            index = self._create_flat_index(dimension)
        elif self.config.index_type == "IVF":
            index = self._create_ivf_index(dimension, embeddings)
        elif self.config.index_type == "HNSW":
            index = self._create_hnsw_index(dimension)
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        if self.config.index_type != "IVF":
            index.add(embeddings)
        else:
            index.add(embeddings)

        return index

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = {
            "index_type": self.config.index_type,
            "metric": self.config.metric,
        }

        if self.config.index_type == "IVF":
            config_dict["ivf"] = {
                "nlist": self.config.ivf.nlist,
                "nprobe": self.config.ivf.nprobe,
            }
        elif self.config.index_type == "HNSW":
            config_dict["hnsw"] = {
                "M": self.config.hnsw.M,
                "ef_construction": self.config.hnsw.ef_construction,
                "ef_search": self.config.hnsw.ef_search,
            }

        return config_dict


class FAISSIndexStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        index: faiss.Index,
        chunk_ids: list[str],
        version_id: str,
        metadata: dict[str, Any],
    ) -> Path:
        version_dir = self.base_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        index_path = version_dir / "index.faiss"
        faiss.write_index(index, str(index_path))

        mapping_path = version_dir / "id_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({"vector_id_to_chunk_id": chunk_ids}, f)

        metadata_path = version_dir / "metadata.json"
        full_metadata = {
            **metadata,
            "num_vectors": index.ntotal,
            "dimension": index.d,
        }
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        return version_dir

    def load(self, version_id: str) -> tuple[faiss.Index, list[str], dict[str, Any]]:
        version_dir = self.base_dir / version_id

        index_path = version_dir / "index.faiss"
        index = faiss.read_index(str(index_path))

        mapping_path = version_dir / "id_mapping.json"
        with open(mapping_path) as f:
            mapping = json.load(f)
        chunk_ids = mapping["vector_id_to_chunk_id"]

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        return index, chunk_ids, metadata

    def exists(self, version_id: str) -> bool:
        version_dir = self.base_dir / version_id
        return (
            version_dir.exists()
            and (version_dir / "index.faiss").exists()
            and (version_dir / "id_mapping.json").exists()
        )

    def list_versions(self) -> list[str]:
        versions = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "index.faiss").exists():
                versions.append(path.name)
        return sorted(versions)

    def get_metadata(self, version_id: str) -> dict[str, Any] | None:
        metadata_path = self.base_dir / version_id / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None
