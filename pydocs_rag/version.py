"""Versioning utilities for corpus, chunks, embeddings, and indices."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return compute_content_hash(f.read())


def compute_manifest_hash(manifest_entries: list[dict[str, Any]]) -> str:
    sorted_entries = sorted(manifest_entries, key=lambda x: x.get("url", ""))
    content_hashes = [e.get("content_hash", "") for e in sorted_entries]
    combined = "".join(content_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


@dataclass
class CorpusVersion:
    version_id: str
    manifest_hash: str
    created_at: datetime
    num_pages: int
    sources: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "manifest_hash": self.manifest_hash,
            "created_at": self.created_at.isoformat(),
            "num_pages": self.num_pages,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusVersion":
        return cls(
            version_id=data["version_id"],
            manifest_hash=data["manifest_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            num_pages=data["num_pages"],
            sources=data["sources"],
        )


@dataclass
class ChunkVersion:
    version_id: str
    corpus_version: str
    chunking_config_hash: str
    created_at: datetime
    num_chunks: int
    chunker_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "corpus_version": self.corpus_version,
            "chunking_config_hash": self.chunking_config_hash,
            "created_at": self.created_at.isoformat(),
            "num_chunks": self.num_chunks,
            "chunker_type": self.chunker_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkVersion":
        return cls(
            version_id=data["version_id"],
            corpus_version=data["corpus_version"],
            chunking_config_hash=data["chunking_config_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            num_chunks=data["num_chunks"],
            chunker_type=data["chunker_type"],
        )


@dataclass
class EmbeddingVersion:
    version_id: str
    chunk_version: str
    model_name: str
    model_hash: str
    normalize: bool
    created_at: datetime
    num_vectors: int
    dimensions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "chunk_version": self.chunk_version,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "normalize": self.normalize,
            "created_at": self.created_at.isoformat(),
            "num_vectors": self.num_vectors,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingVersion":
        return cls(
            version_id=data["version_id"],
            chunk_version=data["chunk_version"],
            model_name=data["model_name"],
            model_hash=data["model_hash"],
            normalize=data["normalize"],
            created_at=datetime.fromisoformat(data["created_at"]),
            num_vectors=data["num_vectors"],
            dimensions=data["dimensions"],
        )


@dataclass
class IndexVersion:
    version_id: str
    embedding_version: str
    index_type: str
    index_config_hash: str
    created_at: datetime
    num_vectors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "embedding_version": self.embedding_version,
            "index_type": self.index_type,
            "index_config_hash": self.index_config_hash,
            "created_at": self.created_at.isoformat(),
            "num_vectors": self.num_vectors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexVersion":
        return cls(
            version_id=data["version_id"],
            embedding_version=data["embedding_version"],
            index_type=data["index_type"],
            index_config_hash=data["index_config_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            num_vectors=data["num_vectors"],
        )


def compute_corpus_version(manifest_hash: str, sources: list[str]) -> str:
    combined = manifest_hash + "".join(sorted(sources))
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def compute_chunk_version(corpus_version: str, chunking_config: dict[str, Any]) -> str:
    config_str = json.dumps(chunking_config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    combined = f"{corpus_version}:{config_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def compute_embedding_version(chunk_version: str, model_name: str, normalize: bool) -> str:
    combined = f"{chunk_version}:{model_name}:{normalize}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def compute_index_version(
    embedding_version: str, index_type: str, index_config: dict[str, Any]
) -> str:
    config_str = json.dumps(index_config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    combined = f"{embedding_version}:{index_type}:{config_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def save_version_info(
    version_obj: CorpusVersion | ChunkVersion | EmbeddingVersion | IndexVersion, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(version_obj.to_dict(), f, indent=2)


def load_version_info(
    path: Path,
    version_class: type[CorpusVersion]
    | type[ChunkVersion]
    | type[EmbeddingVersion]
    | type[IndexVersion],
) -> CorpusVersion | ChunkVersion | EmbeddingVersion | IndexVersion:
    with open(path) as f:
        data = json.load(f)
    return version_class.from_dict(data)
