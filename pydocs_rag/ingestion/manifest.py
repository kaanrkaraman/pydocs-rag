"""Manifest data structures for corpus versioning."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydocs_rag.version import compute_corpus_version, compute_manifest_hash


@dataclass
class ManifestEntry:
    url: str
    crawl_timestamp: datetime
    http_status: int
    content_hash: str
    file_path: str
    last_modified: str | None = None
    etag: str | None = None
    content_length: int = 0
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "crawl_timestamp": self.crawl_timestamp.isoformat(),
            "http_status": self.http_status,
            "content_hash": self.content_hash,
            "file_path": self.file_path,
            "last_modified": self.last_modified,
            "etag": self.etag,
            "content_length": self.content_length,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestEntry":
        return cls(
            url=data["url"],
            crawl_timestamp=datetime.fromisoformat(data["crawl_timestamp"]),
            http_status=data["http_status"],
            content_hash=data["content_hash"],
            file_path=data["file_path"],
            last_modified=data.get("last_modified"),
            etag=data.get("etag"),
            content_length=data.get("content_length", 0),
            title=data.get("title", ""),
        )


@dataclass
class Manifest:
    entries: list[ManifestEntry] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    corpus_version: str = ""

    def __post_init__(self) -> None:
        if not self.corpus_version and self.entries:
            self._compute_version()

    def _compute_version(self) -> None:
        entries_dict = [e.to_dict() for e in self.entries]
        manifest_hash = compute_manifest_hash(entries_dict)
        self.corpus_version = compute_corpus_version(manifest_hash, self.sources)

    def add_entry(self, entry: ManifestEntry) -> None:
        self.entries.append(entry)
        self._compute_version()

    def get_entry_by_url(self, url: str) -> ManifestEntry | None:
        for entry in self.entries:
            if entry.url == url:
                return entry
        return None

    def get_successful_entries(self) -> list[ManifestEntry]:
        return [e for e in self.entries if e.http_status == 200]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_version": self.corpus_version,
            "created_at": self.created_at.isoformat(),
            "sources": self.sources,
            "num_entries": len(self.entries),
            "num_successful": len(self.get_successful_entries()),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        entries = [ManifestEntry.from_dict(e) for e in data["entries"]]
        return cls(
            entries=entries,
            sources=data["sources"],
            created_at=datetime.fromisoformat(data["created_at"]),
            corpus_version=data["corpus_version"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
