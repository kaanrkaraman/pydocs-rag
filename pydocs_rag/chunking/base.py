"""Base classes for chunking."""

from abc import ABC, abstractmethod

from pydocs_rag.models import Chunk, ParsedDocument


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: ParsedDocument, corpus_version: str) -> list[Chunk]:
        pass

    @abstractmethod
    def get_config_dict(self) -> dict:
        pass
