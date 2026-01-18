"""Fixed-size chunker baseline implementation."""

import tiktoken

from pydocs_rag.chunking.base import BaseChunker
from pydocs_rag.config import FixedSizeChunkingConfig
from pydocs_rag.models import Chunk, ChunkMetadata, ParsedDocument


class FixedSizeChunker(BaseChunker):
    def __init__(self, config: FixedSizeChunkingConfig) -> None:
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _split_by_tokens(self, text: str) -> list[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + self.config.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            if end >= len(tokens):
                break
            start = end - self.config.overlap

        return chunks

    def _split_by_chars(self, text: str) -> list[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))

            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(text):
                break
            start = max(start + 1, end - self.config.overlap)

        return chunks

    def chunk(self, document: ParsedDocument, corpus_version: str) -> list[Chunk]:
        chunks = []
        full_text = document.get_full_text()

        if self.config.split_by == "tokens":
            text_chunks = self._split_by_tokens(full_text)
        else:
            text_chunks = self._split_by_chars(full_text)

        for idx, text in enumerate(text_chunks):
            has_code = "```" in text or "`" in text

            chunk_id = f"{corpus_version}:{document.doc_id}:fixed:{idx}"

            metadata = ChunkMetadata(
                doc_id=document.doc_id,
                url=document.url,
                title=document.title,
                section_path=[],
                section_anchor=None,
                heading="",
                heading_level=0,
                chunk_index=idx,
                total_chunks_in_section=len(text_chunks),
                has_code=has_code,
                crawl_timestamp=document.metadata.crawl_timestamp,
                source_name=document.metadata.source_name,
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    corpus_version=corpus_version,
                )
            )

        return chunks

    def get_config_dict(self) -> dict:
        return {
            "type": "fixed_size",
            "chunk_size": self.config.chunk_size,
            "overlap": self.config.overlap,
            "split_by": self.config.split_by,
        }
