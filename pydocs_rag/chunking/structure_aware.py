"""Structure-aware chunker that respects document hierarchy."""

import re

import tiktoken

from pydocs_rag.chunking.base import BaseChunker
from pydocs_rag.config import StructureAwareChunkingConfig
from pydocs_rag.models import Chunk, ChunkMetadata, ParsedDocument, ParsedSection


class StructureAwareChunker(BaseChunker):
    def __init__(self, config: StructureAwareChunkingConfig) -> None:
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _extract_code_blocks(self, text: str) -> list[tuple[int, int, str]]:
        code_pattern = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
        blocks = []
        for match in code_pattern.finditer(text):
            blocks.append((match.start(), match.end(), match.group(0)))
        return blocks

    def _split_preserving_code(self, text: str, max_tokens: int) -> list[str]:
        if self._count_tokens(text) <= max_tokens:
            return [text]

        code_blocks = self._extract_code_blocks(text)

        if not code_blocks or not self.config.preserve_code_blocks:
            return self._split_by_paragraph(text, max_tokens)

        chunks = []
        current_pos = 0

        for start, end, code_block in code_blocks:
            before_code = text[current_pos:start].strip()
            if before_code:
                before_chunks = self._split_by_paragraph(before_code, max_tokens)
                chunks.extend(before_chunks)

            code_tokens = self._count_tokens(code_block)
            if code_tokens <= max_tokens:
                chunks.append(code_block)
            else:
                lines = code_block.split("\n")
                current_chunk_lines = []
                current_chunk_tokens = 0

                for line in lines:
                    line_tokens = self._count_tokens(line)
                    if current_chunk_tokens + line_tokens > max_tokens and current_chunk_lines:
                        chunks.append("\n".join(current_chunk_lines))
                        current_chunk_lines = []
                        current_chunk_tokens = 0
                    current_chunk_lines.append(line)
                    current_chunk_tokens += line_tokens

                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))

            current_pos = end

        remaining = text[current_pos:].strip()
        if remaining:
            remaining_chunks = self._split_by_paragraph(remaining, max_tokens)
            chunks.extend(remaining_chunks)

        return [c for c in chunks if c.strip()]

    def _split_by_paragraph(self, text: str, max_tokens: int) -> list[str]:
        paragraphs = re.split(r"\n\n+", text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = self._count_tokens(para)

            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sent_tokens = self._count_tokens(sentence)
                    if current_tokens + sent_tokens > max_tokens and current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = []
                        current_tokens = 0
                    current_chunk.append(sentence)
                    current_tokens += sent_tokens
            elif current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_section(
        self,
        section: ParsedSection,
        document: ParsedDocument,
        corpus_version: str,
    ) -> list[Chunk]:
        heading_text = (
            f"{'#' * section.heading_level} {section.heading}\n\n" if section.heading else ""
        )
        full_content = heading_text + section.content

        token_count = self._count_tokens(full_content)

        if token_count < self.config.min_chunk_size:
            return []

        if token_count <= self.config.max_chunk_size:
            text_chunks = [full_content]
        else:
            text_chunks = self._split_preserving_code(full_content, self.config.max_chunk_size)

        chunks = []
        section_path = (
            section.parent_path + [section.heading] if section.heading else section.parent_path
        )

        for idx, text in enumerate(text_chunks):
            has_code = "```" in text or "`" in text

            section_id = "_".join(section_path).replace(" ", "-").lower()[:50]
            chunk_id = f"{corpus_version}:{document.doc_id}:{section_id}:{idx}"

            metadata = ChunkMetadata(
                doc_id=document.doc_id,
                url=document.url,
                title=document.title,
                section_path=section_path,
                section_anchor=section.anchor,
                heading=section.heading,
                heading_level=section.heading_level,
                chunk_index=idx,
                total_chunks_in_section=len(text_chunks),
                has_code=has_code,
                crawl_timestamp=document.metadata.crawl_timestamp,
                source_name=document.metadata.source_name,
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text.strip(),
                    metadata=metadata,
                    corpus_version=corpus_version,
                )
            )

        return chunks

    def chunk(self, document: ParsedDocument, corpus_version: str) -> list[Chunk]:
        all_chunks = []

        for section in document.sections:
            section_chunks = self._chunk_section(section, document, corpus_version)
            all_chunks.extend(section_chunks)

        if not all_chunks and document.raw_text:
            text_chunks = self._split_preserving_code(document.raw_text, self.config.max_chunk_size)
            for idx, text in enumerate(text_chunks):
                chunk_id = f"{corpus_version}:{document.doc_id}:raw:{idx}"
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
                    has_code="```" in text or "`" in text,
                    crawl_timestamp=document.metadata.crawl_timestamp,
                    source_name=document.metadata.source_name,
                )
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=text.strip(),
                        metadata=metadata,
                        corpus_version=corpus_version,
                    )
                )

        return all_chunks

    def get_config_dict(self) -> dict:
        return {
            "type": "structure_aware",
            "max_chunk_size": self.config.max_chunk_size,
            "min_chunk_size": self.config.min_chunk_size,
            "preserve_code_blocks": self.config.preserve_code_blocks,
            "heading_weight": self.config.heading_weight,
        }
