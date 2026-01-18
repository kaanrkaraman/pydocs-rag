"""Unit tests for chunking."""

from datetime import datetime

import pytest

from pydocs_rag.chunking.fixed_size import FixedSizeChunker
from pydocs_rag.chunking.structure_aware import StructureAwareChunker
from pydocs_rag.config import FixedSizeChunkingConfig, StructureAwareChunkingConfig
from pydocs_rag.models import DocumentMetadata, ParsedDocument, ParsedSection


@pytest.fixture
def sample_document():
    return ParsedDocument(
        doc_id="test123",
        url="http://example.com/doc",
        title="Test Document",
        sections=[
            ParsedSection(
                heading="Introduction",
                heading_level=1,
                content="This is the introduction paragraph. " * 20,
                code_blocks=[],
                anchor="intro",
                parent_path=[],
            ),
            ParsedSection(
                heading="Code Example",
                heading_level=2,
                content="Here is some code that demonstrates Python functions:\n```python\ndef hello():\n    print('Hello')\n```\nThis example shows a simple function definition. Functions are defined using the def keyword followed by the function name and parameters.",
                code_blocks=["def hello():\n    print('Hello')"],
                anchor="code",
                parent_path=["Introduction"],
            ),
        ],
        metadata=DocumentMetadata(
            url="http://example.com/doc",
            title="Test Document",
            source_name="test",
            crawl_timestamp=datetime.now(),
        ),
    )


class TestFixedSizeChunker:
    def test_chunks_created(self, sample_document):
        config = FixedSizeChunkingConfig(chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)
        chunks = chunker.chunk(sample_document, "corpus_v1")
        assert len(chunks) > 0

    def test_chunk_ids_include_version(self, sample_document):
        config = FixedSizeChunkingConfig(chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)
        chunks = chunker.chunk(sample_document, "corpus_v1")
        for chunk in chunks:
            assert "corpus_v1" in chunk.chunk_id

    def test_config_dict(self):
        config = FixedSizeChunkingConfig(chunk_size=512, overlap=64)
        chunker = FixedSizeChunker(config)
        config_dict = chunker.get_config_dict()
        assert config_dict["type"] == "fixed_size"
        assert config_dict["chunk_size"] == 512


class TestStructureAwareChunker:
    def test_preserves_headings(self, sample_document):
        config = StructureAwareChunkingConfig(max_chunk_size=500, min_chunk_size=50)
        chunker = StructureAwareChunker(config)
        chunks = chunker.chunk(sample_document, "corpus_v1")

        has_heading = any("#" in c.text for c in chunks)
        assert has_heading

    def test_preserves_code_blocks(self, sample_document):
        config = StructureAwareChunkingConfig(
            max_chunk_size=500, min_chunk_size=10, preserve_code_blocks=True
        )
        chunker = StructureAwareChunker(config)
        chunks = chunker.chunk(sample_document, "corpus_v1")

        has_code = any("```" in c.text or "def " in c.text or "`" in c.text for c in chunks)
        assert len(chunks) > 0
        assert has_code

    def test_section_path_tracked(self, sample_document):
        config = StructureAwareChunkingConfig(max_chunk_size=500, min_chunk_size=50)
        chunker = StructureAwareChunker(config)
        chunks = chunker.chunk(sample_document, "corpus_v1")

        paths = [c.metadata.section_path for c in chunks]
        assert any(len(p) > 0 for p in paths)
