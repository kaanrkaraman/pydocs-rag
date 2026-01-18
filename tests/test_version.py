"""Unit tests for versioning utilities."""

from pydocs_rag.version import (
    compute_chunk_version,
    compute_content_hash,
    compute_corpus_version,
    compute_embedding_version,
    compute_index_version,
    compute_manifest_hash,
)


class TestContentHash:
    def test_deterministic(self):
        content = b"test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_different_content(self):
        hash1 = compute_content_hash(b"content 1")
        hash2 = compute_content_hash(b"content 2")
        assert hash1 != hash2


class TestManifestHash:
    def test_deterministic(self):
        entries = [
            {"url": "http://a.com", "content_hash": "abc"},
            {"url": "http://b.com", "content_hash": "def"},
        ]
        hash1 = compute_manifest_hash(entries)
        hash2 = compute_manifest_hash(entries)
        assert hash1 == hash2

    def test_order_independent(self):
        entries1 = [
            {"url": "http://a.com", "content_hash": "abc"},
            {"url": "http://b.com", "content_hash": "def"},
        ]
        entries2 = [
            {"url": "http://b.com", "content_hash": "def"},
            {"url": "http://a.com", "content_hash": "abc"},
        ]
        hash1 = compute_manifest_hash(entries1)
        hash2 = compute_manifest_hash(entries2)
        assert hash1 == hash2


class TestVersionChain:
    def test_corpus_version(self):
        version = compute_corpus_version("manifest123", ["source1", "source2"])
        assert len(version) == 12
        assert version.isalnum()

    def test_chunk_version(self):
        version = compute_chunk_version("corpus123", {"type": "fixed", "size": 512})
        assert len(version) == 12

    def test_embedding_version(self):
        version = compute_embedding_version("chunk123", "model-name", True)
        assert len(version) == 12

    def test_index_version(self):
        version = compute_index_version("embed123", "IVF", {"nlist": 256})
        assert len(version) == 12

    def test_version_chain_deterministic(self):
        corpus_v1 = compute_corpus_version("m1", ["s1"])
        chunk_v1 = compute_chunk_version(corpus_v1, {"type": "fixed"})
        embed_v1 = compute_embedding_version(chunk_v1, "model", True)
        index_v1 = compute_index_version(embed_v1, "IVF", {})

        corpus_v2 = compute_corpus_version("m1", ["s1"])
        chunk_v2 = compute_chunk_version(corpus_v2, {"type": "fixed"})
        embed_v2 = compute_embedding_version(chunk_v2, "model", True)
        index_v2 = compute_index_version(embed_v2, "IVF", {})

        assert corpus_v1 == corpus_v2
        assert chunk_v1 == chunk_v2
        assert embed_v1 == embed_v2
        assert index_v1 == index_v2
