"""Deduplication utilities for removing duplicate content."""

import hashlib

from datasketch import MinHash, MinHashLSH

from pydocs_rag.config import DeduplicationConfig
from pydocs_rag.models import ParsedDocument, ParsedSection


class Deduplicator:
    def __init__(self, config: DeduplicationConfig) -> None:
        self.config = config
        self.lsh = MinHashLSH(
            threshold=config.similarity_threshold, num_perm=config.minhash_num_perm
        )
        self.seen_hashes: set[str] = set()
        self.minhashes: dict[str, MinHash] = {}

    def _compute_content_hash(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _create_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.config.minhash_num_perm)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i : i + 3])
            m.update(shingle.encode("utf-8"))
        return m

    def _is_near_duplicate(self, text: str, doc_id: str) -> bool:
        if len(text.split()) < 10:
            return False

        minhash = self._create_minhash(text)
        similar = self.lsh.query(minhash)

        if similar:
            return True

        key = f"{doc_id}:{len(self.minhashes)}"
        self.lsh.insert(key, minhash)
        self.minhashes[key] = minhash
        return False

    def _is_exact_duplicate(self, text: str) -> bool:
        content_hash = self._compute_content_hash(text)
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

    def deduplicate_sections(
        self, sections: list[ParsedSection], doc_id: str
    ) -> list[ParsedSection]:
        if not self.config.enabled:
            return sections

        unique_sections = []
        for section in sections:
            if self._is_exact_duplicate(section.content):
                continue

            if self._is_near_duplicate(section.content, doc_id):
                continue

            unique_sections.append(section)

        return unique_sections

    def deduplicate_documents(self, documents: list[ParsedDocument]) -> list[ParsedDocument]:
        if not self.config.enabled:
            return documents

        deduped_docs = []
        for doc in documents:
            deduped_sections = self.deduplicate_sections(doc.sections, doc.doc_id)
            if deduped_sections:
                doc.sections = deduped_sections
                deduped_docs.append(doc)

        return deduped_docs

    def reset(self) -> None:
        self.lsh = MinHashLSH(
            threshold=self.config.similarity_threshold,
            num_perm=self.config.minhash_num_perm,
        )
        self.seen_hashes.clear()
        self.minhashes.clear()


def deduplicate_by_url(documents: list[ParsedDocument]) -> list[ParsedDocument]:
    url_to_doc: dict[str, ParsedDocument] = {}

    for doc in documents:
        if doc.url in url_to_doc:
            existing = url_to_doc[doc.url]
            if doc.metadata.crawl_timestamp > existing.metadata.crawl_timestamp:
                url_to_doc[doc.url] = doc
        else:
            url_to_doc[doc.url] = doc

    return list(url_to_doc.values())
