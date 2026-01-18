"""Query pipeline orchestrating rewrite, retrieval, and reranking."""

import time
import uuid
from datetime import datetime

import tiktoken

from pydocs_rag.config import (
    ContextConfig,
    RAGConfig,
)
from pydocs_rag.models import Chunk, QueryTrace, RetrievalResult
from pydocs_rag.query.reranker import CrossEncoderReranker
from pydocs_rag.query.retriever import FAISSRetriever
from pydocs_rag.query.rewriter import QueryRewriter


class QueryPipeline:
    def __init__(
        self,
        retriever: FAISSRetriever,
        config: RAGConfig,
        rewriter: QueryRewriter | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config
        self.rewriter = rewriter
        self.reranker = reranker
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _assemble_context(
        self, results: list[RetrievalResult], context_config: ContextConfig
    ) -> list[Chunk]:
        selected_chunks = []
        total_tokens = 0
        seen_texts = set()

        for result in results:
            if len(selected_chunks) >= context_config.max_chunks:
                break

            if context_config.dedup_chunks:
                text_hash = hash(result.chunk.text[:200])
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)

            chunk_tokens = self._count_tokens(result.chunk.text)
            if total_tokens + chunk_tokens > context_config.max_tokens:
                continue

            selected_chunks.append(result.chunk)
            total_tokens += chunk_tokens

        return selected_chunks

    def _is_chitchat(self, query: str) -> bool:
        """Simple heuristic to detect conversational queries."""
        q = query.strip().lower()
        greetings = {
            "hello",
            "hi",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
        }
        gratitude = {"thanks", "thank you", "thx", "thank you very much", "thanks for your help"}
        farewells = {"bye", "goodbye", "see ya", "cya"}

        # Check for exact matches or reliable prefixes
        if q in greetings or q in gratitude or q in farewells:
            return True

        return q.startswith("thank you") or q.startswith("thanks")

    def run(
        self,
        query: str,
        corpus_version: str = "",
        index_version: str = "",
    ) -> QueryTrace:
        trace = QueryTrace(
            trace_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            original_query=query,
            corpus_version=corpus_version,
            index_version=index_version,
            config_hash=self.config.get_config_hash(),
        )

        # Check for chitchat/conversational queries to bypass expensive retrieval
        if self._is_chitchat(query):
            trace.is_conversational = True
            return trace

        effective_query = query
        if self.rewriter and self.config.query.rewrite_enabled:
            start = time.perf_counter()
            effective_query, rationale = self.rewriter.rewrite(query)
            trace.timing_ms["rewrite"] = (time.perf_counter() - start) * 1000
            trace.rewritten_query = effective_query
            trace.rewrite_rationale = rationale

        start = time.perf_counter()
        retrieval_results = self.retriever.retrieve(effective_query)
        trace.timing_ms["retrieval"] = (time.perf_counter() - start) * 1000
        trace.retrieval_results = retrieval_results

        final_results = retrieval_results
        if self.reranker and self.config.reranker.enabled and retrieval_results:
            start = time.perf_counter()
            final_results = self.reranker.rerank(query, retrieval_results)
            trace.timing_ms["rerank"] = (time.perf_counter() - start) * 1000
            trace.reranked_results = final_results

        start = time.perf_counter()
        selected_chunks = self._assemble_context(final_results, self.config.context)
        trace.timing_ms["context_assembly"] = (time.perf_counter() - start) * 1000
        trace.selected_chunks = selected_chunks

        return trace

    def run_with_expanded_retrieval(
        self,
        query: str,
        k_multiplier: int = 2,
        corpus_version: str = "",
        index_version: str = "",
    ) -> QueryTrace:
        trace = QueryTrace(
            trace_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(),
            original_query=query,
            corpus_version=corpus_version,
            index_version=index_version,
            config_hash=self.config.get_config_hash(),
        )

        effective_query = query
        if self.rewriter and self.config.query.rewrite_enabled:
            start = time.perf_counter()
            effective_query, rationale = self.rewriter.rewrite(query)
            trace.timing_ms["rewrite"] = (time.perf_counter() - start) * 1000
            trace.rewritten_query = effective_query
            trace.rewrite_rationale = rationale

        start = time.perf_counter()
        retrieval_results = self.retriever.retrieve_with_expanded_k(effective_query, k_multiplier)
        trace.timing_ms["retrieval"] = (time.perf_counter() - start) * 1000
        trace.retrieval_results = retrieval_results

        final_results = retrieval_results
        if self.reranker and self.config.reranker.enabled and retrieval_results:
            start = time.perf_counter()
            final_results = self.reranker.rerank(query, retrieval_results)
            trace.timing_ms["rerank"] = (time.perf_counter() - start) * 1000
            trace.reranked_results = final_results

        start = time.perf_counter()
        selected_chunks = self._assemble_context(final_results, self.config.context)
        trace.timing_ms["context_assembly"] = (time.perf_counter() - start) * 1000
        trace.selected_chunks = selected_chunks

        return trace
