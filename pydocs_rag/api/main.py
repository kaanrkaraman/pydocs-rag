"""FastAPI application for serving the RAG system."""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydocs_rag.api.models import (
    CitationResponse,
    HealthResponse,
    IndexInfoResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)
from pydocs_rag.config import RAGConfig
from pydocs_rag.embedding.embedder import Embedder
from pydocs_rag.generation.generator import Generator
from pydocs_rag.indexing.faiss_builder import FAISSIndexStore
from pydocs_rag.models import Chunk
from pydocs_rag.observability.logging import QueryLogger, configure_logging
from pydocs_rag.observability.timing import TimingStats, get_global_stats
from pydocs_rag.query.pipeline import QueryPipeline
from pydocs_rag.query.reranker import CrossEncoderReranker
from pydocs_rag.query.retriever import FAISSRetriever
from pydocs_rag.query.rewriter import QueryRewriter


class RAGService:
    def __init__(self) -> None:
        self.config: RAGConfig | None = None
        self.pipeline: QueryPipeline | None = None
        self.generator: Generator | None = None
        self.query_logger: QueryLogger | None = None
        self.corpus_version: str = ""
        self.index_version: str = ""
        self.timing_stats = TimingStats()
        self.query_count = 0
        self.refusal_count = 0

    def load(
        self,
        config_path: Path,
        index_version: str | None = None,
    ) -> None:
        self.config = RAGConfig.from_yaml(config_path)
        configure_logging(self.config.logging)

        base_path = config_path.parent.parent
        indices_dir = base_path / self.config.paths.indices_dir
        chunks_dir = base_path / self.config.paths.chunks_dir
        print(f"DEBUG: base_path={base_path.absolute()}")
        print(f"DEBUG: indices_dir={indices_dir.absolute()}")
        if indices_dir.exists():
            print(f"DEBUG: indices_dir contents: {[p.name for p in indices_dir.iterdir()]}")
        else:
            print("DEBUG: indices_dir does not exist")

        index_store = FAISSIndexStore(indices_dir)
        available_versions = index_store.list_versions()

        if not available_versions:
            raise RuntimeError("No index versions available")

        self.index_version = index_version or available_versions[-1]

        if self.index_version not in available_versions:
            raise ValueError(f"Index version {self.index_version} not found")

        index, chunk_id_mapping, index_metadata = index_store.load(self.index_version)

        self.corpus_version = index_metadata.get("corpus_version", "unknown")

        chunks_file = chunks_dir / f"{self.corpus_version}_chunks.jsonl"
        if not chunks_file.exists():
            for f in chunks_dir.glob("*_chunks.jsonl"):
                chunks_file = f
                break

        chunks_by_id: dict[str, Chunk] = {}
        if chunks_file.exists():
            with open(chunks_file) as f:
                for line in f:
                    chunk_data = json.loads(line)
                    chunk = Chunk.from_dict(chunk_data)
                    chunks_by_id[chunk.chunk_id] = chunk

        embedder = Embedder(self.config.embedding)

        retriever = FAISSRetriever(
            index=index,
            chunk_id_mapping=chunk_id_mapping,
            chunks_by_id=chunks_by_id,
            embedder=embedder,
            config=self.config.retrieval,
        )

        rewriter = (
            QueryRewriter(self.config.query, self.config.llm)
            if self.config.query.rewrite_enabled
            else None
        )
        reranker = (
            CrossEncoderReranker(self.config.reranker) if self.config.reranker.enabled else None
        )

        self.pipeline = QueryPipeline(
            retriever=retriever,
            config=self.config,
            rewriter=rewriter,
            reranker=reranker,
        )

        self.generator = Generator(
            gen_config=self.config.generation,
            refusal_config=self.config.refusal,
            llm_config=self.config.llm,
        )

        logs_dir = base_path / self.config.paths.logs_dir
        self.query_logger = QueryLogger(self.config.logging, logs_dir)


service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = Path(os.environ.get("RAG_CONFIG_PATH", "configs/default.yaml"))
    index_version = os.environ.get("RAG_INDEX_VERSION")

    if config_path.exists():
        try:
            service.load(config_path, index_version)
            print(f"Loaded index version: {service.index_version}")
            print(f"Corpus version: {service.corpus_version}")
        except Exception as e:
            print(f"Warning: Failed to load service: {e}")

    yield


app = FastAPI(
    title="Python Docs RAG API",
    description="Python documentation RAG with failure handling and evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")


@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    if service.config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return HealthResponse(
        status="healthy",
        corpus_version=service.corpus_version,
        index_version=service.index_version,
        config_hash=service.config.get_config_hash(),
    )


@api_router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if service.pipeline is None or service.generator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    trace = service.pipeline.run(
        request.query,
        corpus_version=service.corpus_version,
        index_version=service.index_version,
    )

    # Prepare generation arguments
    gen_result = None
    results = []

    if trace.is_conversational:
        gen_result = service.generator.generate_conversational(request.query)
    else:
        results = trace.reranked_results if trace.reranked_results else trace.retrieval_results
        gen_result = service.generator.generate(
            request.query,
            trace.selected_chunks,
            results,
        )

    service.query_count += 1
    if gen_result.refusal_triggered:
        service.refusal_count += 1

    trace.refusal_triggered = gen_result.refusal_triggered
    trace.refusal_reason = gen_result.refusal_reason
    trace.citations = gen_result.citations

    if service.query_logger:
        service.query_logger.log_query(
            trace=trace,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
            rewrite_enabled=service.config.query.rewrite_enabled,
            reranker_enabled=service.config.reranker.enabled,
            retrieval_k=service.config.retrieval.top_k,
        )

    answer = gen_result.answer if not gen_result.refusal_triggered else gen_result.fallback_response

    citations = [
        CitationResponse(
            chunk_id=c.chunk_id,
            url=c.url,
            title=c.title,
            section=c.section,
            snippet=c.snippet,
            score=c.relevance_score,
        )
        for c in gen_result.citations
    ]

    # Calculate global confidence from retrieval results if available
    confidence = 0.0
    if trace.is_conversational:
        confidence = 1.0
    elif results:
        top_score = (
            results[0].rerank_score if results[0].rerank_score is not None else results[0].score
        )
        confidence = float(top_score)

    return QueryResponse(
        answer=answer or "",
        citations=citations,
        refusal_triggered=gen_result.refusal_triggered,
        refusal_reason=gen_result.refusal_reason.value if gen_result.refusal_reason else None,
        confidence=confidence,
        warning=None,
        timing_ms=trace.timing_ms if request.include_trace else None,
        trace_id=trace.trace_id,
    )


@api_router.get("/index/info", response_model=IndexInfoResponse)
async def get_index_info():
    if service.config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return IndexInfoResponse(
        index_version=service.index_version,
        corpus_version=service.corpus_version,
        num_vectors=0,
        embedding_model=service.config.embedding.model_name,
        index_type=service.config.faiss.index_type,
        created_at=datetime.now().isoformat(),
    )


@api_router.get("/stats", response_model=StatsResponse)
async def get_stats():
    avg_latency = 0.0
    if service.query_count > 0:
        stats = get_global_stats()
        total_stats = stats.get_stats("total")
        avg_latency = total_stats.get("mean", 0.0)

    refusal_rate = service.refusal_count / service.query_count if service.query_count > 0 else 0.0

    return StatsResponse(
        total_queries=service.query_count,
        avg_latency_ms=avg_latency,
        refusal_rate=refusal_rate,
        stage_timings=get_global_stats().get_all_stats(),
    )


app.include_router(api_router)
