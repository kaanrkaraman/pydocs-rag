"""Configuration models for the RAG system using Pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    url: str
    name: str
    include_patterns: list[str] = Field(default_factory=list)


class CorpusConfig(BaseModel):
    sources: list[SourceConfig]
    crawl_delay_seconds: float = 1.0
    max_pages_per_source: int = 500
    respect_robots_txt: bool = True
    user_agent: str = "FailureAwareRAG/0.1"


class ParsingConfig(BaseModel):
    remove_nav: bool = True
    remove_footer: bool = True
    remove_sidebar: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    min_paragraph_length: int = 20


class DeduplicationConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.85
    minhash_num_perm: int = 128


class FixedSizeChunkingConfig(BaseModel):
    chunk_size: int = 512
    overlap: int = 64
    split_by: Literal["tokens", "characters"] = "tokens"


class StructureAwareChunkingConfig(BaseModel):
    max_chunk_size: int = 1024
    min_chunk_size: int = 128
    preserve_code_blocks: bool = True
    heading_weight: float = 1.5


class ChunkingConfig(BaseModel):
    type: Literal["fixed_size", "structure_aware"] = "structure_aware"
    fixed_size: FixedSizeChunkingConfig = Field(default_factory=FixedSizeChunkingConfig)
    structure_aware: StructureAwareChunkingConfig = Field(
        default_factory=StructureAwareChunkingConfig
    )


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    batch_size: int = 64
    max_seq_length: int = 256
    show_progress: bool = True


class FlatIndexConfig(BaseModel):
    pass


class IVFIndexConfig(BaseModel):
    nlist: int = 256
    nprobe: int = 16


class HNSWIndexConfig(BaseModel):
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128


class FAISSConfig(BaseModel):
    index_type: Literal["Flat", "IVF", "HNSW"] = "IVF"
    metric: Literal["inner_product", "l2"] = "inner_product"
    flat: FlatIndexConfig = Field(default_factory=FlatIndexConfig)
    ivf: IVFIndexConfig = Field(default_factory=IVFIndexConfig)
    hnsw: HNSWIndexConfig = Field(default_factory=HNSWIndexConfig)


class RetrievalConfig(BaseModel):
    top_k: int = 20
    score_threshold: float = 0.0


class LLMProviderConfig(BaseModel):
    provider: Literal["openai", "ollama"] = "ollama"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1024


class QueryConfig(BaseModel):
    rewrite_enabled: bool = False
    rewrite_strategy: Literal["expansion", "clarification", "multi_query"] = "expansion"
    max_rewrites: int = 3


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 16


class ContextConfig(BaseModel):
    max_tokens: int = 4000
    max_chunks: int = 10
    include_metadata: bool = True
    dedup_chunks: bool = True


class GenerationConfig(BaseModel):
    prompt_version: str = "v1"
    require_citations: bool = True
    min_citations_per_paragraph: int = 1


class RefusalConfig(BaseModel):
    enabled: bool = True
    min_top_score: float = 0.4
    min_score_margin: float = 0.1
    min_citations: int = 1
    ambiguity_detection: bool = True
    fallback_strategy: Literal["extractive", "clarification", "warning"] = "extractive"


class RetrievalMissConfig(BaseModel):
    expand_k_multiplier: int = 2
    try_rewrite_if_off: bool = True
    max_retries: int = 2


class HallucinationConfig(BaseModel):
    require_citations: bool = True
    trigger_refusal: bool = True


class StaleDocsConfig(BaseModel):
    show_crawl_timestamp: bool = True
    prefer_newer: bool = True
    max_age_days: int = 90


class AmbiguousQueryConfig(BaseModel):
    ask_clarification: bool = True
    max_clarification_options: int = 3


class ContextOverflowConfig(BaseModel):
    strategy: Literal["relevance_ranked", "diversity", "recency"] = "relevance_ranked"
    preserve_diversity: bool = True


class FailureModesConfig(BaseModel):
    retrieval_miss: RetrievalMissConfig = Field(default_factory=RetrievalMissConfig)
    hallucination: HallucinationConfig = Field(default_factory=HallucinationConfig)
    stale_docs: StaleDocsConfig = Field(default_factory=StaleDocsConfig)
    ambiguous_query: AmbiguousQueryConfig = Field(default_factory=AmbiguousQueryConfig)
    context_overflow: ContextOverflowConfig = Field(default_factory=ContextOverflowConfig)


class EvaluationMetricsConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["hit_at_k", "mrr", "ndcg"])
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])


class AnswerRubricConfig(BaseModel):
    correctness_weight: float = 0.4
    groundedness_weight: float = 0.4
    completeness_weight: float = 0.2


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["hit_at_k", "mrr", "ndcg"])
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    answer_rubric: AnswerRubricConfig = Field(default_factory=AnswerRubricConfig)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    output_dir: str = "logs"
    include_traces: bool = True
    log_retrieval_results: bool = True
    log_token_usage: bool = True


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_dir: str = "data/corpus/raw"
    parsed_dir: str = "data/corpus/parsed"
    chunks_dir: str = "data/pipeline/chunks"
    embeddings_dir: str = "data/pipeline/embeddings"
    indices_dir: str = "data/pipeline/indices"
    cache_dir: str = "data/cache"
    evaluation_dir: str = "evaluation"
    logs_dir: str = "logs"


class RAGConfig(BaseModel):
    corpus: CorpusConfig
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    refusal: RefusalConfig = Field(default_factory=RefusalConfig)
    failure_modes: FailureModesConfig = Field(default_factory=FailureModesConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_config_hash(self) -> str:
        import hashlib
        import json

        config_str = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
