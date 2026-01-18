/**
 * API types for the Failure-Aware RAG system
 */

export interface Citation {
  chunk_id: string;
  url: string;
  title: string;
  section: string;
  source_id?: string;
  text: string;
  score: number;
}

export interface FailureMode {
  type: string;
  message: string;
  description: string;
  severity: "warning" | "error";
  metadata?: Record<string, any>;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  failure_modes: FailureMode[];
  confidence: number;
  query_trace?: {
    original_query: string;
    rewritten_query?: string;
    retrieval_time_ms: number;
    generation_time_ms: number;
  };
}

export interface HealthResponse {
  status: string;
  corpus_version: string;
  index_version: string;
  config_hash: string;
}

export interface IndexInfo {
  index_version: string;
  corpus_version: string;
  num_vectors: number;
  embedding_model: string;
  index_type: string;
  created_at: string;
}

export interface Stats {
  total_queries: number;
  avg_latency_ms: number;
  refusal_rate: number;
  stage_timings: Record<string, Record<string, number>>;
}
