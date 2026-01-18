# Python Docs RAG: Technical Report

**Date:** January 18, 2026
**Project:** Python Docs RAG

## 1. Executive Summary

Python Docs RAG is a production-grade Retrieval-Augmented Generation (RAG) system designed to answer questions about the Python programming language with high accuracy. The system explicitly models failure modes like retrieval misses and hallucinations to ensure robustness.

It indexes documentation from official sources using a structure-aware chunking strategy and serves answers via a React frontend and FastAPI backend. It features an evaluation framework to measure retrieval performance and generation quality.

## 2. System Architecture

The architecture follows a standard RAG pipeline enhanced with failure-aware components.

```
[Indexing Pipeline]
Crawler -> HTML Parser -> Structure-Aware Chunker -> Embedding Model -> FAISS Index

[Query Pipeline]
User Query
    |
    v
Is Chitchat? --Yes--> Conversational Response
    |
    No
    v
Query Rewriter
    |
    v
FAISS Retriever (Top-20)
    |
    v
Cross-Encoder Reranker (Top-5)
    |
    v
LLM Generator (Prompt + Context)
    |
    v
Citation Validation -> Final Response
```

### 2.1 Indexing Pipeline

1.  **Crawling:** Fetches documentation pages from official Python sources.
2.  **Parsing:** Specialized parser removes navigation and footers while preserving code blocks.
3.  **Chunking:** **Structure-Aware Chunker** respects document hierarchy.
    - **Logic:** Splits by sections and keeps code blocks intact.
    - **Parameters:** Max 1024 tokens.
4.  **Embedding:** Uses `BAAI/bge-base-en-v1.5` for generic embeddings.
5.  **Storage:** Vectors stored in a **FAISS** index.

### 2.2 Query Pipeline

1.  **Chitchat Detection:** Heuristic layer intercepts greetings to save compute.
2.  **Retrieval:**
    - **Dense Retrieval:** Queries FAISS for top-k chunks.
    - **Reranking:** Cross-encoder re-scores results to improve precision.
3.  **Generation:**
    - **Prompting:** Injects retrieved context and enforces format.
    - **Citation Enforcement:** Model must cite sources.
    - **Failure Handling:** System falls back to refusal if confidence is low.

## 3. Technology Stack

### Backend

- **Language:** Python 3.12+
- **Framework:** FastAPI, Typer
- **Package Management:** `uv`
- **Vector Database:** FAISS
- **LLM Integration:** Custom Client (OpenAI/Ollama)

### Frontend

- **Framework:** React 19, Vite 7
- **Styling:** Tailwind CSS v4, Framer Motion
- **Rendering:** `react-markdown` with `react-syntax-highlighter`

## 4. Key Technical Innovations

### 4.1 Structure-Aware Chunking

Standard chunking often breaks code blocks. Our chunker parses the DOM to identify logic sections. It treats code tags as atomic units to keep function signatures with their usage.

### 4.2 Failure-Aware Mechanisms

- **Retrieval Confidence:** Reranker scores below a threshold flag the result as unreliable.
- **Citation Validation:** Validators check if the answer cites provided chunks.
- **Fallback Strategies:**
  - _Retrieval Miss:_ Expands search and retries.
  - _Hallucination:_ Returns a refusal message.

## 5. Evaluation & Comparisons

We designed an evaluation harness to benchmark performance.

### 5.1 Metrics

- **Hit@k:** Probability the correct answer is in top k results.
- **MRR:** Measures rank of the correct answer.
- **NDCG:** Accounts for result relevance.

### 5.2 Comparative Analysis

- **Chunking:** Structure-aware chunking improved MRR over fixed-size chunking by preserving context.
- **Reranking:** Cross-encoder reranking significantly improved precision at the cost of slight latency.

## 6. Future Work

1.  **Streaming Responses:** Implement SSE for lower latency.
2.  **Hybrid Search:** Combine dense embeddings with BM25.
