# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python Docs RAG is a Python documentation retrieval system with explicit failure mode detection, citation enforcement, and evaluation capabilities. It indexes Python docs (docs.python.org, peps.python.org, packaging.python.org) into a FAISS vector store and serves queries via FastAPI.

## Commands

### Build Pipeline (sequential, run in order)

```bash
make crawl        # Crawl Python doc sources → data/raw/
make parse        # Parse HTML → data/parsed/
make chunk        # Chunk documents → data/chunks/
make embed        # Compute embeddings → data/embeddings/
make index        # Build FAISS index → data/indices/
make build-all    # Run all above steps
```

### Development

```bash
make install      # uv sync
make dev          # uv sync --all-extras
make serve        # Start FastAPI on :8000
make evaluate     # Run evaluation harness
```

### Quality Checks

```bash
make test         # pytest tests/ -v
make lint         # ruff check pydocs_rag/ tests/
make format       # ruff format + ruff check --fix
make typecheck    # ty check pydocs_rag/
make check        # lint + typecheck + test (all)
```

### Single Test

```bash
uv run pytest tests/test_metrics.py -v
uv run pytest tests/test_chunking.py::test_structure_aware_chunker -v
```

## Architecture

```
Query → Rewriter → Retriever → Reranker → Context Assembly → Generator → Response
                      ↓              ↓              ↓              ↓
                  FAISS Index    Cross-encoder   Token limit    Citations
                                                              + Refusal
```

### Key Modules

- **`pydocs_rag/cli.py`**: Typer CLI entry point (`pydocs` command)
- **`pydocs_rag/config.py`**: Pydantic config models loaded from `configs/default.yaml`
- **`pydocs_rag/models.py`**: Core data models (Chunk, Document, Citation, QueryTrace)
- **`pydocs_rag/query/pipeline.py`**: Orchestrates rewrite→retrieve→rerank flow
- **`pydocs_rag/generation/generator.py`**: LLM generation with citation extraction
- **`pydocs_rag/generation/refusal.py`**: Refusal policy (low confidence, missing citations)
- **`pydocs_rag/failure_modes/handlers.py`**: Detects retrieval_miss, ambiguous_query, stale_docs, context_overflow
- **`pydocs_rag/api/main.py`**: FastAPI app with `/query`, `/health`, `/index/info`, `/stats`

### Failure Modes (in `failure_modes/handlers.py`)

1. **RETRIEVAL_MISS**: top score < 0.3 → expand k, retry with rewrite
2. **HALLUCINATION_RISK**: insufficient citations → trigger refusal
3. **STALE_DOCS**: crawl > 90 days old → show timestamp warning
4. **AMBIGUOUS_QUERY**: score margin < 0.05 → ask clarification
5. **CONTEXT_OVERFLOW**: exceeds token limit → relevance-ranked selection

### Versioning

Each pipeline stage produces versioned outputs (corpus → chunk → embedding → index). Versions are deterministic hashes ensuring reproducibility.

## Configuration

All settings in `configs/default.yaml`. Key sections:

- `chunking.type`: "structure_aware" (default) or "fixed_size"
- `llm.provider`: "ollama" (local) or "openai"
- `refusal.enabled`: citation/confidence-based refusal
- `reranker.enabled`: cross-encoder reranking

Environment variables:

- `OPENAI_API_KEY`: Required when `llm.provider: openai`
- `RAG_CONFIG_PATH`: Override config file path
- `RAG_INDEX_VERSION`: Load specific index version

## Code Style

- Python 3.11+
- Ruff for linting/formatting (100 char line length)
- ty for type checking (permissive "warn" mode)
- Double quotes, space indentation
- First-party imports: `pydocs_rag`

## Frontend Design Principles (2025 Best Practices)

When building web interfaces for this project, follow these modern design guidelines to create distinctive, elegant, and production-ready UIs:

### Core Philosophy

Avoid generic "AI slop" aesthetics. Create creative, context-specific designs that surprise and delight users. Every interface should feel purposefully designed, not cookie-cutter.

### Typography

- **Choose distinctive fonts**: Avoid overused choices (Inter, Roboto, Arial, system fonts)
- **Be creative**: Select beautiful, unique typefaces that elevate the aesthetic
- **Vary choices**: Don't converge on common alternatives (e.g., Space Grotesk); think outside the box
- **Context matters**: Typography should match the overall design vision

### Color & Theme

- **Commit to cohesion**: Develop a strong aesthetic identity
- **Dominant + accent**: Use dominant colors with sharp accents, not timid evenly-distributed palettes
- **Draw inspiration**: Consider IDE themes, cultural aesthetics, and contextual elements
- **Vary approaches**: Mix light and dark themes; avoid always defaulting to the same palette
- **Avoid clichés**: Skip predictable schemes (purple gradients on white backgrounds)

### Motion & Animation

- **Prioritize impact**: One well-orchestrated page load > scattered micro-interactions
- **Staggered reveals**: Use `animation-delay` for orchestrated entrance animations
- **CSS-first**: Prefer CSS-only solutions for HTML elements
- **React Motion**: Use Motion library for React components when available
- **Purposeful animation**: Animate for effects and meaningful micro-interactions

### Backgrounds & Depth

- **Create atmosphere**: Layer CSS gradients, use geometric patterns, or add contextual effects
- **Avoid flat colors**: Build depth rather than defaulting to solid backgrounds
- **Be creative**: Go beyond standard patterns; invent unique treatments that match the aesthetic
- **Context-specific**: Backgrounds should enhance the overall design narrative

### Technical Stack

- **Tailwind 4**: Primary styling framework
- **Responsive design**: Mobile-first approach, elegant across all devices
- **Modern CSS**: Use latest features (container queries, CSS grid, custom properties)
- **Performance**: Optimize animations, lazy-load assets, minimize bundle size

### What to Avoid

- Generic layouts and component patterns
- Predictable cookie-cutter designs lacking character
- Same design system across different contexts
- Convergence toward "on distribution" outputs
- Timid design choices that play it safe

### Execution

- **Think outside the box**: Make unexpected, creative choices
- **Context-specific character**: Design should feel native to its purpose
- **Surprise and delight**: Aim for memorable user experiences
- **Professional quality**: Production-grade code and design standards
- **2025-ready**: Use cutting-edge techniques and modern best practices
