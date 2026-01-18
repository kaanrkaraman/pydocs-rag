# Python Docs RAG

A production-grade Python documentation RAG system with reproducibility, measurable evaluation, observability, and explicit failure-mode handling.

## Features

- **FAISS-based retrieval** with configurable index types (Flat, IVF, HNSW)
- **Corpus versioning** for reproducible builds
- **Configurable chunking** (fixed-size and structure-aware)
- **Optional query rewrite** and cross-encoder reranking
- **Citation-enforced generation** with refusal policy
- **Failure mode handling** for retrieval miss, ambiguity, staleness
- **Evaluation harness** with regression tracking
- **Structured logging** and timing instrumentation
- **FastAPI service** with Docker deployment

## Quick Start

### Installation

```bash
# Install the package
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Build the Pipeline

```bash
# Run the full build pipeline
make build-all

# Or run steps individually:
make crawl   # Crawl Python docs
make parse   # Parse HTML to structured text
make chunk   # Chunk documents
make embed   # Compute embeddings
make index   # Build FAISS index
```

### Serve the API

````bash
make serve
# Serve the API
make serve
# or
python -m pydocs_rag.cli serve --config configs/default.yaml

### Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I use Python decorators?"}'
````

## Configuration

All settings are in `configs/default.yaml`:

```yaml
chunking:
  type: "structure_aware" # or "fixed_size"

query:
  rewrite_enabled: false # Toggle query rewrite

reranker:
  enabled: true # Toggle cross-encoder reranking

refusal:
  enabled: true
  min_top_score: 0.4 # Minimum confidence threshold
```

## Evaluation

```bash
# Run evaluation
python -m pydocs_rag.cli evaluate \
  --config configs/default.yaml \
  --dataset evaluation/datasets/smoke.jsonl

# Compare runs for regression
python -c "
from pydocs_rag.evaluation.regression import RegressionAnalyzer
analyzer = RegressionAnalyzer()
report = analyzer.compare('evaluation/runs/baseline.json', 'evaluation/runs/current.json')
print(report.generate_markdown())
"
```

## Project Structure

```
src/pydocs_rag/
├── config.py           # Configuration models
├── version.py          # Versioning utilities
├── models.py           # Data models
├── cli.py              # CLI commands
├── ingestion/          # Crawling and manifest
├── parsing/            # HTML parsing and dedup
├── chunking/           # Fixed-size and structure-aware
├── embedding/          # Embedding computation
├── indexing/           # FAISS index building
├── query/              # Pipeline, retriever, rewriter, reranker
├── generation/         # Generator, prompts, citations, refusal
├── failure_modes/      # Failure detection and handling
├── evaluation/         # Metrics, rubric, harness, regression
├── observability/      # Logging and timing
└── api/                # FastAPI service
```

## Environment Variables

- `RAG_CONFIG_PATH`: Path to config file (default: `configs/default.yaml`)
- `RAG_INDEX_VERSION`: Specific index version to load
- `OPENAI_API_KEY`: Required for query rewrite and generation

## Docker

```bash
# Build
docker build -t failure-aware-rag .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  failure-aware-rag
```

## License

MIT
