.PHONY: install sync dev crawl parse chunk embed index build-all serve evaluate test lint format typecheck clean frontend-install frontend-dev frontend-build frontend-lint frontend-format frontend-typecheck

# ============================================================================
# Package Management (uv)
# ============================================================================
install:
	uv sync
	uv pip install -e .

sync:
	uv sync --all-extras
	uv pip install -e .

dev:
	uv sync --all-extras
	uv pip install -e .

# ============================================================================
# RAG Pipeline Commands (using pydocs CLI)
# ============================================================================
crawl:
	uv run pydocs crawl --config configs/default.yaml

parse:
	uv run pydocs parse --config configs/default.yaml

chunk:
	uv run pydocs chunk --config configs/default.yaml

embed:
	uv run pydocs embed --config configs/default.yaml

index:
	uv run pydocs build-index --config configs/default.yaml

build-all: crawl parse chunk embed index
	@echo "Full pipeline complete"

serve:
	uv pip install -e .
	uv run pydocs serve --config configs/default.yaml

evaluate:
	uv run pydocs evaluate --config configs/default.yaml --dataset evaluation/datasets/smoke.jsonl

# ============================================================================
# Quality & Testing
# ============================================================================
test:
	uv run pytest tests/ -v

lint:
	uv run ruff check pydocs_rag/ tests/

format:
	uv run ruff format pydocs_rag/ tests/
	uv run ruff check --fix pydocs_rag/ tests/

typecheck:
	uv run ty check pydocs_rag/

check: lint typecheck test
	@echo "All checks passed!"

# ============================================================================
# Frontend
# ============================================================================
frontend-install:
	cd frontend && bun install

frontend-dev:
	cd frontend && bun run dev

frontend-build:
	cd frontend && bun run build

frontend-lint:
	cd frontend && bun run lint

frontend-format:
	cd frontend && bun run format

frontend-typecheck:
	cd frontend && bun run typecheck

# ============================================================================
# Cleanup
# ============================================================================
clean:
	rm -rf data/corpus/*
	rm -rf data/pipeline/*
	rm -rf logs/*
	rm -rf evaluation/runs/*
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# ============================================================================
# Docker
# ============================================================================
docker-build:
	docker build -t pydocs-rag:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data pydocs-rag:latest
