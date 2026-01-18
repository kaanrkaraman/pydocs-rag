FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY pydocs_rag/ pydocs_rag/

RUN pip install --no-cache-dir .


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY configs/ configs/
COPY pydocs_rag/ pydocs_rag/

RUN mkdir -p data/corpus/raw data/corpus/parsed data/pipeline/chunks data/pipeline/embeddings data/pipeline/indices logs/queries logs/pipeline

ENV RAG_CONFIG_PATH=/app/configs/default.yaml
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "pydocs_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
