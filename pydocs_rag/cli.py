"""CLI commands for the RAG system."""

import json
from datetime import datetime
from pathlib import Path

import typer


app = typer.Typer(help="Python Docs RAG CLI")


@app.command()
def crawl(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    output_dir: Path = typer.Option("data/corpus/raw", help="Output directory for raw HTML"),
    resume: bool = typer.Option(True, help="Resume from existing crawl"),
):
    """Crawl Python documentation sources."""
    from pydocs_rag.config import RAGConfig
    from pydocs_rag.ingestion.crawler import run_crawler

    cfg = RAGConfig.from_yaml(config)
    manifest = run_crawler(cfg.corpus, output_dir, resume)
    typer.echo(f"Crawl complete. Corpus version: {manifest.corpus_version}")


@app.command()
def parse(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    raw_dir: Path = typer.Option("data/corpus/raw", help="Raw HTML directory"),
    output_dir: Path = typer.Option("data/corpus/parsed", help="Output directory"),
):
    """Parse raw HTML into structured documents."""
    from pydocs_rag.config import RAGConfig
    from pydocs_rag.ingestion.manifest import Manifest
    from pydocs_rag.parsing.dedup import Deduplicator
    from pydocs_rag.parsing.html_parser import HTMLParser

    cfg = RAGConfig.from_yaml(config)
    manifest = Manifest.load(raw_dir / "manifest.json")

    parser = HTMLParser(cfg.parsing)
    dedup = Deduplicator(cfg.deduplication)

    parsed_docs = []
    for entry in manifest.get_successful_entries():
        file_path = raw_dir / entry.file_path
        if file_path.exists():
            doc = parser.parse_file(file_path, entry)
            parsed_docs.append(doc)

    parsed_docs = dedup.deduplicate_documents(parsed_docs)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{manifest.corpus_version}_parsed.jsonl"

    with open(output_file, "w") as f:
        for doc in parsed_docs:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "url": doc.url,
                        "title": doc.title,
                        "sections": [
                            {
                                "heading": s.heading,
                                "heading_level": s.heading_level,
                                "content": s.content,
                                "code_blocks": s.code_blocks,
                                "anchor": s.anchor,
                                "parent_path": s.parent_path,
                            }
                            for s in doc.sections
                        ],
                        "metadata": doc.metadata.url,
                    }
                )
                + "\n"
            )

    typer.echo(f"Parsed {len(parsed_docs)} documents to {output_file}")


@app.command()
def chunk(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    parsed_dir: Path = typer.Option("data/corpus/parsed", help="Parsed documents directory"),
    output_dir: Path = typer.Option("data/pipeline/chunks", help="Output directory"),
    corpus_version: str | None = typer.Option(None, help="Corpus version to process"),
):
    """Chunk parsed documents."""
    from pydocs_rag.chunking.fixed_size import FixedSizeChunker
    from pydocs_rag.chunking.structure_aware import StructureAwareChunker
    from pydocs_rag.config import RAGConfig
    from pydocs_rag.models import DocumentMetadata, ParsedDocument, ParsedSection

    cfg = RAGConfig.from_yaml(config)

    if cfg.chunking.type == "structure_aware":
        chunker = StructureAwareChunker(cfg.chunking.structure_aware)
    else:
        chunker = FixedSizeChunker(cfg.chunking.fixed_size)

    parsed_files = list(parsed_dir.glob("*_parsed.jsonl"))
    if corpus_version:
        parsed_files = [f for f in parsed_files if corpus_version in f.name]

    if not parsed_files:
        typer.echo("No parsed files found")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for parsed_file in parsed_files:
        cv = parsed_file.stem.replace("_parsed", "")
        all_chunks = []

        with open(parsed_file) as f:
            for line in f:
                data = json.loads(line)
                sections = [
                    ParsedSection(
                        heading=s["heading"],
                        heading_level=s["heading_level"],
                        content=s["content"],
                        code_blocks=s.get("code_blocks", []),
                        anchor=s.get("anchor"),
                        parent_path=s.get("parent_path", []),
                    )
                    for s in data["sections"]
                ]
                doc = ParsedDocument(
                    doc_id=data["doc_id"],
                    url=data["url"],
                    title=data["title"],
                    sections=sections,
                    metadata=DocumentMetadata(
                        url=data["url"],
                        title=data["title"],
                        source_name="",
                        crawl_timestamp=datetime.now(),
                    ),
                )
                chunks = chunker.chunk(doc, cv)
                all_chunks.extend(chunks)

        output_file = output_dir / f"{cv}_chunks.jsonl"
        with open(output_file, "w") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk.to_dict()) + "\n")

        config_file = output_dir / f"{cv}_chunk_config.json"
        with open(config_file, "w") as f:
            json.dump(chunker.get_config_dict(), f, indent=2)

        typer.echo(f"Created {len(all_chunks)} chunks in {output_file}")


@app.command()
def embed(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    chunks_dir: Path = typer.Option("data/pipeline/chunks", help="Chunks directory"),
    output_dir: Path = typer.Option("data/pipeline/embeddings", help="Output directory"),
    corpus_version: str | None = typer.Option(None, help="Corpus version to process"),
):
    """Compute embeddings for chunks."""
    from pydocs_rag.config import RAGConfig
    from pydocs_rag.embedding.embedder import Embedder, EmbeddingStore
    from pydocs_rag.models import Chunk
    from pydocs_rag.version import compute_embedding_version

    cfg = RAGConfig.from_yaml(config)
    embedder = Embedder(cfg.embedding)
    store = EmbeddingStore(output_dir)

    chunk_files = list(chunks_dir.glob("*_chunks.jsonl"))
    if corpus_version:
        chunk_files = [f for f in chunk_files if corpus_version in f.name]

    for chunk_file in chunk_files:
        cv = chunk_file.stem.replace("_chunks", "")
        chunks = []

        with open(chunk_file) as f:
            for line in f:
                chunks.append(Chunk.from_dict(json.loads(line)))

        typer.echo(f"Computing embeddings for {len(chunks)} chunks...")
        embeddings = embedder.embed_chunks(chunks)

        embed_version = compute_embedding_version(
            cv, cfg.embedding.model_name, cfg.embedding.normalize
        )

        chunk_ids = [c.chunk_id for c in chunks]
        metadata = {
            "corpus_version": cv,
            **embedder.get_model_config(),
        }

        store.save(embeddings, chunk_ids, embed_version, metadata)
        typer.echo(f"Saved embeddings to {output_dir / embed_version}")


@app.command()
def build_index(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    embeddings_dir: Path = typer.Option("data/pipeline/embeddings", help="Embeddings directory"),
    output_dir: Path = typer.Option("data/pipeline/indices", help="Output directory"),
    embedding_version: str | None = typer.Option(None, help="Embedding version to use"),
):
    """Build FAISS index from embeddings."""
    from pydocs_rag.config import RAGConfig
    from pydocs_rag.embedding.embedder import EmbeddingStore
    from pydocs_rag.indexing.faiss_builder import FAISSIndexBuilder, FAISSIndexStore
    from pydocs_rag.version import compute_index_version

    cfg = RAGConfig.from_yaml(config)
    embed_store = EmbeddingStore(embeddings_dir)
    index_store = FAISSIndexStore(output_dir)

    versions = embed_store.list_versions()
    if embedding_version:
        versions = [v for v in versions if v == embedding_version]

    if not versions:
        typer.echo("No embedding versions found")
        raise typer.Exit(1)

    builder = FAISSIndexBuilder(cfg.faiss)

    for embed_ver in versions:
        embeddings, chunk_ids, embed_meta = embed_store.load(embed_ver)

        typer.echo(f"Building {cfg.faiss.index_type} index for {len(chunk_ids)} vectors...")
        index = builder.build_index(embeddings)

        index_ver = compute_index_version(
            embed_ver, cfg.faiss.index_type, builder.get_config_dict()
        )

        metadata = {
            "embedding_version": embed_ver,
            "corpus_version": embed_meta.get("corpus_version", ""),
            **builder.get_config_dict(),
        }

        index_store.save(index, chunk_ids, index_ver, metadata)
        typer.echo(f"Saved index to {output_dir / index_ver}")


@app.command()
def evaluate(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    dataset: Path = typer.Option(..., help="Evaluation dataset JSONL"),
    output_dir: Path = typer.Option("evaluation/runs", help="Output directory"),
    index_version: str | None = typer.Option(None, help="Index version to use"),
    run_id: str | None = typer.Option(None, help="Run ID"),
):
    """Run evaluation on a dataset."""
    typer.echo("Evaluation requires a fully initialized pipeline. Use the API or scripts.")


@app.command()
def serve(
    config: Path = typer.Option("configs/default.yaml", help="Config file path"),
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    index_version: str | None = typer.Option(None, help="Index version to load"),
):
    """Start the API server."""
    import os

    import uvicorn

    os.environ["RAG_CONFIG_PATH"] = str(config)
    if index_version:
        os.environ["RAG_INDEX_VERSION"] = index_version

    uvicorn.run(
        "pydocs_rag.api.main:app",
        host=host,
        port=port,
        reload=False,
    )


def main():
    app()


if __name__ == "__main__":
    main()
