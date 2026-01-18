"""Citation extraction and formatting utilities."""

import re

from pydocs_rag.models import Chunk, Citation


def extract_citations_from_text(text: str) -> list[int]:
    pattern = r"\[(\d+)\]"
    matches = re.findall(pattern, text)
    return sorted(set(int(m) for m in matches))


def create_citation_from_chunk(chunk: Chunk, relevance_score: float) -> Citation:
    section = (
        " > ".join(chunk.metadata.section_path)
        if chunk.metadata.section_path
        else chunk.metadata.heading
    )
    snippet = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text

    return Citation(
        chunk_id=chunk.chunk_id,
        url=chunk.metadata.url,
        title=chunk.metadata.title,
        section=section or "Main",
        anchor=chunk.metadata.section_anchor,
        snippet=snippet,
        relevance_score=relevance_score,
    )


def build_citations_from_chunks(
    chunks: list[Chunk],
    used_indices: list[int],
    scores: dict[str, float] | None = None,
) -> list[Citation]:
    citations = []
    scores = scores or {}

    for idx in used_indices:
        if 1 <= idx <= len(chunks):
            chunk = chunks[idx - 1]
            relevance_score = scores.get(chunk.chunk_id, 0.0)
            citation = create_citation_from_chunk(chunk, relevance_score)
            citations.append(citation)

    return citations


def format_citations_for_display(citations: list[Citation]) -> str:
    lines = ["## Sources"]
    for idx, citation in enumerate(citations, 1):
        url = citation.url
        if citation.anchor:
            url = f"{url}#{citation.anchor}"
        lines.append(f"[{idx}] [{citation.title} - {citation.section}]({url})")
    return "\n".join(lines)


def validate_citations(
    answer_text: str,
    num_chunks: int,
    min_citations: int = 1,
) -> tuple[bool, list[str]]:
    issues = []
    cited_indices = extract_citations_from_text(answer_text)

    if len(cited_indices) < min_citations:
        issues.append(f"Expected at least {min_citations} citations, found {len(cited_indices)}")

    for idx in cited_indices:
        if idx < 1 or idx > num_chunks:
            issues.append(f"Invalid citation [{idx}]: only {num_chunks} chunks provided")

    paragraphs = [p.strip() for p in answer_text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        if len(para) > 100 and not re.search(r"\[\d+\]", para):
            if not para.startswith("```") and not para.startswith("#"):
                issues.append(f"Paragraph {i + 1} has no citations")

    return len(issues) == 0, issues
