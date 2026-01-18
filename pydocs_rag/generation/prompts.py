"""Versioned prompt templates for answer generation."""

PROMPTS = {
    "v1": {
        "system": """You are a Python documentation assistant. Answer questions using ONLY the provided context from official Python documentation.

CRITICAL RULES:
1. Base your answer ONLY on the provided context chunks
2. Cite sources using [1], [2], etc. matching the context chunk numbers
3. Every factual claim MUST have a citation
4. If the context doesn't contain enough information, say so clearly
5. For code examples, prefer those from the official documentation
6. Be precise about Python version specifics when relevant

Format your response with:
- Clear, direct answers
- Code examples when helpful (properly formatted)
- Citations after relevant statements [1], [2], etc.""",
        "user": """Context from Python documentation:

{context}

Question: {query}

Provide a comprehensive answer with citations to the context chunks.""",
        "context_template": """[{index}] {title} - {section}
URL: {url}
---
{text}
---""",
    },
    "v2": {
        "system": """You are a precise Python documentation assistant. Your role is to answer questions using ONLY the provided documentation context.

REQUIREMENTS:
1. ONLY use information from the provided context
2. ALWAYS cite sources using numbered references [1], [2], etc.
3. If context is insufficient, acknowledge limitations
4. Prefer official documentation examples over custom code
5. Be specific about Python versions when applicable

CITATION RULES:
- Every paragraph must have at least one citation
- Place citations immediately after the relevant claim
- Use multiple citations when combining information [1][3]""",
        "user": """# Documentation Context

{context}

# User Question
{query}

# Instructions
Answer the question comprehensively. Cite every factual claim with [N] references.""",
        "context_template": """## [{index}] {title}
Section: {section}
Source: {url}

{text}
""",
    },
    "extractive": {
        "system": """You are a Python documentation assistant in EXTRACTIVE mode.
When context is limited or low-confidence, extract the most relevant quotes directly from the documentation.
Present the information as direct quotes with minimal interpretation.""",
        "user": """Context from Python documentation:

{context}

Question: {query}

Extract the most relevant information from the context. Present as direct quotes where possible.""",
        "context_template": """[{index}] From "{title}" ({section}):
"{text}"
Source: {url}""",
    },
}


def get_prompt(version: str) -> dict:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPTS.keys())}")
    return PROMPTS[version]


def format_context(chunks: list, prompt_version: str) -> str:
    template = PROMPTS[prompt_version]["context_template"]
    formatted_chunks = []

    for idx, chunk in enumerate(chunks, 1):
        section = (
            " > ".join(chunk.metadata.section_path)
            if chunk.metadata.section_path
            else chunk.metadata.heading
        )
        formatted = template.format(
            index=idx,
            title=chunk.metadata.title,
            section=section or "Main",
            url=chunk.metadata.url,
            text=chunk.text,
        )
        formatted_chunks.append(formatted)

    return "\n\n".join(formatted_chunks)
