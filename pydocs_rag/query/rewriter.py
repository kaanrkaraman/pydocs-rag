"""Optional query rewriter using LLM."""

from pydocs_rag.config import LLMProviderConfig, QueryConfig
from pydocs_rag.llm import LLMClient, create_llm_client


REWRITE_SYSTEM_PROMPT = """You are a query rewriter for a Python documentation search system.
Your job is to improve search queries to get better results from a vector similarity search over Python docs.

Guidelines:
- Expand abbreviations (e.g., "dict" -> "dictionary")
- Add relevant Python-specific terms
- Make implicit concepts explicit
- Keep the rewritten query concise but comprehensive
- Do not add information the user didn't ask about

Return ONLY the rewritten query, nothing else."""

EXPANSION_PROMPT = """Rewrite this query to improve search results:

Original query: {query}

Rewritten query:"""

MULTI_QUERY_PROMPT = """Generate 3 different search queries for the following question.
Each should capture a different aspect or phrasing of the question.

Original question: {query}

Return one query per line, no numbering or bullets."""


class QueryRewriter:
    def __init__(self, config: QueryConfig, llm_config: LLMProviderConfig) -> None:
        self.config = config
        self.llm_config = llm_config
        self.client: LLMClient = create_llm_client(llm_config)

    def rewrite(self, query: str) -> tuple[str, str | None]:
        if not self.config.rewrite_enabled:
            return query, None

        if self.config.rewrite_strategy == "expansion":
            return self._expansion_rewrite(query)
        elif self.config.rewrite_strategy == "multi_query":
            rewritten_queries = self._multi_query_rewrite(query)
            return rewritten_queries[0] if rewritten_queries else query, None
        else:
            return query, None

    def _expansion_rewrite(self, query: str) -> tuple[str, str | None]:
        try:
            messages = [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": EXPANSION_PROMPT.format(query=query)},
            ]
            response = self.client.chat(messages, temperature=0.3, max_tokens=100)
            rewritten = response.content.strip()
            return rewritten, "expansion"
        except Exception as e:
            return query, f"error: {e!s}"

    def _multi_query_rewrite(self, query: str) -> list[str]:
        try:
            messages = [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)},
            ]
            response = self.client.chat(messages, temperature=0.5, max_tokens=200)
            content = response.content.strip()
            queries = [q.strip() for q in content.split("\n") if q.strip()]
            return queries[: self.config.max_rewrites]
        except Exception:
            return [query]

    def generate_query_variants(self, query: str) -> list[str]:
        if not self.config.rewrite_enabled:
            return [query]

        if self.config.rewrite_strategy == "multi_query":
            return self._multi_query_rewrite(query)

        rewritten, _ = self.rewrite(query)
        if rewritten != query:
            return [query, rewritten]
        return [query]
