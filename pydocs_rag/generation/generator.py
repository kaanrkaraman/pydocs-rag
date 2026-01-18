"""Answer generator with citation enforcement."""

from pydocs_rag.config import GenerationConfig, LLMProviderConfig, RefusalConfig
from pydocs_rag.generation.citations import (
    build_citations_from_chunks,
    extract_citations_from_text,
    validate_citations,
)
from pydocs_rag.generation.prompts import format_context, get_prompt
from pydocs_rag.generation.refusal import (
    check_citation_coverage,
    check_retrieval_confidence,
    generate_refusal_response,
)
from pydocs_rag.llm import LLMClient, create_llm_client
from pydocs_rag.models import (
    Chunk,
    GenerationResult,
    RefusalReason,
    RetrievalResult,
)


class Generator:
    def __init__(
        self,
        gen_config: GenerationConfig,
        refusal_config: RefusalConfig,
        llm_config: LLMProviderConfig,
    ) -> None:
        self.gen_config = gen_config
        self.refusal_config = refusal_config
        self.llm_config = llm_config
        self.client: LLMClient = create_llm_client(llm_config)

    def generate(
        self,
        query: str,
        selected_chunks: list[Chunk],
        retrieval_results: list[RetrievalResult] | None = None,
    ) -> GenerationResult:
        model_name = self.client.get_model_name()

        if self.refusal_config.enabled and retrieval_results:
            refusal_decision = check_retrieval_confidence(retrieval_results, self.refusal_config)
            if refusal_decision.should_refuse:
                fallback = generate_refusal_response(
                    refusal_decision,
                    selected_chunks,
                    self.refusal_config.fallback_strategy,
                )
                return GenerationResult(
                    answer="",
                    citations=[],
                    refusal_triggered=True,
                    refusal_reason=refusal_decision.reason,
                    fallback_response=fallback,
                    input_tokens=0,
                    output_tokens=0,
                    model=model_name,
                    prompt_version=self.gen_config.prompt_version,
                )

        if not selected_chunks:
            return GenerationResult(
                answer="",
                citations=[],
                refusal_triggered=True,
                refusal_reason=RefusalReason.NO_RELEVANT_CONTENT,
                fallback_response="No relevant context found to answer the question.",
                input_tokens=0,
                output_tokens=0,
                model=model_name,
                prompt_version=self.gen_config.prompt_version,
            )

        prompts = get_prompt(self.gen_config.prompt_version)
        context = format_context(selected_chunks, self.gen_config.prompt_version)

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"].format(context=context, query=query)},
        ]

        response = self.client.chat(messages)

        answer = response.content
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens

        cited_indices = extract_citations_from_text(answer)

        is_valid, _ = validate_citations(
            answer, len(selected_chunks), self.refusal_config.min_citations
        )

        if not is_valid and self.refusal_config.enabled:
            refusal_decision = check_citation_coverage(
                len(cited_indices), len(selected_chunks), self.refusal_config
            )
            if refusal_decision.should_refuse:
                fallback = generate_refusal_response(
                    refusal_decision,
                    selected_chunks,
                    "extractive",
                )
                return GenerationResult(
                    answer=answer,
                    citations=[],
                    refusal_triggered=True,
                    refusal_reason=refusal_decision.reason,
                    fallback_response=fallback,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model_name,
                    prompt_version=self.gen_config.prompt_version,
                )

        scores = {}
        if retrieval_results:
            for result in retrieval_results:
                score = result.rerank_score if result.rerank_score is not None else result.score
                scores[result.chunk.chunk_id] = score

        citations = build_citations_from_chunks(selected_chunks, cited_indices, scores)

        return GenerationResult(
            answer=answer,
            citations=citations,
            refusal_triggered=False,
            refusal_reason=None,
            fallback_response=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
            prompt_version=self.gen_config.prompt_version,
        )

    def generate_with_retry(
        self,
        query: str,
        selected_chunks: list[Chunk],
        retrieval_results: list[RetrievalResult] | None = None,
        max_retries: int = 2,
    ) -> GenerationResult:
        last_result: GenerationResult | None = None

        for attempt in range(max_retries + 1):
            result = self.generate(query, selected_chunks, retrieval_results)
            last_result = result

            if not result.refusal_triggered:
                return result

            if (
                result.refusal_reason == RefusalReason.INSUFFICIENT_CITATIONS
                and attempt < max_retries
            ):
                continue

            break

        if last_result is None:
            return self.generate(query, selected_chunks, retrieval_results)
        return last_result

    def generate_conversational(self, query: str) -> GenerationResult:
        """Generate a response for conversational queries without using RAG context."""
        model_name = self.client.get_model_name()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Python documentation assistant. "
                    "The user is engaging in casual conversation (e.g., greeting, thanking, or saying goodbye). "
                    "Reply naturally, politely, and briefly. Do not halluciante documentation or try to answer technical questions if none are asked."
                ),
            },
            {"role": "user", "content": query},
        ]

        response = self.client.chat(messages)

        return GenerationResult(
            answer=response.content,
            citations=[],
            refusal_triggered=False,
            refusal_reason=None,
            fallback_response=None,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=model_name,
            prompt_version=self.gen_config.prompt_version,
        )
