"""Answer quality rubric scoring."""

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from pydocs_rag.config import AnswerRubricConfig


RUBRIC_SYSTEM_PROMPT = """You are an expert evaluator for a Python documentation RAG system.
Evaluate the given answer against the reference answer and context.

Score each dimension from 0.0 to 1.0:

1. CORRECTNESS (0-1): Is the answer factually accurate?
   - 1.0: Completely correct
   - 0.5: Partially correct with minor errors
   - 0.0: Incorrect or misleading

2. GROUNDEDNESS (0-1): Is the answer supported by citations?
   - 1.0: All claims have proper citations
   - 0.5: Some claims lack citations
   - 0.0: No citations or mostly unsupported claims

3. COMPLETENESS (0-1): Does the answer address the question fully?
   - 1.0: Comprehensive answer
   - 0.5: Partial answer, missing some aspects
   - 0.0: Does not address the question

4. HAS_HALLUCINATION (true/false): Does the answer contain information not in the context?

Return a JSON object with: correctness, groundedness, completeness, has_hallucination"""


RUBRIC_USER_PROMPT = """Context provided to the system:
{context}

Question: {question}

System Answer:
{answer}

Reference Answer (if available):
{reference}

Evaluate the system answer. Return JSON with scores."""


@dataclass
class RubricScore:
    correctness: float
    groundedness: float
    completeness: float
    has_hallucination: bool
    weighted_score: float
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "completeness": self.completeness,
            "has_hallucination": self.has_hallucination,
            "weighted_score": self.weighted_score,
        }


class RubricScorer:
    def __init__(self, config: AnswerRubricConfig, model: str = "gpt-4o-mini") -> None:
        self.config = config
        self.model = model
        self.client = OpenAI()

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        reference: str | None = None,
    ) -> RubricScore:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": RUBRIC_USER_PROMPT.format(
                            context=context[:4000],
                            question=question,
                            answer=answer,
                            reference=reference or "Not provided",
                        ),
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            import json

            scores = json.loads(content)

            correctness = float(scores.get("correctness", 0.5))
            groundedness = float(scores.get("groundedness", 0.5))
            completeness = float(scores.get("completeness", 0.5))
            has_hallucination = bool(scores.get("has_hallucination", False))

            weighted_score = (
                correctness * self.config.correctness_weight
                + groundedness * self.config.groundedness_weight
                + completeness * self.config.completeness_weight
            )

            return RubricScore(
                correctness=correctness,
                groundedness=groundedness,
                completeness=completeness,
                has_hallucination=has_hallucination,
                weighted_score=weighted_score,
                raw_response=content,
            )

        except Exception as e:
            return RubricScore(
                correctness=0.0,
                groundedness=0.0,
                completeness=0.0,
                has_hallucination=True,
                weighted_score=0.0,
                raw_response=f"Error: {e!s}",
            )

    def score_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[RubricScore]:
        return [
            self.score(
                question=s["question"],
                answer=s["answer"],
                context=s["context"],
                reference=s.get("reference"),
            )
            for s in samples
        ]
