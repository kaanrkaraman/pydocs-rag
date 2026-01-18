"""LLM client abstraction supporting OpenAI and Ollama."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from pydocs_rag.config import LLMProviderConfig


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMProviderConfig) -> None:
        from openai import OpenAI

        self.config = config
        self.client = OpenAI()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.openai_model,
        )

    def get_model_name(self) -> str:
        return self.config.openai_model


class OllamaClient(LLMClient):
    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.base_url = config.ollama_base_url.rstrip("/")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.config.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            },
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        content = data.get("message", {}).get("content", "")

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.config.ollama_model,
        )

    def get_model_name(self) -> str:
        return self.config.ollama_model

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


def create_llm_client(config: LLMProviderConfig) -> LLMClient:
    if config.provider == "ollama":
        return OllamaClient(config)
    elif config.provider == "openai":
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
