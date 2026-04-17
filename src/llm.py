from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import Settings


class QuotaExceededError(RuntimeError):
    pass


class EmbeddingConfigError(RuntimeError):
    pass


class LocalLLMUnavailableError(RuntimeError):
    pass


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, QuotaExceededError):
        return False
    if isinstance(exc, RateLimitError):
        code = getattr(exc, "code", None)
        if code == "insufficient_quota":
            return False
    if isinstance(exc, LocalLLMUnavailableError):
        return False
    return True


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: OpenAI | None = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self._local_client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        clean = [t[:8000] for t in texts]
        if not clean:
            return []
        if self.settings.embedding_provider == "local":
            model = _load_sentence_transformer(self.settings.local_embedding_model)
            return model.encode(clean, normalize_embeddings=True).tolist()
        if self.settings.embedding_provider != "openai":
            raise EmbeddingConfigError(
                f"Unsupported EMBEDDING_PROVIDER '{self.settings.embedding_provider}'. Use 'local' or 'openai'."
            )
        client = self._require_client()
        try:
            response = client.embeddings.create(model=self.settings.openai_embedding_model, input=clean)
        except RateLimitError as exc:
            self._raise_if_quota_exhausted(exc)
            raise
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
        if self.settings.answer_provider == "local":
            return self._chat_json_local(system_prompt, user_prompt, temperature)

        client = self._require_client()
        try:
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except RateLimitError as exc:
            self._raise_if_quota_exhausted(exc)
            raise
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def chat_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        if self.settings.answer_provider == "local":
            return self._chat_text_local(system_prompt, user_prompt, temperature)

        client = self._require_client()
        try:
            response = client.chat.completions.create(
                model=self.settings.openai_chat_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except RateLimitError as exc:
            self._raise_if_quota_exhausted(exc)
            raise
        return (response.choices[0].message.content or "").strip()

    def supports_chat_json(self) -> bool:
        if self.settings.answer_provider == "openai":
            return self.client is not None
        try:
            self._get_local_client()
            return True
        except LocalLLMUnavailableError:
            return False

    def _require_client(self) -> OpenAI:
        if self.client is None:
            raise ValueError("OPENAI_API_KEY is missing. Add it in Streamlit secrets or .env")
        return self.client

    def _chat_json_local(self, system_prompt: str, user_prompt: str, temperature: float) -> dict:
        client = self._get_local_client()
        try:
            response = client.chat(
                model=self.settings.ollama_chat_model,
                format="json",
                options={"temperature": temperature},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise LocalLLMUnavailableError(
                f"Local Ollama chat failed for model '{self.settings.ollama_chat_model}'. "
                "Start Ollama and pull the configured model, or switch ANSWER_PROVIDER."
            ) from exc

        message = response.get("message", {}) if isinstance(response, dict) else {}
        content = message.get("content", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LocalLLMUnavailableError(
                f"Local Ollama model '{self.settings.ollama_chat_model}' did not return valid JSON."
            ) from exc

    def _chat_text_local(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        client = self._get_local_client()
        try:
            response = client.chat(
                model=self.settings.ollama_chat_model,
                options={"temperature": temperature},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise LocalLLMUnavailableError(
                f"Local Ollama chat failed for model '{self.settings.ollama_chat_model}'. "
                "Start Ollama and pull the configured model, or switch ANSWER_PROVIDER."
            ) from exc

        message = response.get("message", {}) if isinstance(response, dict) else {}
        return str(message.get("content", "")).strip()

    def _get_local_client(self):
        if self._local_client is not None:
            return self._local_client
        try:
            import ollama
        except ImportError as exc:
            raise LocalLLMUnavailableError(
                "The 'ollama' Python package is not installed. Add it to your environment to use local chat."
            ) from exc

        self._local_client = ollama.Client(host=self.settings.ollama_base_url)
        try:
            self._local_client.list()
        except Exception as exc:
            raise LocalLLMUnavailableError(
                f"Could not connect to Ollama at {self.settings.ollama_base_url}. Make sure Ollama is running."
            ) from exc
        return self._local_client

    @staticmethod
    def _raise_if_quota_exhausted(exc: RateLimitError) -> None:
        code = getattr(exc, "code", None)
        if code == "insufficient_quota":
            raise QuotaExceededError(
                "OpenAI API quota is exhausted for the configured key. Update billing or use a key with available credits."
            ) from exc


OpenAIService = LLMService
