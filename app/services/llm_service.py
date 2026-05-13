"""LLM abstraction. Two providers: `mock` (deterministic, offline) and `gemini`.

Usage:
    from app.services.llm_service import llm
    data = llm.generate_json(prompt, schema_hint="...")
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.exceptions import LLMError

log = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, prompt: str) -> str: ...

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:  # noqa: ARG002
        raise NotImplementedError("This LLM does not support vision")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("This LLM does not support embeddings")

    def generate_json(self, prompt: str, schema_hint: str = "") -> Any:
        """Ask the LLM for JSON. Appends schema hint + strict instruction."""
        full = prompt
        if schema_hint:
            full += f"\n\nReturn ONLY valid JSON matching this shape:\n{schema_hint}"
        full += "\n\nNo any starting text such as 'Here is the JSON:', no markdown fences. JSON only."
        raw = self.generate_text(full)
        try:
            return _parse_json(raw)
        except json.JSONDecodeError as e:
            raise LLMError(
                "LLM returned invalid JSON",
                details={"raw_preview": raw[:300]},
            ) from e

    def generate_chat_json(self, messages: list[dict[str, str]]) -> Any:
        """Multi-turn chat returning JSON.

        `messages` is a list of {"role": "system|user|assistant", "content": "..."} dicts.
        Default impl serializes them and calls generate_text. Subclasses with native
        chat APIs can override.
        """
        serialized = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        serialized += "\n\nRespond with ONLY valid JSON. No prose, no markdown fences."
        raw = self.generate_text(serialized)
        try:
            return _parse_json(raw)
        except json.JSONDecodeError as e:
            raise LLMError(
                "LLM returned invalid JSON",
                details={"raw_preview": raw[:300]},
            ) from e


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise


# ---------- Mock ----------

class MockLLM(LLMClient):
    """Deterministic responses for offline dev and tests."""

    def generate_text(self, prompt: str) -> str:
        p = prompt.lower()
        # Chat agent: detect the system prompt and respond with a tool call once,
        # then a final answer. We can't keep state, so use a simple heuristic:
        # if a TOOL RESULT is already present, return final answer; otherwise tool call.
        if "research-paper assistant" in p and "available tools" in p:
            if "tool result" in p:
                return json.dumps({
                    "action": "answer",
                    "content": "Mock answer based on the paper and tool results.",
                })
            return json.dumps({
                "action": "tool",
                "tool": "get_section",
                "args": {"name": "abstract"},
            })
        if "task plan" in p or "plan of tasks" in p:
            return json.dumps([
                {"task": "summarize"},
                {"task": "contributions"},
                {"task": "methodology"},
                {"task": "limitations"},
                {"task": "future_work"},
            ])
        if "summarize" in p or "summary" in p:
            return json.dumps({"summary": "Mock summary: the paper proposes X, evaluates on Y, shows Z."})
        if "contribution" in p:
            return json.dumps({"contributions": ["Mock contribution A", "Mock contribution B"]})
        if "methodolog" in p:
            return json.dumps({"methodology": "Mock methodology: approach described in 3 stages."})
        if "limitation" in p:
            return json.dumps({"limitations": ["Mock limitation 1", "Mock limitation 2"]})
        if "future" in p or "improvement" in p or "idea" in p:
            return json.dumps({"future_work": ["Mock idea 1", "Mock idea 2"]})
        if "question" in p or "answer" in p:
            return json.dumps({"answer": "Mock answer based on the paper."})
        return json.dumps({"text": "mock response"})

    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        return "Mock figure: a chart or diagram from the paper."

    def embed(self, texts: list[str]) -> list[list[float]]:
        import random
        return [[random.random() for _ in range(768)] for _ in texts]


# ---------- Gemini ----------

class GeminiLLM(LLMClient):
    def __init__(self, api_key: str, model: str):
        # Imported lazily so the mock path works without the package.
        from google import genai

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def generate_text(self, prompt: str) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self._model, contents=prompt
            )
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e
        return resp.text or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def describe_image(self, image_bytes: bytes, prompt: str) -> str:
        from google.genai import types

        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
        except Exception as e:
            raise LLMError(f"Gemini vision call failed: {e}") from e
        return resp.text or ""

    def embed(self, texts: list[str]) -> list[list[float]]:  # type: ignore[reportUnusedFunction]
        try:
            resp = self._client.models.embed_content(  # type: ignore
                model="text-embedding-004",
                contents=texts,
            )
            result: list[list[float]] = []
            for emb in (resp.embeddings or []):  # type: ignore
                values = list(emb.values) if hasattr(emb, "values") else []
                result.append(values)
            return result
        except Exception as e:
            raise LLMError(f"Gemini embed call failed: {e}") from e


# ---------- Factory ----------

def _build_llm() -> LLMClient:
    if settings.llm_provider == "gemini":
        try:
            return GeminiLLM(settings.gemini_api_key, settings.gemini_model)
        except Exception as e:
            log.warning("Falling back to MockLLM: %s", e)
            return MockLLM()
    return MockLLM()


llm: LLMClient = _build_llm()
