"""LLM abstraction. Two providers: `mock` (deterministic, offline) and `gemini`.

Usage:
    from app.services.llm_service import llm
    data = llm.generate_json(prompt, schema_hint="...")
"""
from __future__ import annotations

import json
import logging
import re
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

    def generate_json(self, prompt: str, schema_hint: str = "") -> Any:
        """Ask the LLM for JSON. Appends schema hint + strict instruction."""
        full = prompt
        if schema_hint:
            full += f"\n\nReturn ONLY valid JSON matching this shape:\n{schema_hint}"
        full += "\n\nNo prose, no markdown fences. JSON only."
        raw = self.generate_text(full)
        try:
            return _parse_json(raw)
        except json.JSONDecodeError as e:
            raise LLMError(
                "LLM returned invalid JSON",
                details={"raw_preview": raw[:300]},
            ) from e


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    # Strip ```json fences if the model added them anyway.
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: find the first {...} or [...] block.
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


# ---------- Mock ----------

class MockLLM(LLMClient):
    """Deterministic responses for offline dev and tests."""

    def generate_text(self, prompt: str) -> str:
        p = prompt.lower()
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
