"""Factory for the LangChain chat model used by both LangGraph flows.

Embeddings and vision stay on app.services.llm_service.llm (LLMClient); only
text generation / chat / structured output moves to a LangChain BaseChatModel.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from app.config import settings
from app.services.fake_chat_model import DeterministicFakeChat


@lru_cache
def get_chat_model() -> BaseChatModel:
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
    # mock provider, or gemini selected without a key (mirrors the old fallback).
    return DeterministicFakeChat()


def structured_invoke(model: BaseChatModel, schema: type, prompt: str):
    """Invoke `model` and return a populated instance of pydantic `schema`.

    The offline fake is short-circuited (it can't honor LangChain's structured
    wrapper); real models use native structured output.
    """
    if isinstance(model, DeterministicFakeChat):
        return DeterministicFakeChat.mock_structured(schema, prompt)
    return model.with_structured_output(schema).invoke(prompt)
