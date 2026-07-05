"""Deterministic, offline chat model for `llm_provider=mock`.

Mirrors the heuristics of the old MockLLM (app/services/llm_service.py) but as a
LangChain BaseChatModel so it works with create_react_agent and the pipeline
structured-output helper. No external calls.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class DeterministicFakeChat(BaseChatModel):
    """Offline fake. For the ReAct chat path it emits one tool call, then a final
    answer once a ToolMessage is present. Structured output is handled out-of-band
    by `mock_structured` (see app/services/chat_model.structured_invoke)."""

    @property
    def _llm_type(self) -> str:
        return "deterministic-fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DeterministicFakeChat":
        # We decide tool-vs-answer by inspecting the incoming messages, so we can
        # ignore the bound tools and just return ourselves.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        already_used_tool = any(isinstance(m, ToolMessage) for m in messages)
        if already_used_tool:
            msg = AIMessage(content="Mock answer based on the paper and tool results.")
        else:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_section", "args": {"name": "abstract"}, "id": "call_mock_1"}
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @staticmethod
    def mock_structured(schema: type, prompt: str):
        """Return a deterministic instance of `schema`, dispatched on its fields."""
        fields = set(getattr(schema, "model_fields", {}).keys())
        if "tasks" in fields:
            return schema(tasks=["summarize", "contributions", "methodology",
                                 "limitations", "future_work"])
        if "summary" in fields:
            return schema(summary="Mock summary: the paper proposes X, evaluates on Y, shows Z.")
        if "contributions" in fields:
            return schema(contributions=["Mock contribution A", "Mock contribution B"])
        if "methodology" in fields:
            return schema(methodology="Mock methodology: approach described in 3 stages.")
        if "limitations" in fields:
            return schema(limitations=["Mock limitation 1", "Mock limitation 2"])
        if "future_work" in fields:
            return schema(future_work=["Mock idea 1", "Mock idea 2"])
        # Fallback: best-effort empty instance.
        return schema()
