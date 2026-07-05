"""LangGraph ReAct chat agent.

One user turn = one agent.invoke(). The Postgres checkpointer (keyed by
conversation_id) holds the agent's memory across turns, so we pass only the new
user message each call. The Message table is kept as a UI read-model: we still
write user / assistant-tool-call / tool-result / answer rows so GET
/conversations/{id} and the Streamlit trace render unchanged.

Sync; routes wrap it with asyncio.to_thread.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from app.agents.langgraph_tools import build_tools
from app.agents.prompts import CHAT_SYSTEM_PROMPT
from app.db.base import session_scope
from app.db.checkpointer import get_checkpointer
from app.models import Conversation, Message, MessageRole, Paper
from app.services.chat_model import get_chat_model

log = logging.getLogger(__name__)

MAX_ITERATIONS = 6
# Each ReAct cycle is ~2 graph steps (model node + tool node).
RECURSION_LIMIT = MAX_ITERATIONS * 2 + 1


def chat_step(conversation_id: UUID, user_input: str) -> dict[str, Any]:
    """Run one user turn end-to-end, persisting all new messages."""
    with session_scope() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            raise ValueError(f"conversation {conversation_id} not found")
        paper = db.get(Paper, conv.paper_id)
        if paper is None:
            raise ValueError(f"paper {conv.paper_id} not found")
        paper_id = conv.paper_id
        paper_title = paper.title

        user_msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=user_input,
        )
        db.add(user_msg)
        db.flush()
        db.refresh(user_msg)
        user_dict = _message_to_dict(user_msg)

    agent = create_react_agent(
        get_chat_model(),
        build_tools(paper_id),
        state_modifier=CHAT_SYSTEM_PROMPT.format(paper_title=paper_title),
        checkpointer=get_checkpointer(),
    )
    config = {
        "configurable": {"thread_id": str(conversation_id)},
        "recursion_limit": RECURSION_LIMIT,
    }

    # The checkpointer accumulates history; everything after our input message
    # this turn is new.
    snapshot = agent.get_state(config)
    prev_count = len((snapshot.values or {}).get("messages", [])) if snapshot else 0

    try:
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        new_lc = result["messages"][prev_count + 1:]
        new_rows = _persist_trace(conversation_id, new_lc)
        answer = _final_text(new_lc)
        iterations = sum(1 for m in new_lc if isinstance(m, AIMessage)) or 1
    except GraphRecursionError:
        log.warning("Chat agent hit recursion limit for conversation %s", conversation_id)
        answer = "(agent reached max iterations without a final answer)"
        new_rows = _persist_answer(conversation_id, answer)
        iterations = MAX_ITERATIONS

    return {
        "answer": answer,
        "messages": [user_dict, *new_rows],
        "iterations": iterations,
    }


def _persist_trace(conversation_id: UUID, lc_messages: list) -> list[dict[str, Any]]:
    """Translate the turn's LangChain messages into Message rows for the UI."""
    with session_scope() as db:
        rows: list[Message] = []
        for m in lc_messages:
            if isinstance(m, AIMessage):
                if m.tool_calls:
                    for tc in m.tool_calls:
                        rows.append(Message(
                            conversation_id=conversation_id,
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_name=tc.get("name"),
                            tool_args=tc.get("args") or {},
                        ))
                else:
                    content = _as_text(m.content)
                    if content.strip():
                        rows.append(Message(
                            conversation_id=conversation_id,
                            role=MessageRole.ASSISTANT,
                            content=content,
                        ))
            elif isinstance(m, ToolMessage):
                rows.append(Message(
                    conversation_id=conversation_id,
                    role=MessageRole.TOOL,
                    content=_as_text(m.content),
                    tool_name=getattr(m, "name", None),
                ))
        for r in rows:
            db.add(r)
        db.flush()
        for r in rows:
            db.refresh(r)
        return [_message_to_dict(r) for r in rows]


def _persist_answer(conversation_id: UUID, answer: str) -> list[dict[str, Any]]:
    with session_scope() as db:
        row = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=answer,
        )
        db.add(row)
        db.flush()
        db.refresh(row)
        return [_message_to_dict(row)]


def _final_text(lc_messages: list) -> str:
    for m in reversed(lc_messages):
        if isinstance(m, AIMessage):
            text = _as_text(m.content)
            if text.strip():
                return text
    return "(empty answer)"


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {
        "id": m.id,
        "role": m.role.value if hasattr(m.role, "value") else str(m.role),
        "content": m.content,
        "tool_name": m.tool_name,
        "tool_args": m.tool_args,
        "created_at": m.created_at,
    }
