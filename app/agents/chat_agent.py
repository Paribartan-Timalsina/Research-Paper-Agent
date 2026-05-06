"""ReAct-style chat agent.

One user turn drives a short loop:
    LLM(messages + tools) → JSON
        action == "tool"   → run tool, feed result back, loop
        action == "answer" → persist final answer, done

Sync (uses sync session_scope + sync LLM). Routes wrap with `asyncio.to_thread`.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from app.agents.prompts import CHAT_SYSTEM_PROMPT
from app.agents.tools import render_tool_list, run_tool
from app.core.exceptions import LLMError
from app.db.base import session_scope
from app.models import Conversation, Message, MessageRole, Paper
from app.services.llm_service import llm

log = logging.getLogger(__name__)

MAX_ITERATIONS = 6
HISTORY_TURNS_KEPT = 30  # keep last N messages from prior turns to bound prompt size


def _llm_messages_from_history(history: list[Message]) -> list[dict[str, str]]:
    """Convert persisted Message rows into the LLM's role/content format."""
    out: list[dict[str, str]] = []
    for m in history:
        if m.role == MessageRole.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == MessageRole.ASSISTANT:
            if m.tool_name:
                # Replay a tool-call turn as the assistant's last action
                payload = {"action": "tool", "tool": m.tool_name, "args": m.tool_args or {}}
                out.append({"role": "assistant", "content": json.dumps(payload)})
            else:
                # Final answer turn
                payload = {"action": "answer", "content": m.content}
                out.append({"role": "assistant", "content": json.dumps(payload)})
        elif m.role == MessageRole.TOOL:
            out.append({
                "role": "user",
                "content": f"TOOL RESULT ({m.tool_name}): {m.content}",
            })
        # SYSTEM messages from history are ignored — system prompt is rebuilt each turn
    return out


def chat_step(conversation_id: UUID, user_input: str) -> dict[str, Any]:
    """Run one user turn end-to-end. Persists all new messages.

    Returns: {
        "answer": str,
        "messages": [Message rows that were just appended (in order)],
        "iterations": int,
    }
    """
    with session_scope() as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None:
            raise ValueError(f"conversation {conversation_id} not found")
        paper = db.get(Paper, conv.paper_id)
        if paper is None:
            raise ValueError(f"paper {conv.paper_id} not found")

        # Pull history (already ordered by created_at via the relationship)
        history = list(conv.messages)[-HISTORY_TURNS_KEPT:]

        # Persist the user's input first
        user_msg = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=user_input,
        )
        db.add(user_msg)
        db.flush()    # so subsequent rows have created_at after this one

        # Build LLM context
        llm_msgs: list[dict[str, str]] = [
            {
                "role": "system",
                "content": CHAT_SYSTEM_PROMPT.format(
                    paper_title=paper.title,
                    tool_list=render_tool_list(),
                ),
            },
        ]
        llm_msgs.extend(_llm_messages_from_history(history))
        llm_msgs.append({"role": "user", "content": user_input})

        new_msgs: list[Message] = [user_msg]
        final_answer: str | None = None

        for iteration in range(1, MAX_ITERATIONS + 1):
            try:
                decision = llm.generate_chat_json(llm_msgs)
            except LLMError as e:
                log.warning("LLM failed during chat: %s", e)
                final_answer = f"(LLM error: {e.message})"
                break

            action = (decision or {}).get("action")

            if action == "answer":
                content = (decision.get("content") or "").strip()
                final_answer = content or "(empty answer)"
                msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=final_answer,
                )
                db.add(msg)
                db.flush()
                new_msgs.append(msg)
                break

            if action == "tool":
                tool_name = decision.get("tool", "")
                tool_args = decision.get("args") or {}

                # Persist the assistant tool-call message
                call_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                db.add(call_msg)
                db.flush()
                new_msgs.append(call_msg)

                # Run the tool
                result = run_tool(tool_name, tool_args, paper, db)
                tool_msg = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.TOOL,
                    content=result,
                    tool_name=tool_name,
                )
                db.add(tool_msg)
                db.flush()
                new_msgs.append(tool_msg)

                # Feed back into LLM context for the next iteration
                llm_msgs.append({
                    "role": "assistant",
                    "content": json.dumps({"action": "tool", "tool": tool_name, "args": tool_args}),
                })
                llm_msgs.append({
                    "role": "user",
                    "content": f"TOOL RESULT ({tool_name}): {result}",
                })
                continue

            # Unknown action: bail with a friendly message
            log.warning("Unknown LLM action: %r", action)
            final_answer = "(agent returned an unknown action)"
            msg = Message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=final_answer,
            )
            db.add(msg)
            db.flush()
            new_msgs.append(msg)
            break

        else:
            # Loop exhausted without an "answer" action
            final_answer = "(agent reached max iterations without a final answer)"
            msg = Message(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=final_answer,
            )
            db.add(msg)
            db.flush()
            new_msgs.append(msg)
            iteration = MAX_ITERATIONS

        # Detach from session so callers can read fields after the session closes
        for m in new_msgs:
            db.refresh(m)
        return {
            "answer": final_answer or "",
            "messages": [_message_to_dict(m) for m in new_msgs],
            "iterations": iteration,
        }


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {
        "id": m.id,
        "role": m.role.value if hasattr(m.role, "value") else str(m.role),
        "content": m.content,
        "tool_name": m.tool_name,
        "tool_args": m.tool_args,
        "created_at": m.created_at,
    }
