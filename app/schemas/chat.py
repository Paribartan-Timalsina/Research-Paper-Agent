from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class StartConversationRequest(BaseModel):
    paper_id: UUID
    title: str | None = None


class ChatRequest(BaseModel):
    content: str


class MessageOut(BaseModel):
    id: UUID
    role: str
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: UUID
    paper_id: UUID
    title: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut]


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    trace: list[MessageOut]   # all new messages (user + tool calls + tool results + assistant answer)
    iterations: int           # how many LLM round-trips it took


class ToolCallTrace(BaseModel):
    tool: str
    args: dict[str, Any]
    result: str
