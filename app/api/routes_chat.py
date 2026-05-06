import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chat_agent import chat_step
from app.core.exceptions import ConversationNotFoundError, PaperNotFoundError
from app.db.async_base import get_db_async
from app.models import Conversation, Message, Paper
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationDetailOut,
    ConversationOut,
    MessageOut,
    StartConversationRequest,
)

router = APIRouter(tags=["chat"])


@router.post("/conversations", response_model=ConversationOut)
async def start_conversation(
    req: StartConversationRequest,
    db: Annotated[AsyncSession, Depends(get_db_async)],
) -> ConversationOut:
    if (await db.get(Paper, req.paper_id)) is None:
        raise PaperNotFoundError(details={"paper_id": str(req.paper_id)})

    conv = Conversation(paper_id=req.paper_id, title=req.title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return ConversationOut.model_validate(conv)


@router.post("/conversations/{conversation_id}/messages", response_model=ChatResponse)
async def send_message(
    conversation_id: UUID,
    req: ChatRequest,
    db: Annotated[AsyncSession, Depends(get_db_async)],
) -> ChatResponse:
    if (await db.get(Conversation, conversation_id)) is None:
        raise ConversationNotFoundError(details={"conversation_id": str(conversation_id)})

    # The agent loop is sync (sync DB session, sync LLM). Run off the event loop.
    result = await asyncio.to_thread(chat_step, conversation_id, req.content)

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        trace=[MessageOut.model_validate(m) for m in result["messages"]],
        iterations=result["iterations"],
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db_async)],
) -> ConversationDetailOut:
    conv = await db.get(Conversation, conversation_id)
    if conv is None:
        raise ConversationNotFoundError(details={"conversation_id": str(conversation_id)})

    msgs = (await db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )).all()

    return ConversationDetailOut(
        id=conv.id,
        paper_id=conv.paper_id,
        title=conv.title,
        created_at=conv.created_at,
        messages=[MessageOut.model_validate(m) for m in msgs],
    )
