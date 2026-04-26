import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.executor import start_agent
from app.agents.prompts import QA_PROMPT
from app.api.deps import get_llm
from app.core.exceptions import PaperNotFoundError
from app.db.async_base import get_db_async
from app.models import Paper
from app.schemas.agent import (
    AskQuestionRequest,
    AskQuestionResponse,
    RunAgentRequest,
    RunAgentResponse,
)
from app.services.llm_service import LLMClient

router = APIRouter(tags=["agent"])


@router.post("/run-agent", response_model=RunAgentResponse)
async def run_agent(
    req: RunAgentRequest,
    db: Annotated[AsyncSession, Depends(get_db_async)],
) -> RunAgentResponse:
    # paper_id is in the body, not the path, so we can't reuse get_paper_or_404 here.
    if (await db.get(Paper, req.paper_id)) is None:
        raise PaperNotFoundError(details={"paper_id": str(req.paper_id)})
    # start_agent is sync (sync DB session, sync Redis, sync planner LLM,
    # Celery .apply_async). Run it off the event loop.
    result = await asyncio.to_thread(start_agent, req.paper_id, req.goal)
    return RunAgentResponse(**result)


@router.post("/ask-question", response_model=AskQuestionResponse)
async def ask_question(
    req: AskQuestionRequest,
    db: Annotated[AsyncSession, Depends(get_db_async)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> AskQuestionResponse:
    paper = await db.get(Paper, req.paper_id)
    if paper is None:
        raise PaperNotFoundError(details={"paper_id": str(req.paper_id)})

    # MVP: pass (truncated) paper text as context. Swap for RAG later.
    prompt = QA_PROMPT.format(question=req.question, context=paper.raw_text[:15000])
    answer = await asyncio.to_thread(
        llm.generate_json, prompt, '{"answer": "..."}',
    )
    return AskQuestionResponse(paper_id=paper.id, question=req.question, answer=answer)
