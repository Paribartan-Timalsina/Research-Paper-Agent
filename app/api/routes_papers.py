import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_paper_or_404
from app.core.exceptions import InvalidFileError
from app.db.async_base import get_db_async
from app.models import AgentTask, Insight, Paper, PaperFigure
from app.schemas.agent import InsightOut, ResultsResponse, TaskOut
from app.schemas.paper import UploadResponse
from app.services.llm_service import llm
from app.services.pdf_service import ExtractedPDF, extract_pdf
from app.services.rag_service import index_paper

router = APIRouter(tags=["papers"])


@router.post("/upload-paper", response_model=UploadResponse)
async def upload_paper(
    db: Annotated[AsyncSession, Depends(get_db_async)],
    file: UploadFile = File(...),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise InvalidFileError("Only PDF files are accepted")

    content = await file.read()

    # PDF parsing is CPU-bound (PyMuPDF). Operating without blocking the event loop.
    # with_figures=True triggers VLM calls per embedded image — slower but populates
    # the figures table so the chat agent's describe_figure tool has data.
    parsed: ExtractedPDF = await asyncio.to_thread(
        extract_pdf, content, file.filename, False,
    )

    paper = Paper(title=parsed.title, filename=file.filename, raw_text=parsed.text)
    db.add(paper)
    await db.flush()   # so paper.id is available for the figure rows

    for page, desc in parsed.figures:
        db.add(PaperFigure(paper_id=paper.id, page=page, description=desc))

    await db.commit()
    await db.refresh(paper)

    await index_paper(paper.id, db, llm)

    return UploadResponse(paper_id=paper.id, title=paper.title, char_count=len(paper.raw_text))


@router.get("/paper/{paper_id}/results", response_model=ResultsResponse)
async def get_results(
    paper: Annotated[Paper, Depends(get_paper_or_404)],
    db: Annotated[AsyncSession, Depends(get_db_async)],
) -> ResultsResponse:
    tasks = (await db.scalars(
        select(AgentTask)
        .where(AgentTask.paper_id == paper.id)
        .order_by(AgentTask.order_index)
    )).all()
    insight = await db.scalar(select(Insight).where(Insight.paper_id == paper.id))

    return ResultsResponse(
        paper_id=paper.id,
        title=paper.title,
        tasks=[TaskOut.model_validate(t) for t in tasks],
        insight=InsightOut.model_validate(insight) if insight else None,
    )
