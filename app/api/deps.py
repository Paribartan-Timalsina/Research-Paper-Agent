"""FastAPI dependencies.

Routes inline their `Annotated[..., Depends(...)]` declarations directly. This
module just provides the underlying provider functions. Override in tests with
`app.dependency_overrides[...]`.
"""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import PaperNotFoundError
from app.db.async_base import get_db_async
from app.models import Paper
from app.services.llm_service import LLMClient, llm


__all__ = ["get_llm", "get_paper_or_404"]


def get_llm() -> LLMClient:
    """Provide the shared LLM client. Overridable in tests."""
    return llm


async def get_paper_or_404(
    paper_id: Annotated[UUID, Path(...)],
    db: Annotated[AsyncSession, Depends(get_db_async)],
) -> Paper:
    """Fetch a Paper by id or raise a domain-level 404."""
    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise PaperNotFoundError(details={"paper_id": str(paper_id)})
    return paper
