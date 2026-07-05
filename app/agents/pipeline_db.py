"""Postgres helpers shared by the pipeline graph and the orchestrator.

These keep the AgentTask rows and the Insight row up to date so the existing
GET /paper/{id}/results polling UI works unchanged.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.exceptions import PaperNotFoundError
from app.db.base import session_scope
from app.models import AgentTask, Insight, Paper, TaskStatus


def load_paper_text(paper_id: UUID) -> str:
    with session_scope() as db:
        paper = db.get(Paper, paper_id)
        if not paper:
            raise PaperNotFoundError(details={"paper_id": str(paper_id)})
        return paper.raw_text


def find_task_row(db, paper_id: UUID, task_name: str) -> AgentTask | None:
    return db.scalar(
        select(AgentTask)
        .where(AgentTask.paper_id == paper_id, AgentTask.task_name == task_name)
    )


def mark(paper_id: UUID, task_name: str, status: TaskStatus, **fields: Any) -> None:
    with session_scope() as db:
        row = find_task_row(db, paper_id, task_name)
        if row is None:
            return
        row.status = status
        for k, v in fields.items():
            setattr(row, k, v)


def upsert_insight(paper_id: UUID, **fields: Any) -> None:
    with session_scope() as db:
        ins = db.scalar(select(Insight).where(Insight.paper_id == paper_id))
        if ins is None:
            ins = Insight(paper_id=paper_id, **fields)
            db.add(ins)
        else:
            for k, v in fields.items():
                setattr(ins, k, v)
