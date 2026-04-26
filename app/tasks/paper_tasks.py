"""Celery tasks for paper analysis.

Each task follows the same shape:
    1. Mark the AgentTask row RUNNING.
    2. Read paper text + prior context from Redis.
    3. Call the LLM with a task-specific prompt.
    4. Write the structured result back to both Redis (context) and Postgres.
    5. Return the paper_id so Celery chains can pass it to the next task.

Tasks are designed to be chained: the return value (paper_id) flows into the
next task's first argument, and shared state travels via Redis.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from celery import shared_task
from sqlalchemy import select

from app.agents import prompts
from app.core.exceptions import PaperNotFoundError, TaskExecutionError
from app.db.base import session_scope
from app.models import AgentTask, Insight, Paper, TaskStatus
from app.services import context_service
from app.services.llm_service import llm
from app.tasks.celery_app import celery_app  # noqa: F401  (ensures app is registered)

log = logging.getLogger(__name__)


# ---------- helpers ----------

def _load_paper_text(paper_id: UUID) -> str:
    with session_scope() as db:
        paper = db.get(Paper, paper_id)
        if not paper:
            raise PaperNotFoundError(details={"paper_id": str(paper_id)})
        return paper.raw_text


def _find_task_row(db, paper_id: UUID, task_name: str) -> AgentTask | None:
    return db.scalar(
        select(AgentTask)
        .where(AgentTask.paper_id == paper_id, AgentTask.task_name == task_name)
    )


def _mark(paper_id: UUID, task_name: str, status: TaskStatus, **fields: Any) -> None:
    with session_scope() as db:
        row = _find_task_row(db, paper_id, task_name)
        if row is None:
            return
        row.status = status
        for k, v in fields.items():
            setattr(row, k, v)


def _upsert_insight(paper_id: UUID, **fields: Any) -> None:
    with session_scope() as db:
        ins = db.scalar(select(Insight).where(Insight.paper_id == paper_id))
        if ins is None:
            ins = Insight(paper_id=paper_id, **fields)
            db.add(ins)
        else:
            for k, v in fields.items():
                setattr(ins, k, v)


def _run_step(task_name: str, paper_id: str, build_prompt, schema_hint: str, insight_field: str):
    """Shared wrapper for every analysis step."""
    pid = UUID(paper_id)
    _mark(pid, task_name, TaskStatus.RUNNING)
    try:
        paper_text = _load_paper_text(pid)
        ctx = context_service.get_all(pid)
        prompt = build_prompt(paper_text, ctx)
        result = llm.generate_json(prompt, schema_hint=schema_hint)

        context_service.set_field(pid, task_name, result)
        _mark(pid, task_name, TaskStatus.COMPLETED, result=result)
        _upsert_insight(pid, **{insight_field: result.get(insight_field)})
        return paper_id
    except Exception as e:
        log.exception("Task %s failed for paper %s", task_name, paper_id)
        _mark(pid, task_name, TaskStatus.FAILED, error=str(e)[:500])
        # Wrap unknown errors so downstream handlers get a consistent type.
        raise TaskExecutionError(
            f"Task '{task_name}' failed: {e}",
            details={"paper_id": paper_id, "task_name": task_name},
        ) from e


# ---------- analysis tasks ----------

@shared_task(name="paper.summarize", bind=True)
def summarize(self, paper_id: str) -> str:
    return _run_step(
        "summarize",
        paper_id,
        build_prompt=lambda paper, ctx: prompts.SUMMARIZE_PROMPT.format(paper=paper[:12000]),
        schema_hint='{"summary": "..."}',
        insight_field="summary",
    )


@shared_task(name="paper.contributions", bind=True)
def contributions(self, paper_id: str) -> str:
    def build(paper: str, ctx: dict) -> str:
        summary = (ctx.get("summarize") or {}).get("summary", "")
        return prompts.CONTRIBUTIONS_PROMPT.format(paper=paper[:12000], summary=summary)
    return _run_step(
        "contributions",
        paper_id,
        build_prompt=build,
        schema_hint='{"contributions": ["..."]}',
        insight_field="contributions",
    )


@shared_task(name="paper.methodology", bind=True)
def methodology(self, paper_id: str) -> str:
    def build(paper: str, ctx: dict) -> str:
        summary = (ctx.get("summarize") or {}).get("summary", "")
        contribs = (ctx.get("contributions") or {}).get("contributions", [])
        return prompts.METHODOLOGY_PROMPT.format(
            paper=paper[:12000], summary=summary, contributions=contribs
        )
    return _run_step(
        "methodology",
        paper_id,
        build_prompt=build,
        schema_hint='{"methodology": "..."}',
        insight_field="methodology",
    )


@shared_task(name="paper.limitations", bind=True)
def limitations(self, paper_id: str) -> str:
    def build(paper: str, ctx: dict) -> str:
        summary = (ctx.get("summarize") or {}).get("summary", "")
        method = (ctx.get("methodology") or {}).get("methodology", "")
        return prompts.LIMITATIONS_PROMPT.format(
            paper=paper[:12000], summary=summary, methodology=method
        )
    return _run_step(
        "limitations",
        paper_id,
        build_prompt=build,
        schema_hint='{"limitations": ["..."]}',
        insight_field="limitations",
    )


@shared_task(name="paper.future_work", bind=True)
def future_work(self, paper_id: str) -> str:
    def build(paper: str, ctx: dict) -> str:
        summary = (ctx.get("summarize") or {}).get("summary", "")
        lims = (ctx.get("limitations") or {}).get("limitations", [])
        return prompts.FUTURE_WORK_PROMPT.format(
            paper=paper[:12000], summary=summary, limitations=lims
        )
    return _run_step(
        "future_work",
        paper_id,
        build_prompt=build,
        schema_hint='{"future_work": ["..."]}',
        insight_field="future_work",
    )


# Map task-name -> Celery signature builder. Used by the orchestrator.
TASK_REGISTRY = {
    "summarize": summarize,
    "contributions": contributions,
    "methodology": methodology,
    "limitations": limitations,
    "future_work": future_work,
}
