"""Agent orchestrator.

Takes a paper_id + goal, resets the AgentTask rows, and dispatches a single
Celery task that runs the LangGraph analysis pipeline. The planner now lives
inside the graph; here we just pre-create one PENDING row per allowed task so
the UI shows the full task list immediately.
"""
from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import delete

from app.agents.prompts import ALLOWED_TASKS
from app.core.exceptions import PaperNotFoundError
from app.db.base import session_scope
from app.models import AgentTask, Paper, TaskStatus
from app.tasks.paper_tasks import run_pipeline

log = logging.getLogger(__name__)


def start_agent(paper_id: UUID, goal: str = "") -> dict:
    with session_scope() as db:
        paper = db.get(Paper, paper_id)
        if paper is None:
            raise PaperNotFoundError(details={"paper_id": str(paper_id)})

    # Reset any previous run: fresh PENDING rows for every allowed task.
    with session_scope() as db:
        db.execute(delete(AgentTask).where(AgentTask.paper_id == paper_id))
        for idx, name in enumerate(ALLOWED_TASKS):
            db.add(AgentTask(
                paper_id=paper_id,
                task_name=name,
                order_index=idx,
                status=TaskStatus.PENDING,
            ))

    async_result = run_pipeline.apply_async(args=[str(paper_id), goal])
    log.info("Dispatched pipeline for %s (task %s)", paper_id, async_result.id)

    return {
        "paper_id": str(paper_id),
        "plan": list(ALLOWED_TASKS),
        "chain_id": async_result.id,
    }
