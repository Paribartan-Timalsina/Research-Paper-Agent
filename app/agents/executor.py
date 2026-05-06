"""Agent orchestrator.

Takes a paper_id + user goal, asks the planner for an ordered task list,
creates AgentTask rows in Postgres (so status is queryable), and dispatches
the tasks to Celery as a chain. Context flows via Redis (see context_service).
"""
from __future__ import annotations

import logging
from uuid import UUID

from celery import chain
from sqlalchemy import delete

from app.agents.planner import plan_tasks
from app.core.exceptions import PaperNotFoundError
from app.db.base import session_scope
from app.models import AgentTask, Paper, TaskStatus
from app.services import context_service
from app.tasks.paper_tasks import TASK_REGISTRY

log = logging.getLogger(__name__)


def start_agent(paper_id: float, goal: str = "") -> dict:
    # 1. Load paper + plan
    with session_scope() as db:
        paper = db.get(Paper, paper_id)
        if paper is None:
            raise PaperNotFoundError(details={"paper_id": str(paper_id)})
        paper_text = paper.raw_text

    task_names = plan_tasks(goal, paper_text)
    log.info("Planned tasks for %s: %s", paper_id, task_names)

    # 2. Reset any previous run for this paper (fresh context + task rows)
    context_service.clear(paper_id)
    with session_scope() as db:
        db.execute(delete(AgentTask).where(AgentTask.paper_id == paper_id))
        for idx, name in enumerate(task_names):
            db.add(AgentTask(
                paper_id=paper_id,
                task_name=name,
                order_index=idx,
                status=TaskStatus.PENDING,
            ))

    # 3. Build a Celery chain: each task takes the prior task's return value
    #    (paper_id string) as its first argument.
    signatures = [TASK_REGISTRY[name].s() for name in task_names]
    # Seed the chain with the paper_id as the first arg.
    if signatures:
        signatures[0] = TASK_REGISTRY[task_names[0]].s(str(paper_id))

    async_result = chain(*signatures).apply_async() if signatures else None

    return {
        "paper_id": str(paper_id),
        "plan": task_names,
        "chain_id": async_result.id if async_result else None,
    }
