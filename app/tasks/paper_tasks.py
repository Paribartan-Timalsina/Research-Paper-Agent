"""Celery task that runs the analysis pipeline.

The whole LangGraph StateGraph runs inside this single task — Celery is the
distributed execution substrate; LangGraph handles the per-step orchestration
and state. Kept as a module so celery_app.py's `include` can import it.
"""
from __future__ import annotations

import logging

from celery import shared_task

from app.tasks.celery_app import celery_app  # noqa: F401  (ensures app is registered)

log = logging.getLogger(__name__)


@shared_task(name="paper.run_pipeline", bind=True)
def run_pipeline(self, paper_id: str, goal: str = "") -> str:
    # Lazy import keeps Celery startup light and avoids importing LangGraph in
    # processes that only enqueue the task.
    from app.agents.pipeline_runtime import run_pipeline_graph

    thread_id = self.request.id or paper_id
    log.info("Running analysis pipeline for paper %s (thread %s)", paper_id, thread_id)
    run_pipeline_graph(paper_id, goal, thread_id=str(thread_id))
    return paper_id
