"""Entry point that runs the analysis-pipeline graph once, inside a Celery task."""
from __future__ import annotations

from uuid import UUID

from app.agents.pipeline_db import load_paper_text
from app.agents.pipeline_graph import build_pipeline_graph
from app.db.checkpointer import get_checkpointer


def run_pipeline_graph(paper_id: str, goal: str, thread_id: str) -> None:
    graph = build_pipeline_graph(get_checkpointer())
    paper_text = load_paper_text(UUID(paper_id))
    graph.invoke(
        {"paper_id": paper_id, "goal": goal, "paper_text": paper_text},
        config={"configurable": {"thread_id": thread_id}},
    )
