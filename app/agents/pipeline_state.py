"""Shared state for the analysis-pipeline StateGraph.

Replaces the Redis context hash the old Celery chain used to pass results
between steps — the graph accumulates everything here instead.
"""
from __future__ import annotations

from typing import TypedDict


class PipelineState(TypedDict, total=False):
    paper_id: str
    goal: str
    paper_text: str
    plan: list[str]
    # Accumulated structured outputs, keyed by task name.
    summarize: dict
    contributions: dict
    methodology: dict
    limitations: dict
    future_work: dict
