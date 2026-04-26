"""Agent planner: asks the LLM to produce an ordered list of task names."""
from __future__ import annotations

import logging

from app.agents.prompts import ALLOWED_TASKS, PLANNER_SYSTEM
from app.core.exceptions import LLMError
from app.services.llm_service import llm

log = logging.getLogger(__name__)

_SCHEMA = '[{"task": "summarize"}, {"task": "contributions"}]'


def plan_tasks(goal: str, paper_text: str) -> list[str]:
    """Return an ordered list of task names, filtered to ALLOWED_TASKS."""
    prompt = PLANNER_SYSTEM.format(
        allowed=", ".join(ALLOWED_TASKS),
        goal=goal or "Help me deeply understand this paper",
        excerpt=paper_text[:1500],
    )
    try:
        raw = llm.generate_json(prompt, schema_hint=_SCHEMA)
    except LLMError as e:
        # Graceful degradation: fall back to the full default plan.
        log.warning("Planner LLM call failed, using default plan: %s", e)
        return list(ALLOWED_TASKS)
    except Exception as e:  # unexpected — surface so callers can handle
        log.exception("Unexpected planner failure")
        raise LLMError("Planner produced an unexpected error") from e

    plan: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        name = (item.get("task") if isinstance(item, dict) else str(item)).strip()
        if name in ALLOWED_TASKS and name not in seen:
            plan.append(name)
            seen.add(name)

    return plan or list(ALLOWED_TASKS)
