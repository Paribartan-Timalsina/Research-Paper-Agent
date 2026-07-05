"""Analysis-pipeline StateGraph.

planner → summarize → contributions → methodology → limitations → future_work → END

The planner picks the subset of tasks to run; each analysis node early-returns
if its task wasn't planned. Dependency order is enforced structurally by the
linear edges. Every node keeps the AgentTask rows and the Insight row current
so the existing /results polling UI is unaffected.
"""
from __future__ import annotations

from uuid import UUID

from langgraph.graph import END, StateGraph

from app.agents import prompts
from app.agents.pipeline_db import mark, upsert_insight
from app.agents.pipeline_state import PipelineState
from app.agents.schemas import (
    Contributions,
    FutureWork,
    Limitations,
    Methodology,
    PlanSchema,
    Summary,
)
from app.models import TaskStatus
from app.services.chat_model import get_chat_model, structured_invoke

ORDER = ["summarize", "contributions", "methodology", "limitations", "future_work"]

_PAPER_CAP = 12000


def _summarize_prompt(s: PipelineState) -> str:
    return prompts.SUMMARIZE_PROMPT.format(paper=s["paper_text"][:_PAPER_CAP])


def _contributions_prompt(s: PipelineState) -> str:
    summary = (s.get("summarize") or {}).get("summary", "")
    return prompts.CONTRIBUTIONS_PROMPT.format(paper=s["paper_text"][:_PAPER_CAP], summary=summary)


def _methodology_prompt(s: PipelineState) -> str:
    summary = (s.get("summarize") or {}).get("summary", "")
    contribs = (s.get("contributions") or {}).get("contributions", [])
    return prompts.METHODOLOGY_PROMPT.format(
        paper=s["paper_text"][:_PAPER_CAP], summary=summary, contributions=contribs
    )


def _limitations_prompt(s: PipelineState) -> str:
    summary = (s.get("summarize") or {}).get("summary", "")
    method = (s.get("methodology") or {}).get("methodology", "")
    return prompts.LIMITATIONS_PROMPT.format(
        paper=s["paper_text"][:_PAPER_CAP], summary=summary, methodology=method
    )


def _future_work_prompt(s: PipelineState) -> str:
    summary = (s.get("summarize") or {}).get("summary", "")
    lims = (s.get("limitations") or {}).get("limitations", [])
    return prompts.FUTURE_WORK_PROMPT.format(
        paper=s["paper_text"][:_PAPER_CAP], summary=summary, limitations=lims
    )


# task_name -> (schema, prompt builder, insight field)
_NODES = {
    "summarize":     (Summary,       _summarize_prompt,     "summary"),
    "contributions": (Contributions, _contributions_prompt, "contributions"),
    "methodology":   (Methodology,   _methodology_prompt,   "methodology"),
    "limitations":   (Limitations,   _limitations_prompt,   "limitations"),
    "future_work":   (FutureWork,    _future_work_prompt,   "future_work"),
}


def _planner_node(state: PipelineState) -> dict:
    prompt = prompts.PLANNER_SYSTEM.format(
        allowed=", ".join(prompts.ALLOWED_TASKS),
        goal=state.get("goal") or "Help me deeply understand this paper",
        excerpt=state["paper_text"][:1500],
    )
    chosen: set[str] = set()
    try:
        result = structured_invoke(get_chat_model(), PlanSchema, prompt)
        chosen = {t for t in result.tasks if t in prompts.ALLOWED_TASKS}
    except Exception:
        chosen = set(ORDER)
    plan = [t for t in ORDER if t in chosen] or list(ORDER)

    # Tasks the planner skipped are marked COMPLETED (no result) so the
    # frontend progress bar still reaches 100%.
    pid = UUID(state["paper_id"])
    for name in ORDER:
        if name not in plan:
            mark(pid, name, TaskStatus.COMPLETED)
    return {"plan": plan}


def _make_analysis_node(name: str):
    schema, prompt_fn, insight_field = _NODES[name]

    def node(state: PipelineState) -> dict:
        if name not in state.get("plan", []):
            return {}
        pid = UUID(state["paper_id"])
        mark(pid, name, TaskStatus.RUNNING)
        try:
            result = structured_invoke(get_chat_model(), schema, prompt_fn(state))
            data = result.model_dump()
            mark(pid, name, TaskStatus.COMPLETED, result=data)
            upsert_insight(pid, **{insight_field: data.get(insight_field)})
            return {name: data}
        except Exception as e:
            mark(pid, name, TaskStatus.FAILED, error=str(e)[:500])
            raise

    return node


def _node_id(name: str) -> str:
    # Node ids must differ from state keys (which are the task names).
    return f"do_{name}"


def build_pipeline_graph(checkpointer):
    g = StateGraph(PipelineState)
    g.add_node("planner", _planner_node)
    for name in ORDER:
        g.add_node(_node_id(name), _make_analysis_node(name))
    g.set_entry_point("planner")
    g.add_edge("planner", _node_id(ORDER[0]))
    for a, b in zip(ORDER, ORDER[1:]):
        g.add_edge(_node_id(a), _node_id(b))
    g.add_edge(_node_id(ORDER[-1]), END)
    return g.compile(checkpointer=checkpointer)
