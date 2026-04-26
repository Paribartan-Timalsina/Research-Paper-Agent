"""Reusable prompt templates. Keep them small and explicit."""
from __future__ import annotations

# The set of tasks the agent is allowed to plan. Celery task names must match.
ALLOWED_TASKS: list[str] = [
    "summarize",
    "contributions",
    "methodology",
    "limitations",
    "future_work",
]


PLANNER_SYSTEM = """You are a research-paper analysis planner.

Given the user's goal and a short excerpt of the paper, decide an ordered
subset of analysis tasks to run. Each task must come from this allowed list:
{allowed}

Order them so that later tasks benefit from earlier ones (e.g. summarize
before limitations). Do NOT invent new tasks.

User goal: {goal}

Paper excerpt (first ~1500 chars):
\"\"\"
{excerpt}
\"\"\"
"""


SUMMARIZE_PROMPT = """Summarize this research paper in 4-6 sentences.
Focus on the problem, approach, and main result.

PAPER:
\"\"\"
{paper}
\"\"\"
"""


CONTRIBUTIONS_PROMPT = """List the paper's main contributions as concise bullet points (3-6 items).

Prior summary (for context): {summary}

PAPER:
\"\"\"
{paper}
\"\"\"
"""


METHODOLOGY_PROMPT = """Explain the paper's methodology clearly in 1-2 paragraphs.
Assume the reader is technical but not a specialist.

Prior summary: {summary}
Prior contributions: {contributions}

PAPER:
\"\"\"
{paper}
\"\"\"
"""


LIMITATIONS_PROMPT = """Identify the paper's limitations and weaknesses.
Include both stated and implied limitations. Return 3-6 bullets.

Prior summary: {summary}
Prior methodology: {methodology}

PAPER:
\"\"\"
{paper}
\"\"\"
"""


FUTURE_WORK_PROMPT = """Suggest 3-5 concrete future-work / improvement ideas for this paper.
Ground each idea in something stated or implied in the paper.

Prior summary: {summary}
Prior limitations: {limitations}

PAPER:
\"\"\"
{paper}
\"\"\"
"""


QA_PROMPT = """Answer the user's question using only the paper context below.
If the answer is not in the paper, say so.

QUESTION: {question}

PAPER CONTEXT:
\"\"\"
{context}
\"\"\"
"""
