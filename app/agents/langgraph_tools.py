"""LangChain tool adapters for the chat agent.

The underlying tool bodies live in app.agents.tools and still take
(args, paper, db). Here we wrap them as @tool callables bound to a paper_id;
each opens its own short-lived session so the LLM never has to supply a DB
handle or Paper object.
"""
from __future__ import annotations

from uuid import UUID

from langchain_core.tools import tool

from app.agents import tools as legacy
from app.db.base import session_scope
from app.models import Paper


def build_tools(paper_id: UUID):
    """Return the chat tools bound to a single paper."""

    @tool
    def get_section(name: str) -> str:
        """Fetch a named section of the paper. `name` is one of: abstract,
        introduction, background, methods, results, discussion, conclusion,
        references."""
        with session_scope() as db:
            paper = db.get(Paper, paper_id)
            return legacy.get_section({"name": name}, paper, db)

    @tool
    def get_citations() -> str:
        """List the bibliography / cited references from the paper."""
        with session_scope() as db:
            paper = db.get(Paper, paper_id)
            return legacy.get_citations({}, paper, db)

    @tool
    def describe_figure(page: int) -> str:
        """Get a textual description of figures on a given page (1-indexed)."""
        with session_scope() as db:
            paper = db.get(Paper, paper_id)
            return legacy.describe_figure({"page": page}, paper, db)

    return [get_section, get_citations, describe_figure]
