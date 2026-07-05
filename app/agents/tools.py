"""Tools the chat agent can call. Each takes (args, paper, db) and returns a
string that is fed back into the LLM as a tool result."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Paper, PaperFigure


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, str] = field(default_factory=dict)   # arg_name -> human description
    fn: Callable[[dict[str, Any], Paper, Session], str] = lambda *_: ""

    def render(self) -> str:
        arg_lines = "\n".join(f"      - {k}: {v}" for k, v in self.args.items())
        return f"  {self.name}: {self.description}\n    args:\n{arg_lines}" if self.args else (
            f"  {self.name}: {self.description}"
        )


_SECTION_ALIASES = {
    "abstract":     [r"\babstract\b"],
    "introduction": [r"\b1\.?\s*introduction\b", r"\bintroduction\b"],
    "background":   [r"\b(?:related work|background)\b"],
    "methods":      [r"\b(?:methods?|methodology|approach|model|architecture)\b"],
    "results":      [r"\b(?:results?|experiments?|evaluation)\b"],
    "discussion":   [r"\bdiscussion\b"],
    "conclusion":   [r"\b(?:conclusion|conclusions)\b"],
    "references":   [r"\b(?:references|bibliography)\b"],
}

_MAX_SECTION_CHARS = 3500


def _find_section(text: str, name: str) -> str | None:
    """Return the slice of `text` that corresponds to a heading-named section.

    Heuristic: find the first line matching one of the aliases, then read up to
    the next line that looks like another section heading (or ~3500 chars).
    """
    aliases = _SECTION_ALIASES.get(name.lower())
    if not aliases:
        return None

    lines = text.splitlines()
    heading_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if 1 <= len(stripped) <= 80:   # plausible heading length
            if any(re.search(p, stripped) for p in aliases):
                heading_idx = i
                break
    if heading_idx == -1:
        return None

    # Read until next plausible heading or char cap
    body: list[str] = []
    body_chars = 0
    next_heading_patterns = [p for patterns in _SECTION_ALIASES.values() for p in patterns]
    for line in lines[heading_idx + 1:]:
        stripped = line.strip().lower()
        # Stop if we hit another section heading
        if 1 <= len(stripped) <= 80 and any(re.search(p, stripped) for p in next_heading_patterns):
            break
        body.append(line)
        body_chars += len(line) + 1
        if body_chars >= _MAX_SECTION_CHARS:
            break

    return "\n".join(body).strip() or None


def get_section(args: dict[str, Any], paper: Paper, db: Session) -> str:  # noqa: ARG001
    name = (args.get("name") or "").strip()
    if not name:
        return "Error: 'name' arg is required (e.g. 'abstract', 'methods', 'results')."
    section = _find_section(paper.raw_text, name)
    if not section:
        valid = ", ".join(_SECTION_ALIASES.keys())
        return f"Section '{name}' not found. Try one of: {valid}."
    return f"--- Section: {name} ---\n{section}"


_CITE_SPLIT = re.compile(r"\n(?=\s*(?:\[\d+\]|\d+\.\s|\(\d+\)))")


def get_citations(args: dict[str, Any], paper: Paper, db: Session) -> str:  # noqa: ARG001
    refs = _find_section(paper.raw_text, "references")
    if not refs:
        return "No references section found in the paper."
    items = [chunk.strip() for chunk in _CITE_SPLIT.split(refs) if chunk.strip()]
    if not items:
        return refs[:_MAX_SECTION_CHARS]
    items = items[:30]   # cap to avoid huge dumps
    return "References:\n" + "\n".join(f"- {item[:300]}" for item in items)


def describe_figure(args: dict[str, Any], paper: Paper, db: Session) -> str:
    raw_page = args.get("page")
    if raw_page is None:
        return "Error: 'page' arg is required (integer, 1-indexed)."
    try:
        page = int(raw_page)
    except (TypeError, ValueError):
        return "Error: 'page' must be an integer page number."

    figs = db.scalars(
        select(PaperFigure)
        .where(PaperFigure.paper_id == paper.id, PaperFigure.page == page)
    ).all()
    if not figs:
        # Fall back: any figure on any page
        any_figs = db.scalars(
            select(PaperFigure).where(PaperFigure.paper_id == paper.id)
        ).all()
        if not any_figs:
            return "No figures were extracted from this paper."
        avail_pages = sorted({f.page for f in any_figs})
        return f"No figure on page {page}. Available pages with figures: {avail_pages}."
    return "\n".join(f"[Figure on page {f.page}]: {f.description}" for f in figs)


TOOLS: dict[str, Tool] = {
    "get_section": Tool(
        name="get_section",
        description="Fetch a named section of the paper (heuristic match).",
        args={"name": 'one of: abstract, introduction, background, methods, '
                       'results, discussion, conclusion, references'},
        fn=get_section,
    ),
    "get_citations": Tool(
        name="get_citations",
        description="List the bibliography / cited references from the paper.",
        args={},
        fn=get_citations,
    ),
    "describe_figure": Tool(
        name="describe_figure",
        description="Get a textual description of figures on a given page.",
        args={"page": "integer page number (1-indexed)"},
        fn=describe_figure,
    ),
}


def render_tool_list() -> str:
    return "\n".join(t.render() for t in TOOLS.values())


def run_tool(name: str, args: dict[str, Any], paper: Paper, db: Session) -> str:
    tool = TOOLS.get(name)
    if not tool:
        return f"Error: unknown tool '{name}'. Available: {list(TOOLS)}."
    try:
        return tool.fn(args, paper, db)
    except Exception as e:
        return f"Error running '{name}': {e}"
