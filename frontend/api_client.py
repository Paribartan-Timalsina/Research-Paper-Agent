"""Thin httpx wrapper around the FastAPI backend.

All HTTP plumbing lives here. Streamlit code calls these functions and gets
back plain dicts/lists. If we ever swap Streamlit for React, this file is
what stays.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class APIError(Exception):
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"API {status_code}: {body}")


def _client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND_URL, timeout=TIMEOUT)


def _unwrap(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise APIError(resp.status_code, body)
    return resp.json()


# ---------- Health ----------

def health() -> dict:
    with _client() as c:
        return _unwrap(c.get("/health"))


# ---------- Papers ----------

def upload_paper(file_name: str, file_bytes: bytes, mime: str = "application/pdf") -> dict:
    """POST /upload-paper. Returns {paper_id, title, char_count}."""
    with _client() as c:
        files = {"file": (file_name, file_bytes, mime)}
        return _unwrap(c.post("/upload-paper", files=files))


def get_results(paper_id: str) -> dict:
    """GET /paper/{id}/results. Returns {paper_id, title, tasks, insight}."""
    with _client() as c:
        return _unwrap(c.get(f"/paper/{paper_id}/results"))


# ---------- Agent ----------

def run_agent(paper_id: str, goal: str = "") -> dict:
    """POST /run-agent. Returns {paper_id, plan, chain_id}."""
    with _client() as c:
        payload = {"paper_id": paper_id}
        if goal:
            payload["goal"] = goal
        return _unwrap(c.post("/run-agent", json=payload))


def ask_question(paper_id: str, question: str) -> dict:
    """POST /ask-question. Returns {paper_id, question, answer}."""
    with _client() as c:
        return _unwrap(c.post("/ask-question", json={
            "paper_id": paper_id,
            "question": question,
        }))


# ---------- Chat ----------

def start_conversation(paper_id: str, title: str | None = None) -> dict:
    """POST /conversations. Returns {id, paper_id, title, created_at}."""
    with _client() as c:
        payload: dict[str, Any] = {"paper_id": paper_id}
        if title:
            payload["title"] = title
        return _unwrap(c.post("/conversations", json=payload))


def send_message(conversation_id: str, content: str) -> dict:
    """POST /conversations/{id}/messages. Returns {answer, trace, iterations}."""
    with _client() as c:
        return _unwrap(c.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": content},
        ))


def get_conversation(conversation_id: str) -> dict:
    """GET /conversations/{id}. Returns {id, paper_id, title, messages: [...]}."""
    with _client() as c:
        return _unwrap(c.get(f"/conversations/{conversation_id}"))
