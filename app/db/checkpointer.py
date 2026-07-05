"""Shared LangGraph Postgres checkpointer.

Both agentic entry points (the Celery pipeline task and chat_step) are sync, so
we use the sync PostgresSaver over a psycopg3 connection pool. One saver per
process; thread_ids isolate the pipeline runs and chat conversations from each
other.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from app.config import settings


def _conn_str() -> str:
    # SQLAlchemy uses postgresql+psycopg://...; psycopg wants postgresql://...
    return settings.database_url.replace("postgresql+psycopg://", "postgresql://")


@lru_cache
def _pool() -> ConnectionPool:
    return ConnectionPool(
        conninfo=_conn_str(),
        max_size=10,
        # PostgresSaver requires autocommit and disabled prepared statements.
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=True,
    )


@lru_cache
def get_checkpointer() -> PostgresSaver:
    saver = PostgresSaver(_pool())
    saver.setup()  # idempotent: CREATE TABLE IF NOT EXISTS for checkpoint tables
    return saver
