"""Async SQLAlchemy engine + session.

Used by FastAPI route handlers so DB I/O doesn't block the event loop.
Celery workers continue to use the sync stack in `app.db.base`.

Both share the same `Base` metadata, so models are defined once.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings

async_engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    expire_on_commit=False,
)


async def get_db_async() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an AsyncSession scoped to the request."""
    async with AsyncSessionLocal() as db:
        yield db
