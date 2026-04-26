"""Redis-backed short-term context store.

Keyed per paper. Each completed agent step writes its structured output here so
downstream tasks can read prior results without re-hitting Postgres.

    key: agent:ctx:{paper_id}
    type: HASH (field -> json-encoded value)
    TTL : 24h (refreshed on every write)
"""
from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import redis
from redis.exceptions import RedisError

from app.config import settings
from app.core.exceptions import ContextStoreError

_TTL_SECONDS = 60 * 60 * 24  # 24h

_redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _key(paper_id: UUID | str) -> str:
    return f"agent:ctx:{paper_id}"


def set_field(paper_id: UUID | str, field: str, value: Any) -> None:
    k = _key(paper_id)
    try:
        _redis.hset(k, field, json.dumps(value))
        _redis.expire(k, _TTL_SECONDS)
    except RedisError as e:
        raise ContextStoreError(f"Redis HSET failed: {e}") from e


def get_field(paper_id: UUID | str, field: str) -> Any | None:
    try:
        raw = _redis.hget(_key(paper_id), field)
    except RedisError as e:
        raise ContextStoreError(f"Redis HGET failed: {e}") from e
    return json.loads(raw) if raw else None


def get_all(paper_id: UUID | str) -> dict[str, Any]:
    try:
        raw = _redis.hgetall(_key(paper_id))
    except RedisError as e:
        raise ContextStoreError(f"Redis HGETALL failed: {e}") from e
    return {k: json.loads(v) for k, v in raw.items()}


def clear(paper_id: UUID | str) -> None:
    try:
        _redis.delete(_key(paper_id))
    except RedisError as e:
        raise ContextStoreError(f"Redis DEL failed: {e}") from e
