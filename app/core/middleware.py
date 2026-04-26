"""HTTP middleware: request-id, access logging, timing.

Ordering matters. In FastAPI, middlewares execute in REVERSE of the order they
are added. We register them so the outermost (first to see the request) is
RequestIDMiddleware, then AccessLogMiddleware inside it — so every log line
can reference the request id.
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

log = logging.getLogger("app.access")

_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a stable request id to request.state and response headers.

    Honors an incoming `X-Request-ID` so clients / tracing systems can thread
    ids end-to-end.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get(_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[_HEADER] = rid
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Log one line per request with method, path, status, duration, rid."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            status = response.status_code if response else 500
            rid = getattr(request.state, "request_id", "-")
            log.info(
                '%s %s %d %.1fms rid=%s',
                request.method, request.url.path, status, duration_ms, rid,
            )
            if response is not None:
                response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"


def install_middleware(app) -> None:
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)
