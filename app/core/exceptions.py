"""Custom exception hierarchy.

All domain errors derive from AppException. Each carries:
  - code          stable machine-readable string
  - message       human-readable
  - status_code   HTTP status the handler will map to
  - details       optional dict for extra context (never secrets)

Services / agents raise these; FastAPI's exception handlers translate them
to consistent JSON responses. Celery tasks log them and keep AgentTask.error.
"""
from __future__ import annotations

from typing import Any


class AppException(Exception):
    code: str = "app_error"
    message: str = "Application error"
    status_code: int = 500

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


# ---------- 4xx client errors ----------

class NotFoundError(AppException):
    code = "not_found"
    message = "Resource not found"
    status_code = 404


class PaperNotFoundError(NotFoundError):
    code = "paper_not_found"
    message = "Paper not found"


class InvalidFileError(AppException):
    code = "invalid_file"
    message = "Invalid file"
    status_code = 400


class PDFParseError(AppException):
    code = "pdf_parse_error"
    message = "Failed to parse PDF"
    status_code = 400


class ValidationError(AppException):
    code = "validation_error"
    message = "Validation failed"
    status_code = 422


# ---------- 5xx / agent errors ----------

class LLMError(AppException):
    code = "llm_error"
    message = "LLM call failed"
    status_code = 502


class PlanningError(AppException):
    code = "planning_error"
    message = "Agent planner failed"
    status_code = 502


class TaskExecutionError(AppException):
    code = "task_execution_error"
    message = "Agent task execution failed"
    status_code = 500


class ContextStoreError(AppException):
    code = "context_store_error"
    message = "Context store (Redis) unavailable"
    status_code = 503
