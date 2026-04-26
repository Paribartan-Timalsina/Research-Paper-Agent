from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class RunAgentRequest(BaseModel):
    paper_id: UUID
    goal: str = "Help me deeply understand this paper"


class RunAgentResponse(BaseModel):
    paper_id: UUID
    plan: list[str]
    chain_id: str | None


class TaskOut(BaseModel):
    task_name: str
    order_index: int
    status: str
    result: dict | None
    error: str | None
    updated_at: datetime

    class Config:
        from_attributes = True


class InsightOut(BaseModel):
    summary: str | None = None
    contributions: list | None = None
    methodology: str | None = None
    limitations: list | None = None
    future_work: list | None = None

    class Config:
        from_attributes = True


class ResultsResponse(BaseModel):
    paper_id: UUID
    title: str
    tasks: list[TaskOut]
    insight: InsightOut | None


class AskQuestionRequest(BaseModel):
    paper_id: UUID
    question: str


class AskQuestionResponse(BaseModel):
    paper_id: UUID
    question: str
    answer: Any
