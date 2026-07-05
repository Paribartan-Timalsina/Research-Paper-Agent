"""Structured-output schemas for the analysis pipeline.

Field names match the Insight columns and the shapes the frontend expects, so
the existing /results UI keeps working unchanged.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PlanSchema(BaseModel):
    tasks: list[str] = Field(description="Ordered subset of allowed analysis task names")


class Summary(BaseModel):
    summary: str


class Contributions(BaseModel):
    contributions: list[str]


class Methodology(BaseModel):
    methodology: str


class Limitations(BaseModel):
    limitations: list[str]


class FutureWork(BaseModel):
    future_work: list[str]
