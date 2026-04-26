from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PaperOut(BaseModel):
    id: UUID
    title: str
    filename: str
    created_at: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    paper_id: UUID
    title: str
    char_count: int
