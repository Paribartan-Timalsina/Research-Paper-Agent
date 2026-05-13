"""RAG (Retrieval-Augmented Generation) service for paper chunks.

Provides:
  - index_paper: chunk text, embed, store in DB with FTS index
  - retrieve: hybrid search (semantic + keyword) via Reciprocal Rank Fusion
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Paper, PaperChunk
from app.services.llm_service import LLMClient
from app.services.pdf_service import chunk_text


async def index_paper(
    paper_id: UUID,
    db: AsyncSession,
    llm: LLMClient,
) -> None:
    """Chunk paper text, embed chunks, store in DB with FTS index.

    Runs synchronously (CPU-bound embedding) in a thread pool.
    Called from upload_paper route after inserting the Paper row.
    """
    paper = await db.get(Paper, paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")

    chunks = chunk_text(paper.raw_text, max_chars=4000, overlap=200)
    embeddings = await asyncio.to_thread(llm.embed, chunks)

    for i, (chunk_text_val, embedding) in enumerate(zip(chunks, embeddings)):
        chunk = PaperChunk(
            paper_id=paper_id,
            chunk_index=i,
            content=chunk_text_val,
            embedding=embedding,
        )
        db.add(chunk)

    await db.commit()


async def retrieve(
    paper_id: UUID,
    query: str,
    db: AsyncSession,
    llm: LLMClient,
    top_k: int = 5,
) -> list[PaperChunk]:
    """Hybrid search: semantic (pgvector) + keyword (FTS) via RRF.

    Returns top_k chunks ranked by Reciprocal Rank Fusion.
    """
    query_embedding = (await asyncio.to_thread(llm.embed, [query]))[0]

    stmt_semantic = (
        select(PaperChunk.id, PaperChunk.content)
        .where(PaperChunk.paper_id == paper_id)
        .order_by(PaperChunk.embedding.cosine_distance(query_embedding))  # type: ignore
        .limit(top_k * 2)
    )
    semantic_results = (await db.execute(stmt_semantic)).all()
    semantic_map = {row[0]: (i + 1) for i, row in enumerate(semantic_results)}

    tsvector = func.to_tsvector("english", PaperChunk.content)
    tsquery = func.plainto_tsquery("english", query)
    stmt_keyword = (
        select(PaperChunk.id, PaperChunk.content, func.ts_rank(tsvector, tsquery).label("rank"))
        .where(and_(PaperChunk.paper_id == paper_id, tsvector.match(tsquery)))
        .order_by(text("rank DESC"))
        .limit(top_k * 2)
    )
    keyword_results = (await db.execute(stmt_keyword)).all()
    keyword_map = {row[0]: (i + 1) for i, row in enumerate(keyword_results)}

    rrf_scores: dict[UUID, float] = {}
    for chunk_id in set(semantic_map.keys()) | set(keyword_map.keys()):
        semantic_rank = semantic_map.get(chunk_id, top_k * 2 + 1)
        keyword_rank = keyword_map.get(chunk_id, top_k * 2 + 1)
        rrf_scores[chunk_id] = 1.0 / (semantic_rank + 60) + 1.0 / (keyword_rank + 60)

    top_chunk_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    stmt_fetch = (
        select(PaperChunk)
        .where(PaperChunk.id.in_(top_chunk_ids))
        .order_by(PaperChunk.chunk_index)
    )
    chunks = (await db.scalars(stmt_fetch)).all()
    return list(chunks)
