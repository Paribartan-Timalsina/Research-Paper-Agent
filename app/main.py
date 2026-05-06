import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import routes_agent, routes_chat, routes_papers
from app.config import settings
from app.core.error_handlers import register_exception_handlers
from app.core.middleware import install_middleware
from app.db.base import init_db

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    # init_db is sync (CREATE TABLE via sync engine). Run off the loop.
    await asyncio.to_thread(init_db)
    yield


app = FastAPI(
    title="Research Paper LLM Agent",
    version="0.1.0",
    description="Multi-step agent that analyzes research papers asynchronously.",
    lifespan=lifespan,
)

# Order: middleware -> exception handlers -> routes.
install_middleware(app)
register_exception_handlers(app)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}


app.include_router(routes_papers.router)
app.include_router(routes_agent.router)
app.include_router(routes_chat.router)
