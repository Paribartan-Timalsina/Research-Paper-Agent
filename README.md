# Research Paper LLM Agent

A multi-step, context-aware LLM agent that ingests a research paper (PDF),
plans analysis tasks with an LLM, and executes them asynchronously on Celery
workers with Redis-backed context passing and PostgreSQL persistence.

```
Client ──► FastAPI ──► Planner (LLM) ──► Celery chain
                                 │           │
                                 ▼           ▼
                              Redis      Postgres
                           (context)    (papers/tasks/insights)
```

## Quickstart

```bash
cp .env.example .env
# (optional) set LLM_PROVIDER=gemini and GEMINI_API_KEY=... to use real LLM
docker compose up --build
```

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## End-to-end flow

```bash
# 1. Upload a paper
curl -F "file=@paper.pdf" http://localhost:8000/upload-paper
# → { "paper_id": "...", "title": "...", "char_count": 48210 }

# 2. Run the agent (planner + async execution)
curl -X POST http://localhost:8000/run-agent \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "<id>", "goal": "Help me deeply understand this paper"}'
# → { "plan": ["summarize","contributions",...], "chain_id": "..." }

# 3. Poll results — shows PENDING/RUNNING/COMPLETED per task + aggregated insight
curl http://localhost:8000/paper/<id>/results

# 4. Ask follow-up questions grounded in the paper
curl -X POST http://localhost:8000/ask-question \
  -H "Content-Type: application/json" \
  -d '{"paper_id": "<id>", "question": "What datasets did they use?"}'
```

## Module map

| Module | Responsibility |
|---|---|
| `app/api/` | FastAPI routes. Thin — delegate to services/agents. |
| `app/agents/` | `planner.py` (LLM → ordered task list), `prompts.py`, `executor.py` (build Celery chain). |
| `app/tasks/` | Celery app + `paper_tasks.py` (summarize, contributions, methodology, limitations, future_work). |
| `app/services/` | `pdf_service` (pdfplumber extract), `llm_service` (Gemini + mock), `context_service` (Redis hash per paper). |
| `app/db/` | SQLAlchemy engine, session, `init_db()`. |
| `app/models/` | `Paper`, `AgentTask`, `Insight`. |
| `app/schemas/` | Pydantic request/response models. |

## How context flows between tasks

Each Celery task reads the full accumulated context from Redis
(`agent:ctx:{paper_id}`), builds its prompt using prior step outputs, writes
its own structured result back to Redis **and** Postgres, then returns
`paper_id` so the next task in the chain can pick up.

That means `methodology` sees the output of `summarize` + `contributions`,
`limitations` sees the methodology, and `future_work` sees limitations —
tasks are **not** independent.

## Swapping the LLM

Set `LLM_PROVIDER=gemini` + `GEMINI_API_KEY`. `MockLLM` stays as the default
so the whole pipeline runs offline for dev/tests. Add a new provider by
subclassing `LLMClient` in `app/services/llm_service.py` and registering it in
`_build_llm()`.

## Scaling knobs

- `docker compose up -d --scale worker=4` — more parallel Celery workers.
- Tasks inside the plan run **sequentially** (chain) so later tasks see
  earlier results. If you add independent tasks (e.g. keyword extraction),
  wrap them in a Celery `group` / `chord` inside `executor.py`.
- `worker_prefetch_multiplier=1` + `task_acks_late=True` for long LLM calls.

## Next steps (not yet implemented)

- Replace truncated-paper Q&A with real RAG (pgvector + chunk embeddings).
- Compare multiple papers.
- Per-task latency metrics (Prometheus middleware).
- Minimal UI dashboard.
