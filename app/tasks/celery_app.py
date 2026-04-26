from celery import Celery

from app.config import settings

celery_app = Celery(
    "research_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.paper_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_retry_delay=10,
    task_default_max_retries=2,
)
