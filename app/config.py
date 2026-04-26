from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    postgres_user: str = "agent"
    postgres_password: str = "agent"
    postgres_db: str = "research"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # Redis / Celery
    redis_url: str = "redis://redis:6379/0"

    # LLM
    llm_provider: Literal["mock", "gemini"] = "mock"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # App
    app_env: str = "dev"
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
