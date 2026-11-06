from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "INFO"
    database_url: str
    redis_url: str
    api_key: str | None = None
    storage_dir: str = "./data/uploads"
    embedding_dimensions: int = 384
    max_upload_bytes: int = 20 * 1024 * 1024
    chunk_max_tokens: int = 256
    chunk_overlap_tokens: int = 32
    tokenizer_path: str = "./models/bge-small-en-v1.5/tokenizer.json"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = "./models/fastembed"
    search_candidate_limit: int = 20
    rrf_k: int = 60
    llm_base_url: str = ""
    llm_api_key: str | None = None
    llm_model: str = ""
    llm_timeout_seconds: float = 30

    @field_validator("api_key", "llm_api_key", mode="before")
    @classmethod
    def blank_api_key_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def dimensions_match_model(cls, value: int) -> int:
        if value != 384:
            raise ValueError("embedding_dimensions must be 384 for BAAI/bge-small-en-v1.5")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
