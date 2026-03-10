import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl

class Settings(BaseSettings):
    target_llm_url: str = Field(default="https://api.openai.com/v1/chat/completions")
    proxy_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    model_type: str = Field(default="bert", description="Must be 'scratch' or 'bert'")
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    admin_secret_key: str = Field(default="dev-secret-key")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()