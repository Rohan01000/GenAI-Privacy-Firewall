from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    target_llm_url: str = "https://api.openai.com/v1/chat/completions"
    proxy_port: int = 8000
    log_level: str = "INFO"
    model_type: str = "scratch"
    confidence_threshold: float = 0.85
    admin_secret_key: str = "changeme"

    model_config = SettingsConfigDict(env_file=".env")

# Module-level singleton to be imported across the application
settings = Settings()