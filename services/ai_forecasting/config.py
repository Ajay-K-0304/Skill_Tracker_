"""
Configuration settings for the AI Forecasting & Curriculum Recommendation service.
Loads values from environment variables or .env file with safe production defaults.
"""

from pathlib import Path
from typing import Optional
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Module-level settings for Member 3 service."""

    # Service environment
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Forecasting configurations
    DEFAULT_FORECAST_HORIZON: int = 6  # Supported: 6 or 12 months
    MIN_OBSERVATIONS_FOR_FORECAST: int = 4
    MIN_OBSERVATIONS_FOR_HOLDOUT: int = 12

    # Data Source defaults
    DATA_SOURCE_MODE: str = "synthetic_demo"  # "synthetic_demo", "csv", "live_database"
    CSV_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"

    # Database connection (optional, for Phase 2 dynamic aggregation with Member 1)
    DATABASE_URL: Optional[str] = None
    DB_ECHO: bool = False

    # LLM adapter settings (Phase 7 & Groq cloud / Ollama local integration)
    LOCAL_LLM_PROVIDER: str = "groq"  # "groq", "ollama", "mock"
    GROQ_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "AI_FORECASTING_GROQ_API_KEY"),
        description="Groq Cloud API key for curriculum recommendation synthesis",
    )
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434"  # For Ollama provider
    LOCAL_LLM_MODEL: str = "llama-3.3-70b-versatile"  # Groq production default; e.g. "llama3:8b" for Ollama
    LOCAL_LLM_TIMEOUT: float = 10.0  # seconds timeout for API/inference calls
    ENABLE_LLM_FALLBACK: bool = True

    # Trajectory classification thresholds (Phase 4)
    EMERGING_GROWTH_THRESHOLD: float = 0.20  # >= 20% annual/period growth
    DECLINING_GROWTH_THRESHOLD: float = -0.15  # <= -15% decline
    NICHE_MAX_SHARE_THRESHOLD: float = 0.05  # <= 5% share of total jobs
    MIN_VOLUME_FOR_CORE: int = 20  # min monthly count to be considered Core/Stable (vs Niche)

    # Document upload limits (Phase 5)
    MAX_SYLLABUS_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    model_config = SettingsConfigDict(
        env_prefix="AI_FORECASTING_",
        env_file=(Path(__file__).resolve().parent.parent.parent / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def configure_module_logging() -> None:
    """Configures standard output logging for Member 3 service loggers."""
    import logging
    import sys

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    app_logger = logging.getLogger("services.ai_forecasting")
    app_logger.setLevel(log_level)

    # Attach stream handler to services.ai_forecasting if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in app_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("%(levelname)s:     [%(name)s] %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        app_logger.addHandler(handler)


# Auto-initialize logging on configuration load
configure_module_logging()

