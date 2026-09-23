"""CES-76 · the one module that reads the environment.

Every `os.getenv` / `os.environ` read in the package resolves here, through the cached
`get_settings()`. Fields are typed as the environment delivers them (`str`), and callers keep
whatever coercion they already did -- this module centralizes the reads without changing what
any of them mean.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,  # IAX_RUNS_DIR and iax_runs_dir both bind `runs_dir`
        extra="ignore",
    )

    # No env_prefix: MLFLOW_TRACKING_URI and RAY_ADDRESS are third-party contracts that cannot
    # be renamed, so every field names its variable outright.
    runs_dir: str = Field(default="outputs/experiments/runs", validation_alias="IAX_RUNS_DIR")
    artifacts_dir: str | None = Field(default=None, validation_alias="IAX_ARTIFACTS_DIR")
    clusters_config: str | None = Field(default=None, validation_alias="IAX_CLUSTERS")
    notify_webhook: str | None = Field(default=None, validation_alias="IAX_NOTIFY_WEBHOOK")
    notify_command: str | None = Field(default=None, validation_alias="IAX_NOTIFY_COMMAND")
    mlflow_tracking_uri: str = Field(default="", validation_alias="MLFLOW_TRACKING_URI")
    ray_address: str | None = Field(default=None, validation_alias="RAY_ADDRESS")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, constructed (and validated) once."""
    return Settings()
