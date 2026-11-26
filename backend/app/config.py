"""Backend configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Application settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="FAKECHECK_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://127.0.0.1:5500,http://localhost:5500"
    models_dir: Path = PROJECT_ROOT / "models"
    max_image_mb: int = 15
    max_video_mb: int = 100

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
