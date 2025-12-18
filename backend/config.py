from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys (Gemini / Google GenAI) ---
    # Primär: GOOGLE_API_KEY (rekommenderas av nya google-genai SDK)
    # Fallback: GEMINI_API_KEY (för bakåtkompatibilitet)
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    @property
    def api_key(self) -> str:
        """Returnera en fungerande API-nyckel oavsett om användaren satte GOOGLE_API_KEY eller GEMINI_API_KEY."""
        key = self.google_api_key or self.gemini_api_key
        if not key:
            raise ValueError(
                "Ingen API-nyckel hittades. Sätt GOOGLE_API_KEY (rekommenderat) eller GEMINI_API_KEY i din .env eller miljövariabler."
            )
        return key

    # --- LanceDB ---
    lancedb_dir: Path = ROOT_DIR / "db" / "lancedb"
    lancedb_table: str = "segments"

    # --- Models ---
    embed_model: str = "models/text-embedding-004"
    chat_model: str = "gemini-1.5-flash"
    # chat_model: str = "gemini-2.0-flash"


settings = Settings()
