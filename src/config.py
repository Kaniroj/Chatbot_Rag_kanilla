from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATA_PATH: Path = Path("data")

settings = Settings()
