"""Environment-backed application settings."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY", "local-cifar-vision-development-key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE", 5 * 1024 * 1024))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "cifar10_model.keras"))
    LOG_PATH = Path(os.getenv("LOG_PATH", BASE_DIR / "logs" / "app.log"))
    JSON_SORT_KEYS = False