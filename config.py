import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Copyright:
    """Konfigurasi hak cipta"""

    PROJECT_NAME = "XenfoAi"
    VERSION = "Beta Test"
    AUTHOR = "Xentrovt"
    CREATED_AT = "2025-05-09 17:45:34"


class KonfigurasiPath:
    """Konfigurasi untuk mengelola path direktori"""

    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = BASE_DIR / "saved_models"
    FOREX_MODEL_DIR = MODEL_DIR / "forex"
    LOG_DIR = BASE_DIR / "logs"

    REQUIRED_DIR = [MODEL_DIR, LOG_DIR]
    for directory in REQUIRED_DIR:
        directory.mkdir(parents=True, exist_ok=True)


class KonfigurasiTelegram:
    """Konfigurasi untuk bot telegram"""

    BOT_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
    ADMIN_ID = [1198920849]


class KonfigurasiMT5:
    LOGIN = 5035979858
    PASSWORD = "N@Al6mVk"
    SERVER = "MetaQuotes-Demo"
