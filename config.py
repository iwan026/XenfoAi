import os
import logging
import MetaTrader5 as mt5
from pathlib import Path

logger = logging.getLogger(__name__)

# KONFIGURASI PATH
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Buat direktori yang diperlukan
REQUIRED_DIR = [MODELS_DIR, LOGS_DIR, DATASETS_DIR]
for directory in REQUIRED_DIR:
    directory.mkdir(parents=True, exist_ok=True)

# Timeframe MT5
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
}

# Konfigurasi Telegram
BOT_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
ADMIN_ID = [1198920849]

# Konfigurasi MT5
MT5_LOGIN = 5035979858
MT5_PASSWORD = "N@Al6mVk"
MT5_SERVER = "MetaQuotes-Demo"

# Resource Management
MAX_MEMORY_USAGE = 6.0  # GB
MAX_CPU_USAGE = 75  # Percent
ENABLE_GPU = False  # Set True jika GPU tersedia

# Cache Management
CACHE_EXPIRY = 3600  # Detik
MAX_CACHE_SIZE = 1.0  # GB
