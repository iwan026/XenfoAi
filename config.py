import os
import logging
import MetaTrader5 as mt5
from pathlib import Path

logger = logging.getLogger(__name__)

# KONFIGURASI PATH
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

REQUIRED_DIR = [MODELS_DIR, LOG_DIR]
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

# Konfigurasi Model
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
EPOCHS = 50
TRAIN_TEST_SPLIT = 0.2

# Konfigurasi Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN", "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI")
ADMIN_ID = [1198920849]

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "rsi",
    "MACD_12_26_9",
    "MACDs_12_26_9",
    "BBL_5_2.0",
    "BBM_5_2.0",
    "BBU_5_2.0",
    "sma_20",
    "sma_50",
    "ema_9",
    "ema_21",
]

# Konfigurasi MT5
MT5_LOGIN = 5035979858
MT5_PASSWORD = "N@Al6mVk"
MT5_SERVER = "MetaQuotes-Demo"
