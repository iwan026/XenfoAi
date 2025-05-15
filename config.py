import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# KONFIGURASI PATH
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "XenfoAi.log"

REQUIRED_DIR = [MODEL_DIR, LOG_DIR]
for directory in REQUIRED_DIR:
    directory.mkdir(parents=True, exist_ok=True)

# Konfigurasi Sistem
SEQUENCE_LENGTH = 60
INPUT_FEATURES = 15
LEARNING_RATE = 0.001
TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}
FEATURE_COLUMNS = {
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
}

# Konfigurasi Telegram
BOT_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
ADMIN_ID = [1198920849]

# Konfigurasi MT5
MT5_LOGIN = 5035979858
MT5_PASSWORD = "N@Al6mVk"
MT5_SERVER = "MetaQuotes-Demo"
