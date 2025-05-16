import os
import logging
from pathlib import Path

# Path Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"

# Create required directories
REQUIRED_DIRS = [DATASETS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]
for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# Timeframe Configuration
TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}

# Trading Configuration
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

# MetaTrader5 Configuration
MT5_LOGIN = 5035979858
MT5_PASSWORD = "N@Al6mVk"
MT5_SERVER = "MetaQuotes-Demo"

# Telegram Configuration
BOT_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
ADMIN_IDS = [1198920849]


# Model Configuration
class ModelConfig:
    # Training
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    PATIENCE = 7
    L2_REG = 0.001
    DROPOUT_RATE = 0.3

    # Model Architecture
    CNN_FILTERS = [16, 32]
    CNN_KERNEL_SIZES = [(3, 1), (3, 1)]
    CNN_POOL_SIZES = [(2, 1), (2, 1)]
    LSTM_UNITS = 32

    # Feature Engineering
    TECHNICAL_INDICATORS = [
        "rsi_14",
        "macd",
        "bb_upper",
        "bb_lower",
        "ema_9",
        "ema_21",
        "atr_14",
        "adx_14",
    ]

    PRICE_FEATURES = ["open", "high", "low", "close", "volume"]

    # Target threshold (0.1% change)
    TARGET_THRESHOLD = 0.002
