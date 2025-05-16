import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = BASE_DIR / "datasets" / "forex"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"

# Create required directories
REQUIRED_DIRS = [DATASETS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]
for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

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
    SEQUENCE_LENGTH = 60  # 60 hours (2.5 days)
    PREDICTION_WINDOW = 24  # Predict next 24 hours (1 day)
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    PATIENCE = 10
    L2_REG = 0.001
    DROPOUT_RATE = 0.4

    # Model Architecture
    CNN_FILTERS = [32, 64]
    CNN_KERNEL_SIZES = [(3, 1), (3, 1)]
    CNN_POOL_SIZES = [(2, 1), (2, 1)]
    LSTM_UNITS = 64

    # Feature Engineering
    TECHNICAL_INDICATORS = [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "ema_9",
        "ema_21",
        "ema_50",
        "atr_14",
        "adx_14",
        "stoch_k",
        "stoch_d",
    ]

    PRICE_FEATURES = ["open", "high", "low", "close", "volume"]

    # Target thresholds
    SHORT_TERM_THRESHOLD = 0.002  # 0.2%
    LONG_TERM_THRESHOLD = 0.005  # 0.5%
