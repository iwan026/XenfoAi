import os
import logging
from pathlib import Path

# Path Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

# Create required directories
REQUIRED_DIRS = [DATASETS_DIR, MODELS_DIR]
for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)

# MT5 Configuration
TIMEFRAMES = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
}

# Trading Configuration
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]  # Tambahkan simbol yang diperlukan

# MetaTrader5 Credentials
MT5_LOGIN = "5035979858"
MT5_PASSWORD = "N@Al6mVk"
MT5_SERVER = "MetaQuotes-Demo"

# Telegram Configuration
BOT_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
ADMIN_IDS = [1198920849]  # List admin user IDs


# Model Configuration
class ModelConfig:
    # Training
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 0.001
    PATIENCE = 10

    # Model Architecture
    CNN_FILTERS = [32, 64, 128]
    CNN_KERNEL_SIZES = [(3, 1), (3, 1), (3, 1)]
    CNN_POOL_SIZES = [(2, 1), (2, 1), (2, 1)]
    CNN_DROPOUT = 0.2

    LSTM_UNITS = [64, 32]
    LSTM_DROPOUT = 0.2

    TRANSFORMER_HEAD_SIZE = 256
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_FF_DIM = 4
    TRANSFORMER_DROPOUT = 0.1

    # Feature Engineering
    TECHNICAL_FEATURES = [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "sma_20",
        "sma_50",
        "ema_9",
        "ema_21",
        "atr_14",
        "adx_14",
        "cci_20",
        "stoch_k",
        "stoch_d",
    ]

    PRICE_FEATURES = ["open", "high", "low", "close", "volume"]


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(BASE_DIR / "app.log")],
)

# Resource Management
MAX_MEMORY_GB = 6.0  # Maximum memory usage in GB
MAX_CPU_PERCENT = 75  # Maximum CPU usage in percent
CACHE_EXPIRY = 3600  # Cache expiry in seconds
