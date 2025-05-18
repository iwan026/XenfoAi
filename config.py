import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Path Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = BASE_DIR / "datasets" / "forex"
MODELS_DIR = BASE_DIR / "models" / "forex"
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"

# Create required directories
for directory in [DATASETS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Trading Configuration
SYMBOLS = [
    "BTCUSD",
    "ETHUSD",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "XAUUSD",
]

# MetaTrader5 Configuration
MT5_LOGIN = 204313635
MT5_PASSWORD = "123@Demo"
MT5_SERVER = "Exness-MT5Trial7"

# Telegram Configuration
BOT_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
ADMIN_IDS = [1198920849]


# Model Configuration
class ModelConfig:
    # Data parameters
    SEQUENCE_LENGTH = 60  # 60 candles (60 hours)
    PREDICTION_HORIZONS = [1, 3, 8, 24]  # Predictions for 1h, 3h, 8h, and 24h ahead
    PRICE_FEATURES = ["open", "high", "low", "close", "volume"]
    MARKET_STRUCTURE_FEATURES = ["atr", "dist_from_high", "dist_from_low"]

    @property
    def ALL_FEATURES(self):
        return self.PRICE_FEATURES + self.MARKET_STRUCTURE_FEATURES

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    PATIENCE = 10
    DROPOUT_RATE = 0.3
