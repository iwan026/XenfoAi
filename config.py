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
    # Auto-detected parameters will override these defaults
    SEQUENCE_LENGTH = 60
    PREDICTION_HORIZON = 4
    MAX_HORIZON = 24

    # Extended features for direction prediction
    PRICE_FEATURES = ["open", "high", "low", "close", "volume"]
    TECHNICAL_FEATURES = [
        "atr",  # Volatility indicator
        "price_momentum_1",  # Short-term momentum
        "price_momentum_3",  # Medium-term momentum
        "ma_fast",  # Fast moving average
        "ma_slow",  # Slow moving average
        "ma_crossover",  # Moving average crossover signal
        "direction",  # Current price direction
    ]

    @property
    def ALL_FEATURES(self):
        return self.PRICE_FEATURES + self.TECHNICAL_FEATURES

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    PATIENCE = 5
    DROPOUT_RATE = 0.3  # Slightly increased dropout for better generalization
