import os
from pathlib import Path

# Create necessary directories
for directory in ["models", "logs", "data", "charts", "logs/tensorboard"]:
    Path(directory).mkdir(exist_ok=True, parents=True)


class TradingConfig:
    # Telegram Configuration
    TELEGRAM_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
    ALLOWED_USERS = [1198920849]

    # MetaTrader5 Configuration
    MT5_LOGIN = 204313635
    MT5_PASSWORD = "123@Demo"
    MT5_SERVER = "Exness-MT5Trial7"

    # Model File Paths
    MODEL_PATH = "models/forex_model.h5"
    RF_MODEL_PATH = "models/rf_model.pkl"
    XGB_MODEL_PATH = "models/xgb_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    FEATURE_IMPORTANCE_PATH = "models/feature_importance.json"

    # Log Configuration
    LOG_FILE = "logs/forex_model.log"
    DATABASE = "data/predictions.db"

    # Model Parameters
    LOOKBACK_PERIOD = 60
    TRAINING_YEARS = 5
    FEATURE_SELECTION_THRESHOLD = 0.01
    CONFIDENCE_THRESHOLD = 0.65

    # Training Parameters
    BATCH_SIZE = 32
    INITIAL_LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5

    # Trading Timeframes
    VALID_TIMEFRAMES = {
        "M1": "1 minute",
        "M5": "5 minutes",
        "M15": "15 minutes",
        "M30": "30 minutes",
        "H1": "1 hour",
        "H4": "4 hours",
        "D1": "1 day",
        "W1": "1 week",
    }

    @staticmethod
    def validate_timeframe(timeframe):
        """Validate if timeframe is supported"""
        return timeframe.upper() in TradingConfig.VALID_TIMEFRAMES

    @staticmethod
    def get_model_path(symbol, timeframe):
        """Get model path for specific symbol and timeframe"""
        return os.path.join("models", f"{symbol}_{timeframe}_model.h5")

    @staticmethod
    def get_scaler_path(symbol, timeframe):
        """Get scaler path for specific symbol and timeframe"""
        return os.path.join("models", f"{symbol}_{timeframe}_scaler.pkl")
