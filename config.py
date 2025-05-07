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
    MODEL_DIR = "models"
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
    def validate_symbol_timeframe(symbol: str, timeframe: str) -> bool:
        """Validate symbol and timeframe combination"""
        try:
            # Validate symbol
            if not isinstance(symbol, str) or len(symbol) < 6:
                logger.error(f"Invalid symbol format: {symbol}")
                return False

            # Validate timeframe
            valid_timeframes = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"}
            if timeframe.upper() not in valid_timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating symbol/timeframe: {str(e)}")
            return False

    @staticmethod
    def get_model_path(symbol: str, timeframe: str) -> str:
        """Get model file path"""
        try:
            # Validate inputs
            if not TradingConfig.validate_symbol_timeframe(symbol, timeframe):
                return ""

            # Create standardized filename
            symbol = symbol.upper()
            timeframe = timeframe.upper()
            filename = f"{symbol}_{timeframe}_model.h5"

            # Construct full path
            model_path = os.path.join(TradingConfig.MODEL_DIR, filename)

            return model_path

        except Exception as e:
            logger.error(f"Error getting model path: {str(e)}")
            return ""

    @staticmethod
    def get_scaler_path(symbol: str, timeframe: str) -> str:
        """Get scaler file path"""
        try:
            # Validate inputs
            if not TradingConfig.validate_symbol_timeframe(symbol, timeframe):
                return ""

            # Create standardized filename
            symbol = symbol.upper()
            timeframe = timeframe.upper()
            filename = f"{symbol}_{timeframe}_scaler.pkl"

            # Construct full path
            scaler_path = os.path.join(TradingConfig.MODEL_DIR, filename)

            return scaler_path

        except Exception as e:
            logger.error(f"Error getting scaler path: {str(e)}")
            return ""
