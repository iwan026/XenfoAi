import os
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json
import pytz

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Model architecture and training parameters"""

    lookback_period: int = 60
    feature_dims: int = 32
    lstm_units: list = (64, 32)
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class SignalParameters:
    """Signal generation parameters"""

    confidence_threshold: float = 0.75
    min_prediction_interval: int = 5  # minutes
    signal_expiry: int = 240  # minutes
    signal_types: list = ("BUY", "SELL", "HOLD")
    minimum_volatility: float = 0.0001
    trend_confirmation_period: int = 3


@dataclass
class BacktestParameters:
    """Backtesting configuration"""

    initial_date: str = "2024-01-01"
    signal_delay: int = 1  # candle delay for signal validation
    minimum_trades: int = 30
    accuracy_threshold: float = 0.6


class TradingConfig:
    """Enhanced configuration for signal generation"""

    # Basic paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    REPORT_DIR = os.path.join(BASE_DIR, "reports")
    SIGNAL_DIR = os.path.join(BASE_DIR, "signals")

    # Ensure directories exist
    for directory in [MODEL_DIR, LOG_DIR, DATA_DIR, REPORT_DIR, SIGNAL_DIR]:
        os.makedirs(directory, exist_ok=True)

    # MetaTrader5 Configuration (for market data only)
    MT5_LOGIN = 204313635
    MT5_PASSWORD = "123@Demo"
    MT5_SERVER = "Exness-MT5Trial7"

    # Telegram Configuration
    TELEGRAM_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
    ALLOWED_USERS = [1198920849]

    # Model Configuration
    MODEL_PARAMS = ModelParameters()
    SIGNAL_PARAMS = SignalParameters()
    BACKTEST_PARAMS = BacktestParameters()

    # Time Settings
    TIMEZONE = pytz.UTC
    MARKET_HOURS = {
        "sydney": {"start": 22, "end": 7},
        "tokyo": {"start": 0, "end": 9},
        "london": {"start": 8, "end": 17},
        "new_york": {"start": 13, "end": 22},
    }

    # Currency Pairs Configuration
    CURRENCY_PAIRS = {
        "major": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
        "minor": ["EURGBP", "EURJPY", "GBPJPY"],
        "exotic": ["EURNZD", "GBPCAD", "AUDNZD"],
    }

    # Feature Groups for Signal Generation
    FEATURE_GROUPS = {
        "price": ["open", "high", "low", "close", "returns"],
        "momentum": ["rsi", "stoch_k", "stoch_d", "cci"],
        "trend": ["sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50"],
        "volatility": ["atr", "bb_width", "volatility"],
    }

    # Market Regimes
    MARKET_REGIMES = {
        "volatile_trend": {
            "atr_multiplier": 1.5,
            "trend_strength_threshold": 25,
            "volatility_threshold": 0.0015,
        },
        "stable_trend": {
            "atr_multiplier": 1.2,
            "trend_strength_threshold": 25,
            "volatility_threshold": 0.0010,
        },
        "volatile_range": {
            "atr_multiplier": 1.3,
            "trend_strength_threshold": 20,
            "volatility_threshold": 0.0015,
        },
        "stable_range": {
            "atr_multiplier": 1.0,
            "trend_strength_threshold": 20,
            "volatility_threshold": 0.0010,
        },
    }

    @classmethod
    def get_model_path(cls, symbol: str, timeframe: str) -> str:
        """Get model file path"""
        try:
            base_path = os.path.join(cls.MODEL_DIR, f"{symbol}_{timeframe}")
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, "model.h5")
        except Exception as e:
            logger.error(f"Error getting model path: {str(e)}")
            return ""

    @classmethod
    def get_signal_path(cls, symbol: str, timeframe: str, date: str) -> str:
        """Get signal file path"""
        try:
            base_path = os.path.join(cls.SIGNAL_DIR, symbol, timeframe)
            os.makedirs(base_path, exist_ok=True)
            return os.path.join(base_path, f"signals_{date}.json")
        except Exception as e:
            logger.error(f"Error getting signal path: {str(e)}")
            return ""

    @classmethod
    def save_signals(cls, signals: List[Dict], symbol: str, timeframe: str) -> bool:
        """Save generated signals"""
        try:
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            signal_path = cls.get_signal_path(symbol, timeframe, current_date)

            with open(signal_path, "w") as f:
                json.dump(signals, f, indent=4)

            return True
        except Exception as e:
            logger.error(f"Error saving signals: {str(e)}")
            return False

    @classmethod
    def load_signals(
        cls, symbol: str, timeframe: str, date: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """Load historical signals"""
        try:
            if date is None:
                date = datetime.now(timezone.utc).strftime("%Y%m%d")

            signal_path = cls.get_signal_path(symbol, timeframe, date)
            if not os.path.exists(signal_path):
                return None

            with open(signal_path, "r") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading signals: {str(e)}")
            return None

    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate timeframe"""
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
        return timeframe.upper() in valid_timeframes

    @classmethod
    def get_log_path(cls, symbol: str, timeframe: str) -> str:
        """Get log file path"""
        try:
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            log_dir = os.path.join(cls.LOG_DIR, f"{symbol}_{timeframe}")
            os.makedirs(log_dir, exist_ok=True)
            return os.path.join(log_dir, f"{current_date}.log")
        except Exception as e:
            logger.error(f"Error getting log path: {str(e)}")
            return ""

    @classmethod
    def is_market_open(cls, symbol: str) -> bool:
        """Check if market is open for symbol"""
        try:
            current_time = datetime.now(timezone.utc)
            hour = current_time.hour

            # Define trading sessions for currency pairs
            pair_sessions = {
                "USD": cls.MARKET_HOURS["new_york"],
                "EUR": cls.MARKET_HOURS["london"],
                "GBP": cls.MARKET_HOURS["london"],
                "JPY": cls.MARKET_HOURS["tokyo"],
                "AUD": cls.MARKET_HOURS["sydney"],
                "NZD": cls.MARKET_HOURS["sydney"],
            }

            # Get currencies from symbol
            curr1, curr2 = symbol[:3], symbol[3:]
            relevant_sessions = [
                session
                for curr, session in pair_sessions.items()
                if curr in [curr1, curr2]
            ]

            # Check if current hour is in any relevant session
            for session in relevant_sessions:
                if session["start"] <= hour < session["end"]:
                    return True
                if session["start"] > session["end"]:  # Session crosses midnight
                    if hour >= session["start"] or hour < session["end"]:
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")
            return False
