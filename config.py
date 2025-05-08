import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Model training and prediction parameters"""

    lookback_period: int = 20
    batch_size: int = 48
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    dropout_rate: float = 0.4
    l2_lambda: float = 0.02
    attention_heads: int = 8
    min_delta: float = 0.0001


@dataclass
class SignalParameters:
    """Signal generation parameters"""

    confidence_threshold: float = 0.75
    signal_expiry: int = 240  # minutes
    min_signal_interval: int = 30  # minutes
    volatility_threshold: float = 0.002
    trend_strength_threshold: float = 0.6


class TradingConfig:
    """Main configuration class for the trading system"""

    # System Constants
    VERSION = "2.0.0"
    AUTHOR = "Xentrovt"
    CREATED_AT = "2025-05-08 13:34:11"

    # Directory Structure
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"
    REPORT_DIR = BASE_DIR / "reports"
    SIGNAL_DIR = BASE_DIR / "signals"
    BACKTEST_DIR = BASE_DIR / "backtest"

    # Ensure all directories exist
    REQUIRED_DIRS = [MODEL_DIR, LOG_DIR, DATA_DIR, REPORT_DIR, SIGNAL_DIR, BACKTEST_DIR]
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    # MetaTrader5 Configuration
    MT5_LOGIN = 204313635
    MT5_PASSWORD = "123@Demo"
    MT5_SERVER = "Exness-MT5Trial7"

    # Telegram Configuration
    TELEGRAM_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
    ALLOWED_USERS = [1198920849]

    # Trading Parameters
    MODEL_PARAMS = ModelParameters()
    SIGNAL_PARAMS = SignalParameters()

    # Market Regime Parameters
    REGIME_PARAMS = {
        "volatile_trend": {
            "atr_multiplier": 1.5,
            "trend_strength_threshold": 40,
            "volatility_threshold": 0.0020,
        },
        "stable_trend": {
            "atr_multiplier": 1.2,
            "trend_strength_threshold": 30,
            "volatility_threshold": 0.0015,
        },
        "volatile_range": {
            "atr_multiplier": 1.3,
            "trend_strength_threshold": 25,
            "volatility_threshold": 0.0012,
        },
        "stable_range": {
            "atr_multiplier": 1.0,
            "trend_strength_threshold": 20,
            "volatility_threshold": 0.0010,
        },
    }

    @classmethod
    def get_current_timestamp(cls) -> str:
        """Get current UTC timestamp in standard format"""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def get_model_path(cls, symbol: str, timeframe: str) -> Path:
        """Get model file path"""
        try:
            base_path = cls.MODEL_DIR / f"{symbol}_{timeframe}"
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path / "model.h5"
        except Exception as e:
            logger.error(f"Error getting model path: {str(e)}")
            return cls.MODEL_DIR / "default.h5"

    @classmethod
    def get_signal_path(cls, symbol: str, timeframe: str, date: str) -> Path:
        """Get signal file path"""
        try:
            base_path = cls.SIGNAL_DIR / symbol / timeframe
            base_path.mkdir(parents=True, exist_ok=True)
            return base_path / f"signals_{date}.json"
        except Exception as e:
            logger.error(f"Error getting signal path: {str(e)}")
            return cls.SIGNAL_DIR / "error_signals.json"

    @classmethod
    def save_signals(cls, signals: List[Dict], symbol: str, timeframe: str) -> bool:
        """Save generated signals"""
        try:
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            signal_path = cls.get_signal_path(symbol, timeframe, current_date)

            # Add metadata to signals
            for signal in signals:
                signal.update(
                    {
                        "saved_at": cls.get_current_timestamp(),
                        "version": cls.VERSION,
                        "author": cls.AUTHOR,
                    }
                )

            signal_path.write_text(json.dumps(signals, indent=4))
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
            if not signal_path.exists():
                return None

            return json.loads(signal_path.read_text())

        except Exception as e:
            logger.error(f"Error loading signals: {str(e)}")
            return None

    @staticmethod
    def is_market_open(symbol: str) -> bool:
        """Check if market is open for symbol"""
        try:
            # TODO: Implement proper market hours checking
            # For now return True as placeholder
            return True

        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
