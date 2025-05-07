import os
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json
import pytz

logger = logging.getLogger(__name__)


@dataclass
class ModelParameters:
    """Model architecture and training parameters"""

    lookback_period: int = 60
    feature_dims: int = 32
    lstm_units: list = (128, 64)
    attention_heads: int = 4
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
class TradingParameters:
    """Trading strategy parameters"""

    min_pip_movement: float = 10.0
    risk_reward_ratio: float = 2.0
    max_risk_per_trade: float = 0.02
    max_correlation_threshold: float = 0.75
    position_sizing_atr_multiple: float = 1.5
    max_positions_per_regime: Dict[str, int] = None

    def __post_init__(self):
        if self.max_positions_per_regime is None:
            self.max_positions_per_regime = {
                "volatile_trend": 3,
                "stable_trend": 4,
                "volatile_range": 2,
                "stable_range": 3,
            }


@dataclass
class BacktestParameters:
    """Backtesting configuration"""

    initial_balance: float = 10000.0
    commission_rate: float = 0.0001
    slippage_pips: float = 1.0
    spread_pips: float = 2.0
    leverage: float = 100.0
    margin_requirement: float = 0.01


class TradingConfig:
    """Enhanced trading configuration with environment awareness"""

    # Basic paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    REPORT_DIR = os.path.join(BASE_DIR, "reports")

    # Ensure directories exist
    for directory in [MODEL_DIR, LOG_DIR, DATA_DIR, REPORT_DIR]:
        os.makedirs(directory, exist_ok=True)

    # MetaTrader5 Configuration
    MT5_LOGIN = 204313635
    MT5_PASSWORD = "123@Demo"
    MT5_SERVER = "Exness-MT5Trial7"

    # Telegram Configuration (if needed)
    TELEGRAM_TOKEN = "7667262262:AAFYkfcdd8OZQskNYQPJ9KbVO8rGE3rvouI"
    ALLOWED_USERS = [1198920849]

    # Model Configuration
    MODEL_PARAMS = ModelParameters()
    TRADING_PARAMS = TradingParameters()
    BACKTEST_PARAMS = BacktestParameters()

    # Time Settings
    TIMEZONE = pytz.UTC
    TRADING_HOURS = {
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

    # Feature Groups
    FEATURE_GROUPS = {
        "price": ["open", "high", "low", "close", "returns", "log_returns"],
        "volume": ["volume", "volume_sma", "volume_ratio"],
        "momentum": ["rsi", "stoch_k", "stoch_d", "cci"],
        "trend": ["sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50"],
        "volatility": ["atr", "bb_width", "volatility"],
    }

    # Risk Management
    RISK_MANAGEMENT = {
        "max_drawdown": 0.20,  # 20% maximum drawdown
        "daily_risk": 0.02,  # 2% daily risk
        "position_risk": 0.01,  # 1% risk per position
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
        """Get model file path with enhanced validation"""
        try:
            if not cls.validate_symbol_timeframe(symbol, timeframe):
                return ""

            # Create standardized filename
            symbol = symbol.upper()
            timeframe = timeframe.upper()

            # Create model directory if it doesn't exist
            model_dir = os.path.join(cls.MODEL_DIR, f"{symbol}_{timeframe}")
            os.makedirs(model_dir, exist_ok=True)

            return os.path.join(model_dir, "model.h5")

        except Exception as e:
            logger.error(f"Error getting model path: {str(e)}")
            return ""

    @classmethod
    def get_scaler_path(cls, symbol: str, timeframe: str) -> str:
        """Get scaler file path with enhanced validation"""
        try:
            if not cls.validate_symbol_timeframe(symbol, timeframe):
                return ""

            symbol = symbol.upper()
            timeframe = timeframe.upper()
            model_dir = os.path.join(cls.MODEL_DIR, f"{symbol}_{timeframe}")
            os.makedirs(model_dir, exist_ok=True)

            return os.path.join(model_dir, "scaler.pkl")

        except Exception as e:
            logger.error(f"Error getting scaler path: {str(e)}")
            return ""

    @staticmethod
    def validate_symbol_timeframe(symbol: str, timeframe: str) -> bool:
        """Validate symbol and timeframe"""
        try:
            valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

            if not symbol or not timeframe:
                logger.error("Symbol and timeframe cannot be empty")
                return False

            if timeframe.upper() not in valid_timeframes:
                logger.error(f"Invalid timeframe: {timeframe}")
                return False

            # Additional symbol validation could be added here

            return True

        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return False

    @classmethod
    def get_log_path(cls, symbol: str, timeframe: str) -> str:
        """Get log file path"""
        try:
            if not cls.validate_symbol_timeframe(symbol, timeframe):
                return ""

            symbol = symbol.upper()
            timeframe = timeframe.upper()
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")

            log_dir = os.path.join(cls.LOG_DIR, f"{symbol}_{timeframe}")
            os.makedirs(log_dir, exist_ok=True)

            return os.path.join(log_dir, f"{current_date}.log")

        except Exception as e:
            logger.error(f"Error getting log path: {str(e)}")
            return ""

    @classmethod
    def save_trading_session(
        cls, session_data: Dict[str, Any], symbol: str, timeframe: str
    ) -> bool:
        """Save trading session data"""
        try:
            if not cls.validate_symbol_timeframe(symbol, timeframe):
                return False

            symbol = symbol.upper()
            timeframe = timeframe.upper()
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")

            session_dir = os.path.join(
                cls.DATA_DIR, "sessions", f"{symbol}_{timeframe}"
            )
            os.makedirs(session_dir, exist_ok=True)

            session_file = os.path.join(session_dir, f"session_{current_date}.json")

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=4)

            return True

        except Exception as e:
            logger.error(f"Error saving trading session: {str(e)}")
            return False

    @classmethod
    def load_trading_session(
        cls, symbol: str, timeframe: str, date: Optional[str] = None
    ) -> Optional[Dict]:
        """Load trading session data"""
        try:
            if not cls.validate_symbol_timeframe(symbol, timeframe):
                return None

            symbol = symbol.upper()
            timeframe = timeframe.upper()

            if date is None:
                date = datetime.now(timezone.utc).strftime("%Y%m%d")

            session_file = os.path.join(
                cls.DATA_DIR,
                "sessions",
                f"{symbol}_{timeframe}",
                f"session_{date}.json",
            )

            if not os.path.exists(session_file):
                logger.error(f"Session file not found: {session_file}")
                return None

            with open(session_file, "r") as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading trading session: {str(e)}")
            return None

    @classmethod
    def is_trading_time(cls, pair: str) -> bool:
        """Check if it's trading time for a given pair"""
        try:
            current_time = datetime.now(timezone.utc)
            hour = current_time.hour

            # Define trading sessions for currency pairs
            pair_sessions = {
                "USD": cls.TRADING_HOURS["new_york"],
                "EUR": cls.TRADING_HOURS["london"],
                "GBP": cls.TRADING_HOURS["london"],
                "JPY": cls.TRADING_HOURS["tokyo"],
                "AUD": cls.TRADING_HOURS["sydney"],
                "NZD": cls.TRADING_HOURS["sydney"],
            }

            # Get relevant sessions for the pair
            currencies = [pair[:3], pair[3:]]
            relevant_sessions = [
                pair_sessions[curr] for curr in currencies if curr in pair_sessions
            ]

            # Check if current hour is in any relevant session
            for session in relevant_sessions:
                start = session["start"]
                end = session["end"]

                if start < end:
                    if start <= hour < end:
                        return True
                else:  # Session crosses midnight
                    if hour >= start or hour < end:
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking trading time: {str(e)}")
            return False
