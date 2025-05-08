import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class TechnicalLevelParameters:
    """Parameters for technical level calculations"""

    # Price Levels
    pivot_period: int = 20  # Period for pivot point calculation
    fibonacci_levels: List[float] = field(
        default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786]
    )

    # Moving Averages
    ma_periods: List[int] = field(default_factory=lambda: [20, 50, 100, 200])

    # Volatility
    atr_period: int = 14  # Period for ATR calculation
    std_dev_period: int = 20  # Period for standard deviation

    # Statistical
    zscore_period: int = 20  # Period for z-score calculation
    outlier_threshold: float = 2.0  # Z-score threshold for outliers


class FeatureUtils:
    """Utility class for feature calculations and analysis"""

    def __init__(self, params: Optional[TechnicalLevelParameters] = None):
        """Initialize Feature utilities with parameters"""
        self.params = params or TechnicalLevelParameters()
        self.metadata = {
            "version": TradingConfig.VERSION,
            "created_at": TradingConfig.get_current_timestamp(),
            "created_by": TradingConfig.AUTHOR,
        }

    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate DataFrame structure and content

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            bool: True if validation passes
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.error("DataFrame is empty")
                return False

            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for NaN values
            nan_columns = (
                df[required_columns].columns[df[required_columns].isna().any()].tolist()
            )
            if nan_columns:
                logger.error(f"NaN values found in columns: {nan_columns}")
                return False

            # Check index
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex")

            return True

        except Exception as e:
            logger.error(f"Error validating DataFrame: {str(e)}")
            return False

    def calculate_technical_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate various technical levels"""
        try:
            levels = {}

            # Calculate pivot points
            levels.update(self._calculate_pivot_points(df))

            # Calculate Fibonacci levels
            levels.update(self._calculate_fibonacci_levels(df))

            # Calculate moving averages
            levels.update(self._calculate_moving_averages(df))

            # Calculate volatility levels
            levels.update(self._calculate_volatility_levels(df))

            return levels

        except Exception as e:
            logger.error(f"Error calculating technical levels: {str(e)}")
            return {}

    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate pivot points and associated levels"""
        try:
            period = self.params.pivot_period

            # Get high, low, close for period
            high = df["high"].rolling(period).max().iloc[-1]
            low = df["low"].rolling(period).min().iloc[-1]
            close = df["close"].iloc[-1]

            # Calculate pivot point
            pivot = (high + low + close) / 3

            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)

            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)

            return {
                "pivot_point": pivot,
                "resistance_levels": [r1, r2, r3],
                "support_levels": [s1, s2, s3],
            }

        except Exception as e:
            logger.error(f"Error calculating pivot points: {str(e)}")
            return {}

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        try:
            # Get high and low
            high = df["high"].max()
            low = df["low"].min()
            trend_range = high - low

            # Calculate levels
            levels = {
                f"fib_{int(level * 1000)}": low + trend_range * level
                for level in self.params.fibonacci_levels
            }

            return {"fibonacci_levels": levels}

        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {}

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Calculate multiple moving averages"""
        try:
            mas = {}

            for period in self.params.ma_periods:
                ma = df["close"].rolling(period).mean().iloc[-1]
                mas[f"ma_{period}"] = ma

            return {"moving_averages": mas}

        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return {}

    def _calculate_volatility_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility-based levels"""
        try:
            # Calculate ATR
            atr = self._calculate_atr(df).iloc[-1]

            # Calculate standard deviation
            std_dev = df["close"].rolling(self.params.std_dev_period).std().iloc[-1]

            current_price = df["close"].iloc[-1]

            return {
                "volatility_levels": {
                    "atr": atr,
                    "std_dev": std_dev,
                    "upper_band": current_price + 2 * std_dev,
                    "lower_band": current_price - 2 * std_dev,
                }
            }

        except Exception as e:
            logger.error(f"Error calculating volatility levels: {str(e)}")
            return {}

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        try:
            period = self.params.atr_period

            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift(1))
            tr3 = abs(df["low"] - df["close"].shift(1))

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()

            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_zscore(
        self, series: pd.Series, period: Optional[int] = None
    ) -> pd.Series:
        """Calculate rolling z-score for a series"""
        try:
            if period is None:
                period = self.params.zscore_period

            mean = series.rolling(period).mean()
            std = series.rolling(period).std()

            return (series - mean) / std

        except Exception as e:
            logger.error(f"Error calculating z-score: {str(e)}")
            return pd.Series(0, index=series.index)

    def detect_outliers(
        self, series: pd.Series, threshold: Optional[float] = None
    ) -> pd.Series:
        """Detect outliers using z-score method"""
        try:
            if threshold is None:
                threshold = self.params.outlier_threshold

            zscore = self.calculate_zscore(series)
            return abs(zscore) > threshold

        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return pd.Series(False, index=series.index)

    def log_calculation_metadata(
        self, operation: str, calculation_time: float, metadata: Dict
    ) -> None:
        """Log feature calculation metadata"""
        try:
            log_entry = {
                "timestamp": TradingConfig.get_current_timestamp(),
                "operation": operation,
                "calculation_time": calculation_time,
                "version": TradingConfig.VERSION,
                "user": TradingConfig.AUTHOR,
                **metadata,
            }

            log_file = TradingConfig.LOG_DIR / "feature_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    def get_metadata(self) -> Dict:
        """Get utility metadata"""
        return {**self.metadata, "parameters": self.params.__dict__}
