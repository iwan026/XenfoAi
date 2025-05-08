import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timezone
import talib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CURRENT_UTC = "2025-05-08 12:26:13"
CURRENT_USER = "Xentrovt"


@dataclass
class TechnicalLevels:
    """Container for key technical levels"""

    support: float
    resistance: float
    daily_pivot: float
    weekly_pivot: float
    monthly_pivot: float
    atr: float
    volatility: float


class FeatureUtils:
    """Utility functions for feature calculations"""

    @staticmethod
    def normalize_feature(series: pd.Series, method: str = "minmax") -> pd.Series:
        """Normalize feature to 0-1 range"""
        try:
            if method == "minmax":
                min_val = series.min()
                max_val = series.max()
                if max_val == min_val:
                    return pd.Series(0, index=series.index)
                return (series - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    return pd.Series(0, index=series.index)
                return (series - mean) / std
            else:
                raise ValueError(f"Unknown normalization method: {method}")

        except Exception as e:
            logger.error(f"Error normalizing feature: {str(e)}")
            return pd.Series(0, index=series.index)

    @staticmethod
    def calculate_technical_levels(df: pd.DataFrame) -> TechnicalLevels:
        """Calculate key technical levels"""
        try:
            # Support and Resistance
            support = df["low"].rolling(20).min().iloc[-1]
            resistance = df["high"].rolling(20).max().iloc[-1]

            # Pivot Points
            typical_price = (df["high"] + df["low"] + df["close"]) / 3

            daily_pivot = typical_price.iloc[-1]
            weekly_pivot = typical_price.tail(5).mean()
            monthly_pivot = typical_price.tail(20).mean()

            # Volatility Metrics
            atr = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values, timeperiod=14
            )[-1]

            volatility = df["close"].pct_change().std()

            return TechnicalLevels(
                support=support,
                resistance=resistance,
                daily_pivot=daily_pivot,
                weekly_pivot=weekly_pivot,
                monthly_pivot=monthly_pivot,
                atr=atr,
                volatility=volatility,
            )

        except Exception as e:
            logger.error(f"Error calculating technical levels: {str(e)}")
            return None

    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling z-score"""
        try:
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
            return (series - mean) / std

        except Exception as e:
            logger.error(f"Error calculating z-score: {str(e)}")
            return pd.Series(0, index=series.index)

    @staticmethod
    def detect_divergence(
        price: pd.Series, indicator: pd.Series, window: int = 20
    ) -> pd.Series:
        """Detect regular and hidden divergences"""
        try:
            # Price and Indicator Trends
            price_trend = price.diff()
            indicator_trend = indicator.diff()

            # Regular Bullish Divergence
            reg_bull_div = (
                (price_trend < 0)
                & (indicator_trend > 0)
                & (price < price.shift(window))
                & (indicator > indicator.shift(window))
            )

            # Regular Bearish Divergence
            reg_bear_div = (
                (price_trend > 0)
                & (indicator_trend < 0)
                & (price > price.shift(window))
                & (indicator < indicator.shift(window))
            )

            # Hidden Bullish Divergence
            hid_bull_div = (
                (price_trend > 0)
                & (indicator_trend < 0)
                & (price > price.shift(window))
                & (indicator < indicator.shift(window))
            )

            # Hidden Bearish Divergence
            hid_bear_div = (
                (price_trend < 0)
                & (indicator_trend > 0)
                & (price < price.shift(window))
                & (indicator > indicator.shift(window))
            )

            # Combine all divergences
            divergence = pd.Series(0, index=price.index)
            divergence[reg_bull_div] = 1  # Regular Bullish
            divergence[reg_bear_div] = -1  # Regular Bearish
            divergence[hid_bull_div] = 2  # Hidden Bullish
            divergence[hid_bear_div] = -2  # Hidden Bearish

            return divergence

        except Exception as e:
            logger.error(f"Error detecting divergence: {str(e)}")
            return pd.Series(0, index=price.index)

    @staticmethod
    def calculate_efficiency_ratio(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Market Efficiency Ratio"""
        try:
            direction = abs(prices - prices.shift(window))
            volatility = pd.Series(0, index=prices.index)

            for i in range(window):
                volatility += abs(prices - prices.shift(1))

            return direction / volatility

        except Exception as e:
            logger.error(f"Error calculating efficiency ratio: {str(e)}")
            return pd.Series(0, index=prices.index)

    @staticmethod
    def detect_volume_anomaly(volume: pd.Series, threshold: float = 2.0) -> pd.Series:
        """Detect volume anomalies"""
        try:
            volume_sma = volume.rolling(window=20).mean()
            volume_std = volume.rolling(window=20).std()

            return (volume - volume_sma) > (volume_std * threshold)

        except Exception as e:
            logger.error(f"Error detecting volume anomaly: {str(e)}")
            return pd.Series(False, index=volume.index)

    @staticmethod
    def get_session_times(timestamp: pd.Timestamp) -> Dict[str, bool]:
        """Determine active trading sessions"""
        try:
            hour = timestamp.hour

            sessions = {"asian": False, "london": False, "new_york": False}

            # Asian Session (Tokyo)
            if 0 <= hour < 9:
                sessions["asian"] = True

            # London Session
            if 8 <= hour < 16:
                sessions["london"] = True

            # New York Session
            if 13 <= hour < 22:
                sessions["new_york"] = True

            return sessions

        except Exception as e:
            logger.error(f"Error getting session times: {str(e)}")
            return {"asian": False, "london": False, "new_york": False}

    @staticmethod
    def calculate_momentum_quality(
        close: pd.Series, volume: pd.Series, window: int = 14
    ) -> pd.Series:
        """Calculate momentum quality"""
        try:
            # Price momentum
            returns = close.pct_change()
            momentum = returns.rolling(window).mean()

            # Volume momentum
            volume_ratio = volume / volume.rolling(window).mean()

            # Combine price and volume momentum
            quality = momentum * np.sqrt(volume_ratio)

            return FeatureUtils.normalize_feature(quality)

        except Exception as e:
            logger.error(f"Error calculating momentum quality: {str(e)}")
            return pd.Series(0, index=close.index)

    @staticmethod
    def log_calculation_metadata(
        feature_name: str, calculation_time: float, params: Dict[str, Any]
    ) -> None:
        """Log feature calculation metadata"""
        try:
            metadata = {
                "feature": feature_name,
                "calculation_time": calculation_time,
                "parameters": params,
                "timestamp": CURRENT_UTC,
                "calculated_by": CURRENT_USER,
            }

            logger.info(f"Feature calculation metadata: {metadata}")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame has required columns"""
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating DataFrame: {str(e)}")
            return False

    @staticmethod
    def get_timeframe_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        try:
            timeframe_dict = {
                "M1": 1,
                "M5": 5,
                "M15": 15,
                "M30": 30,
                "H1": 60,
                "H4": 240,
                "D1": 1440,
                "W1": 10080,
            }

            return timeframe_dict.get(timeframe.upper(), 0)

        except Exception as e:
            logger.error(f"Error converting timeframe: {str(e)}")
            return 0
