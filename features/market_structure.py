import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketStructureParameters:
    """Parameters for Market Structure analysis"""

    # Trend Analysis
    trend_period: int = 20  # Period for trend calculation
    trend_threshold: float = 0.6  # Threshold for trend strength
    swing_threshold: float = 0.002  # Minimum swing size

    # Support/Resistance
    sr_lookback: int = 50  # Lookback period for S/R levels
    sr_touch_count: int = 3  # Minimum touches for valid level
    sr_proximity: float = 0.001  # Proximity threshold for level grouping

    # Structure Break
    structure_break_threshold: float = 0.003  # Min size for structure break
    confirmation_candles: int = 3  # Candles to confirm break

    # Volatility
    volatility_period: int = 20  # Period for volatility calculation
    regime_threshold: float = 0.5  # Threshold for regime classification

    # Momentum
    momentum_period: int = 14  # Period for momentum indicators
    momentum_threshold: float = 0.7  # Threshold for momentum significance


class MarketStructureAnalyzer:
    """Main class for Market Structure analysis"""

    def __init__(self, params: Optional[MarketStructureParameters] = None):
        """Initialize Market Structure analyzer with parameters"""
        self.params = params or MarketStructureParameters()
        self.feature_columns: List[str] = []
        self._initialize_feature_columns()

        self.metadata = {
            "version": TradingConfig.VERSION,
            "created_at": TradingConfig.get_current_timestamp(),
            "created_by": TradingConfig.AUTHOR,
        }

    def _initialize_feature_columns(self) -> None:
        """Initialize list of feature column names"""
        self.feature_columns = [
            # Trend Analysis
            "trend_state",
            "trend_strength",
            "trend_duration",
            # Support/Resistance
            "nearest_support",
            "nearest_resistance",
            "sr_strength",
            # Structure Break
            "structure_break",
            "break_direction",
            "break_strength",
            # Volatility
            "volatility_regime",
            "volatility_score",
            # Momentum
            "momentum_state",
            "momentum_strength",
        ]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Market Structure related features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional Market Structure features
        """
        try:
            # Validate input
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in input DataFrame")

            # Create copy to prevent modifications to original
            df = df.copy()

            # Calculate main features
            df = self._analyze_trend(df)
            df = self._identify_support_resistance(df)
            df = self._detect_structure_breaks(df)
            df = self._analyze_volatility(df)
            df = self._analyze_momentum(df)

            # Log calculation metadata
            self._log_calculation(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Market Structure features: {str(e)}")
            return df

    def _analyze_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze market trend state and characteristics"""
        try:
            period = self.params.trend_period

            # Calculate moving averages
            df["ma_fast"] = df["close"].rolling(period).mean()
            df["ma_slow"] = df["close"].rolling(period * 2).mean()

            # Calculate trend direction
            df["trend_direction"] = np.where(
                df["ma_fast"] > df["ma_slow"],
                1,  # Uptrend
                np.where(
                    df["ma_fast"] < df["ma_slow"],
                    -1,  # Downtrend
                    0,  # Sideways
                ),
            )

            # Calculate trend strength
            df["trend_strength"] = abs((df["ma_fast"] - df["ma_slow"]) / df["ma_slow"])

            # Determine trend state
            conditions = [
                (df["trend_direction"] == 1)
                & (df["trend_strength"] > self.params.trend_threshold),
                (df["trend_direction"] == -1)
                & (df["trend_strength"] > self.params.trend_threshold),
                (df["trend_strength"] <= self.params.trend_threshold),
            ]
            choices = ["uptrend", "downtrend", "ranging"]
            df["trend_state"] = np.select(conditions, choices, default="ranging")

            # Calculate trend duration
            df["trend_duration"] = (
                df.groupby(
                    (df["trend_state"] != df["trend_state"].shift(1)).cumsum()
                ).cumcount()
                + 1
            )

            # Clean up
            df = df.drop(["ma_fast", "ma_slow", "trend_direction"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return df

    def _identify_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify support and resistance levels"""
        try:
            lookback = self.params.sr_lookback

            # Find swing highs and lows
            df["swing_high"] = df["high"].rolling(lookback, center=True).max()
            df["swing_low"] = df["low"].rolling(lookback, center=True).min()

            # Group nearby levels
            def group_levels(series: pd.Series) -> List[float]:
                levels = []
                current_group = []

                sorted_values = sorted(series.unique())

                for value in sorted_values:
                    if (
                        not current_group
                        or abs(value - current_group[-1]) <= self.params.sr_proximity
                    ):
                        current_group.append(value)
                    else:
                        if len(current_group) >= self.params.sr_touch_count:
                            levels.append(np.mean(current_group))
                        current_group = [value]

                if len(current_group) >= self.params.sr_touch_count:
                    levels.append(np.mean(current_group))

                return levels

            # Find support and resistance levels
            resistance_levels = group_levels(df["swing_high"])
            support_levels = group_levels(df["swing_low"])

            # Find nearest levels
            current_price = df["close"].iloc[-1]

            df["nearest_resistance"] = min(
                [level for level in resistance_levels if level > current_price],
                default=current_price,
            )

            df["nearest_support"] = max(
                [level for level in support_levels if level < current_price],
                default=current_price,
            )

            # Calculate S/R strength
            df["sr_strength"] = (
                df["volume"].rolling(lookback).mean()
                / df["volume"].rolling(lookback).std()
            )

            # Clean up
            df = df.drop(["swing_high", "swing_low"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error identifying support/resistance: {str(e)}")
            return df

    def _detect_structure_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect market structure breaks"""
        try:
            threshold = self.params.structure_break_threshold
            confirmation = self.params.confirmation_candles

            # Identify potential breaks
            df["higher_high"] = df["high"] > df["high"].rolling(
                confirmation
            ).max().shift(1)

            df["lower_low"] = df["low"] < df["low"].rolling(confirmation).min().shift(1)

            # Confirm breaks with size threshold
            df["structure_break"] = (
                df["higher_high"] & (df["high"] - df["high"].shift(1)) > threshold
            ) | (df["lower_low"] & (df["low"].shift(1) - df["low"]) > threshold)

            # Determine break direction
            df["break_direction"] = np.where(
                df["higher_high"] & df["structure_break"],
                1,  # Bullish break
                np.where(
                    df["lower_low"] & df["structure_break"],
                    -1,  # Bearish break
                    0,  # No break
                ),
            )

            # Calculate break strength
            df["break_strength"] = np.where(
                df["structure_break"],
                abs(df["close"] - df["close"].shift(confirmation))
                / df["close"].shift(confirmation),
                0,
            )

            # Clean up
            df = df.drop(["higher_high", "lower_low"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error detecting structure breaks: {str(e)}")
            return df

    def _analyze_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze volatility regimes"""
        try:
            period = self.params.volatility_period

            # Calculate different volatility measures
            df["atr"] = self._calculate_atr(df, period)
            df["volatility"] = (
                df["close"].rolling(period).std() / df["close"].rolling(period).mean()
            )

            # Normalize volatility score
            df["volatility_score"] = (
                df["volatility"] / df["volatility"].rolling(period * 2).max()
            )

            # Classify volatility regime
            conditions = [
                df["volatility_score"] > self.params.regime_threshold,
                df["volatility_score"] <= self.params.regime_threshold * 0.5,
                True,
            ]
            choices = ["high", "low", "normal"]

            df["volatility_regime"] = np.select(conditions, choices, default="normal")

            # Clean up
            df = df.drop(["atr", "volatility"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
            return df

    def _analyze_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze market momentum"""
        try:
            period = self.params.momentum_period

            # Calculate momentum indicators
            df["roc"] = (df["close"] - df["close"].shift(period)) / df["close"].shift(
                period
            )

            df["momentum"] = df["close"] - df["close"].shift(period)

            # Calculate momentum strength
            df["momentum_strength"] = abs(
                df["momentum"] / df["momentum"].rolling(period).std()
            )

            # Determine momentum state
            conditions = [
                (df["momentum"] > 0)
                & (df["momentum_strength"] > self.params.momentum_threshold),
                (df["momentum"] < 0)
                & (df["momentum_strength"] > self.params.momentum_threshold),
                True,
            ]
            choices = ["bullish", "bearish", "neutral"]

            df["momentum_state"] = np.select(conditions, choices, default="neutral")

            # Clean up
            df = df.drop(["roc", "momentum"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing momentum: {str(e)}")
            return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift(1))
            tr3 = abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(0, index=df.index)

    def _log_calculation(self, df: pd.DataFrame) -> None:
        """Log calculation metadata"""
        try:
            log_entry = {
                "timestamp": TradingConfig.get_current_timestamp(),
                "rows_processed": len(df),
                "parameters": self.params.__dict__,
                "version": TradingConfig.VERSION,
                "stats": {
                    "structure_breaks": df["structure_break"].sum(),
                    "avg_trend_strength": df["trend_strength"].mean(),
                    "volatility_regimes": df["volatility_regime"]
                    .value_counts()
                    .to_dict(),
                },
            }

            log_file = TradingConfig.LOG_DIR / "market_structure_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    def get_feature_names(self) -> List[str]:
        """Get list of all Market Structure feature names"""
        return self.feature_columns

    def get_feature_snapshot(self, row: pd.Series) -> Dict:
        """Get snapshot of Market Structure features for a single row"""
        try:
            return {
                feature: row[feature]
                for feature in self.feature_columns
                if feature in row
            }
        except Exception as e:
            logger.error(f"Error getting feature snapshot: {str(e)}")
            return {}

    def get_metadata(self) -> Dict:
        """Get analyzer metadata"""
        return {
            **self.metadata,
            "parameters": self.params.__dict__,
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
        }
