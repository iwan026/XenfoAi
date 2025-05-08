import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OrderBlockFeatures:
    """Container for Order Block features"""

    bull_ob: bool
    bear_ob: bool
    ob_strength: float
    ob_volume_ratio: float
    ob_momentum: float
    ob_quality: float
    reversal_zone: bool
    ob_retest: bool
    ob_distance: float
    ob_confirmation: bool


class OrderBlockAnalyzer:
    """Enhanced Order Block feature generator"""

    def __init__(self):
        self.feature_columns = []
        self.lookback_period = 20
        self.volume_threshold = 1.5
        self.atr_multiplier = 1.2
        self.min_ob_size = 0.5

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive Order Block features"""
        try:
            # Validate input data
            if df.empty or not all(
                col in df.columns
                for col in ["open", "high", "low", "close", "volume", "atr"]
            ):
                logger.error("Missing required columns in input data")
                return df

            # Basic Order Block Detection
            df = self._detect_order_blocks(df)

            # Order Block Strength and Quality
            df = self._calculate_ob_metrics(df)

            # Order Block Context
            df = self._analyze_ob_context(df)

            # Order Block Validation
            df = self._validate_order_blocks(df)

            # Update feature columns list
            self.feature_columns = [
                "bull_ob",
                "bear_ob",
                "ob_strength",
                "ob_volume_ratio",
                "ob_momentum",
                "ob_quality",
                "reversal_zone",
                "ob_retest",
                "ob_distance",
                "ob_confirmation",
            ]

            # Validate calculated features
            df = self._validate_features(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Order Block features: {str(e)}")
            return df

    def _detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Bullish and Bearish Order Blocks"""
        try:
            # Bullish Order Block
            df["bull_ob"] = (
                (df["close"] < df["open"])  # Bearish candle
                & (df["close"].shift(-1) > df["high"])  # Next candle breaks high
                & (
                    df["volume"]
                    > df["volume"].rolling(self.lookback_period).mean()
                    * self.volume_threshold
                )  # High volume
                & (
                    abs(df["close"] - df["open"]) > df["atr"] * self.min_ob_size
                )  # Significant size
            )

            # Bearish Order Block
            df["bear_ob"] = (
                (df["close"] > df["open"])  # Bullish candle
                & (df["close"].shift(-1) < df["low"])  # Next candle breaks low
                & (
                    df["volume"]
                    > df["volume"].rolling(self.lookback_period).mean()
                    * self.volume_threshold
                )  # High volume
                & (
                    abs(df["close"] - df["open"]) > df["atr"] * self.min_ob_size
                )  # Significant size
            )

            return df

        except Exception as e:
            logger.error(f"Error detecting order blocks: {str(e)}")
            return df

    def _calculate_ob_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Order Block strength and related metrics"""
        try:
            # Order Block Strength
            df["ob_strength"] = np.where(
                df["bull_ob"],
                (df["high"].shift(-1) - df["low"]) / df["atr"],
                np.where(
                    df["bear_ob"], (df["high"] - df["low"].shift(-1)) / df["atr"], 0
                ),
            )

            # Volume Ratio
            df["ob_volume_ratio"] = (
                df["volume"] / df["volume"].rolling(self.lookback_period).mean()
            )

            # Momentum
            df["ob_momentum"] = np.where(
                df["bull_ob"] | df["bear_ob"],
                abs(df["close"] - df["open"]) * df["ob_volume_ratio"],
                0,
            )

            # Quality Score
            df["ob_quality"] = np.where(
                df["bull_ob"] | df["bear_ob"],
                (
                    df["ob_strength"] * 0.4
                    + df["ob_volume_ratio"] * 0.3
                    + df["ob_momentum"] * 0.3
                ),
                0,
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating OB metrics: {str(e)}")
            return df

    def _analyze_ob_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze Order Block context and market conditions"""
        try:
            # Reversal Zone Detection
            df["reversal_zone"] = (
                (df["bull_ob"])
                & (
                    df["close"].rolling(self.lookback_period).min().shift(1)
                    == df["close"].shift(1)
                )
            ) | (
                (df["bear_ob"])
                & (
                    df["close"].rolling(self.lookback_period).max().shift(1)
                    == df["close"].shift(1)
                )
            )

            # Order Block Retest
            df["ob_retest"] = self._detect_retests(df)

            # Distance from Order Block
            df["ob_distance"] = self._calculate_ob_distance(df)

            # Order Block Confirmation
            df["ob_confirmation"] = (
                (df["ob_quality"] > 0.7)
                & (df["ob_volume_ratio"] > self.volume_threshold)
                & (df["ob_strength"] > 1.0)
            )

            return df

        except Exception as e:
            logger.error(f"Error analyzing OB context: {str(e)}")
            return df

    def _detect_retests(self, df: pd.DataFrame) -> pd.Series:
        """Detect Order Block retests"""
        try:
            return (
                (df["bull_ob"].shift(1))
                & (df["low"] <= df["low"].shift(1))
                & (df["close"] > df["open"])
            ) | (
                (df["bear_ob"].shift(1))
                & (df["high"] >= df["high"].shift(1))
                & (df["close"] < df["open"])
            )

        except Exception as e:
            logger.error(f"Error detecting retests: {str(e)}")
            return pd.Series(False, index=df.index)

    def _calculate_ob_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance from nearest Order Block"""
        try:
            bull_distance = np.where(
                df["bull_ob"],
                0,
                (df["close"] - df["low"].rolling(self.lookback_period).min())
                / df["atr"],
            )

            bear_distance = np.where(
                df["bear_ob"],
                0,
                (df["high"].rolling(self.lookback_period).max() - df["close"])
                / df["atr"],
            )

            return pd.Series(np.minimum(bull_distance, bear_distance), index=df.index)

        except Exception as e:
            logger.error(f"Error calculating OB distance: {str(e)}")
            return pd.Series(0, index=df.index)

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean calculated features"""
        try:
            for column in self.feature_columns:
                # Replace infinities
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values
                df[column] = df[column].fillna(method="ffill").fillna(0)

                # Ensure proper range for normalized features
                if column in ["ob_quality", "ob_strength", "ob_momentum"]:
                    df[column] = df[column].clip(0, 1)

            return df

        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns

    def get_ob_features(self, row: pd.Series) -> OrderBlockFeatures:
        """Get Order Block features for a single row"""
        try:
            return OrderBlockFeatures(
                bull_ob=bool(row["bull_ob"]),
                bear_ob=bool(row["bear_ob"]),
                ob_strength=float(row["ob_strength"]),
                ob_volume_ratio=float(row["ob_volume_ratio"]),
                ob_momentum=float(row["ob_momentum"]),
                ob_quality=float(row["ob_quality"]),
                reversal_zone=bool(row["reversal_zone"]),
                ob_retest=bool(row["ob_retest"]),
                ob_distance=float(row["ob_distance"]),
                ob_confirmation=bool(row["ob_confirmation"]),
            )
        except Exception as e:
            logger.error(f"Error creating OB features: {str(e)}")
            return None

    def get_recent_order_blocks(
        self, df: pd.DataFrame, lookback: int = 5
    ) -> List[Dict]:
        """Get recent Order Block information"""
        try:
            recent_obs = []

            for i in range(min(lookback, len(df))):
                row = df.iloc[-i - 1]
                if row["bull_ob"] or row["bear_ob"]:
                    ob_info = {
                        "type": "bullish" if row["bull_ob"] else "bearish",
                        "timestamp": row.name,
                        "price": row["close"],
                        "strength": row["ob_strength"],
                        "quality": row["ob_quality"],
                        "confirmed": row["ob_confirmation"],
                    }
                    recent_obs.append(ob_info)

            return recent_obs

        except Exception as e:
            logger.error(f"Error getting recent order blocks: {str(e)}")
            return []
