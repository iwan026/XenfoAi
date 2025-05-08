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
class OrderBlockParameters:
    """Parameters for Order Block detection and analysis"""

    # Order Block Detection
    min_ob_size: float = 0.001  # Minimum size relative to ATR
    max_ob_size: float = 0.05  # Maximum size relative to ATR
    lookback_period: int = 20  # Periods to look back
    volume_threshold: float = 1.5  # Volume multiplier for significance

    # Break of Structure
    bos_threshold: float = 0.002  # Minimum break size
    bos_confirmation: int = 3  # Candles for confirmation

    # Mitigation
    mitigation_threshold: float = 0.7  # Percentage for mitigation
    partial_mitigation: float = 0.5  # Threshold for partial mitigation

    # Strength Analysis
    volume_weight: float = 0.4  # Weight for volume component
    size_weight: float = 0.3  # Weight for size component
    time_weight: float = 0.3  # Weight for recency component


class OrderBlockAnalyzer:
    """Main class for Order Block detection and analysis"""

    def __init__(self, params: Optional[OrderBlockParameters] = None):
        """Initialize Order Block analyzer with parameters"""
        self.params = params or OrderBlockParameters()
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
            # Order Block Detection
            "bull_ob",
            "bear_ob",
            "ob_size",
            "ob_volume_ratio",
            # Break of Structure
            "bos_up",
            "bos_down",
            "bos_strength",
            # Mitigation
            "ob_mitigation",
            "mitigation_type",
            # Strength Analysis
            "ob_strength",
            "ob_quality",
            "ob_age",
        ]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Order Block related features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional Order Block features
        """
        try:
            # Validate input
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in input DataFrame")

            # Create copy to prevent modifications to original
            df = df.copy()

            # Calculate ATR for relative size measurements
            df["atr"] = self._calculate_atr(df)

            # Calculate main features
            df = self._detect_order_blocks(df)
            df = self._analyze_break_of_structure(df)
            df = self._track_mitigation(df)
            df = self._calculate_strength(df)

            # Clean up temporary columns
            df = df.drop(["atr"], axis=1)

            # Log calculation metadata
            self._log_calculation(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Order Block features: {str(e)}")
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

    def _detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Bullish and Bearish Order Blocks"""
        try:
            # Calculate volume moving average
            df["volume_ma"] = df["volume"].rolling(self.params.lookback_period).mean()

            # Bullish Order Blocks
            df["bull_ob"] = (
                (df["close"] < df["open"])  # Bearish candle
                & (df["close"].shift(-1) > df["high"])  # Break of high
                & (
                    df["volume"] > df["volume_ma"] * self.params.volume_threshold
                )  # High volume
                & (
                    abs(df["close"] - df["open"]) > df["atr"] * self.params.min_ob_size
                )  # Minimum size
                & (
                    abs(df["close"] - df["open"]) < df["atr"] * self.params.max_ob_size
                )  # Maximum size
            )

            # Bearish Order Blocks
            df["bear_ob"] = (
                (df["close"] > df["open"])  # Bullish candle
                & (df["close"].shift(-1) < df["low"])  # Break of low
                & (
                    df["volume"] > df["volume_ma"] * self.params.volume_threshold
                )  # High volume
                & (
                    abs(df["close"] - df["open"]) > df["atr"] * self.params.min_ob_size
                )  # Minimum size
                & (
                    abs(df["close"] - df["open"]) < df["atr"] * self.params.max_ob_size
                )  # Maximum size
            )

            # Calculate Order Block size
            df["ob_size"] = abs(df["close"] - df["open"]) / df["atr"]

            # Calculate volume ratio
            df["ob_volume_ratio"] = df["volume"] / df["volume_ma"]

            # Clean up
            df = df.drop(["volume_ma"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error detecting Order Blocks: {str(e)}")
            return df

    def _analyze_break_of_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze breaks of market structure"""
        try:
            # Calculate local highs and lows
            df["local_high"] = df["high"].rolling(self.params.bos_confirmation).max()
            df["local_low"] = df["low"].rolling(self.params.bos_confirmation).min()

            # Detect breaks
            df["bos_up"] = (
                (df["high"] > df["local_high"].shift(1))
                & (df["close"] > df["open"])
                & (
                    (df["high"] - df["local_high"].shift(1))
                    > df["atr"] * self.params.bos_threshold
                )
            )

            df["bos_down"] = (
                (df["low"] < df["local_low"].shift(1))
                & (df["close"] < df["open"])
                & (
                    (df["local_low"].shift(1) - df["low"])
                    > df["atr"] * self.params.bos_threshold
                )
            )

            # Calculate break strength
            df["bos_strength"] = np.where(
                df["bos_up"],
                (df["high"] - df["local_high"].shift(1)) / df["atr"],
                np.where(
                    df["bos_down"],
                    (df["local_low"].shift(1) - df["low"]) / df["atr"],
                    0,
                ),
            )

            # Clean up
            df = df.drop(["local_high", "local_low"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing break of structure: {str(e)}")
            return df

    def _track_mitigation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track Order Block mitigation"""
        try:
            # Initialize mitigation columns
            df["ob_mitigation"] = 0.0
            df["mitigation_type"] = "none"

            # Track bullish OB mitigation
            bull_ob_mask = df["bull_ob"]
            if bull_ob_mask.any():
                for idx in df.index[bull_ob_mask]:
                    ob_low = df.loc[idx, "low"]
                    ob_high = df.loc[idx, "high"]
                    future_prices = df.loc[idx:, "low"]

                    mitigation = (ob_high - future_prices.min()) / (ob_high - ob_low)
                    df.loc[idx, "ob_mitigation"] = max(min(mitigation, 1.0), 0.0)

                    if mitigation >= self.params.mitigation_threshold:
                        df.loc[idx, "mitigation_type"] = "full"
                    elif mitigation >= self.params.partial_mitigation:
                        df.loc[idx, "mitigation_type"] = "partial"

            # Track bearish OB mitigation
            bear_ob_mask = df["bear_ob"]
            if bear_ob_mask.any():
                for idx in df.index[bear_ob_mask]:
                    ob_low = df.loc[idx, "low"]
                    ob_high = df.loc[idx, "high"]
                    future_prices = df.loc[idx:, "high"]

                    mitigation = (future_prices.max() - ob_low) / (ob_high - ob_low)
                    df.loc[idx, "ob_mitigation"] = max(min(mitigation, 1.0), 0.0)

                    if mitigation >= self.params.mitigation_threshold:
                        df.loc[idx, "mitigation_type"] = "full"
                    elif mitigation >= self.params.partial_mitigation:
                        df.loc[idx, "mitigation_type"] = "partial"

            return df

        except Exception as e:
            logger.error(f"Error tracking mitigation: {str(e)}")
            return df

    def _calculate_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Order Block strength and quality metrics"""
        try:
            # Calculate components
            volume_component = df["ob_volume_ratio"] * self.params.volume_weight
            size_component = df["ob_size"] * self.params.size_weight

            # Calculate age factor (decreasing with time)
            df["ob_age"] = np.arange(len(df))[::-1]
            time_component = (1 - (df["ob_age"] / len(df))) * self.params.time_weight

            # Calculate overall strength
            df["ob_strength"] = volume_component + size_component + time_component

            # Calculate quality score
            df["ob_quality"] = df["ob_strength"] * (1 - df["ob_mitigation"])

            return df

        except Exception as e:
            logger.error(f"Error calculating strength metrics: {str(e)}")
            return df

    def _log_calculation(self, df: pd.DataFrame) -> None:
        """Log calculation metadata"""
        try:
            log_entry = {
                "timestamp": TradingConfig.get_current_timestamp(),
                "rows_processed": len(df),
                "bull_obs_detected": df["bull_ob"].sum(),
                "bear_obs_detected": df["bear_ob"].sum(),
                "parameters": self.params.__dict__,
                "version": TradingConfig.VERSION,
            }

            log_file = TradingConfig.LOG_DIR / "order_block_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    def get_feature_names(self) -> List[str]:
        """Get list of all Order Block feature names"""
        return self.feature_columns

    def get_feature_snapshot(self, row: pd.Series) -> Dict:
        """Get snapshot of Order Block features for a single row"""
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
