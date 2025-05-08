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
class SMCParameters:
    """Parameters for Smart Money Concepts calculations"""

    # Fair Value Gaps
    fvg_threshold: float = 0.0015  # Minimum gap size (0.15%)
    fvg_lookback: int = 20  # Periods to look back for gaps

    # Price Inefficiency
    inefficiency_threshold: float = 0.001  # Minimum inefficiency (0.1%)
    inefficiency_window: int = 14  # Window for inefficiency calc

    # Money Flow
    flow_period: int = 20  # Period for money flow calculation
    flow_threshold: float = 0.7  # Threshold for significant flow (70%)

    # Divergence
    divergence_lookback: int = 14  # Periods for divergence detection
    divergence_threshold: float = 0.5  # Minimum divergence significance

    # Volume
    volume_ma_period: int = 20  # Moving average period for volume
    volume_threshold: float = 1.5  # Volume spike threshold


class SmartMoneyFeatures:
    """Main class for Smart Money Concepts feature calculations"""

    def __init__(self, params: Optional[SMCParameters] = None):
        """Initialize SMC feature calculator with parameters"""
        self.params = params or SMCParameters()
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
            # Fair Value Gaps
            "bull_fvg",
            "bear_fvg",
            "fvg_size",
            "fvg_distance",
            # Price Inefficiency
            "price_inefficiency",
            "inefficiency_score",
            # Money Flow
            "smart_money_flow",
            "flow_strength",
            "flow_direction",
            # Divergence
            "price_divergence",
            "divergence_strength",
            # Volume Analysis
            "institutional_volume",
            "volume_score",
        ]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Smart Money Concepts features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional SMC features
        """
        try:
            # Validate input
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in input DataFrame")

            # Create copy to prevent modifications to original
            df = df.copy()

            # Calculate individual feature components
            df = self._calculate_fair_value_gaps(df)
            df = self._calculate_price_inefficiency(df)
            df = self._calculate_money_flow(df)
            df = self._calculate_smart_money_divergence(df)
            df = self._calculate_volume_analysis(df)

            # Log calculation metadata
            self._log_calculation(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating SMC features: {str(e)}")
            # Return original DataFrame if calculation fails
            return df

    def _calculate_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fair Value Gaps (FVG)"""
        try:
            # Bullish FVG: Low[1] > High[-1]
            df["bull_fvg"] = (
                (df["low"].shift(1) > df["high"].shift(-1))
                & (df["close"].shift(1) > df["open"].shift(1))  # Bearish candle
                & (df["close"].shift(-1) < df["open"].shift(-1))  # Bearish candle
            )

            # Bearish FVG: High[1] < Low[-1]
            df["bear_fvg"] = (
                (df["high"].shift(1) < df["low"].shift(-1))
                & (df["close"].shift(1) < df["open"].shift(1))  # Bullish candle
                & (df["close"].shift(-1) > df["open"].shift(-1))  # Bullish candle
            )

            # Calculate FVG size
            df["fvg_size"] = np.where(
                df["bull_fvg"],
                df["low"].shift(1) - df["high"].shift(-1),
                np.where(df["bear_fvg"], df["low"].shift(-1) - df["high"].shift(1), 0),
            )

            # Calculate distance from current price to FVG
            df["fvg_distance"] = self._calculate_fvg_distance(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Fair Value Gaps: {str(e)}")
            return df

    def _calculate_price_inefficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Price Inefficiencies"""
        try:
            window = self.params.inefficiency_window

            # Calculate price swings
            df["swing_high"] = df["high"].rolling(window, center=True).max()
            df["swing_low"] = df["low"].rolling(window, center=True).min()

            # Calculate inefficiency
            df["price_inefficiency"] = (df["swing_high"] - df["swing_low"]) / df[
                "close"
            ].rolling(window).mean()

            # Calculate inefficiency score
            df["inefficiency_score"] = (
                df["price_inefficiency"].rolling(window).mean()
                / df["price_inefficiency"].rolling(window).std()
            )

            # Clean up temporary columns
            df = df.drop(["swing_high", "swing_low"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error calculating Price Inefficiency: {str(e)}")
            return df

    def _calculate_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Smart Money Flow"""
        try:
            period = self.params.flow_period

            # Calculate typical price
            df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

            # Calculate money flow
            df["money_flow"] = (
                df["typical_price"]
                * df["volume"]
                * ((df["close"] - df["open"]) / (df["high"] - df["low"]))
            )

            # Calculate positive and negative money flow
            df["positive_flow"] = np.where(
                df["typical_price"] > df["typical_price"].shift(1), df["money_flow"], 0
            )

            df["negative_flow"] = np.where(
                df["typical_price"] < df["typical_price"].shift(1), df["money_flow"], 0
            )

            # Calculate money flow index
            df["smart_money_flow"] = (
                df["positive_flow"].rolling(period).sum()
                / df["negative_flow"].rolling(period).sum()
            )

            # Calculate flow strength and direction
            df["flow_strength"] = abs(df["smart_money_flow"] - 1)
            df["flow_direction"] = np.sign(df["smart_money_flow"] - 1)

            # Clean up temporary columns
            df = df.drop(
                ["typical_price", "money_flow", "positive_flow", "negative_flow"],
                axis=1,
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating Money Flow: {str(e)}")
            return df

    def _calculate_smart_money_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Smart Money Divergence"""
        try:
            lookback = self.params.divergence_lookback

            # Calculate price momentum
            df["price_momentum"] = (df["close"] - df["close"].shift(lookback)) / df[
                "close"
            ].shift(lookback)

            # Calculate volume momentum
            df["volume_momentum"] = (df["volume"] - df["volume"].shift(lookback)) / df[
                "volume"
            ].shift(lookback)

            # Calculate divergence
            df["price_divergence"] = (
                df["price_momentum"] * -1 * np.sign(df["volume_momentum"])
            )

            # Calculate divergence strength
            df["divergence_strength"] = abs(
                df["price_momentum"] - df["volume_momentum"]
            )

            # Clean up temporary columns
            df = df.drop(["price_momentum", "volume_momentum"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error calculating Smart Money Divergence: {str(e)}")
            return df

    def _calculate_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Institutional Volume Analysis"""
        try:
            period = self.params.volume_ma_period
            threshold = self.params.volume_threshold

            # Calculate volume moving average
            df["volume_ma"] = df["volume"].rolling(period).mean()

            # Identify institutional volume
            df["institutional_volume"] = df["volume"] > (df["volume_ma"] * threshold)

            # Calculate volume score
            df["volume_score"] = (
                df["volume"]
                / df["volume_ma"]
                * np.where(df["close"] > df["open"], 1, -1)
            )

            # Clean up temporary columns
            df = df.drop(["volume_ma"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error calculating Volume Analysis: {str(e)}")
            return df

    def _calculate_fvg_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance from current price to nearest FVG"""
        try:
            current_price = df["close"]

            # Find nearest bullish FVG
            bull_fvg_price = np.where(
                df["bull_fvg"], (df["low"].shift(1) + df["high"].shift(-1)) / 2, np.nan
            )

            # Find nearest bearish FVG
            bear_fvg_price = np.where(
                df["bear_fvg"], (df["high"].shift(1) + df["low"].shift(-1)) / 2, np.nan
            )

            # Calculate distances
            bull_distance = abs(current_price - bull_fvg_price)
            bear_distance = abs(current_price - bear_fvg_price)

            # Return minimum distance
            return pd.Series(
                np.nanmin([bull_distance, bear_distance], axis=0), index=df.index
            )

        except Exception as e:
            logger.error(f"Error calculating FVG distance: {str(e)}")
            return pd.Series(0, index=df.index)

    def _log_calculation(self, df: pd.DataFrame) -> None:
        """Log calculation metadata"""
        try:
            log_entry = {
                "timestamp": TradingConfig.get_current_timestamp(),
                "rows_processed": len(df),
                "features_calculated": len(self.feature_columns),
                "parameters": self.params.__dict__,
                "version": TradingConfig.VERSION,
            }

            log_file = TradingConfig.LOG_DIR / "smc_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    def get_feature_names(self) -> List[str]:
        """Get list of all SMC feature names"""
        return self.feature_columns

    def get_feature_snapshot(self, row: pd.Series) -> Dict:
        """Get snapshot of SMC features for a single row"""
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
        """Get calculator metadata"""
        return {
            **self.metadata,
            "parameters": self.params.__dict__,
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
        }
