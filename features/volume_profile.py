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
class VolumeProfileParameters:
    """Parameters for Volume Profile analysis"""

    # Value Area
    value_area_volume: float = 0.70  # Percentage of volume for VA (70%)
    price_levels: int = 24  # Number of price levels
    min_volume_threshold: float = 0.01  # Minimum volume for level significance

    # Volume Delta
    delta_period: int = 20  # Period for delta calculations
    delta_threshold: float = 0.6  # Threshold for significant deltas

    # Acceptance/Rejection
    acceptance_threshold: float = 0.65  # Threshold for price acceptance
    rejection_period: int = 5  # Periods to confirm rejection

    # Distribution
    distribution_period: int = 50  # Period for distribution analysis
    distribution_threshold: float = 0.4  # Threshold for distribution patterns


class VolumeProfileAnalyzer:
    """Main class for Volume Profile analysis"""

    def __init__(self, params: Optional[VolumeProfileParameters] = None):
        """Initialize Volume Profile analyzer with parameters"""
        self.params = params or VolumeProfileParameters()
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
            # Value Area
            "vah",
            "val",
            "poc",
            "value_area_range",
            # Volume Delta
            "volume_delta",
            "delta_strength",
            "cumulative_delta",
            # Acceptance/Rejection
            "price_acceptance",
            "rejection_strength",
            # Distribution
            "distribution_type",
            "distribution_strength",
            # Volume Nodes
            "high_volume_node",
            "low_volume_node",
        ]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Volume Profile related features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional Volume Profile features
        """
        try:
            # Validate input
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in input DataFrame")

            # Create copy to prevent modifications to original
            df = df.copy()

            # Calculate main features
            df = self._calculate_value_area(df)
            df = self._calculate_volume_delta(df)
            df = self._analyze_acceptance_rejection(df)
            df = self._analyze_distribution(df)
            df = self._identify_volume_nodes(df)

            # Log calculation metadata
            self._log_calculation(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Volume Profile features: {str(e)}")
            return df

    def _calculate_value_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Value Area High, Low, and Point of Control"""
        try:
            window = self.params.distribution_period

            for i in range(len(df) - window + 1):
                subset = df.iloc[i : i + window]

                # Create price levels
                price_range = subset["high"].max() - subset["low"].min()
                level_height = price_range / self.params.price_levels

                # Initialize volume array for each price level
                volumes = np.zeros(self.params.price_levels)

                # Distribute volume across price levels
                for _, row in subset.iterrows():
                    price_span = row["high"] - row["low"]
                    vol_per_level = row["volume"] / (price_span / level_height)

                    start_level = int((row["low"] - subset["low"].min()) / level_height)
                    end_level = int((row["high"] - subset["low"].min()) / level_height)

                    volumes[start_level : end_level + 1] += vol_per_level

                # Find POC (Point of Control)
                poc_level = np.argmax(volumes)
                poc_price = subset["low"].min() + (poc_level + 0.5) * level_height

                # Calculate Value Area
                total_volume = np.sum(volumes)
                target_volume = total_volume * self.params.value_area_volume
                current_volume = volumes[poc_level]

                upper_level = lower_level = poc_level
                while current_volume < target_volume and (
                    upper_level < len(volumes) - 1 or lower_level > 0
                ):
                    if upper_level < len(volumes) - 1 and (
                        lower_level == 0
                        or volumes[upper_level + 1] > volumes[lower_level - 1]
                    ):
                        upper_level += 1
                        current_volume += volumes[upper_level]
                    elif lower_level > 0:
                        lower_level -= 1
                        current_volume += volumes[lower_level]

                # Set values for the current row
                current_idx = df.index[i + window - 1]
                df.loc[current_idx, "poc"] = poc_price
                df.loc[current_idx, "vah"] = (
                    subset["low"].min() + (upper_level + 1) * level_height
                )
                df.loc[current_idx, "val"] = (
                    subset["low"].min() + lower_level * level_height
                )
                df.loc[current_idx, "value_area_range"] = (
                    df.loc[current_idx, "vah"] - df.loc[current_idx, "val"]
                )

            return df

        except Exception as e:
            logger.error(f"Error calculating Value Area: {str(e)}")
            return df

    def _calculate_volume_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Delta metrics using optimized tick volume"""
        try:
            # Calculate buying and selling volume based on optimized tick volume
            df["buy_volume"] = np.where(
                df["close"] > df["open"],
                df["tick_volume"]
                * (df["close"] - df["open"])
                / (df["high"] - df["low"]),
                0,
            )

            df["sell_volume"] = np.where(
                df["close"] < df["open"],
                df["tick_volume"]
                * (df["open"] - df["close"])
                / (df["high"] - df["low"]),
                0,
            )

            # Calculate delta with price impact
            df["volume_delta"] = (
                df["buy_volume"].rolling(self.params.delta_period).sum()
                - df["sell_volume"].rolling(self.params.delta_period).sum()
            ) / df["tick_volume"].rolling(self.params.delta_period).sum()

            # Calculate delta strength considering price movement
            df["delta_strength"] = abs(df["volume_delta"]) * (
                (df["high"] - df["low"]) / df["close"]
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating Volume Delta: {str(e)}")
            return df


def _analyze_acceptance_rejection(self, df: pd.DataFrame) -> pd.DataFrame:
    """Analyze price acceptance and rejection zones"""
    try:
        window = self.params.rejection_period

        # Calculate price movement persistence
        df["price_persistence"] = (
            (df["close"] - df["close"].shift(window)) / df["close"].shift(window)
        ).abs()

        # Calculate volume contribution using tick volume
        df["volume_contribution"] = (
            df["tick_volume"] / df["tick_volume"].rolling(window).sum()
        )

        # Calculate price acceptance
        df["price_acceptance"] = np.where(
            (df["price_persistence"] < self.params.acceptance_threshold)
            & (df["volume_contribution"] > self.params.min_volume_threshold),
            1,
            0,
        )

        return df

    except Exception as e:
        logger.error(f"Error analyzing acceptance/rejection: {str(e)}")
        return df

    def _analyze_acceptance_rejection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze price acceptance and rejection zones"""
        try:
            window = self.params.rejection_period

            # Calculate price movement persistence
            df["price_persistence"] = (
                (df["close"] - df["close"].shift(window)) / df["close"].shift(window)
            ).abs()

            # Calculate volume contribution
            df["volume_contribution"] = (
                df["volume"] / df["volume"].rolling(window).sum()
            )

            # Calculate price acceptance
            df["price_acceptance"] = np.where(
                (df["price_persistence"] < self.params.acceptance_threshold)
                & (df["volume_contribution"] > self.params.min_volume_threshold),
                1,
                0,
            )

            # Calculate rejection strength
            df["rejection_strength"] = np.where(
                df["price_acceptance"] == 0,
                df["volume_contribution"] * df["price_persistence"],
                0,
            )

            # Clean up
            df = df.drop(["price_persistence", "volume_contribution"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing acceptance/rejection: {str(e)}")
            return df

    def _analyze_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze volume distribution patterns"""
        try:
            window = self.params.distribution_period

            # Calculate volume distribution metrics
            df["vol_std"] = df["volume"].rolling(window).std()
            df["vol_mean"] = df["volume"].rolling(window).mean()
            df["vol_skew"] = df["volume"].rolling(window).skew()

            # Classify distribution patterns
            conditions = [
                (df["vol_skew"] > self.params.distribution_threshold),
                (df["vol_skew"] < -self.params.distribution_threshold),
                (abs(df["vol_skew"]) <= self.params.distribution_threshold),
            ]
            choices = ["accumulation", "distribution", "balanced"]

            df["distribution_type"] = np.select(conditions, choices, default="unknown")

            # Calculate distribution strength
            df["distribution_strength"] = abs(df["vol_skew"])

            # Clean up
            df = df.drop(["vol_std", "vol_mean", "vol_skew"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing distribution: {str(e)}")
            return df

    def _identify_volume_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify high and low volume nodes"""
        try:
            window = self.params.distribution_period

            # Calculate volume profile
            df["vol_sma"] = df["volume"].rolling(window).mean()
            df["vol_std"] = df["volume"].rolling(window).std()

            # Identify high volume nodes
            df["high_volume_node"] = df["volume"] > (df["vol_sma"] + df["vol_std"])

            # Identify low volume nodes
            df["low_volume_node"] = df["volume"] < (df["vol_sma"] - df["vol_std"])

            # Clean up
            df = df.drop(["vol_sma", "vol_std"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error identifying volume nodes: {str(e)}")
            return df

    def _log_calculation(self, df: pd.DataFrame) -> None:
        """Log calculation metadata"""
        try:
            log_entry = {
                "timestamp": TradingConfig.get_current_timestamp(),
                "rows_processed": len(df),
                "parameters": self.params.__dict__,
                "version": TradingConfig.VERSION,
                "stats": {
                    "high_volume_nodes": df["high_volume_node"].sum(),
                    "low_volume_nodes": df["low_volume_node"].sum(),
                    "acceptance_zones": df["price_acceptance"].sum(),
                },
            }

            log_file = TradingConfig.LOG_DIR / "volume_profile_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    def get_feature_names(self) -> List[str]:
        """Get list of all Volume Profile feature names"""
        return self.feature_columns

    def get_feature_snapshot(self, row: pd.Series) -> Dict:
        """Get snapshot of Volume Profile features for a single row"""
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
