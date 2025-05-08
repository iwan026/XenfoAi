import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileFeatures:
    """Container for Volume Profile features"""

    poc_price: float
    poc_strength: float
    value_area_high: float
    value_area_low: float
    in_value_area: bool
    volume_delta: float
    buying_pressure: float
    selling_pressure: float
    volume_imbalance: float
    volume_trend: int  # 1: increasing, -1: decreasing, 0: neutral


class VolumeProfileAnalyzer:
    """Enhanced Volume Profile feature generator"""

    def __init__(self, n_levels: int = 12):
        self.feature_columns = []
        self.n_levels = n_levels
        self.lookback_period = 20
        self.value_area_percentage = 0.70  # 70% of volume
        self.volume_threshold = 1.5

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive Volume Profile features"""
        try:
            # Validate input data
            if df.empty or not all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            ):
                logger.error("Missing required columns in input data")
                return df

            # Calculate basic volume profile
            df = self._calculate_volume_distribution(df)

            # Calculate Point of Control and Value Area
            df = self._calculate_poc_and_value_area(df)

            # Calculate Volume Delta and Pressure
            df = self._calculate_volume_metrics(df)

            # Calculate Volume Patterns
            df = self._analyze_volume_patterns(df)

            # Update feature columns list
            self.feature_columns = [
                "poc_price",
                "poc_strength",
                "value_area_high",
                "value_area_low",
                "in_value_area",
                "volume_delta",
                "buying_pressure",
                "selling_pressure",
                "volume_imbalance",
                "volume_trend",
            ]

            # Validate calculated features
            df = self._validate_features(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Volume Profile features: {str(e)}")
            return df

    def _calculate_volume_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume distribution across price levels"""
        try:
            # Calculate price range for the period
            price_range = (
                df["high"].rolling(self.lookback_period).max()
                - df["low"].rolling(self.lookback_period).min()
            )
            level_height = price_range / self.n_levels

            # Initialize volume levels
            for i in range(self.n_levels):
                level_name = f"vol_level_{i}"
                df[level_name] = 0.0

            # Calculate volume at each price level
            for i in range(self.n_levels):
                level_low = df["low"] + (i * level_height)
                level_high = level_low + level_height

                # Volume contribution to this level
                df[f"vol_level_{i}"] = np.where(
                    (df["low"] <= level_high) & (df["high"] >= level_low),
                    df["volume"]
                    * (
                        np.minimum(df["high"], level_high)
                        - np.maximum(df["low"], level_low)
                    )
                    / (df["high"] - df["low"]),
                    0,
                )

            return df

        except Exception as e:
            logger.error(f"Error calculating volume distribution: {str(e)}")
            return df

    def _calculate_poc_and_value_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Point of Control and Value Area"""
        try:
            # Get volume level columns
            vol_columns = [f"vol_level_{i}" for i in range(self.n_levels)]

            # Point of Control
            df["poc_level"] = df[vol_columns].idxmax(axis=1)
            df["poc_price"] = df.apply(
                lambda row: self._get_price_from_level(row, row["poc_level"]), axis=1
            )

            # POC Strength
            df["poc_strength"] = df.apply(
                lambda row: row[row["poc_level"]] / row["volume"], axis=1
            )

            # Value Area Calculation
            total_volume = df[vol_columns].sum(axis=1)
            target_volume = total_volume * self.value_area_percentage

            # Calculate VAH and VAL
            df["value_area_high"] = df.apply(
                lambda row: self._calculate_value_area_boundary(
                    row, vol_columns, target_volume.loc[row.name], "high"
                ),
                axis=1,
            )

            df["value_area_low"] = df.apply(
                lambda row: self._calculate_value_area_boundary(
                    row, vol_columns, target_volume.loc[row.name], "low"
                ),
                axis=1,
            )

            # Check if price is within Value Area
            df["in_value_area"] = (df["close"] >= df["value_area_low"]) & (
                df["close"] <= df["value_area_high"]
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating POC and Value Area: {str(e)}")
            return df

    def _calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Delta and Pressure metrics"""
        try:
            # Volume Delta (Buying vs Selling Volume)
            df["volume_delta"] = np.where(
                df["close"] >= df["open"],
                df["volume"],  # Buying volume
                -df["volume"],  # Selling volume
            )

            # Buying Pressure
            df["buying_pressure"] = np.where(
                df["close"] > df["open"],
                df["volume"] * (df["close"] - df["open"]) / (df["high"] - df["low"]),
                0,
            )

            # Selling Pressure
            df["selling_pressure"] = np.where(
                df["close"] < df["open"],
                df["volume"] * (df["open"] - df["close"]) / (df["high"] - df["low"]),
                0,
            )

            # Volume Imbalance
            df["volume_imbalance"] = (
                df["buying_pressure"] - df["selling_pressure"]
            ) / (df["buying_pressure"] + df["selling_pressure"] + 1e-7)

            return df

        except Exception as e:
            logger.error(f"Error calculating volume metrics: {str(e)}")
            return df

    def _analyze_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze Volume Patterns and Trends"""
        try:
            # Volume Trend
            df["volume_trend"] = np.where(
                df["volume"]
                > df["volume"].rolling(self.lookback_period).mean()
                * self.volume_threshold,
                1,  # Increasing volume
                np.where(
                    df["volume"]
                    < df["volume"].rolling(self.lookback_period).mean() * 0.5,
                    -1,  # Decreasing volume
                    0,  # Neutral
                ),
            )

            return df

        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {str(e)}")
            return df

    def _get_price_from_level(self, row: pd.Series, level: str) -> float:
        """Convert volume level to price"""
        try:
            level_idx = int(level.split("_")[-1])
            price_range = row["high"] - row["low"]
            level_height = price_range / self.n_levels
            return row["low"] + (level_idx * level_height) + (level_height / 2)

        except Exception as e:
            logger.error(f"Error getting price from level: {str(e)}")
            return row["close"]

    def _calculate_value_area_boundary(
        self,
        row: pd.Series,
        vol_columns: List[str],
        target_volume: float,
        boundary_type: str,
    ) -> float:
        """Calculate Value Area High or Low boundary"""
        try:
            volumes = [row[col] for col in vol_columns]
            total_vol = 0

            if boundary_type == "high":
                for i, vol in enumerate(volumes):
                    total_vol += vol
                    if total_vol >= target_volume:
                        return self._get_price_from_level(row, f"vol_level_{i}")

            else:  # low
                for i in range(len(volumes) - 1, -1, -1):
                    total_vol += volumes[i]
                    if total_vol >= target_volume:
                        return self._get_price_from_level(row, f"vol_level_{i}")

            return row["close"]

        except Exception as e:
            logger.error(f"Error calculating value area boundary: {str(e)}")
            return row["close"]

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean calculated features"""
        try:
            for column in self.feature_columns:
                # Replace infinities
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values
                df[column] = df[column].fillna(method="ffill").fillna(0)

                # Normalize specific features
                if column in ["volume_imbalance"]:
                    df[column] = df[column].clip(-1, 1)
                elif column in ["poc_strength", "buying_pressure", "selling_pressure"]:
                    df[column] = df[column].clip(0, 1)

            return df

        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns

    def get_volume_profile_features(self, row: pd.Series) -> VolumeProfileFeatures:
        """Get Volume Profile features for a single row"""
        try:
            return VolumeProfileFeatures(
                poc_price=float(row["poc_price"]),
                poc_strength=float(row["poc_strength"]),
                value_area_high=float(row["value_area_high"]),
                value_area_low=float(row["value_area_low"]),
                in_value_area=bool(row["in_value_area"]),
                volume_delta=float(row["volume_delta"]),
                buying_pressure=float(row["buying_pressure"]),
                selling_pressure=float(row["selling_pressure"]),
                volume_imbalance=float(row["volume_imbalance"]),
                volume_trend=int(row["volume_trend"]),
            )
        except Exception as e:
            logger.error(f"Error creating Volume Profile features: {str(e)}")
            return None
