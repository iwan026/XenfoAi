import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MarketStructureFeatures:
    """Container for Market Structure features"""

    trend_state: str  # uptrend, downtrend, ranging
    structure_break: bool
    higher_high: bool
    lower_low: bool
    swing_high: float
    swing_low: float
    trend_strength: float
    breakout_level: float
    market_context: str
    structure_quality: float


class MarketStructureAnalyzer:
    """Enhanced Market Structure feature generator"""

    def __init__(self):
        self.feature_columns = []
        self.lookback_period = 20
        self.swing_threshold = 2  # Number of candles for swing point confirmation
        self.trend_threshold = 0.6  # Minimum trend strength threshold
        self.created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        self.created_by = "iwan026"

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive Market Structure features"""
        try:
            # Validate input data
            if df.empty or not all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            ):
                logger.error("Missing required columns in input data")
                return df

            # Calculate Swing Points
            df = self._identify_swing_points(df)

            # Analyze Market Structure
            df = self._analyze_market_structure(df)

            # Calculate Structure Breaks
            df = self._calculate_structure_breaks(df)

            # Analyze Market Context
            df = self._analyze_market_context(df)

            # Update feature columns list
            self.feature_columns = [
                "trend_state",
                "structure_break",
                "higher_high",
                "lower_low",
                "swing_high",
                "swing_low",
                "trend_strength",
                "breakout_level",
                "market_context",
                "structure_quality",
            ]

            # Validate features
            df = self._validate_features(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Market Structure features: {str(e)}")
            return df

    def _identify_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify swing highs and lows"""
        try:
            # Swing High Detection
            df["swing_high"] = (
                (df["high"] > df["high"].shift(1))
                & (df["high"] > df["high"].shift(-1))
                & (df["high"] > df["high"].shift(2))
                & (df["high"] > df["high"].shift(-2))
            )

            # Swing Low Detection
            df["swing_low"] = (
                (df["low"] < df["low"].shift(1))
                & (df["low"] < df["low"].shift(-1))
                & (df["low"] < df["low"].shift(2))
                & (df["low"] < df["low"].shift(-2))
            )

            # Store swing point values
            df["swing_high_value"] = np.where(df["swing_high"], df["high"], np.nan)
            df["swing_low_value"] = np.where(df["swing_low"], df["low"], np.nan)

            # Forward fill swing points
            df["swing_high_value"] = df["swing_high_value"].fillna(method="ffill")
            df["swing_low_value"] = df["swing_low_value"].fillna(method="ffill")

            return df

        except Exception as e:
            logger.error(f"Error identifying swing points: {str(e)}")
            return df

    def _analyze_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze market structure and trends"""
        try:
            # Higher Highs & Lower Lows
            df["higher_high"] = df["swing_high"] & (
                df["high"] > df["swing_high_value"].shift(1)
            )

            df["lower_low"] = df["swing_low"] & (
                df["low"] < df["swing_low_value"].shift(1)
            )

            # Trend State Analysis
            df["trend_state"] = self._determine_trend_state(df)

            # Trend Strength
            df["trend_strength"] = self._calculate_trend_strength(df)

            return df

        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return df

    def _determine_trend_state(self, df: pd.DataFrame) -> pd.Series:
        """Determine current trend state"""
        try:
            # Calculate trend based on higher highs and lower lows
            trend_state = pd.Series(index=df.index, data="ranging")  # default state

            # Uptrend conditions
            uptrend = (
                df["higher_high"]
                & (df["close"] > df["close"].rolling(self.lookback_period).mean())
                & (df["low"] > df["low"].shift(1))
            )

            # Downtrend conditions
            downtrend = (
                df["lower_low"]
                & (df["close"] < df["close"].rolling(self.lookback_period).mean())
                & (df["high"] < df["high"].shift(1))
            )

            trend_state[uptrend] = "uptrend"
            trend_state[downtrend] = "downtrend"

            return trend_state

        except Exception as e:
            logger.error(f"Error determining trend state: {str(e)}")
            return pd.Series("ranging", index=df.index)

    def _calculate_structure_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate structure breaks and breakout levels"""
        try:
            # Structure Break Detection
            df["structure_break"] = (
                df["trend_state"] != df["trend_state"].shift(1)
            ) & (df["volume"] > df["volume"].rolling(self.lookback_period).mean())

            # Breakout Level Calculation
            df["breakout_level"] = np.where(
                df["structure_break"] & (df["trend_state"] == "uptrend"),
                df["high"],
                np.where(
                    df["structure_break"] & (df["trend_state"] == "downtrend"),
                    df["low"],
                    np.nan,
                ),
            )

            # Forward fill breakout levels
            df["breakout_level"] = df["breakout_level"].fillna(method="ffill")

            return df

        except Exception as e:
            logger.error(f"Error calculating structure breaks: {str(e)}")
            return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength metric"""
        try:
            # Initialize strength
            strength = pd.Series(0.0, index=df.index)

            # Calculate components
            price_trend = (
                df["close"] - df["close"].rolling(self.lookback_period).mean()
            ) / df["close"].rolling(self.lookback_period).std()

            volume_trend = (
                df["volume"] - df["volume"].rolling(self.lookback_period).mean()
            ) / df["volume"].rolling(self.lookback_period).std()

            # Combine components
            strength = np.abs(price_trend) * 0.7 + np.abs(volume_trend) * 0.3

            # Normalize to 0-1
            strength = (strength - strength.min()) / (strength.max() - strength.min())

            return strength

        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return pd.Series(0.0, index=df.index)

    def _analyze_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze broader market context"""
        try:
            # Market Context Analysis
            df["market_context"] = np.where(
                (df["trend_strength"] > self.trend_threshold)
                & (df["trend_state"] == "uptrend"),
                "strong_uptrend",
                np.where(
                    (df["trend_strength"] > self.trend_threshold)
                    & (df["trend_state"] == "downtrend"),
                    "strong_downtrend",
                    np.where(
                        df["trend_strength"] <= self.trend_threshold,
                        "weak_trend",
                        "ranging",
                    ),
                ),
            )

            # Structure Quality Score
            df["structure_quality"] = (
                df["trend_strength"] * 0.4
                + (df["volume"] / df["volume"].rolling(self.lookback_period).mean())
                * 0.3
                + (df["structure_break"].astype(float) * 0.3)
            )

            return df

        except Exception as e:
            logger.error(f"Error analyzing market context: {str(e)}")
            return df

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean calculated features"""
        try:
            for column in self.feature_columns:
                # Handle categorical columns
                if column in ["trend_state", "market_context"]:
                    df[column] = df[column].fillna("ranging")
                    continue

                # Handle numeric columns
                if column in ["trend_strength", "structure_quality"]:
                    df[column] = df[column].clip(0, 1)

                # Replace infinities
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values
                df[column] = df[column].fillna(method="ffill").fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns

    def get_market_structure_features(self, row: pd.Series) -> MarketStructureFeatures:
        """Get Market Structure features for a single row"""
        try:
            return MarketStructureFeatures(
                trend_state=str(row["trend_state"]),
                structure_break=bool(row["structure_break"]),
                higher_high=bool(row["higher_high"]),
                lower_low=bool(row["lower_low"]),
                swing_high=float(row["swing_high_value"]),
                swing_low=float(row["swing_low_value"]),
                trend_strength=float(row["trend_strength"]),
                breakout_level=float(row["breakout_level"]),
                market_context=str(row["market_context"]),
                structure_quality=float(row["structure_quality"]),
            )
        except Exception as e:
            logger.error(f"Error creating Market Structure features: {str(e)}")
            return None
