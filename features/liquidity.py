import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LiquidityFeatures:
    """Container for Liquidity Analysis features"""

    buy_liquidity: float
    sell_liquidity: float
    liquidity_imbalance: float
    liquidity_sweep_up: bool
    liquidity_sweep_down: bool
    stop_hunt_level: float
    manipulation_score: float
    institutional_level: float
    liquidity_efficiency: float
    smart_money_activity: float


class LiquidityAnalyzer:
    """Enhanced Liquidity Analysis feature generator"""

    def __init__(self):
        self.feature_columns = []
        self.lookback_period = 20
        self.volume_threshold = 1.5
        self.sweep_threshold = 2.0
        self.created_at = "2025-05-08 12:24:10"
        self.created_by = "Xentrovt"

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive Liquidity Analysis features"""
        try:
            # Validate input data
            if df.empty or not all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            ):
                logger.error("Missing required columns in input data")
                return df

            # Calculate Basic Liquidity Levels
            df = self._calculate_liquidity_levels(df)

            # Detect Liquidity Sweeps
            df = self._detect_liquidity_sweeps(df)

            # Analyze Manipulation Zones
            df = self._analyze_manipulation(df)

            # Calculate Smart Money Activity
            df = self._calculate_smart_money_activity(df)

            # Update feature columns list
            self.feature_columns = [
                "buy_liquidity",
                "sell_liquidity",
                "liquidity_imbalance",
                "liquidity_sweep_up",
                "liquidity_sweep_down",
                "stop_hunt_level",
                "manipulation_score",
                "institutional_level",
                "liquidity_efficiency",
                "smart_money_activity",
            ]

            # Validate features
            df = self._validate_features(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Liquidity Analysis features: {str(e)}")
            return df

    def _calculate_liquidity_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic liquidity levels"""
        try:
            # Buy Side Liquidity
            df["buy_liquidity"] = df["volume"].where(
                (df["close"] > df["open"])
                & (df["volume"] > df["volume"].rolling(self.lookback_period).mean()),
                0,
            )

            # Sell Side Liquidity
            df["sell_liquidity"] = df["volume"].where(
                (df["close"] < df["open"])
                & (df["volume"] > df["volume"].rolling(self.lookback_period).mean()),
                0,
            )

            # Liquidity Imbalance
            df["liquidity_imbalance"] = (df["buy_liquidity"] - df["sell_liquidity"]) / (
                df["buy_liquidity"] + df["sell_liquidity"] + 1e-7
            )

            # Liquidity Efficiency
            df["liquidity_efficiency"] = df.apply(
                lambda x: self._calculate_efficiency(x), axis=1
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating liquidity levels: {str(e)}")
            return df

    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect liquidity sweeps and stop hunts"""
        try:
            # Upward Sweep Detection
            df["liquidity_sweep_up"] = (
                (df["high"] > df["high"].rolling(self.lookback_period).max().shift(1))
                & (df["close"] < df["open"])
                & (
                    df["volume"]
                    > df["volume"].rolling(self.lookback_period).mean()
                    * self.sweep_threshold
                )
            )

            # Downward Sweep Detection
            df["liquidity_sweep_down"] = (
                (df["low"] < df["low"].rolling(self.lookback_period).min().shift(1))
                & (df["close"] > df["open"])
                & (
                    df["volume"]
                    > df["volume"].rolling(self.lookback_period).mean()
                    * self.sweep_threshold
                )
            )

            # Stop Hunt Levels
            df["stop_hunt_level"] = np.where(
                df["liquidity_sweep_up"],
                df["high"],
                np.where(df["liquidity_sweep_down"], df["low"], np.nan),
            )

            return df

        except Exception as e:
            logger.error(f"Error detecting liquidity sweeps: {str(e)}")
            return df

    def _analyze_manipulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze manipulation zones and institutional activity"""
        try:
            # Manipulation Score
            df["manipulation_score"] = (
                (df["liquidity_sweep_up"] | df["liquidity_sweep_down"]).astype(float)
                * 0.4
                + (df["volume"] / df["volume"].rolling(self.lookback_period).mean())
                * 0.3
                + abs(df["liquidity_imbalance"]) * 0.3
            )

            # Institutional Levels
            df["institutional_level"] = self._calculate_institutional_levels(df)

            return df

        except Exception as e:
            logger.error(f"Error analyzing manipulation: {str(e)}")
            return df

    def _calculate_smart_money_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate smart money activity indicators"""
        try:
            # Smart Money Activity Score
            df["smart_money_activity"] = (
                df["manipulation_score"] * 0.3
                + abs(df["liquidity_imbalance"]) * 0.3
                + df["liquidity_efficiency"] * 0.4
            )

            # Normalize score
            df["smart_money_activity"] = (
                df["smart_money_activity"]
                - df["smart_money_activity"].rolling(self.lookback_period).min()
            ) / (
                df["smart_money_activity"].rolling(self.lookback_period).max()
                - df["smart_money_activity"].rolling(self.lookback_period).min()
                + 1e-7
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating smart money activity: {str(e)}")
            return df

    def _calculate_efficiency(self, row: pd.Series) -> float:
        """Calculate liquidity efficiency for a single row"""
        try:
            price_range = row["high"] - row["low"]
            if price_range == 0:
                return 0

            price_move = abs(row["close"] - row["open"])
            volume_ratio = (
                row["volume"] / row["volume_sma"] if "volume_sma" in row else 1
            )

            efficiency = (price_move / price_range) * volume_ratio
            return min(efficiency, 1.0)

        except Exception as e:
            logger.error(f"Error calculating efficiency: {str(e)}")
            return 0.0

    def _calculate_institutional_levels(self, df: pd.DataFrame) -> pd.Series:
        """Calculate institutional activity levels"""
        try:
            # Identify high volume nodes
            volume_threshold = (
                df["volume"].rolling(self.lookback_period).mean()
                * self.volume_threshold
            )

            # Calculate institutional levels based on high volume and price rejection
            inst_levels = np.where(
                (df["volume"] > volume_threshold)
                & (abs(df["close"] - df["open"]) < abs(df["high"] - df["low"]) * 0.3),
                (df["high"] + df["low"]) / 2,
                np.nan,
            )

            return pd.Series(inst_levels, index=df.index)

        except Exception as e:
            logger.error(f"Error calculating institutional levels: {str(e)}")
            return pd.Series(np.nan, index=df.index)

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean calculated features"""
        try:
            for column in self.feature_columns:
                # Replace infinities
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values
                df[column] = df[column].fillna(method="ffill").fillna(0)

                # Normalize specific features
                if column in ["liquidity_imbalance"]:
                    df[column] = df[column].clip(-1, 1)
                elif column in [
                    "manipulation_score",
                    "smart_money_activity",
                    "liquidity_efficiency",
                ]:
                    df[column] = df[column].clip(0, 1)

            return df

        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns

    def get_liquidity_features(self, row: pd.Series) -> LiquidityFeatures:
        """Get Liquidity Analysis features for a single row"""
        try:
            return LiquidityFeatures(
                buy_liquidity=float(row["buy_liquidity"]),
                sell_liquidity=float(row["sell_liquidity"]),
                liquidity_imbalance=float(row["liquidity_imbalance"]),
                liquidity_sweep_up=bool(row["liquidity_sweep_up"]),
                liquidity_sweep_down=bool(row["liquidity_sweep_down"]),
                stop_hunt_level=float(row["stop_hunt_level"]),
                manipulation_score=float(row["manipulation_score"]),
                institutional_level=float(row["institutional_level"]),
                liquidity_efficiency=float(row["liquidity_efficiency"]),
                smart_money_activity=float(row["smart_money_activity"]),
            )
        except Exception as e:
            logger.error(f"Error creating Liquidity features: {str(e)}")
            return None

    def analyze_liquidity_profile(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        """Analyze recent liquidity profile"""
        try:
            recent_data = df.tail(lookback)

            profile = {
                "timestamp": self.created_at,
                "analysis_by": self.created_by,
                "liquidity_state": {
                    "buy_side_dominance": (
                        recent_data["buy_liquidity"] > recent_data["sell_liquidity"]
                    ).mean(),
                    "average_imbalance": recent_data["liquidity_imbalance"].mean(),
                    "sweep_activity": (
                        recent_data["liquidity_sweep_up"]
                        | recent_data["liquidity_sweep_down"]
                    ).sum(),
                    "manipulation_risk": recent_data["manipulation_score"].mean(),
                    "smart_money_presence": recent_data["smart_money_activity"].mean(),
                },
                "key_levels": {
                    "recent_sweep_levels": recent_data[
                        recent_data["stop_hunt_level"].notna()
                    ]["stop_hunt_level"].tolist(),
                    "institutional_levels": recent_data[
                        recent_data["institutional_level"].notna()
                    ]["institutional_level"].tolist(),
                },
            }

            return profile

        except Exception as e:
            logger.error(f"Error analyzing liquidity profile: {str(e)}")
            return {}
