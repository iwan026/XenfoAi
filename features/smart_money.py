import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SMCFeatures:
    """Container for Smart Money Concept features"""

    fvg_up: bool
    fvg_down: bool
    fvg_up_size: float
    fvg_down_size: float
    price_inefficiency: float
    inst_money_flow: float
    smart_money_div: int
    accumulation_score: float
    distribution_score: float
    institutional_activity: float


class SmartMoneyFeatures:
    """Enhanced Smart Money Concepts feature generator"""

    def __init__(self):
        self.feature_columns = []
        self.lookback_period = 20
        self.volume_threshold = 1.5
        self.price_threshold = 0.001

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive Smart Money Concepts features"""
        try:
            # Validate input data
            if df.empty or not all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            ):
                logger.error("Missing required columns in input data")
                return df

            # Fair Value Gaps (FVG)
            df = self._calculate_fair_value_gaps(df)

            # Price Inefficiency
            df = self._calculate_price_inefficiency(df)

            # Institutional Money Flow
            df = self._calculate_money_flow(df)

            # Smart Money Divergence
            df = self._calculate_smart_money_divergence(df)

            # Accumulation/Distribution Analysis
            df = self._calculate_accumulation_distribution(df)

            # Institutional Activity Score
            df = self._calculate_institutional_activity(df)

            # Update feature columns list
            self.feature_columns = [
                "fvg_up",
                "fvg_down",
                "fvg_up_size",
                "fvg_down_size",
                "price_inefficiency",
                "inst_money_flow",
                "smart_money_div",
                "accumulation_score",
                "distribution_score",
                "institutional_activity",
            ]

            # Validate calculated features
            df = self._validate_features(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Smart Money features: {str(e)}")
            return df

    def _calculate_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fair Value Gaps and their characteristics"""
        try:
            # Bullish FVG
            df["fvg_up"] = (df["low"].shift(2) > df["high"].shift(1)) & (
                df["low"] > df["high"].shift(1)
            )

            # Bearish FVG
            df["fvg_down"] = (df["high"].shift(2) < df["low"].shift(1)) & (
                df["high"] < df["low"].shift(1)
            )

            # FVG Sizes
            df["fvg_up_size"] = np.where(
                df["fvg_up"], df["low"].shift(2) - df["high"].shift(1), 0
            )

            df["fvg_down_size"] = np.where(
                df["fvg_down"], df["low"].shift(1) - df["high"].shift(2), 0
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating Fair Value Gaps: {str(e)}")
            return df

    def _calculate_price_inefficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price inefficiency metrics"""
        try:
            # Basic price inefficiency
            df["price_inefficiency"] = abs(df["close"] - df["open"]) / (
                df["high"] - df["low"] + 1e-7
            )

            # Rolling inefficiency
            df["price_inefficiency"] = (
                df["price_inefficiency"].rolling(window=self.lookback_period).mean()
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating price inefficiency: {str(e)}")
            return df

    def _calculate_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate institutional money flow indicators"""
        try:
            # Money Flow calculation
            df["money_flow"] = (
                ((df["close"] - df["low"]) - (df["high"] - df["close"]))
                / (df["high"] - df["low"] + 1e-7)
                * df["volume"]
            )

            # Institutional Money Flow Index
            df["inst_money_flow"] = (
                df["money_flow"].rolling(window=self.lookback_period).sum()
            )

            # Normalize
            df["inst_money_flow"] = (
                df["inst_money_flow"] - df["inst_money_flow"].min()
            ) / (df["inst_money_flow"].max() - df["inst_money_flow"].min() + 1e-7)

            return df

        except Exception as e:
            logger.error(f"Error calculating money flow: {str(e)}")
            return df

    def _calculate_smart_money_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Smart Money Divergence signals"""
        try:
            # Price and Volume Deltas
            df["price_delta"] = df["close"] - df["close"].shift(1)
            df["volume_delta"] = (
                df["volume"] - df["volume"].rolling(window=self.lookback_period).mean()
            )

            # Smart Money Divergence
            df["smart_money_div"] = np.where(
                (df["price_delta"] > 0) & (df["volume_delta"] < 0),
                -1,  # Distribution
                np.where(
                    (df["price_delta"] < 0) & (df["volume_delta"] < 0),
                    1,  # Accumulation
                    0,
                ),
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating smart money divergence: {str(e)}")
            return df

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate accumulation and distribution patterns"""
        try:
            # Accumulation Score
            df["accumulation_score"] = (
                np.where(
                    (df["close"] > df["open"])
                    & (df["volume"] > df["volume"].rolling(self.lookback_period).mean())
                    & (df["close"] > df["close"].shift(1)),
                    1,
                    0,
                )
                .rolling(self.lookback_period)
                .mean()
            )

            # Distribution Score
            df["distribution_score"] = (
                np.where(
                    (df["close"] < df["open"])
                    & (df["volume"] > df["volume"].rolling(self.lookback_period).mean())
                    & (df["close"] < df["close"].shift(1)),
                    1,
                    0,
                )
                .rolling(self.lookback_period)
                .mean()
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating accumulation/distribution: {str(e)}")
            return df

    def _calculate_institutional_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate institutional activity composite score"""
        try:
            # Combine various institutional indicators
            df["institutional_activity"] = (
                df["accumulation_score"] * 0.3
                + df["inst_money_flow"] * 0.3
                + (1 - df["price_inefficiency"]) * 0.2
                + (df["volume"] / df["volume"].rolling(self.lookback_period).mean())
                * 0.2
            )

            # Normalize to 0-1 range
            df["institutional_activity"] = (
                df["institutional_activity"]
                - df["institutional_activity"].rolling(self.lookback_period).min()
            ) / (
                df["institutional_activity"].rolling(self.lookback_period).max()
                - df["institutional_activity"].rolling(self.lookback_period).min()
                + 1e-7
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating institutional activity: {str(e)}")
            return df

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean calculated features"""
        try:
            for column in self.feature_columns:
                # Replace infinities
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

                # Fill NaN values
                df[column] = df[column].fillna(method="ffill").fillna(0)

                # Ensure proper range for normalized features
                if column in ["inst_money_flow", "institutional_activity"]:
                    df[column] = df[column].clip(0, 1)

            return df

        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_columns

    def get_smc_features(self, row: pd.Series) -> SMCFeatures:
        """Get SMC features for a single row"""
        try:
            return SMCFeatures(
                fvg_up=bool(row["fvg_up"]),
                fvg_down=bool(row["fvg_down"]),
                fvg_up_size=float(row["fvg_up_size"]),
                fvg_down_size=float(row["fvg_down_size"]),
                price_inefficiency=float(row["price_inefficiency"]),
                inst_money_flow=float(row["inst_money_flow"]),
                smart_money_div=int(row["smart_money_div"]),
                accumulation_score=float(row["accumulation_score"]),
                distribution_score=float(row["distribution_score"]),
                institutional_activity=float(row["institutional_activity"]),
            )
        except Exception as e:
            logger.error(f"Error creating SMC features: {str(e)}")
            return None
