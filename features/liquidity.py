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
class LiquidityParameters:
    """Parameters for Liquidity analysis"""

    # Liquidity Pools
    pool_detection_period: int = 20  # Period for pool detection
    pool_volume_threshold: float = 2.0  # Volume multiplier for pool significance
    min_pool_touches: int = 3  # Minimum touches for valid pool

    # Swept Levels
    sweep_threshold: float = 0.001  # Minimum size for level sweep
    sweep_confirmation: int = 3  # Candles to confirm sweep

    # Order Flow
    flow_window: int = 50  # Window for order flow analysis
    imbalance_threshold: float = 0.7  # Threshold for flow imbalance

    # Market Depth
    depth_levels: int = 10  # Number of depth levels to analyze
    depth_volume_threshold: float = 1.5  # Volume threshold for depth significance

    # Gaps
    min_gap_size: float = 0.002  # Minimum size for liquidity gap
    gap_volume_threshold: float = 0.5  # Volume threshold for gap confirmation


class LiquidityAnalyzer:
    """Main class for Liquidity analysis"""

    def __init__(self, params: Optional[LiquidityParameters] = None):
        """Initialize Liquidity analyzer with parameters"""
        self.params = params or LiquidityParameters()
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
            # Liquidity Pools
            "liquidity_pool",
            "pool_strength",
            "pool_distance",
            # Swept Levels
            "level_swept",
            "sweep_direction",
            "sweep_strength",
            # Order Flow
            "flow_imbalance",
            "buy_pressure",
            "sell_pressure",
            # Market Depth
            "depth_score",
            "depth_imbalance",
            "depth_ratio",
            # Gaps
            "liquidity_gap",
            "gap_size",
            "gap_significance",
        ]

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all Liquidity related features

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional Liquidity features
        """
        try:
            # Validate input
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in input DataFrame")

            # Create copy to prevent modifications to original
            df = df.copy()

            # Calculate main features
            df = self._detect_liquidity_pools(df)
            df = self._analyze_swept_levels(df)
            df = self._analyze_order_flow(df)
            df = self._estimate_market_depth(df)
            df = self._detect_liquidity_gaps(df)

            # Log calculation metadata
            self._log_calculation(df)

            return df

        except Exception as e:
            logger.error(f"Error calculating Liquidity features: {str(e)}")
            return df

    def _detect_liquidity_pools(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and analyze liquidity pools"""
        try:
            period = self.params.pool_detection_period

            # Calculate volume profile
            df["volume_ma"] = df["volume"].rolling(period).mean()
            df["price_levels"] = pd.qcut(
                df["close"], q=10, labels=False, duplicates="drop"
            )

            # Identify high volume areas
            df["high_volume"] = df["volume"] > (
                df["volume_ma"] * self.params.pool_volume_threshold
            )

            # Count touches at price levels
            df["level_touches"] = df.groupby("price_levels")["high_volume"].transform(
                "sum"
            )

            # Identify liquidity pools
            df["liquidity_pool"] = df["level_touches"] >= self.params.min_pool_touches

            # Calculate pool strength
            df["pool_strength"] = np.where(
                df["liquidity_pool"], df["volume"] / df["volume_ma"], 0
            )

            # Calculate distance to nearest pool
            df["pool_distance"] = self._calculate_pool_distance(df)

            # Clean up
            df = df.drop(
                ["volume_ma", "price_levels", "high_volume", "level_touches"], axis=1
            )

            return df

        except Exception as e:
            logger.error(f"Error detecting liquidity pools: {str(e)}")
            return df

    def _analyze_swept_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze swept price levels"""
        try:
            threshold = self.params.sweep_threshold
            confirmation = self.params.sweep_confirmation

            # Detect level sweeps
            df["high_sweep"] = (
                (df["high"] > df["high"].rolling(confirmation).max().shift(1))
                & (df["close"] < df["open"])
                & ((df["high"] - df["high"].shift(1)) > threshold)
            )

            df["low_sweep"] = (
                (df["low"] < df["low"].rolling(confirmation).min().shift(1))
                & (df["close"] > df["open"])
                & ((df["low"].shift(1) - df["low"]) > threshold)
            )

            # Determine sweep characteristics
            df["level_swept"] = df["high_sweep"] | df["low_sweep"]

            df["sweep_direction"] = np.where(
                df["high_sweep"], 1, np.where(df["low_sweep"], -1, 0)
            )

            # Calculate sweep strength
            df["sweep_strength"] = np.where(
                df["high_sweep"],
                (df["high"] - df["high"].shift(1)) / df["high"].shift(1),
                np.where(
                    df["low_sweep"],
                    (df["low"].shift(1) - df["low"]) / df["low"].shift(1),
                    0,
                ),
            )

            # Clean up
            df = df.drop(["high_sweep", "low_sweep"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing swept levels: {str(e)}")
            return df

    def _analyze_order_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze order flow with optimized tick volume"""
        try:
            window = self.params.flow_window

            # Calculate buying and selling pressure
            df["buy_volume"] = np.where(
                df["close"] > df["open"],
                df["volume"] * ((df["close"] - df["open"]) / (df["high"] - df["low"])),
                0,
            )

            df["sell_volume"] = np.where(
                df["close"] < df["open"],
                df["volume"] * ((df["open"] - df["close"]) / (df["high"] - df["low"])),
                0,
            )

            # Calculate cumulative pressures
            price_impact = (df["high"] - df["low"]) / df["close"]

            df["buy_pressure"] = (df["buy_volume"] * price_impact).rolling(
                window
            ).sum() / (df["volume"] * price_impact).rolling(window).sum()

            df["sell_pressure"] = (df["sell_volume"] * price_impact).rolling(
                window
            ).sum() / (df["volume"] * price_impact).rolling(window).sum()

            # Calculate flow imbalance
            df["flow_imbalance"] = abs(df["buy_pressure"] - df["sell_pressure"])

            # Clean up
            df = df.drop(["buy_volume", "sell_volume"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error analyzing order flow: {str(e)}")
            return df

    def _estimate_market_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate market depth metrics"""
        try:
            levels = self.params.depth_levels
            threshold = self.params.depth_volume_threshold

            # Calculate volume at different price levels
            df["price_buckets"] = pd.qcut(
                df["close"], q=levels, labels=False, duplicates="drop"
            )

            # Calculate depth metrics
            df["depth_volume"] = df.groupby("price_buckets")["volume"].transform("sum")
            df["depth_ratio"] = df["depth_volume"] / df["volume"].rolling(levels).mean()

            # Calculate depth score
            df["depth_score"] = np.where(
                df["depth_ratio"] > threshold,
                df["depth_ratio"] / threshold,
                df["depth_ratio"] / (2 * threshold),
            )

            # Calculate depth imbalance
            upper_depth = df[df["price_buckets"] > levels / 2]["depth_volume"].sum()
            lower_depth = df[df["price_buckets"] <= levels / 2]["depth_volume"].sum()

            df["depth_imbalance"] = (upper_depth - lower_depth) / (
                upper_depth + lower_depth
            )

            # Clean up
            df = df.drop(["price_buckets", "depth_volume"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error estimating market depth: {str(e)}")
            return df

    def _detect_liquidity_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and analyze liquidity gaps"""
        try:
            min_size = self.params.min_gap_size
            vol_threshold = self.params.gap_volume_threshold

            # Calculate price gaps
            df["price_gap"] = (df["low"] - df["high"].shift(1)) / df["close"]

            # Identify significant gaps
            df["liquidity_gap"] = (abs(df["price_gap"]) > min_size) & (
                df["volume"] < df["volume"].rolling(20).mean() * vol_threshold
            )

            # Calculate gap characteristics
            df["gap_size"] = np.where(df["liquidity_gap"], abs(df["price_gap"]), 0)

            df["gap_significance"] = np.where(
                df["liquidity_gap"],
                df["gap_size"] * (1 - df["volume"] / df["volume"].rolling(20).mean()),
                0,
            )

            # Clean up
            df = df.drop(["price_gap"], axis=1)

            return df

        except Exception as e:
            logger.error(f"Error detecting liquidity gaps: {str(e)}")
            return df

    def _calculate_pool_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to nearest liquidity pool"""
        try:
            pool_prices = df[df["liquidity_pool"]]["close"]
            current_price = df["close"]

            distances = abs(
                current_price.values.reshape(-1, 1) - pool_prices.values.reshape(1, -1)
            )

            return pd.Series(np.min(distances, axis=1), index=df.index)

        except Exception as e:
            logger.error(f"Error calculating pool distance: {str(e)}")
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
                    "liquidity_pools": df["liquidity_pool"].sum(),
                    "swept_levels": df["level_swept"].sum(),
                    "avg_flow_imbalance": df["flow_imbalance"].mean(),
                    "liquidity_gaps": df["liquidity_gap"].sum(),
                },
            }

            log_file = TradingConfig.LOG_DIR / "liquidity_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")

    def get_feature_names(self) -> List[str]:
        """Get list of all Liquidity feature names"""
        return self.feature_columns

    def get_feature_snapshot(self, row: pd.Series) -> Dict:
        """Get snapshot of Liquidity features for a single row"""
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
