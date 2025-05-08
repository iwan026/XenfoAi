import logging
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timezone
from pathlib import Path

from .smart_money import SmartMoneyFeatures
from .order_blocks import OrderBlockAnalyzer
from .volume_profile import VolumeProfileAnalyzer
from .market_structure import MarketStructureAnalyzer
from .liquidity import LiquidityAnalyzer
from .utils import FeatureUtils

from config import TradingConfig

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """Main class for calculating all trading features"""

    def __init__(self):
        """Initialize feature analyzers and metadata"""
        self.smc = SmartMoneyFeatures()
        self.order_blocks = OrderBlockAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.market_structure = MarketStructureAnalyzer()
        self.liquidity = LiquidityAnalyzer()
        self.utils = FeatureUtils()

        self.feature_columns: List[str] = []
        self.metadata = {
            "version": TradingConfig.VERSION,
            "created_at": TradingConfig.get_current_timestamp(),
            "created_by": TradingConfig.AUTHOR,
        }

        # Cache for feature calculations
        self._cache: Dict = {}
        self._cache_expiry: Dict = {}
        self.CACHE_DURATION = 300  # 5 minutes in seconds

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        include_features: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all or selected features with caching support

        Args:
            df: DataFrame with OHLCV data
            include_features: List of features to calculate, None for all
            use_cache: Whether to use calculation caching

        Returns:
            DataFrame with calculated features
        """
        try:
            # Validate input data
            required_columns = ["open", "high", "low", "close", "volume"]
            if not self.utils.validate_dataframe(df, required_columns):
                raise ValueError("Missing required columns in input data")

            # Check cache if enabled
            cache_key = self._generate_cache_key(df, include_features)
            if use_cache and self._check_cache(cache_key):
                logger.info("Using cached feature calculations")
                return self._cache[cache_key]

            # Initialize feature calculation
            calculation_start = datetime.now(timezone.utc)

            # Default to all features if none specified
            if include_features is None:
                include_features = [
                    "smc",
                    "order_blocks",
                    "volume_profile",
                    "market_structure",
                    "liquidity",
                ]

            # Calculate selected features
            df = df.copy()  # Prevent modifications to original DataFrame

            feature_calculators = {
                "smc": (self.smc, self.smc.get_feature_names()),
                "order_blocks": (
                    self.order_blocks,
                    self.order_blocks.get_feature_names(),
                ),
                "volume_profile": (
                    self.volume_profile,
                    self.volume_profile.get_feature_names(),
                ),
                "market_structure": (
                    self.market_structure,
                    self.market_structure.get_feature_names(),
                ),
                "liquidity": (self.liquidity, self.liquidity.get_feature_names()),
            }

            self.feature_columns = []  # Reset feature columns

            for feature in include_features:
                if feature not in feature_calculators:
                    logger.warning(f"Unknown feature type: {feature}")
                    continue

                calculator, feature_names = feature_calculators[feature]
                df = calculator.calculate_features(df)
                self.feature_columns.extend(feature_names)

            # Log calculation metadata
            calculation_time = (
                datetime.now(timezone.utc) - calculation_start
            ).total_seconds()
            self._log_calculation_metadata(
                operation="all_features",
                calculation_time=calculation_time,
                metadata={"included_features": include_features},
            )

            # Update cache if enabled
            if use_cache:
                self._update_cache(cache_key, df)

            return df

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise

    def get_feature_snapshot(
        self, df: pd.DataFrame, timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Get snapshot of all features at specific timestamp

        Args:
            df: DataFrame with calculated features
            timestamp: Specific timestamp for snapshot, latest if None

        Returns:
            Dictionary containing feature snapshot
        """
        try:
            # Use latest timestamp if none provided
            if timestamp is None:
                row = df.iloc[-1]
            else:
                row = df.loc[timestamp]

            snapshot = {
                "metadata": {
                    "timestamp": row.name.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(row.name, "strftime")
                    else str(row.name),
                    "captured_at": TradingConfig.get_current_timestamp(),
                    "captured_by": TradingConfig.AUTHOR,
                    "version": TradingConfig.VERSION,
                },
                "features": {
                    "smart_money": self.smc.get_feature_snapshot(row),
                    "order_blocks": self.order_blocks.get_feature_snapshot(row),
                    "volume_profile": self.volume_profile.get_feature_snapshot(row),
                    "market_structure": self.market_structure.get_feature_snapshot(row),
                    "liquidity": self.liquidity.get_feature_snapshot(row),
                },
                "technical_levels": self.utils.calculate_technical_levels(df),
            }

            return snapshot

        except Exception as e:
            logger.error(f"Error getting feature snapshot: {str(e)}")
            return {}

    def get_feature_names(self) -> List[str]:
        """Get list of all calculated feature names"""
        return self.feature_columns

    def get_metadata(self) -> Dict:
        """Get calculator metadata"""
        return {
            **self.metadata,
            "last_updated": TradingConfig.get_current_timestamp(),
            "feature_count": len(self.feature_columns),
            "features": self.feature_columns,
        }

    def _generate_cache_key(
        self, df: pd.DataFrame, features: Optional[List[str]]
    ) -> str:
        """Generate unique cache key for DataFrame and feature set"""
        try:
            features_str = "_".join(sorted(features)) if features else "all"
            last_timestamp = (
                df.index[-1].strftime("%Y%m%d%H%M%S") if len(df) > 0 else "empty"
            )
            return f"{features_str}_{last_timestamp}_{len(df)}"
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    def _check_cache(self, cache_key: str) -> bool:
        """Check if valid cached calculation exists"""
        try:
            if cache_key not in self._cache or cache_key not in self._cache_expiry:
                return False

            now = datetime.now(timezone.utc).timestamp()
            return now < self._cache_expiry[cache_key]
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            return False

    def _update_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Update cache with new calculation results"""
        try:
            self._cache[cache_key] = df
            self._cache_expiry[cache_key] = (
                datetime.now(timezone.utc).timestamp() + self.CACHE_DURATION
            )

            # Clean expired cache entries
            self._clean_cache()
        except Exception as e:
            logger.error(f"Error updating cache: {str(e)}")

    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        try:
            now = datetime.now(timezone.utc).timestamp()
            expired_keys = [k for k, v in self._cache_expiry.items() if v < now]

            for k in expired_keys:
                self._cache.pop(k, None)
                self._cache_expiry.pop(k, None)
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")

    def _log_calculation_metadata(
        self, operation: str, calculation_time: float, metadata: Dict
    ) -> None:
        """Log feature calculation metadata"""
        try:
            log_entry = {
                "operation": operation,
                "calculation_time": calculation_time,
                "timestamp": TradingConfig.get_current_timestamp(),
                "version": TradingConfig.VERSION,
                **metadata,
            }

            log_file = TradingConfig.LOG_DIR / "feature_calculations.jsonl"

            with open(log_file, "a") as f:
                f.write(f"{log_entry}\n")

        except Exception as e:
            logger.error(f"Error logging calculation metadata: {str(e)}")
