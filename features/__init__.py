"""
Forex Trading Features Package
Created at: 2025-05-08 12:28:04
Created by: Xentrovt

This package provides comprehensive feature generation for forex trading analysis.
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime

from .smart_money import SmartMoneyFeatures, SMCFeatures
from .order_blocks import OrderBlockAnalyzer, OrderBlockFeatures
from .volume_profile import VolumeProfileAnalyzer, VolumeProfileFeatures
from .market_structure import MarketStructureAnalyzer, MarketStructureFeatures
from .liquidity import LiquidityAnalyzer, LiquidityFeatures
from .utils import FeatureUtils, TechnicalLevels

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = "iwan026"
__created_at__ = "2025-05-08 12:28:04"

__all__ = [
    "SmartMoneyFeatures",
    "SMCFeatures",
    "OrderBlockAnalyzer",
    "OrderBlockFeatures",
    "VolumeProfileAnalyzer",
    "VolumeProfileFeatures",
    "MarketStructureAnalyzer",
    "MarketStructureFeatures",
    "LiquidityAnalyzer",
    "LiquidityFeatures",
    "FeatureUtils",
    "TechnicalLevels",
    "FeatureCalculator",
]


class FeatureCalculator:
    """Main class for calculating all features"""

    def __init__(self):
        self.smc = SmartMoneyFeatures()
        self.order_blocks = OrderBlockAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.market_structure = MarketStructureAnalyzer()
        self.liquidity = LiquidityAnalyzer()
        self.utils = FeatureUtils()

        self.feature_columns = []
        self.metadata = {
            "version": __version__,
            "created_at": __created_at__,
            "created_by": __author__,
        }

    def calculate_all_features(
        self, df: pd.DataFrame, include_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate all or selected features"""
        try:
            # Validate input data
            required_columns = ["open", "high", "low", "close", "volume"]
            if not self.utils.validate_dataframe(df, required_columns):
                raise ValueError("Missing required columns in input data")

            # Initialize feature calculation
            calculation_start = datetime.utcnow()

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
            if "smc" in include_features:
                df = self.smc.calculate_features(df)
                self.feature_columns.extend(self.smc.get_feature_names())

            if "order_blocks" in include_features:
                df = self.order_blocks.calculate_features(df)
                self.feature_columns.extend(self.order_blocks.get_feature_names())

            if "volume_profile" in include_features:
                df = self.volume_profile.calculate_features(df)
                self.feature_columns.extend(self.volume_profile.get_feature_names())

            if "market_structure" in include_features:
                df = self.market_structure.calculate_features(df)
                self.feature_columns.extend(self.market_structure.get_feature_names())

            if "liquidity" in include_features:
                df = self.liquidity.calculate_features(df)
                self.feature_columns.extend(self.liquidity.get_feature_names())

            # Log calculation metadata
            calculation_time = (datetime.utcnow() - calculation_start).total_seconds()
            self.utils.log_calculation_metadata(
                "all_features",
                calculation_time,
                {"included_features": include_features},
            )

            return df

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            raise

    def get_feature_snapshot(
        self, df: pd.DataFrame, timestamp: Optional[datetime] = None
    ) -> Dict:
        """Get snapshot of all features at specific timestamp"""
        try:
            # Use latest timestamp if none provided
            if timestamp is None:
                row = df.iloc[-1]
            else:
                row = df.loc[timestamp]

            snapshot = {
                "metadata": {
                    "timestamp": row.name,
                    "captured_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "captured_by": __author__,
                },
                "features": {
                    "smart_money": self.smc.get_smc_features(row),
                    "order_blocks": self.order_blocks.get_ob_features(row),
                    "volume_profile": self.volume_profile.get_volume_profile_features(
                        row
                    ),
                    "market_structure": self.market_structure.get_market_structure_features(
                        row
                    ),
                    "liquidity": self.liquidity.get_liquidity_features(row),
                },
                "technical_levels": self.utils.calculate_technical_levels(df),
            }

            return snapshot

        except Exception as e:
            logger.error(f"Error getting feature snapshot: {str(e)}")
            return {}

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_columns

    def get_metadata(self) -> Dict:
        """Get calculator metadata"""
        return self.metadata
