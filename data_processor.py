import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import ta
from features import (
    FeatureCalculator,
    SmartMoneyFeatures,
    OrderBlockAnalyzer,
    VolumeProfileAnalyzer,
    MarketStructureAnalyzer,
    LiquidityAnalyzer,
    FeatureUtils,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketCondition:
    """Container for market condition data"""

    volatility: float
    trend_strength: float
    market_regime: str
    current_timeframe: str


class ForexDataProcessor:
    """Class for processing forex market data for signal generation"""

    def __init__(self):
        self.price_scaler = RobustScaler()
        self.feature_calculator = FeatureCalculator()
        self.utils = FeatureUtils()
        self.is_fitted = False
        self.market_condition = None
        self.feature_columns = []

        # Initialize feature processors
        self.smc = SmartMoneyFeatures()
        self.order_blocks = OrderBlockAnalyzer()
        self.volume_profile = VolumeProfileAnalyzer()
        self.market_structure = MarketStructureAnalyzer()
        self.liquidity = LiquidityAnalyzer()

    def get_market_data(
        self, symbol: str, timeframe: str, num_candles: int = 1000
    ) -> Optional[pd.DataFrame]:
        """Fetch and process market data for signal generation"""
        try:
            # Initialize MT5 if needed
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return None

            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
            }

            if timeframe not in timeframe_map:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None

            # Fetch data
            rates = mt5.copy_rates_from_pos(
                symbol, timeframe_map[timeframe], 0, num_candles
            )

            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get data for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Check if volume data exists, if not create synthetic volume
            if "volume" not in df.columns or df["volume"].isnull().any():
                logger.warning(
                    f"Volume data missing for {symbol}, using tick volume instead"
                )
                # Get tick volume if real volume is not available
                if "tick_volume" in df.columns:
                    df["volume"] = df["tick_volume"]
                else:
                    # Create synthetic volume based on price movement
                    df["volume"] = ((df["high"] - df["low"]) * 1000).round()

            # Process data with new feature module
            df = self.process_market_data(df)

            return df

        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return None
        finally:
            mt5.shutdown()

    def process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process market data with all features"""
        try:
            # Validate input data
            if not self.utils.validate_dataframe(
                df, ["open", "high", "low", "close", "volume"]
            ):
                raise ValueError("Missing required columns in input data")

            # Calculate all features using FeatureCalculator
            df = self.feature_calculator.calculate_all_features(df)

            # Update feature columns
            self.feature_columns = self.feature_calculator.get_feature_names()

            # Calculate additional technical indicators
            df = self._add_technical_indicators(df)

            # Add market regime features
            df = self._add_market_regime_features(df)

            # Clean up NaN values
            df = df.dropna()

            # Update market condition
            self._update_market_condition(df)

            return df

        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
            return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators"""
        try:
            # Basic price features
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log1p(df["returns"])
            df["volatility"] = df["returns"].rolling(window=20).std()

            # Trend indicators
            for period in [10, 20, 50]:
                df[f"sma_{period}"] = ta.trend.sma_indicator(df["close"], period)
                df[f"ema_{period}"] = ta.trend.ema_indicator(df["close"], period)

            # Momentum indicators
            df["rsi"] = ta.momentum.rsi(df["close"])
            df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
            df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"])
            df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])

            # Trend strength
            df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
            df["macd"] = ta.trend.macd_diff(df["close"])

            # Volatility indicators
            bb = ta.volatility.BollingerBands(df["close"])
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]
            df["atr"] = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"]
            )

            return df

        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime identification features"""
        try:
            # Trend strength using ADX
            df["trend_strength"] = ta.trend.adx(df["high"], df["low"], df["close"])

            # Volatility regime using ATR
            df["volatility_regime"] = pd.qcut(
                df["atr"].rolling(window=20).mean(),
                q=3,
                labels=["low", "medium", "high"],
            )

            # Market regime classification
            df["market_regime"] = df.apply(
                lambda x: self._classify_market_regime(x["trend_strength"], x["atr"]),
                axis=1,
            )

            return df

        except Exception as e:
            logger.error(f"Error adding market regime features: {str(e)}")
            return df

    @staticmethod
    def _classify_market_regime(trend_strength: float, volatility: float) -> str:
        """Classify market regime based on trend strength and volatility"""
        if trend_strength > 25:
            if volatility > 0.0015:
                return "volatile_trend"
            return "stable_trend"
        else:
            if volatility > 0.0015:
                return "volatile_range"
            return "stable_range"

    def _update_market_condition(self, df: pd.DataFrame) -> None:
        """Update current market condition based on recent data"""
        try:
            recent_data = df.iloc[-20:]

            self.market_condition = MarketCondition(
                volatility=recent_data["volatility"].mean(),
                trend_strength=recent_data["trend_strength"].mean(),
                market_regime=recent_data["market_regime"].mode()[0],
                current_timeframe=recent_data.index.freq
                if hasattr(recent_data.index, "freq")
                else "unknown",
            )

        except Exception as e:
            logger.error(f"Error updating market condition: {str(e)}")

    def get_training_data(
        self,
        symbol: str,
        timeframe: str,
        lookback_period: int = 60,
        training_days: int = 365,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get and prepare data for model training"""
        try:
            # Calculate number of candles needed
            candles_per_day = {
                "M1": 1440,
                "M5": 288,
                "M15": 96,
                "M30": 48,
                "H1": 24,
                "H4": 6,
                "D1": 1,
            }
            num_candles = candles_per_day.get(timeframe, 24) * training_days

            # Get historical data
            df = self.get_market_data(symbol, timeframe, num_candles)
            if df is None:
                logger.error(f"Failed to get market data for {symbol} {timeframe}")
                return np.array([]), np.array([])

            # Prepare training data
            X, y = self.prepare_training_data(df, lookback_period)

            if len(X) == 0 or len(y) == 0:
                logger.error(
                    f"Failed to prepare training data for {symbol} {timeframe}"
                )
                return np.array([]), np.array([])

            logger.info(
                f"Successfully prepared training data: X shape {X.shape}, y shape {y.shape}"
            )
            return X, y

        except Exception as e:
            logger.error(f"Error getting training data: {str(e)}")
            return np.array([]), np.array([])

    def prepare_data_for_signal(
        self, df: pd.DataFrame, lookback_period: int = 60
    ) -> Tuple[np.ndarray, List[datetime]]:
        """Prepare data for signal generation"""
        try:
            # Get all feature names from feature calculator
            feature_columns = self.feature_calculator.get_feature_names()

            # Add technical indicator columns
            feature_columns.extend(
                [
                    "returns",
                    "volatility",
                    "rsi",
                    "stoch_k",
                    "stoch_d",
                    "cci",
                    "adx",
                    "macd",
                    "atr",
                    "bb_width",
                    "sma_10",
                    "sma_20",
                    "sma_50",
                    "ema_10",
                    "ema_20",
                    "ema_50",
                ]
            )

            # Ensure all features are numeric
            for col in feature_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Handle missing values
            df = df.ffill().bfill()

            # Extract and scale features
            features_df = df[feature_columns].astype(np.float32)
            if not self.is_fitted:
                self.price_scaler.fit(features_df)
                self.is_fitted = True
            scaled_data = self.price_scaler.transform(features_df)

            # Create sequences
            sequences = []
            timestamps = []

            for i in range(len(scaled_data) - lookback_period):
                seq = scaled_data[i : (i + lookback_period)]
                if len(seq) == lookback_period:
                    sequences.append(seq)
                    timestamps.append(df.index[i + lookback_period])

            # Convert to numpy array
            X = np.array(sequences, dtype=np.float32)

            # Validate output
            if len(X) == 0:
                logger.warning("No valid sequences created")
                return np.array([], dtype=np.float32), []

            logger.debug(f"Prepared sequences shape: {X.shape}")
            return X, timestamps

        except Exception as e:
            logger.error(f"Error preparing data for signal: {str(e)}")
            return np.array([], dtype=np.float32), []

    def prepare_training_data(
        self, df: pd.DataFrame, lookback_period: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        try:
            # Create labels based on future price movement
            df["future_returns"] = df["close"].pct_change(periods=3).shift(-3)
            df["target"] = np.where(
                df["future_returns"] > 0.001,
                1,  # BUY
                np.where(
                    df["future_returns"] < -0.001,
                    0,  # SELL
                    2,  # HOLD
                ),
            )

            # Remove NaN values from target that occur due to shift operation
            df = df.iloc[:-3].copy()

            # Prepare feature sequences
            X, timestamps = self.prepare_data_for_signal(df, lookback_period)
            if len(X) == 0:
                logger.error("Failed to create feature sequences")
                return np.array([]), np.array([])

            # Create target array aligned with feature sequences
            y = df["target"].values[lookback_period:]

            # Final validation of shapes
            if len(X) != len(y):
                logger.error(f"Shape mismatch: X: {X.shape}, y: {len(y)}")
                if len(X) > len(y):
                    X = X[: len(y)]
                else:
                    y = y[: len(X)]

            # Additional sanity checks
            if len(X) == 0 or len(y) == 0:
                logger.error("Empty arrays after preparation")
                return np.array([]), np.array([])

            if not np.any(np.isnan(X)) and not np.any(np.isnan(y)):
                logger.info(
                    f"Successfully prepared training data: X shape {X.shape}, y shape {y.shape}"
                )
                return X, y
            else:
                logger.error("NaN values detected in prepared data")
                return np.array([]), np.array([])

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])

    def create_train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple:
        """Create train-test split with time series consideration"""
        try:
            split_idx = int(len(X) * (1 - test_size))

            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error creating train-test split: {str(e)}")
            return None, None, None, None

    def get_market_summary(self, df: pd.DataFrame) -> Dict:
        """Get current market summary"""
        try:
            latest = df.iloc[-1]

            # Get feature snapshots
            feature_snapshot = self.feature_calculator.get_feature_snapshot(df)

            summary = {
                "timestamp": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
                "market_regime": latest["market_regime"],
                "trend_strength": latest["trend_strength"],
                "volatility": latest["volatility"],
                "rsi": latest["rsi"],
                "adx": latest["adx"],
                "features": feature_snapshot,
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting market summary: {str(e)}")
            return {}
