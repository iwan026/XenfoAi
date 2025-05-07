import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import ta

logger = logging.getLogger(__name__)


@dataclass
class MarketCondition:
    volatility: float
    trend_strength: float
    market_regime: str
    current_timeframe: str


class ForexDataProcessor:
    """Enhanced class for handling data preprocessing for forex trading"""

    def __init__(self):
        self.price_scaler = RobustScaler()
        self.indicator_scaler = MinMaxScaler()
        self.is_fitted = False
        self.market_condition = None

    def get_mt5_historical_data(
        self, symbol: str, timeframe: str, num_candles: int = 5000
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data from MetaTrader 5 with enhanced error handling"""
        try:
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

            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return None

            rates = mt5.copy_rates_from_pos(
                symbol, timeframe_map[timeframe], 0, num_candles
            )

            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get data for {symbol}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Enhanced data validation
            if df.isnull().any().any():
                logger.warning(f"Missing values found in {symbol} data")
                df = self.handle_missing_values(df)

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced missing value handling"""
        # Forward fill price data
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].ffill()

        # Interpolate volume data
        if "volume" in df.columns:
            df["volume"] = df["volume"].interpolate(method="linear")

        # Check for remaining NaN values
        if df.isnull().any().any():
            logger.warning("Some NaN values could not be handled - dropping rows")
            df = df.dropna()

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicator calculation"""
        try:
            # Basic price action features
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log1p(df["returns"])
            df["volatility"] = df["returns"].rolling(window=20).std()

            # Volume-based features (if available)
            if "volume" in df.columns:
                df["volume_sma"] = df["volume"].rolling(window=20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma"]

            # Trend indicators
            for period in [10, 20, 50, 100, 200]:
                df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
                df[f"ema_{period}"] = df["close"].ewm(span=period).mean()

            # Advanced indicators using the 'ta' library
            # Momentum
            df["rsi"] = ta.momentum.rsi(df["close"], window=14)
            df["stoch_k"] = ta.momentum.stoch(df["high"], df["low"], df["close"])
            df["stoch_d"] = ta.momentum.stoch_signal(df["high"], df["low"], df["close"])
            df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])

            # Trend
            df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
            df["macd"] = ta.trend.macd_diff(df["close"])
            df["vwap"] = ta.volume.volume_weighted_average_price(
                df["high"], df["low"], df["close"], df.get("volume", df["close"])
            )

            # Volatility
            bb = ta.volatility.BollingerBands(df["close"])
            df["bb_high"] = bb.bollinger_hband()
            df["bb_low"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]
            df["atr"] = ta.volatility.average_true_range(
                df["high"], df["low"], df["close"]
            )

            # Fibonacci retracements
            df = self.add_fibonacci_levels(df)

            # Market regime features
            df = self.add_market_regime_features(df)

            # Clean up any NaN values
            df = df.dropna()

            # Update market condition
            self.update_market_condition(df)

            return df

        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return None

    def add_fibonacci_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fibonacci retracement levels"""
        window = 20
        for i in range(len(df) - window):
            period = df.iloc[i : i + window]
            high = period["high"].max()
            low = period["low"].min()
            diff = high - low

            df.loc[df.index[i + window], "fib_236"] = high - (diff * 0.236)
            df.loc[df.index[i + window], "fib_382"] = high - (diff * 0.382)
            df.loc[df.index[i + window], "fib_618"] = high - (diff * 0.618)

        return df

    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime identification features"""
        # Trend strength using ADX
        df["trend_strength"] = ta.trend.adx(df["high"], df["low"], df["close"])

        # Volatility regime using ATR
        df["volatility_regime"] = pd.qcut(
            df["atr"].rolling(window=20).mean(), q=3, labels=["low", "medium", "high"]
        )

        # Market regime based on both trend and volatility
        df["market_regime"] = df.apply(
            lambda x: self.classify_market_regime(x["trend_strength"], x["atr"]), axis=1
        )

        return df

    @staticmethod
    def classify_market_regime(trend_strength: float, volatility: float) -> str:
        """Classify market regime based on trend strength and volatility"""
        if trend_strength > 25:
            if volatility > 0.0015:
                return "volatile_trend"
            return "stable_trend"
        else:
            if volatility > 0.0015:
                return "volatile_range"
            return "stable_range"

    def update_market_condition(self, df: pd.DataFrame) -> None:
        """Update current market condition based on recent data"""
        recent_data = df.iloc[-20:]

        self.market_condition = MarketCondition(
            volatility=recent_data["volatility"].mean(),
            trend_strength=recent_data["trend_strength"].mean(),
            market_regime=recent_data["market_regime"].mode()[0],
            current_timeframe=recent_data.index.freq
            if hasattr(recent_data.index, "freq")
            else "unknown",
        )

    def create_advanced_labels(
        self,
        df: pd.DataFrame,
        min_pip_movement: float = 10,
        forecast_window: int = 20,
        risk_reward_ratio: float = 2.0,
    ) -> pd.DataFrame:
        """Create sophisticated trading signals based on future price movements"""
        try:
            pip_value = 0.0001  # Adjust for JPY pairs (0.01)

            # Calculate future price movements
            future_prices = df["close"].shift(-forecast_window)
            current_prices = df["close"]

            # Calculate potential profit and loss in pips
            potential_profit = (future_prices - current_prices) / pip_value
            stop_loss = min_pip_movement
            take_profit = min_pip_movement * risk_reward_ratio

            # Create labels based on risk-reward and price movement
            conditions = [
                (potential_profit >= take_profit),  # Strong Buy
                (potential_profit >= min_pip_movement),  # Buy
                (potential_profit <= -take_profit),  # Strong Sell
                (potential_profit <= -min_pip_movement),  # Sell
            ]
            choices = [3, 2, 0, 1]  # 0:Strong Sell, 1:Sell, 2:Buy, 3:Strong Buy

            df["target"] = np.select(conditions, choices, default=4)  # 4:Hold

            # Additional label features
            df["price_movement"] = potential_profit
            df["risk_reward_ratio"] = abs(potential_profit / stop_loss)

            return df

        except Exception as e:
            logger.error(f"Error creating labels: {str(e)}")
            return None

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        lookback_period: int = 60,
        feature_groups: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences with enhanced feature grouping and scaling"""
        try:
            # Create labels first
            df = self.create_advanced_labels(df)

            # Default feature groups if none provided
            if feature_groups is None:
                feature_groups = {
                    "price": ["open", "high", "low", "close", "returns", "log_returns"],
                    "volume": ["volume", "volume_sma", "volume_ratio"]
                    if "volume" in df.columns
                    else [],
                    "momentum": ["rsi", "stoch_k", "stoch_d", "cci"],
                    "trend": ["adx", "macd", "trend_strength"],
                    "volatility": ["atr", "bb_width", "volatility"],
                }

            # Scale each feature group separately
            scaled_data = {}
            for group, features in feature_groups.items():
                if features:  # Only process non-empty feature groups
                    if group == "price":
                        scaled_data[group] = self.price_scaler.fit_transform(
                            df[features]
                        )
                    else:
                        scaled_data[group] = self.indicator_scaler.fit_transform(
                            df[features]
                        )

            # Combine all scaled features
            all_features = np.column_stack(
                [scaled_data[group] for group in scaled_data.keys()]
            )

            # Create sequences
            X, y = [], []
            for i in range(len(df) - lookback_period):
                X.append(all_features[i : (i + lookback_period)])
                y.append(df["target"].iloc[i + lookback_period])

            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    def create_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> Tuple:
        """Create train-test split with validation set and time series consideration"""
        try:
            tscv = TimeSeriesSplit(n_splits=5)

            # Get the last split for final training
            splits = list(tscv.split(X))
            train_idx, test_idx = splits[-1]

            # Further split test set into test and validation
            test_split = int(
                len(test_idx) * (1 - validation_size / (test_size + validation_size))
            )
            val_idx = test_idx[test_split:]
            test_idx = test_idx[:test_split]

            # Create the splits
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            return X_train, X_test, X_val, y_train, y_test, y_val

        except Exception as e:
            logger.error(f"Error creating train-test split: {str(e)}")
            return None, None, None, None, None, None
