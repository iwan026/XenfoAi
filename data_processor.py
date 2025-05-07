import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
from config import TradingConfig
from typing import Tuple, Optional, List, Dict

logger = logging.getLogger(__name__)


class ForexDataProcessor:
    """Class for handling data preprocessing for forex trading"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.is_fitted = False

    def get_mt5_historical_data(
        self, symbol: str, timeframe: str, num_candles: int = 5000
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data from MetaTrader 5"""
        try:
            # Map timeframe strings to MT5 timeframe constants
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

            # Initialize MT5 if not already initialized
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return None

            # Get historical data
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

            # Basic data validation
            if df.isnull().any().any():
                logger.warning(f"Missing values found in {symbol} data")
                df = df.ffill()  # Forward fill missing values

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        try:
            # Price action features
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log1p(df["returns"])
            df["volatility"] = df["returns"].rolling(window=20).std()

            # Moving averages
            df["sma_10"] = df["close"].rolling(window=10).mean()
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["ema_10"] = df["close"].ewm(span=10).mean()
            df["ema_20"] = df["close"].ewm(span=20).mean()
            df["ema_50"] = df["close"].ewm(span=50).mean()

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

            # Average True Range (ATR)
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df["atr"] = true_range.rolling(14).mean()

            # Clean up any NaN values
            df = df.dropna()

            return df

        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return None

    def prepare_sequences(
        self, df: pd.DataFrame, lookback_period: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training/prediction"""
        try:
            # Calculate target variable (1 for price up, 0 for price down)
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(np.float32)

            # Select features for training
            feature_columns = [
                "returns",
                "log_returns",
                "volatility",
                "sma_10",
                "sma_20",
                "sma_50",
                "ema_10",
                "ema_20",
                "ema_50",
                "bb_width",
                "rsi",
                "macd",
                "macd_hist",
                "atr",
            ]

            # Scale features
            scaled_data = self.scaler.fit_transform(df[feature_columns])
            scaled_df = pd.DataFrame(
                scaled_data, columns=feature_columns, index=df.index
            )

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_df) - lookback_period):
                X.append(scaled_df.iloc[i : i + lookback_period].values)
                y.append(df["target"].iloc[i + lookback_period])

            # Perbaikan: Convert ke numpy array dengan tipe data yang benar
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            return X, y

        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    def create_train_test_split(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create train-test split with time series consideration"""
        try:
            tscv = TimeSeriesSplit(n_splits=5)

            # Get the last split for final training
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error creating train-test split: {str(e)}")
            return None, None, None, None

    def prepare_latest_sequence(
        self, df: pd.DataFrame, lookback_period: int = TradingConfig.LOOKBACK_PERIOD
    ) -> Optional[np.ndarray]:
        """Prepare the latest sequence for prediction"""
        try:
            # Add technical indicators
            df = self.add_technical_indicators(df)
            if df is None:
                return None

            # Select features for scaling
            feature_columns = [
                "returns",
                "log_returns",
                "volatility",
                "sma_10",
                "sma_20",
                "sma_50",
                "ema_10",
                "ema_20",
                "ema_50",
                "bb_width",
                "rsi",
                "macd",
                "macd_hist",
                "atr",
            ]

            # Fit scaler if not already fitted
            if not self.is_fitted:
                self.scaler.fit(df[feature_columns])
                self.is_fitted = True

            # Select latest data points
            latest_data = df.iloc[-lookback_period:]

            # Scale features
            scaled_data = self.scaler.transform(latest_data[feature_columns])

            # Reshape for model input (batch_size, timesteps, features)
            sequence = np.array([scaled_data], dtype=np.float32)

            return sequence

        except Exception as e:
            logger.error(f"Error preparing latest sequence: {str(e)}")
            return None

    def load_scaler(self, symbol: str, timeframe: str) -> bool:
        """Load saved scaler for specific symbol and timeframe"""
        try:
            scaler_path = TradingConfig.get_scaler_path(symbol, timeframe)
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler file not found: {scaler_path}")
                return False

            self.scaler = joblib.load(scaler_path)
            self.is_fitted = True
            logger.info(f"Scaler loaded successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return False
