import numpy as np
import pandas as pd
import tensorflow as tf
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import joblib
from mplfinance.original_flavor import candlestick_ohlc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.stattools import acf

# TensorFlow/Keras imports
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks

# Config
from config import (
    ModelConfig,
    MODELS_DIR,
    DATASETS_DIR,
    PLOTS_DIR,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
)

logger = logging.getLogger(__name__)


class ForexModel:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.timeframe = "H1"
        self.config = ModelConfig()
        self.scaler = None
        self._is_scaler_fitted = False
        self.model = None
        self.model_dir = MODELS_DIR / self.symbol
        self.dataset_dir = DATASETS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect optimal parameters
        self._auto_detect_parameters()
        self._load_or_create_model()

    def _auto_detect_parameters(self):
        """Auto-detect sequence length and prediction horizon"""
        df = self._load_dataset()

        # Auto-detect sequence length using ACF
        close_prices = df["close"].values
        acf_values = acf(close_prices, nlags=100)
        threshold = 0.2
        optimal_seq_len = next(
            (i for i, val in enumerate(acf_values) if abs(val) < threshold), 60
        )
        self.config.SEQUENCE_LENGTH = max(30, min(optimal_seq_len, 120))

        # Auto-detect horizon based on volatility (using ATR)
        atr = self._calculate_atr(df)
        volatility_score = atr.mean() / df["close"].mean()
        self.config.PREDICTION_HORIZON = min(
            int(4 * (1 + volatility_score)), self.config.MAX_HORIZON
        )

        logger.info(
            f"Auto-detected params for {self.symbol}: seq_len={self.config.SEQUENCE_LENGTH}, horizon={self.config.PREDICTION_HORIZON}"
        )

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate ATR
        df["atr"] = self._calculate_atr(df)

        # Add price momentum features
        df["price_momentum_1"] = df["close"].pct_change(1)
        df["price_momentum_3"] = df["close"].pct_change(3)

        # Add moving averages
        df["ma_fast"] = df["close"].rolling(window=10).mean()
        df["ma_slow"] = df["close"].rolling(window=20).mean()

        # Add MA crossover signal
        df["ma_crossover"] = (
            (df["ma_fast"] > df["ma_slow"])
            & (df["ma_fast"].shift(1) <= df["ma_slow"].shift(1))
        ).astype(int) - (
            (df["ma_fast"] < df["ma_slow"])
            & (df["ma_fast"].shift(1) >= df["ma_slow"].shift(1))
        ).astype(int)

        # Add directional movement
        df["direction"] = np.sign(df["close"] - df["close"].shift(1))

        # Ensure we have all required features
        return df[list(set(self.config.ALL_FEATURES))].fillna(0)

    def _load_dataset(self) -> pd.DataFrame:
        csv_path = self._get_dataset_path()
        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in dataset")
        return self._add_technical_features(df)

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))

        data = df[self.config.ALL_FEATURES].values

        if not self._is_scaler_fitted:
            scaled_data = self.scaler.fit_transform(data)
            self._is_scaler_fitted = True
            joblib.dump(self.scaler, self._get_scaler_path())
        else:
            scaled_data = self.scaler.transform(data)

        X, y = [], []
        close_idx = self.config.ALL_FEATURES.index("close")

        for i in range(
            len(scaled_data)
            - self.config.SEQUENCE_LENGTH
            - self.config.PREDICTION_HORIZON
        ):
            seq = scaled_data[i : i + self.config.SEQUENCE_LENGTH]
            X.append(seq)

            future_close = scaled_data[
                i + self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON - 1,
                close_idx,
            ]
            current_close = seq[-1, close_idx]

            # Direction prediction (1 for up, 0 for down or no change)
            price_direction = 1 if future_close > current_close else 0
            y.append(price_direction)

        return np.array(X), np.array(y)

    def _build_model(self):
        # Direction classification model instead of regression
        input_layer = Input(
            shape=(self.config.SEQUENCE_LENGTH, len(self.config.ALL_FEATURES))
        )

        # LSTM layers
        x = LSTM(64, return_sequences=True)(input_layer)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)

        # Output layer for binary classification (direction)
        output = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=optimizers.Adam(self.config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self) -> bool:
        try:
            df = self._load_dataset()
            self.scaler = None
            self._is_scaler_fitted = False
            X, y = self._prepare_data(df)

            tscv = TimeSeriesSplit(n_splits=3)
            best_val_accuracy = 0

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                history = self.model.fit(
                    X_train,
                    y_train,
                    batch_size=self.config.BATCH_SIZE,
                    epochs=self.config.EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks=[
                        callbacks.EarlyStopping(
                            monitor="val_accuracy",
                            patience=self.config.PATIENCE,
                            restore_best_weights=True,
                        ),
                        callbacks.ModelCheckpoint(
                            filepath=str(self._get_model_path()),
                            save_best_only=True,
                            monitor="val_accuracy",
                        ),
                    ],
                    verbose=1,
                )

                current_val_accuracy = max(history.history["val_accuracy"])
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy

            logger.info(
                f"Training completed with best val accuracy: {best_val_accuracy:.4f}"
            )
            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def predict(self, use_realtime: bool = True) -> Optional[Dict]:
        try:
            if not self._is_scaler_fitted:
                raise NotFittedError("Scaler not fitted. Train model first!")

            df = self._get_realtime_data() if use_realtime else self._load_dataset()
            df = self._add_technical_features(df)

            if len(df) < self.config.SEQUENCE_LENGTH:
                raise ValueError("Not enough data for prediction")

            scaled_data = self.scaler.transform(df.values)
            seq = scaled_data[-self.config.SEQUENCE_LENGTH :]
            seq = np.expand_dims(seq, axis=0)

            # Get direction probability
            direction_prob = float(self.model.predict(seq, verbose=0)[0][0])

            # Determine direction
            predicted_direction = "UP" if direction_prob > 0.5 else "DOWN"
            confidence = max(direction_prob, 1 - direction_prob)

            # Calculate potential price change based on ATR
            current_close = float(df["close"].iloc[-1])
            current_atr = float(df["atr"].iloc[-1])

            # Expected price change based on ATR and confidence
            if predicted_direction == "UP":
                price_change = current_atr * confidence
                predicted_price = current_close + price_change
            else:
                price_change = -current_atr * confidence
                predicted_price = current_close + price_change

            result = {
                "current_close": current_close,
                "prediction": {
                    "horizon": self.config.PREDICTION_HORIZON,
                    "direction": predicted_direction,
                    "confidence": confidence * 100,  # Convert to percentage
                    "next_close": predicted_price,
                    "price_change": price_change,
                },
                "chart_path": self._plot_chart(
                    df, current_close, predicted_price, predicted_direction
                ),
            }
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

    def _plot_chart(
        self,
        df: pd.DataFrame,
        current_price: float,
        predicted_price: float,
        direction: str,
    ) -> Path:
        # TradingView-like style
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#131722")

        # Display more candles for context
        display_df = df.tail(120).copy()
        display_df["date_num"] = mdates.date2num(display_df.index.to_pydatetime())
        ohlc = display_df[["date_num", "open", "high", "low", "close"]].values

        # Draw candlesticks
        candlestick_ohlc(
            ax, ohlc, width=0.6 / 24, colorup="#26a69a", colordown="#ef5350", alpha=0.9
        )

        # Plot moving averages for context
        if "ma_fast" in display_df.columns and "ma_slow" in display_df.columns:
            ax.plot(
                display_df.index,
                display_df["ma_fast"],
                color="#2962FF",
                linewidth=1.2,
                alpha=0.7,
                label=f"MA(10)",
            )
            ax.plot(
                display_df.index,
                display_df["ma_slow"],
                color="#FF6D00",
                linewidth=1.2,
                alpha=0.7,
                label=f"MA(20)",
            )

        # Prediction visualization - TradingView style
        last_date = display_df.index[-1]
        future_date = last_date + timedelta(hours=self.config.PREDICTION_HORIZON)

        # Draw prediction line
        ax.plot(
            [last_date, future_date],
            [current_price, predicted_price],
            color="#4CAF50" if direction == "UP" else "#F44336",
            linestyle="-",
            linewidth=2,
            marker="o",
        )

        # Add prediction target point
        ax.scatter(
            [future_date],
            [predicted_price],
            s=100,
            color="#4CAF50" if direction == "UP" else "#F44336",
            zorder=5,
            edgecolors="#FFFFFF",
        )

        # Add prediction label
        ax.annotate(
            f"{predicted_price:.5f}",
            xy=(future_date, predicted_price),
            xytext=(10, 0),
            textcoords="offset points",
            color="#FFFFFF",
            fontweight="bold",
            fontsize=10,
        )

        # TradingView-style visual enhancements
        ax.set_title(
            f"{self.symbol} {self.timeframe} - Prediction ({self.config.PREDICTION_HORIZON}h)",
            color="white",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_facecolor("#131722")
        ax.grid(color="#2a2e39", linestyle="-", linewidth=0.5, alpha=0.7)

        # Only show a subset of dates on x-axis for cleaner look
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.xticks(rotation=45)

        # Add direction arrow and confidence in the corner
        arrow = "▲" if direction == "UP" else "▼"
        color = "#26a69a" if direction == "UP" else "#ef5350"
        prediction_text = f"{arrow} {direction} ({predicted_price:.5f})"

        # Add text box with prediction in top-left corner
        ax.text(
            0.02,
            0.98,
            prediction_text,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#1E222D",
                edgecolor=color,
                alpha=0.8,
            ),
            color=color,
        )

        # Add legend with custom style
        legend = ax.legend(
            loc="upper right", facecolor="#131722", edgecolor="#2a2e39", framealpha=0.9
        )
        for text in legend.get_texts():
            text.set_color("white")

        # Adjust layout
        plt.tight_layout()

        # Save the chart
        plot_path = PLOTS_DIR / f"{self.symbol}_prediction.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return plot_path

    def _get_realtime_data(self, num_candles: int = 200) -> pd.DataFrame:
        try:
            if not mt5.initialize(
                login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER
            ):
                raise Exception(f"MT5 init failed: {mt5.last_error()}")

            rates = mt5.copy_rates_from_pos(
                self.symbol, mt5.TIMEFRAME_H1, 0, num_candles
            )
            if rates is None:
                raise Exception(f"Failed to get rates: {mt5.last_error()}")

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df = df.rename(columns={"tick_volume": "volume"})
            return df[["open", "high", "low", "close", "volume"]]

        finally:
            mt5.shutdown()

    def _get_scaler_path(self) -> Path:
        return self.model_dir / f"{self.symbol}_scaler.pkl"

    def _get_model_path(self) -> Path:
        return self.model_dir / f"{self.symbol}_model.keras"

    def _get_dataset_path(self) -> Path:
        return self.dataset_dir / f"{self.symbol}.csv"

    def _load_or_create_model(self):
        model_path = self._get_model_path()
        scaler_path = self._get_scaler_path()

        if model_path.exists() and scaler_path.exists():
            self.model = models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self._is_scaler_fitted = True
            logger.info(f"Loaded existing model for {self.symbol}")
        else:
            self.model = self._build_model()
            logger.info(f"Created new model for {self.symbol}")
