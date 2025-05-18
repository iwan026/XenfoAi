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
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

# TensorFlow/Keras imports
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical

# Config
from config import (
    ModelConfig,
    MODELS_DIR,
    DATASETS_DIR,
    LOGS_DIR,
    PLOTS_DIR,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    SYMBOLS,
)

logger = logging.getLogger(__name__)


class ForexModel:
    def __init__(self, symbol: str):
        """Initialize Forex prediction model for H1 timeframe"""
        self.symbol = symbol.upper()
        self.timeframe = "H1"
        self.config = ModelConfig()
        self.scaler = None
        self._is_scaler_fitted = False

        # Model and paths
        self.model = None
        self.model_dir = MODELS_DIR / self.symbol
        self.dataset_dir = DATASETS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load or create model
        self._load_or_create_model()

        logger.info(f"Initialized ForexModel for {self.symbol} {self.timeframe}")

    def _get_scaler_path(self) -> Path:
        """Get path for scaler file"""
        return self.model_dir / f"{self.symbol}_scaler.pkl"

    def _get_model_path(self) -> Path:
        """Get path for model file"""
        return self.model_dir / f"{self.symbol}_model.keras"

    def _get_dataset_path(self) -> Path:
        """Get path for CSV dataset file"""
        return self.dataset_dir / f"{self.symbol}.csv"

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()

    def _calculate_swing_levels(
        self, df: pd.DataFrame, window: int = 20
    ) -> pd.DataFrame:
        """Calculate swing highs and lows"""
        df["swing_high"] = df["high"].rolling(window, center=True).max()
        df["swing_low"] = df["low"].rolling(window, center=True).min()
        df["dist_from_high"] = (df["swing_high"] - df["close"]) / df["swing_high"]
        df["dist_from_low"] = (df["close"] - df["swing_low"]) / df["swing_low"]
        return df

    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features with consistent column order"""
        df = df.copy()

        # Calculate features
        df["atr"] = self._calculate_atr(df)
        df = self._calculate_swing_levels(df)

        # Reorder columns to match ALL_FEATURES order
        features = self.config.ALL_FEATURES
        missing = [f for f in features if f not in df.columns]

        if missing:
            raise ValueError(f"Missing features after processing: {missing}")

        return df[features].copy()

    def _build_model(self):
        """Build CNN-LSTM model with multi-horizon predictions"""
        n_features = len(self.config.ALL_FEATURES)
        input_layer = Input(shape=(self.config.SEQUENCE_LENGTH, n_features))

        # CNN Branch
        cnn = Conv1D(32, 3, activation="relu", padding="same")(input_layer)
        cnn = MaxPooling1D(2)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Conv1D(64, 3, activation="relu", padding="same")(cnn)
        cnn = MaxPooling1D(2)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Flatten()(cnn)

        # LSTM Branch
        lstm = LSTM(64, return_sequences=True)(input_layer)
        lstm = LSTM(32)(lstm)

        # Combined model
        combined = Concatenate()([cnn, lstm])
        combined = Dense(64, activation="relu")(combined)
        combined = Dropout(self.config.DROPOUT_RATE)(combined)

        # Outputs for each horizon
        outputs = []
        for horizon in self.config.PREDICTION_HORIZONS:
            # Direction output (binary)
            direction_output = Dense(
                1, activation="sigmoid", name=f"direction_{horizon}"
            )(combined)

            # Price change output (regression)
            price_output = Dense(1, activation="linear", name=f"price_{horizon}")(
                combined
            )

            outputs.extend([direction_output, price_output])

        model = Model(inputs=input_layer, outputs=outputs)

        # Prepare loss functions and weights for each horizon
        loss_dict = {}
        loss_weights_dict = {}
        for i, horizon in enumerate(self.config.PREDICTION_HORIZONS):
            loss_dict[f"direction_{horizon}"] = "binary_crossentropy"
            loss_dict[f"price_{horizon}"] = "mse"
            loss_weights_dict[f"direction_{horizon}"] = 0.7 / len(
                self.config.PREDICTION_HORIZONS
            )
            loss_weights_dict[f"price_{horizon}"] = 0.3 / len(
                self.config.PREDICTION_HORIZONS
            )

        model.compile(
            optimizer=optimizers.Adam(self.config.LEARNING_RATE),
            loss=loss_dict,
            metrics={
                **{
                    f"direction_{horizon}": "accuracy"
                    for horizon in self.config.PREDICTION_HORIZONS
                },
                **{
                    f"price_{horizon}": "mae"
                    for horizon in self.config.PREDICTION_HORIZONS
                },
            },
            loss_weights=loss_weights_dict,
        )

        return model

    def _load_dataset(self) -> pd.DataFrame:
        """Load and preprocess dataset"""
        csv_path = self._get_dataset_path()
        logger.info(f"Loading dataset from {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        required_cols = ["open", "high", "low", "close", "volume"]

        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing required columns in dataset")

        # Add market structure features
        df = self._add_market_structure_features(df)

        return df

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare sequences and targets for multiple horizons"""
        if self.scaler is None:
            self.scaler = MinMaxScaler()

        # Gunakan ALL_FEATURES bukan PRICE_FEATURES
        feature_columns = self.config.ALL_FEATURES
        data = df[feature_columns].values

        # Normalize data
        if not self._is_scaler_fitted:
            scaled_data = self.scaler.fit_transform(data)
            self._is_scaler_fitted = True
            joblib.dump(self.scaler, self._get_scaler_path())
        else:
            scaled_data = self.scaler.transform(data)

        X, y = [], {}

        # Initialize target dictionaries
        for horizon in self.config.PREDICTION_HORIZONS:
            y[f"direction_{horizon}"] = []
            y[f"price_{horizon}"] = []

        for i in range(
            len(scaled_data)
            - self.config.SEQUENCE_LENGTH
            - max(self.config.PREDICTION_HORIZONS)
        ):
            seq = scaled_data[i : i + self.config.SEQUENCE_LENGTH]
            X.append(seq)

            for horizon in self.config.PREDICTION_HORIZONS:
                if i + self.config.SEQUENCE_LENGTH + horizon >= len(scaled_data):
                    continue

                next_close = scaled_data[
                    i + self.config.SEQUENCE_LENGTH + horizon - 1,
                    self.config.ALL_FEATURES.index("close"),
                ]
                current_close = seq[-1, self.config.ALL_FEATURES.index("close")]

                y[f"direction_{horizon}"].append(1 if next_close > current_close else 0)
                y[f"price_{horizon}"].append(next_close - current_close)

        # Convert to numpy arrays
        X = np.array(X)
        for key in y:
            y[key] = np.array(y[key])

        return X, y

    def train(self) -> bool:
        """Train the model with time series validation"""
        try:
            df = self._load_dataset()

            # Reset scaler for new training
            self.scaler = MinMaxScaler()
            self._is_scaler_fitted = False

            X, y = self._prepare_data(df)

            tscv = TimeSeriesSplit(n_splits=3)
            best_history = None
            best_val_loss = float("inf")

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = {k: v[train_idx] for k, v in y.items()}
                y_val = {k: v[val_idx] for k, v in y.items()}

                history = self.model.fit(
                    X_train,
                    y_train,
                    batch_size=self.config.BATCH_SIZE,
                    epochs=self.config.EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks=[
                        callbacks.EarlyStopping(
                            patience=self.config.PATIENCE, restore_best_weights=True
                        ),
                        callbacks.ModelCheckpoint(
                            filepath=str(self._get_model_path()), save_best_only=True
                        ),
                    ],
                    verbose=1,
                )

                current_val_loss = min(history.history["val_loss"])
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_history = history

            # Save training plots
            if best_history:
                self._plot_training_history(best_history, self.symbol)

            logger.info(f"Training completed with best val loss: {best_val_loss:.4f}")

            # Ensure scaler is saved
            if self._is_scaler_fitted:
                joblib.dump(self.scaler, self._get_scaler_path())

            return True

        except Exception as e:
            self.scaler = None
            self._is_scaler_fitted = False
            logger.error(f"Training failed: {str(e)}")
            return False

    def predict(self, use_realtime: bool = True) -> Optional[Dict]:
        """Make prediction for multiple horizons"""
        try:
            if not self._is_scaler_fitted:
                raise NotFittedError("Scaler not fitted. Train model first!")

            # Get data
            df = self._get_realtime_data() if use_realtime else self._load_dataset()
            df = self._add_market_structure_features(df)

            # Validate data
            if len(df) < self.config.SEQUENCE_LENGTH:
                raise ValueError("Not enough data for prediction")

            # Transform data
            scaled_data = self.scaler.transform(df)
            seq = scaled_data[-self.config.SEQUENCE_LENGTH :]
            seq = np.expand_dims(seq, axis=0)

            # Make prediction
            predictions = self.model.predict(seq, verbose=0)

            # Organize predictions by horizon
            result = {
                "current_close": float(df["close"].iloc[-1]),
                "predictions": {},
                "chart_path": None,
            }

            for i, horizon in enumerate(self.config.PREDICTION_HORIZONS):
                direction_idx = i * 2
                price_idx = i * 2 + 1

                direction_pred = predictions[direction_idx][0][0]
                price_pred = predictions[price_idx][0][0]

                result["predictions"][horizon] = {
                    "direction": float(direction_pred),
                    "price_change": float(price_pred),
                    "next_close": float(df["close"].iloc[-1] + price_pred),
                }

            # Generate chart
            chart_path = self._plot_chart(df, result)
            result["chart_path"] = chart_path

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

    def _get_realtime_data(self, num_candles: int = 180) -> pd.DataFrame:
        """Get realtime data from MT5"""
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
            df = df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "tick_volume": "volume",
                }
            )

            return df[["open", "high", "low", "close", "volume"]]

        finally:
            mt5.shutdown()

    def _plot_training_history(self, history, model_name: str):
        """Plot and save training history metrics"""
        plt.figure(figsize=(15, 10))

        # Plot Losses
        plt.subplot(2, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot Direction Accuracy (first horizon only for simplicity)
        first_horizon = self.config.PREDICTION_HORIZONS[0]
        plt.subplot(2, 2, 2)
        plt.plot(
            history.history[f"direction_{first_horizon}_accuracy"], label="Train Acc"
        )
        plt.plot(
            history.history[f"val_direction_{first_horizon}_accuracy"], label="Val Acc"
        )
        plt.title(f"Direction Accuracy (Horizon {first_horizon})")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot Price MAE (first horizon only for simplicity)
        plt.subplot(2, 2, 3)
        plt.plot(history.history[f"price_{first_horizon}_mae"], label="Train MAE")
        plt.plot(history.history[f"val_price_{first_horizon}_mae"], label="Val MAE")
        plt.title(f"Price MAE (Horizon {first_horizon})")
        plt.ylabel("MAE")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot Direction Loss (first horizon only for simplicity)
        plt.subplot(2, 2, 4)
        plt.plot(history.history[f"direction_{first_horizon}_loss"], label="Train Loss")
        plt.plot(
            history.history[f"val_direction_{first_horizon}_loss"], label="Val Loss"
        )
        plt.title(f"Direction Loss (Horizon {first_horizon})")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plot_path = PLOTS_DIR / f"{model_name}_training.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved training plot to {plot_path}")

    def _plot_chart(self, df: pd.DataFrame, predictions: Dict) -> Path:
        """Generate trading chart with predictions"""
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#131722")

        # Prepare data
        display_df = df.tail(100).copy()
        display_df["date_num"] = mdates.date2num(display_df.index.to_pydatetime())
        ohlc = display_df[["date_num", "open", "high", "low", "close"]].values

        # Plot candlesticks
        candlestick_ohlc(
            ax, ohlc, width=0.6 / 24, colorup="#26a69a", colordown="#ef5350", alpha=1.0
        )

        # Add prediction markers for each horizon
        colors = [
            "#4CAF50",
            "#FFC107",
            "#2196F3",
        ]  # Green, Yellow, Blue for different horizons
        for i, (horizon, pred) in enumerate(predictions["predictions"].items()):
            pred_color = colors[i % len(colors)]
            ax.axhline(
                pred["next_close"],
                color=pred_color,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=f"{horizon}h Pred: {pred['next_close']:.5f}",
            )

        # Style chart
        ax.set_title(
            f"{self.symbol} {self.timeframe} - Multi-Horizon Predictions",
            color="white",
        )
        ax.set_facecolor("#131722")
        ax.grid(color="#2a2e39", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="both", colors="#787b86")
        for spine in ax.spines.values():
            spine.set_color("#2a2e39")

        # Add legend
        legend = ax.legend(loc="upper left", facecolor="#131722", edgecolor="#2a2e39")
        for text in legend.get_texts():
            text.set_color("#787b86")

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save plot
        plot_path = PLOTS_DIR / f"{self.symbol}_prediction.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

    def _load_or_create_model(self):
        """Load existing or create new model"""
        model_path = self._get_model_path()

        if model_path.exists():
            logger.info(f"Loading existing model from {model_path}")
            self.model = models.load_model(model_path)
            scaler_path = self._get_scaler_path()
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self._is_scaler_fitted = True
                logger.info(f"Loaded scaler state for {self.symbol}")
        else:
            logger.info("Creating new model")
            self.model = self._build_model()
            self.model.summary(print_fn=logger.info)
