import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

from config import ModelConfig, MODELS_DIR, DATASETS_DIR, LOGS_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)


class ForexModel:
    def __init__(self, symbol: str, timeframe: str):
        """Initialize Forex prediction model"""
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.config = ModelConfig()

        # Model and paths
        self.model = None
        self.model_dir = MODELS_DIR / self.symbol
        self.dataset_dir = DATASETS_DIR / self.symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load or create model
        self._load_or_create_model()

        logger.info(f"Initialized ForexModel for {self.symbol} {self.timeframe}")

    def _get_model_path(self) -> Path:
        """Get path for model file"""
        return self.model_dir / f"{self.symbol}_{self.timeframe}_model.keras"

    def _get_dataset_path(self) -> Path:
        """Get path for CSV dataset file"""
        return self.dataset_dir / f"{self.symbol}_{self.timeframe}.csv"

    def _build_model(self):
        """Build CNN-LSTM model architecture"""
        # Technical features input (for LSTM)
        tech_input = layers.Input(
            shape=(self.config.SEQUENCE_LENGTH, len(self.config.TECHNICAL_INDICATORS)),
            name="technical_input",
        )

        # Price features input (for CNN)
        price_input = layers.Input(
            shape=(self.config.SEQUENCE_LENGTH, len(self.config.PRICE_FEATURES), 1),
            name="price_input",
        )

        # CNN Branch
        cnn = price_input
        for filters, kernel_size, pool_size in zip(
            self.config.CNN_FILTERS,
            self.config.CNN_KERNEL_SIZES,
            self.config.CNN_POOL_SIZES,
        ):
            cnn = layers.Conv2D(
                filters, kernel_size, activation="relu", padding="same"
            )(cnn)
            cnn = layers.MaxPooling2D(pool_size)(cnn)
            cnn = layers.BatchNormalization()(cnn)

        cnn = layers.Flatten()(cnn)
        cnn = layers.Dropout(self.config.DROPOUT_RATE)(cnn)

        # LSTM Branch
        lstm = layers.LSTM(self.config.LSTM_UNITS, dropout=self.config.DROPOUT_RATE)(
            tech_input
        )

        # Combine branches
        combined = layers.concatenate([cnn, lstm])
        output = layers.Dense(1, activation="sigmoid")(combined)

        # Create model
        model = models.Model(inputs=[tech_input, price_input], outputs=output)

        model.compile(
            optimizer=optimizers.Adam(self.config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _load_or_create_model(self):
        """Load existing model or create new one"""
        model_path = self._get_model_path()

        try:
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                self.model = models.load_model(model_path)
            else:
                logger.info("Creating new model")
                self.model = self._build_model()
                self.model.summary(print_fn=logger.info)

        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from price data"""
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std
        df["bb_lower"] = df["bb_middle"] - 2 * std

        # Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=14).mean()

        # ADX
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = df["high"] - df["low"]
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14).mean() / tr.ewm(alpha=1 / 14).mean())
        minus_di = 100 * (
            minus_dm.ewm(alpha=1 / 14).mean() / tr.ewm(alpha=1 / 14).mean()
        )
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx_14"] = dx.ewm(alpha=1 / 14).mean()

        # CCI
        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["cci_20"] = (tp - tp.rolling(window=20).mean()) / (
            0.015 * tp.rolling(window=20).std()
        )

        # Stochastic
        low_min = df["low"].rolling(window=14).min()
        high_max = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # Clean NaN values
        df = df.fillna(method="ffill").fillna(method="bfill")

        return df

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training/prediction"""
        # Calculate indicators
        df = self._calculate_technical_indicators(df)

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(df) - self.config.SEQUENCE_LENGTH - 1):
            seq = df.iloc[i : i + self.config.SEQUENCE_LENGTH]
            next_close = df["close"].iloc[i + self.config.SEQUENCE_LENGTH]
            current_close = df["close"].iloc[i + self.config.SEQUENCE_LENGTH - 1]

            # Calculate price change percentage
            price_change = (next_close - current_close) / current_close

            # Create binary target
            target = 1 if price_change > self.config.TARGET_THRESHOLD else 0
            targets.append(target)

            sequences.append(seq)

        # Prepare feature arrays
        tech_features = np.array(
            [seq[self.config.TECHNICAL_INDICATORS].values for seq in sequences]
        )

        price_features = np.array(
            [seq[self.config.PRICE_FEATURES].values for seq in sequences]
        ).reshape(-1, self.config.SEQUENCE_LENGTH, len(self.config.PRICE_FEATURES), 1)

        targets = np.array(targets)

        return tech_features, price_features, targets

    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file"""
        csv_path = self._get_dataset_path()
        logger.info(f"Loading dataset from {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def _plot_training_history(self, history, model_name: str):
        """Plot training history and save to file"""
        plt.figure(figsize=(12, 6))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        # Save plot
        plot_path = PLOTS_DIR / f"{model_name}_training.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Saved training plot to {plot_path}")

    def train(self) -> bool:
        """Train the model"""
        try:
            logger.info(f"Starting training for {self.symbol} {self.timeframe}")

            # Load and prepare data
            df = self._load_dataset()
            tech_features, price_features, targets = self._prepare_data(df)

            # Split data
            X_tech_train, X_tech_val, X_price_train, X_price_val, y_train, y_val = (
                train_test_split(
                    tech_features,
                    price_features,
                    targets,
                    test_size=self.config.VALIDATION_SPLIT,
                    shuffle=False,
                )
            )

            # Compute class weights
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights = dict(enumerate(class_weights))

            logger.info(f"Class weights: {class_weights}")
            logger.info(f"Training samples: {len(y_train)}")
            logger.info(f"Validation samples: {len(y_val)}")

            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.PATIENCE,
                    restore_best_weights=True,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
                ),
                callbacks.ModelCheckpoint(
                    filepath=str(self._get_model_path()),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]

            # Train model
            history = self.model.fit(
                [X_tech_train, X_price_train],
                y_train,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_data=([X_tech_val, X_price_val], y_val),
                class_weight=class_weights,
                callbacks=callbacks_list,
                verbose=1,
            )

            # Save training plot
            model_name = f"{self.symbol}_{self.timeframe}"
            self._plot_training_history(history, model_name)

            logger.info("Training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """Make prediction on given data"""
        try:
            # Prepare data
            tech_features, price_features, _ = self._prepare_data(df)

            if len(tech_features) == 0:
                logger.warning("Not enough data for prediction")
                return None

            # Use the most recent sequence
            tech_seq = tech_features[-1:].reshape(1, self.config.SEQUENCE_LENGTH, -1)
            price_seq = price_features[-1:].reshape(
                1, self.config.SEQUENCE_LENGTH, len(self.config.PRICE_FEATURES), 1
            )

            # Make prediction
            prediction = self.model.predict([tech_seq, price_seq], verbose=0)[0][0]

            return float(prediction)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
