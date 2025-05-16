import numpy as np
import pandas as pd
import tensorflow as tf
import MetaTrader5 as mt5
import matplotlib.dates as mdates
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

from config import (
    ModelConfig,
    MODELS_DIR,
    DATASETS_DIR,
    LOGS_DIR,
    PLOTS_DIR,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    TIMEFRAMES,
)

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
        """Build CNN-LSTM model architecture with reduced complexity"""
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

        # CNN Branch with reduced layers and regularization
        cnn = price_input
        for filters, kernel_size, pool_size in zip(
            self.config.CNN_FILTERS,
            self.config.CNN_KERNEL_SIZES,
            self.config.CNN_POOL_SIZES,
        ):
            cnn = layers.Conv2D(
                filters,
                kernel_size,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.l2(self.config.L2_REG),
            )(cnn)
            cnn = layers.MaxPooling2D(pool_size)(cnn)
            cnn = layers.BatchNormalization()(cnn)

        cnn = layers.Flatten()(cnn)
        cnn = layers.Dropout(self.config.DROPOUT_RATE)(cnn)

        # LSTM Branch with regularization
        lstm = layers.LSTM(
            self.config.LSTM_UNITS,
            dropout=self.config.DROPOUT_RATE,
            kernel_regularizer=regularizers.l2(self.config.L2_REG),
        )(tech_input)

        # Combine branches
        combined = layers.concatenate([cnn, lstm])

        # Dense layer with regularization
        dense = layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.config.L2_REG),
        )(combined)
        dense = layers.Dropout(self.config.DROPOUT_RATE)(dense)

        output = layers.Dense(1, activation="sigmoid")(dense)

        # Create model
        model = models.Model(inputs=[tech_input, price_input], outputs=output)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        return model

    def _load_or_create_model(self):
        """Load existing model or create new one"""
        model_path = self._get_model_path()

        try:
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                # Load model tanpa optimizer state
                self.model = models.load_model(model_path, compile=False)
                # Recompile dengan setting baru
                self.model.compile(
                    optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                    loss="binary_crossentropy",
                    metrics=[
                        "accuracy",
                        tf.keras.metrics.Precision(name="precision"),
                        tf.keras.metrics.Recall(name="recall"),
                        tf.keras.metrics.AUC(name="auc"),
                    ],
                )
            else:
                logger.info("Creating new model")
                self.model = self._build_model()
                self.model.summary(print_fn=logger.info)
        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate reduced set of technical indicators from price data"""
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD (simplified - only MACD line)
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std
        df["bb_lower"] = df["bb_middle"] - 2 * std

        # Moving Averages
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

            # Calculate price change percentage with increased threshold
            price_change = (next_close - current_close) / current_close
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

    def _plot_chart(self, df: pd.DataFrame, prediction: float) -> Path:
        """Generate trading chart with proper TradingView-like candlesticks"""
        try:
            # Prepare figure with TradingView-like style
            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                facecolor="#131722",
            )
            fig.suptitle(
                f"{self.symbol} {self.timeframe} - Prediction: {'BUY' if prediction > 0.5 else 'SELL'} ({prediction:.2%})",
                color="white",
                fontsize=14,
            )

            # Convert datetime to mdates
            df = df.copy()
            df["date_num"] = mdates.date2num(df.index.to_pydatetime())

            # Get last 50 candles for display
            display_df = df.tail(50).copy()

            # Create proper OHLC data for candlestick
            ohlc = display_df[["date_num", "open", "high", "low", "close"]].values

            # TradingView colors
            up_color = "#26a69a"  # Green for bullish candles
            down_color = "#ef5350"  # Red for bearish candles

            # Use the proper mplfinance candlestick function
            # Make sure wick and body are properly displayed
            candlestick_ohlc(
                ax1,
                ohlc,
                width=0.6 / 24,  # Adjusted for proper width - depends on timeframe
                colorup=up_color,
                colordown=down_color,
                alpha=1.0,
            )

            # Add Trading View-like grid
            ax1.grid(color="#2a2e39", linestyle="-", linewidth=0.5, alpha=0.5)
            ax1.set_facecolor("#131722")  # TradingView background color

            # Set y-axis to appropriate price range with some padding
            price_min = display_df["low"].min()
            price_max = display_df["high"].max()
            price_range = price_max - price_min
            ax1.set_ylim(price_min - 0.1 * price_range, price_max + 0.1 * price_range)

            # Set axis colors
            ax1.tick_params(axis="both", colors="#787b86", which="both")
            for spine in ax1.spines.values():
                spine.set_color("#2a2e39")

            # Plot EMAs with TradingView-like colors
            ax1.plot(
                display_df.index,
                display_df["ema_9"],
                color="#e91e63",
                label="EMA 9",
                linewidth=1.2,
                alpha=0.9,
            )
            ax1.plot(
                display_df.index,
                display_df["ema_21"],
                color="#9c27b0",
                label="EMA 21",
                linewidth=1.2,
                alpha=0.9,
            )
            ax1.plot(
                display_df.index,
                display_df["sma_50"],
                color="#2196f3",
                label="SMA 50",
                linewidth=1.2,
                alpha=0.9,
            )

            # Add TradingView-like legend
            legend = ax1.legend(loc="upper left", frameon=True, fancybox=True)
            legend.get_frame().set_facecolor("#131722")
            legend.get_frame().set_edgecolor("#2a2e39")
            for text in legend.get_texts():
                text.set_color("#787b86")

            # Plot RSI with TradingView style
            ax2.plot(
                display_df.index,
                display_df["rsi_14"],
                color="#2196f3",
                label="RSI 14",
                linewidth=1.2,
            )

            # Overbought/Oversold lines
            ax2.axhline(70, color="#787b86", linestyle="--", linewidth=0.8, alpha=0.5)
            ax2.axhline(30, color="#787b86", linestyle="--", linewidth=0.8, alpha=0.5)
            ax2.axhline(50, color="#787b86", linestyle="-", linewidth=0.5, alpha=0.3)

            # RSI panel styling
            ax2.set_ylim(0, 100)
            ax2.set_yticks([0, 30, 50, 70, 100])
            ax2.grid(color="#2a2e39", linestyle="-", linewidth=0.5, alpha=0.5)
            ax2.set_facecolor("#131722")

            # Set axis colors for RSI panel
            ax2.tick_params(axis="both", colors="#787b86", which="both")
            for spine in ax2.spines.values():
                spine.set_color("#2a2e39")

            # RSI legend
            legend2 = ax2.legend(loc="upper left", frameon=True, fancybox=True)
            legend2.get_frame().set_facecolor("#131722")
            legend2.get_frame().set_edgecolor("#2a2e39")
            for text in legend2.get_texts():
                text.set_color("#787b86")

            # Format x-axis for both panels
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(rotation=45)

            # Add volume as histogram at the bottom of price chart
            volume_ax = ax1.twinx()
            volume_max = display_df["volume"].max()
            volume_ax.set_ylim(0, volume_max * 3)
            volume_ax.set_yticks([])  # Hide volume scale

            # Calculate volume bar colors based on price change
            colors = []
            for i in range(len(display_df)):
                if display_df["close"].iloc[i] >= display_df["open"].iloc[i]:
                    colors.append(up_color)
                else:
                    colors.append(down_color)

            # Plot volume bars with low alpha to not obscure price action
            for i, (idx, row) in enumerate(display_df.iterrows()):
                volume_ax.bar(
                    idx,
                    row["volume"],
                    color=colors[i],
                    alpha=0.3,
                    width=0.6 / 24,  # Width should match candlesticks
                )

            plt.tight_layout()

            # Save plot with higher resolution
            plot_path = PLOTS_DIR / f"{self.symbol}_{self.timeframe}_chart.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            return plot_path

        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            raise

    def _plot_metrics(self, history, model_name: str):
        """Plot training metrics including confusion matrix"""
        plt.figure(figsize=(18, 12))

        # Plot loss
        plt.subplot(2, 3, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 3, 2)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Val Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot precision
        plt.subplot(2, 3, 3)
        plt.plot(history.history["precision"], label="Train Precision")
        plt.plot(history.history["val_precision"], label="Val Precision")
        plt.title("Model Precision")
        plt.ylabel("Precision")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot recall
        plt.subplot(2, 3, 4)
        plt.plot(history.history["recall"], label="Train Recall")
        plt.plot(history.history["val_recall"], label="Val Recall")
        plt.title("Model Recall")
        plt.ylabel("Recall")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot AUC
        plt.subplot(2, 3, 5)
        plt.plot(history.history["auc"], label="Train AUC")
        plt.plot(history.history["val_auc"], label="Val AUC")
        plt.title("Model AUC")
        plt.ylabel("AUC")
        plt.xlabel("Epoch")
        plt.legend()

        # Save plot
        plot_path = PLOTS_DIR / f"{model_name}_metrics.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Saved metrics plot to {plot_path}")

    def _get_realtime_data(self, num_candles: int = 60) -> pd.DataFrame:
        """Get realtime data from MT5 and calculate indicators"""
        try:
            if not mt5.initialize(
                login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER
            ):
                raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

            # Get current rates
            rates = mt5.copy_rates_from_pos(
                self.symbol, getattr(mt5, TIMEFRAMES[self.timeframe]), 0, num_candles
            )

            if rates is None:
                raise Exception(f"Failed to get rates: {mt5.last_error()}")

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Standardize column names
            df = df.rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "tick_volume": "volume",
                }
            )

            # Ensure we have required OHLCV columns
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    raise Exception(f"Missing required price column: {col}")

            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)

            return df

        except Exception as e:
            logger.error(f"Error getting realtime data: {e}")
            raise
        finally:
            mt5.shutdown()

    def train(self) -> bool:
        """Train the model with time series split validation"""
        try:
            logger.info(f"Starting training for {self.symbol} {self.timeframe}")

            # Load and prepare data
            df = self._load_dataset()
            tech_features, price_features, targets = self._prepare_data(df)

            # Check class distribution
            class_counts = np.bincount(targets)
            logger.info(
                f"Class distribution: 0: {class_counts[0]}, 1: {class_counts[1]}"
            )

            # Time Series Split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            fold = 0
            best_val_loss = float("inf")

            for train_index, val_index in tscv.split(tech_features):
                fold += 1
                logger.info(f"Training fold {fold}")

                X_tech_train, X_tech_val = (
                    tech_features[train_index],
                    tech_features[val_index],
                )
                X_price_train, X_price_val = (
                    price_features[train_index],
                    price_features[val_index],
                )
                y_train, y_val = targets[train_index], targets[val_index]

                # Compute class weights
                class_weights = compute_class_weight(
                    "balanced", classes=np.unique(y_train), y=y_train
                )
                class_weights = dict(enumerate(class_weights))

                logger.info(f"Fold {fold} class weights: {class_weights}")
                logger.info(f"Fold {fold} training samples: {len(y_train)}")
                logger.info(f"Fold {fold} validation samples: {len(y_val)}")

                # Callbacks
                callbacks_list = [
                    callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.config.PATIENCE,
                        restore_best_weights=True,
                        verbose=1,
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=3,
                        min_lr=1e-6,
                        verbose=1,
                    ),
                    callbacks.ModelCheckpoint(
                        filepath=str(self._get_model_path()),
                        monitor="val_loss",
                        save_best_only=True,
                        save_weights_only=False,
                    ),
                    # Tambahkan LearningRateScheduler sebagai callback terpisah
                    callbacks.LearningRateScheduler(
                        lambda epoch, lr: lr * 0.9 if epoch % 5 == 0 else lr
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

                # Track best model
                current_val_loss = min(history.history["val_loss"])
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_history = history

            # Save training plots and metrics
            model_name = f"{self.symbol}_{self.timeframe}"
            self._plot_metrics(best_history, model_name)

            # Generate and save confusion matrix
            y_pred = (self.model.predict([X_tech_val, X_price_val]) > 0.5).astype(int)
            cm = confusion_matrix(y_val, y_pred)

            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            cm_path = PLOTS_DIR / f"{model_name}_cm.png"
            plt.savefig(cm_path)
            plt.close()

            # Log classification report
            report = classification_report(y_val, y_pred, target_names=["Sell", "Buy"])
            logger.info(f"Classification Report:\n{report}")

            logger.info("Training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return False

    def predict(self, use_realtime: bool = True) -> Optional[Dict]:
        """Make prediction and return result with chart path"""
        try:
            # Get data
            if use_realtime:
                df = self._get_realtime_data(self.config.SEQUENCE_LENGTH + 60)
            else:
                df = self._load_dataset()

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

            # Generate chart
            chart_path = self._plot_chart(df, prediction)

            # Get latest indicators
            latest = df.iloc[-1]

            return {
                "prediction": float(prediction),
                "confidence": abs(prediction - 0.5) * 2,
                "chart_path": chart_path,
                "indicators": {
                    "ema_9": latest.get("ema_9", None),
                    "ema_21": latest.get("ema_21", None),
                    "sma_50": latest.get("sma_50", None),
                    "rsi": latest.get("rsi_14", None),
                    "macd": latest.get("macd", None),
                },
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
