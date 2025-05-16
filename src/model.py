import numpy as np
import pandas as pd
import talib
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
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

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
        self.timeframe = "H1"  # Fixed to H1 timeframe
        self.config = ModelConfig()

        # Model and paths
        self.model = None
        self.model_dir = MODELS_DIR / self.symbol
        self.dataset_dir = DATASETS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load or create model
        self._load_or_create_model()

        logger.info(f"Initialized ForexModel for {self.symbol} {self.timeframe}")

    def _get_model_path(self) -> Path:
        """Get path for model file"""
        return self.model_dir / f"{self.symbol}_model.keras"

    def _get_dataset_path(self) -> Path:
        """Get path for CSV dataset file"""
        return self.dataset_dir / f"{self.symbol}.csv"

    def _build_model(self):
        """Build CNN-LSTM model architecture with multi-output"""
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

        # LSTM Branch
        lstm = layers.LSTM(
            self.config.LSTM_UNITS,
            dropout=self.config.DROPOUT_RATE,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(self.config.L2_REG),
        )(tech_input)
        lstm = layers.LSTM(
            self.config.LSTM_UNITS // 2,
            dropout=self.config.DROPOUT_RATE,
            kernel_regularizer=regularizers.l2(self.config.L2_REG),
        )(lstm)

        # Combine branches
        combined = layers.concatenate([cnn, lstm])

        # Shared dense layers
        dense = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.config.L2_REG),
        )(combined)
        dense = layers.Dropout(self.config.DROPOUT_RATE)(dense)

        dense = layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.config.L2_REG),
        )(dense)
        dense = layers.Dropout(self.config.DROPOUT_RATE)(dense)

        # Multi-output
        short_term = layers.Dense(1, activation="sigmoid", name="short_term")(dense)
        long_term = layers.Dense(1, activation="sigmoid", name="long_term")(dense)
        movement = layers.Dense(
            self.config.PREDICTION_WINDOW, activation="tanh", name="movement"
        )(dense)

        # Create model
        model = models.Model(
            inputs=[tech_input, price_input],
            outputs=[short_term, long_term, movement],
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss={
                "short_term": "binary_crossentropy",
                "long_term": "binary_crossentropy",
                "movement": "mse",
            },
            loss_weights={"short_term": 0.4, "long_term": 0.4, "movement": 0.2},
            metrics={
                "short_term": [
                    "accuracy",
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                ],
                "long_term": [
                    "accuracy",
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                ],
                "movement": ["mae"],
            },
        )

        return model

    def _load_or_create_model(self):
        """Load existing model or create new one"""
        model_path = self._get_model_path()

        try:
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                self.model = models.load_model(model_path, compile=False)
                # Recompile with new settings
                self.model.compile(
                    optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                    loss={
                        "short_term": "binary_crossentropy",
                        "long_term": "binary_crossentropy",
                        "movement": "mse",
                    },
                    loss_weights={"short_term": 0.4, "long_term": 0.4, "movement": 0.2},
                    metrics={
                        "short_term": [
                            "accuracy",
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.Recall(name="recall"),
                        ],
                        "long_term": [
                            "accuracy",
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.Recall(name="recall"),
                        ],
                        "movement": ["mae"],
                    },
                )
            else:
                logger.info("Creating new model")
                self.model = self._build_model()
                self.model.summary(print_fn=logger.info)
        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using TA-Lib"""
        df = df.copy()

        # RSI
        df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
            df["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        # Moving Averages
        df["ema_9"] = talib.EMA(df["close"], timeperiod=9)
        df["ema_21"] = talib.EMA(df["close"], timeperiod=21)
        df["ema_50"] = talib.EMA(df["close"], timeperiod=50)

        # ATR
        df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

        # ADX
        df["adx_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

        # Stochastic
        slowk, slowd = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        df["stoch_k"] = slowk
        df["stoch_d"] = slowd

        # Clean NaN values
        df = df.fillna(method="ffill").fillna(method="bfill")

        return df

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Prepare data for training/prediction with multiple targets"""
        # Calculate indicators
        df = self._calculate_technical_indicators(df)

        # Create sequences
        sequences = []
        short_term_targets = []
        long_term_targets = []
        movement_targets = []

        for i in range(
            len(df) - self.config.SEQUENCE_LENGTH - self.config.PREDICTION_WINDOW
        ):
            seq = df.iloc[i : i + self.config.SEQUENCE_LENGTH]
            future_prices = df["close"].iloc[
                i + self.config.SEQUENCE_LENGTH : i
                + self.config.SEQUENCE_LENGTH
                + self.config.PREDICTION_WINDOW
            ]
            current_close = df["close"].iloc[i + self.config.SEQUENCE_LENGTH - 1]

            # Short term target (next 4 hours)
            short_term_change = (
                df["close"].iloc[i + self.config.SEQUENCE_LENGTH + 3] - current_close
            ) / current_close
            short_term_target = (
                1 if short_term_change > self.config.SHORT_TERM_THRESHOLD else 0
            )
            short_term_targets.append(short_term_target)

            # Long term target (next 24 hours)
            long_term_change = (future_prices.iloc[-1] - current_close) / current_close
            long_term_target = (
                1 if long_term_change > self.config.LONG_TERM_THRESHOLD else 0
            )
            long_term_targets.append(long_term_target)

            # Movement target (normalized price changes over prediction window)
            movement_target = (future_prices.values - current_close) / current_close
            movement_targets.append(movement_target)

            sequences.append(seq)

        # Prepare feature arrays
        tech_features = np.array(
            [seq[self.config.TECHNICAL_INDICATORS].values for seq in sequences]
        )

        price_features = np.array(
            [seq[self.config.PRICE_FEATURES].values for seq in sequences]
        ).reshape(-1, self.config.SEQUENCE_LENGTH, len(self.config.PRICE_FEATURES), 1)

        targets = {
            "short_term": np.array(short_term_targets),
            "long_term": np.array(long_term_targets),
            "movement": np.array(movement_targets),
        }

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
        plt.figure(figsize=(18, 12))

        # Plot losses
        plt.subplot(2, 3, 1)
        plt.plot(history.history["loss"], label="Total Loss")
        plt.plot(history.history["short_term_loss"], label="Short Term Loss")
        plt.plot(history.history["long_term_loss"], label="Long Term Loss")
        plt.plot(history.history["movement_loss"], label="Movement Loss")
        plt.title("Model Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot short term metrics
        plt.subplot(2, 3, 2)
        plt.plot(history.history["short_term_accuracy"], label="Accuracy")
        plt.plot(history.history["short_term_precision"], label="Precision")
        plt.plot(history.history["short_term_recall"], label="Recall")
        plt.title("Short Term Metrics")
        plt.ylabel("Score")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot long term metrics
        plt.subplot(2, 3, 3)
        plt.plot(history.history["long_term_accuracy"], label="Accuracy")
        plt.plot(history.history["long_term_precision"], label="Precision")
        plt.plot(history.history["long_term_recall"], label="Recall")
        plt.title("Long Term Metrics")
        plt.ylabel("Score")
        plt.xlabel("Epoch")
        plt.legend()

        # Plot movement MAE
        plt.subplot(2, 3, 4)
        plt.plot(history.history["movement_mae"], label="MAE")
        plt.title("Movement MAE")
        plt.ylabel("MAE")
        plt.xlabel("Epoch")
        plt.legend()

        # Save plot
        plot_path = PLOTS_DIR / f"{model_name}_training.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Saved training plot to {plot_path}")

    def _plot_chart(self, df: pd.DataFrame, predictions: Dict) -> Path:
        """Generate trading chart with predictions"""
        try:
            # Prepare figure with TradingView-like style
            plt.style.use("dark_background")
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(14, 10),
                gridspec_kw={"height_ratios": [3, 1]},
                facecolor="#131722",
            )

            # Main title with predictions
            short_term_pred = predictions["short_term"]
            long_term_pred = predictions["long_term"]
            title = (
                f"{self.symbol} {self.timeframe}\n"
                f"Short Term: {'BUY' if short_term_pred > 0.5 else 'SELL'} ({short_term_pred:.2%}) | "
                f"Long Term: {'BUY' if long_term_pred > 0.5 else 'SELL'} ({long_term_pred:.2%})"
            )
            fig.suptitle(title, color="white", fontsize=14)

            # Convert datetime to mdates
            df = df.copy()
            df["date_num"] = mdates.date2num(df.index.to_pydatetime())

            # Get last 100 candles for display
            display_df = df.tail(100).copy()

            # Create candlestick data
            ohlc = display_df[["date_num", "open", "high", "low", "close"]].values

            # Plot candlesticks
            candlestick_ohlc(
                ax1,
                ohlc,
                width=0.6 / 24,
                colorup="#26a69a",
                colordown="#ef5350",
                alpha=1.0,
            )

            # Add indicators
            ax1.plot(
                display_df.index,
                display_df["ema_9"],
                color="#e91e63",
                label="EMA 9",
                linewidth=1.2,
            )
            ax1.plot(
                display_df.index,
                display_df["ema_21"],
                color="#9c27b0",
                label="EMA 21",
                linewidth=1.2,
            )
            ax1.plot(
                display_df.index,
                display_df["ema_50"],
                color="#ff9800",
                label="EMA 50",
                linewidth=1.2,
            )

            # Style price chart
            ax1.grid(color="#2a2e39", linestyle="-", linewidth=0.5, alpha=0.5)
            ax1.set_facecolor("#131722")
            ax1.tick_params(axis="both", colors="#787b86")
            for spine in ax1.spines.values():
                spine.set_color("#2a2e39")

            # Add legend
            legend = ax1.legend(loc="upper left", frameon=True, fancybox=True)
            legend.get_frame().set_facecolor("#131722")
            legend.get_frame().set_edgecolor("#2a2e39")
            for text in legend.get_texts():
                text.set_color("#787b86")

            # Plot RSI
            ax2.plot(
                display_df.index, display_df["rsi_14"], color="#2196f3", linewidth=1.2
            )
            ax2.axhline(70, color="#787b86", linestyle="--", linewidth=0.8, alpha=0.5)
            ax2.axhline(30, color="#787b86", linestyle="--", linewidth=0.8, alpha=0.5)
            ax2.axhline(50, color="#787b86", linestyle="-", linewidth=0.5, alpha=0.3)

            # Style RSI chart
            ax2.set_ylim(0, 100)
            ax2.set_yticks([0, 30, 50, 70, 100])
            ax2.grid(color="#2a2e39", linestyle="-", linewidth=0.5, alpha=0.5)
            ax2.set_facecolor("#131722")
            ax2.tick_params(axis="both", colors="#787b86")
            for spine in ax2.spines.values():
                spine.set_color("#2a2e39")

            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plot_path = PLOTS_DIR / f"{self.symbol}_chart.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            return plot_path

        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            raise

    def _get_realtime_data(self, num_candles: int = 180) -> pd.DataFrame:
        """Get realtime data from MT5 (180 candles = 7.5 days of H1 data)"""
        try:
            if not mt5.initialize(
                login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER
            ):
                raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

            # Get current rates for H1 timeframe
            rates = mt5.copy_rates_from_pos(
                self.symbol, mt5.TIMEFRAME_H1, 0, num_candles
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
            logger.info(f"Starting training for {self.symbol}")

            # Load and prepare data
            df = self._load_dataset()
            tech_features, price_features, targets = self._prepare_data(df)

            # Check class distribution
            short_term_counts = np.bincount(targets["short_term"])
            long_term_counts = np.bincount(targets["long_term"])
            logger.info(
                f"Class distribution - Short Term: 0={short_term_counts[0]}, 1={short_term_counts[1]} | "
                f"Long Term: 0={long_term_counts[0]}, 1={long_term_counts[1]}"
            )

            # Time Series Split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            fold = 0
            best_val_loss = float("inf")

            for train_index, val_index in tscv.split(tech_features):
                fold += 1
                logger.info(f"Training fold {fold}")

                # Split data
                X_tech_train, X_tech_val = (
                    tech_features[train_index],
                    tech_features[val_index],
                )
                X_price_train, X_price_val = (
                    price_features[train_index],
                    price_features[val_index],
                )
                y_train = {k: v[train_index] for k, v in targets.items()}
                y_val = {k: v[val_index] for k, v in targets.items()}

                # Compute class weights
                short_term_weights = compute_class_weight(
                    "balanced",
                    classes=np.unique(y_train["short_term"]),
                    y=y_train["short_term"],
                )
                long_term_weights = compute_class_weight(
                    "balanced",
                    classes=np.unique(y_train["long_term"]),
                    y=y_train["long_term"],
                )
                class_weights = {
                    "short_term": dict(enumerate(short_term_weights)),
                    "long_term": dict(enumerate(long_term_weights)),
                }

                logger.info(f"Fold {fold} class weights: {class_weights}")

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

            # Save training plots
            self._plot_training_history(best_history, self.symbol)

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
                df = self._get_realtime_data(self.config.SEQUENCE_LENGTH + 100)
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
            short_term_pred, long_term_pred, movement_pred = self.model.predict(
                [tech_seq, price_seq], verbose=0
            )

            # Get latest indicators
            latest = df.iloc[-1]

            # Generate chart
            chart_path = self._plot_chart(
                df,
                {
                    "short_term": short_term_pred[0][0],
                    "long_term": long_term_pred[0][0],
                },
            )

            return {
                "short_term": float(short_term_pred[0][0]),
                "long_term": float(long_term_pred[0][0]),
                "movement": movement_pred[0].tolist(),
                "chart_path": chart_path,
                "indicators": {
                    "ema_9": latest.get("ema_9", None),
                    "ema_21": latest.get("ema_21", None),
                    "ema_50": latest.get("ema_50", None),
                    "rsi": latest.get("rsi_14", None),
                    "macd": latest.get("macd", None),
                    "macd_signal": latest.get("macd_signal", None),
                    "stoch_k": latest.get("stoch_k", None),
                    "stoch_d": latest.get("stoch_d", None),
                },
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
