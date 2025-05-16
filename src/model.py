import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import MetaTrader5 as mt5
from datetime import datetime
import logging
from pathlib import Path
import gc
from typing import Dict, Optional, Tuple

from config import (
    ModelConfig,
    TIMEFRAMES,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MODELS_DIR,
    DATASETS_DIR,
)

logger = logging.getLogger(__name__)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block untuk sequence processing"""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Multi-head attention dengan dimensi yang konsisten
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,  # Pastikan key_dim sesuai dengan num_heads
            value_dim=embed_dim // num_heads,  # Tambahkan value_dim yang sesuai
        )

        # Feed forward network dengan dimensi yang konsisten
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(
                    embed_dim
                ),  # Output dim harus sama dengan input (embed_dim)
            ]
        )

        # Layer normalization dan dropout
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Multi-head attention dengan shape yang konsisten
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed forward network dengan shape yang konsisten
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


class ForexModel:
    def __init__(self, symbol: str, timeframe: str):
        """Initialize Forex model"""
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.config = ModelConfig()

        # Initialize models
        self.xgb_model = None
        self.deep_model = None

        # Setup model and dataset directories
        self.model_dir = MODELS_DIR / self.symbol
        self.dataset_dir = DATASETS_DIR / self.symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # Load or create models
        self._load_or_create_models()

        logger.info(f"Initialized ForexModel for {self.symbol} {self.timeframe}")

    def _get_model_paths(self) -> Tuple[Path, Path]:
        """Get paths for model files"""
        xgb_path = self.model_dir / f"{self.symbol}_{self.timeframe}_xgb.json"
        deep_path = self.model_dir / f"{self.symbol}_{self.timeframe}_deep.keras"
        return xgb_path, deep_path

    def _get_dataset_path(self) -> Path:
        """Get path for CSV dataset file"""
        return self.dataset_dir / f"{self.symbol}_{self.timeframe}.csv"

    def _initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

            if not mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                raise Exception(f"MT5 login failed: {mt5.last_error()}")

            logger.info("MT5 connection successful")
            return True

        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False

    def _load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            xgb_path, deep_path = self._get_model_paths()

            if xgb_path.exists() and deep_path.exists():
                logger.info(f"Loading existing models from {self.model_dir}")
                self._load_models(xgb_path, deep_path)
            else:
                logger.info("Creating new models")
                self._build_models()

        except Exception as e:
            logger.error(f"Error in model initialization: {e}")
            raise

    def _build_models(self):
        """Build arsitektur model"""
        try:
            # XGBoost model
            self.xgb_model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.01,
                n_estimators=100,
                objective="binary:logistic",
                tree_method="hist",
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric=["logloss", "auc"],
                early_stopping_rounds=10,
            )

            # Calculate input dimensions
            num_technical_features = len(self.config.TECHNICAL_FEATURES)
            num_price_features = len(self.config.PRICE_FEATURES)

            # Define embedding dimension yang konsisten
            embed_dim = 64  # Ubah ke 64 untuk dimensi yang lebih besar

            # Input layers
            technical_input = tf.keras.layers.Input(
                shape=(self.config.SEQUENCE_LENGTH, num_technical_features)
            )

            # Add embedding layer untuk technical features
            technical_embedding = tf.keras.layers.Dense(embed_dim)(technical_input)

            price_input = tf.keras.layers.Input(
                shape=(self.config.SEQUENCE_LENGTH, num_price_features, 1)
            )

            # CNN branch
            cnn = price_input
            for filters, kernel_size, pool_size in zip(
                self.config.CNN_FILTERS,
                self.config.CNN_KERNEL_SIZES,
                self.config.CNN_POOL_SIZES,
            ):
                cnn = tf.keras.layers.Conv2D(
                    filters, kernel_size, activation="relu", padding="same"
                )(cnn)
                cnn = tf.keras.layers.MaxPooling2D(pool_size)(cnn)
                cnn = tf.keras.layers.BatchNormalization()(cnn)

            cnn = tf.keras.layers.Flatten()(cnn)
            cnn = tf.keras.layers.Dense(embed_dim, activation="relu")(cnn)
            cnn = tf.keras.layers.Dropout(self.config.CNN_DROPOUT)(cnn)

            # LSTM branch
            lstm = technical_embedding  # Gunakan technical_embedding
            for i, units in enumerate(self.config.LSTM_UNITS):
                return_sequences = i < len(self.config.LSTM_UNITS) - 1
                lstm = tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.LSTM_DROPOUT,
                )(lstm)

            # Transformer branch
            transformer_block = TransformerBlock(
                embed_dim=embed_dim,  # Gunakan embed_dim yang sama
                num_heads=4,
                ff_dim=embed_dim * 4,  # FF dim biasanya 4x embed_dim
                rate=self.config.TRANSFORMER_DROPOUT,
            )

            transformer = transformer_block(technical_embedding)
            transformer = tf.keras.layers.GlobalAveragePooling1D()(transformer)

            # Combine all branches dengan dimensi yang konsisten
            combined = tf.keras.layers.concatenate([cnn, lstm, transformer])
            combined = tf.keras.layers.Dense(embed_dim, activation="relu")(combined)
            combined = tf.keras.layers.Dropout(0.2)(combined)
            output = tf.keras.layers.Dense(1, activation="sigmoid")(combined)

            # Create and compile model
            self.deep_model = tf.keras.Model(
                inputs=[technical_input, price_input], outputs=output
            )

            self.deep_model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config.LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            logger.info("Models built successfully")

        except Exception as e:
            logger.error(f"Error building models: {e}")
            raise

    def _load_models(self, xgb_path: Path, deep_path: Path):
        """Load saved models"""
        try:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(xgb_path))

            self.deep_model = tf.keras.models.load_model(
                str(deep_path), custom_objects={"TransformerBlock": TransformerBlock}
            )

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _save_models(self):
        """Save models to disk"""
        try:
            xgb_path, deep_path = self._get_model_paths()

            # Save XGBoost model
            self.xgb_model.save_model(str(xgb_path))

            # Save Deep Learning model
            self.deep_model.save(str(deep_path), save_format="keras")

            logger.info(f"Models saved to {self.model_dir}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            logger.info("Calculating technical indicators...")
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
            df["bb_upper"] = df["bb_middle"] + (std * 2)
            df["bb_lower"] = df["bb_middle"] - (std * 2)

            # Moving Averages
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
            df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

            # ATR
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df["atr_14"] = true_range.rolling(window=14).mean()

            # ADX
            plus_dm = df["high"].diff()
            minus_dm = df["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (
                plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean()
            )
            minus_di = 100 * (
                minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean()
            )
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df["adx_14"] = dx.rolling(window=14).mean()

            # CCI
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            mean_dev = typical_price.rolling(window=20).apply(
                lambda x: pd.Series(x).mad()
            )
            df["cci_20"] = (typical_price - typical_price.rolling(window=20).mean()) / (
                0.015 * mean_dev
            )

            # Stochastic
            low_min = df["low"].rolling(window=14).min()
            high_max = df["high"].rolling(window=14).max()
            df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
            df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

            # Handle missing/infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method="ffill").fillna(method="bfill")

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> Dict:
        """Prepare data for model"""
        try:
            logger.info("Preparing data...")

            # Calculate technical indicators
            df = self._calculate_technical_indicators(df)

            # Create sequences
            sequences = []
            targets = []

            for i in range(len(df) - self.config.SEQUENCE_LENGTH):
                sequence = df.iloc[i : i + self.config.SEQUENCE_LENGTH]

                if is_training and i + self.config.SEQUENCE_LENGTH < len(df):
                    target = (
                        1
                        if df["close"].iloc[i + self.config.SEQUENCE_LENGTH]
                        > df["close"].iloc[i + self.config.SEQUENCE_LENGTH - 1]
                        else 0
                    )
                    targets.append(target)

                sequences.append(sequence)

            # Prepare feature arrays
            technical_features = np.array(
                [
                    sequence[self.config.TECHNICAL_FEATURES].values
                    for sequence in sequences
                ]
            )

            price_features = np.array(
                [sequence[self.config.PRICE_FEATURES].values for sequence in sequences]
            )

            # Reshape price features for CNN
            price_features_cnn = price_features.reshape(
                price_features.shape[0],
                price_features.shape[1],
                price_features.shape[2],
                1,
            )

            data = {
                "technical_features": technical_features,
                "price_features_cnn": price_features_cnn,
            }

            if is_training:
                data["target"] = np.array(targets)

            return data

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def _get_mt5_timeframe(self) -> int:
        """Convert timeframe string to MT5 timeframe"""
        return getattr(mt5, TIMEFRAMES[self.timeframe])

    def _get_historical_data(self, num_candles: Optional[int] = None) -> pd.DataFrame:
        """Get historical data from MT5"""
        try:
            if not self._initialize_mt5():
                raise Exception("Failed to initialize MT5")

            # Get data from MT5
            rates = mt5.copy_rates_from_pos(
                self.symbol,
                self._get_mt5_timeframe(),
                0,
                num_candles or self.config.SEQUENCE_LENGTH * 2,
            )

            if rates is None or len(rates) == 0:
                raise Exception("No data received from MT5")

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            # Rename columns if needed
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]

            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            raise
        finally:
            mt5.shutdown()

    def _load_csv_data(self) -> pd.DataFrame:
        """Load data from CSV file in datasets directory"""
        try:
            csv_path = self._get_dataset_path()
            logger.info(f"Loading data from {csv_path}")

            if not csv_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {csv_path}")

            # Load the CSV file
            df = pd.read_csv(csv_path)

            # Ensure we have 'time' column and convert to datetime
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
            else:
                # If there's no time column, check if there's a datetime column
                datetime_cols = [
                    col
                    for col in df.columns
                    if "date" in col.lower() or "time" in col.lower()
                ]
                if datetime_cols:
                    df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]])
                    df.set_index(datetime_cols[0], inplace=True)
                else:
                    # If no obvious datetime column, create a simple index
                    logger.warning(
                        "No datetime column found, creating sequential index"
                    )
                    df.index = pd.date_range(start="2000-01-01", periods=len(df))

            # Ensure all required columns exist
            required_columns = self.config.PRICE_FEATURES
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            # Make sure column names are lowercase
            df.columns = [col.lower() for col in df.columns]

            # Ensure volume column exists (some CSVs might have 'volume' or 'tick_volume')
            if "volume" not in df.columns and "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            elif "volume" not in df.columns:
                # If no volume data, create dummy volume data
                logger.warning("No volume data found, creating dummy volume data")
                df["volume"] = 1

            return df

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise

    def train(self) -> bool:
        """Train the model using CSV data"""
        try:
            logger.info(f"Starting training for {self.symbol} {self.timeframe}")

            # Get training data from CSV instead of MT5
            df = self._load_csv_data()
            data = self._prepare_data(df, is_training=True)

            # Split data
            train_idx = int(len(data["target"]) * (1 - self.config.VALIDATION_SPLIT))

            train_data = {
                "technical_features": data["technical_features"][:train_idx],
                "price_features_cnn": data["price_features_cnn"][:train_idx],
                "target": data["target"][:train_idx],
            }

            val_data = {
                "technical_features": data["technical_features"][train_idx:],
                "price_features_cnn": data["price_features_cnn"][train_idx:],
                "target": data["target"][train_idx:],
            }

            # Train XGBoost
            self.xgb_model.fit(
                train_data["technical_features"].reshape(len(train_data["target"]), -1),
                train_data["target"],
                eval_set=[
                    (
                        val_data["technical_features"].reshape(
                            len(val_data["target"]), -1
                        ),
                        val_data["target"],
                    )
                ],
                verbose=True,
            )

            # Clear memory
            gc.collect()

            # Train Deep Learning model
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.config.PATIENCE
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3
                ),
            ]

            self.deep_model.fit(
                [train_data["technical_features"], train_data["price_features_cnn"]],
                train_data["target"],
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_data=(
                    [val_data["technical_features"], val_data["price_features_cnn"]],
                    val_data["target"],
                ),
                callbacks=callbacks,
                verbose=1,
            )

            # Save models
            self._save_models()

            logger.info("Training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in training: {e}")
            return False

    def predict(self) -> Optional[Dict]:
        """Make prediction using latest data"""
        try:
            # Get latest data
            df = self._get_historical_data()
            data = self._prepare_data(df, is_training=False)

            # Make predictions
            xgb_pred = self.xgb_model.predict_proba(
                data["technical_features"].reshape(1, -1)
            )[0, 1]

            deep_pred = self.deep_model.predict(
                [data["technical_features"], data["price_features_cnn"]], verbose=0
            )[0, 0]

            # Combine predictions
            final_pred = (xgb_pred + deep_pred) / 2
            confidence = abs(final_pred - 0.5) * 2

            # Get latest indicators
            latest_data = df.iloc[-1]

            signals = {
                "rsi": "Overbought"
                if latest_data["rsi_14"] > 70
                else "Oversold"
                if latest_data["rsi_14"] < 30
                else "Neutral",
                "macd": "Buy"
                if latest_data["macd"] > latest_data["macd_signal"]
                else "Sell",
                "ma": "Buy"
                if latest_data["sma_20"] > latest_data["sma_50"]
                else "Sell",
            }

            return {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "prediction": float(final_pred),
                "confidence": float(confidence),
                "signals": signals,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None
