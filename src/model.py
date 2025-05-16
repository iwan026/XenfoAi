import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import MetaTrader5 as mt5
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import gc

from config import (
    ModelConfig,
    TIMEFRAMES,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block untuk sequence processing"""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ForexModel:
    def __init__(self, symbol: str, timeframe: str):
        """Inisialisasi model forex"""
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.config = ModelConfig()

        # Inisialisasi model
        self.xgb_model = None
        self.deep_model = None

        # Setup paths
        self.model_dir = MODELS_DIR / self.symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Inisialisasi MT5
        self._initialize_mt5()

        # Load atau buat model baru
        self._load_or_create_models()

    def _initialize_mt5(self) -> bool:
        """Inisialisasi koneksi MT5"""
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

    def _get_model_paths(self) -> Tuple[Path, Path]:
        """Get paths untuk model files"""
        xgb_path = self.model_dir / f"{self.symbol}_{self.timeframe}_xgb.json"
        deep_path = self.model_dir / f"{self.symbol}_{self.timeframe}_deep.h5"
        return xgb_path, deep_path

    def _load_or_create_models(self):
        """Load existing models atau buat baru"""
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

            # Deep Learning model
            technical_input = tf.keras.layers.Input(
                shape=(self.config.SEQUENCE_LENGTH, len(self.config.TECHNICAL_FEATURES))
            )

            price_input = tf.keras.layers.Input(
                shape=(self.config.SEQUENCE_LENGTH, len(self.config.PRICE_FEATURES), 1)
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
            cnn = tf.keras.layers.Dense(64, activation="relu")(cnn)
            cnn = tf.keras.layers.Dropout(self.config.CNN_DROPOUT)(cnn)

            # LSTM branch
            lstm = technical_input
            for i, units in enumerate(self.config.LSTM_UNITS):
                return_sequences = i < len(self.config.LSTM_UNITS) - 1
                lstm = tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.LSTM_DROPOUT,
                )(lstm)

            # Transformer branch
            transformer_block = TransformerBlock(
                self.config.TRANSFORMER_HEAD_SIZE,
                self.config.TRANSFORMER_NUM_HEADS,
                self.config.TRANSFORMER_FF_DIM,
                self.config.TRANSFORMER_DROPOUT,
            )

            transformer = transformer_block(technical_input)
            transformer = tf.keras.layers.GlobalAveragePooling1D()(transformer)

            # Combine all branches
            combined = tf.keras.layers.concatenate([cnn, lstm, transformer])
            combined = tf.keras.layers.Dense(32, activation="relu")(combined)
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

        except Exception as e:
            logger.error(f"Error building models: {e}")
            raise

    def _load_models(self, xgb_path: Path, deep_path: Path):
        """Load saved models"""
        try:
            # Load XGBoost model
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(str(xgb_path))

            # Load Deep Learning model
            self.deep_model = tf.keras.models.load_model(
                str(deep_path), custom_objects={"TransformerBlock": TransformerBlock}
            )

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _get_mt5_timeframe(self) -> int:
        """Konversi timeframe string ke MT5 timeframe"""
        return getattr(mt5, TIMEFRAMES[self.timeframe])

    def _get_historical_data(self, num_candles: int = None) -> Optional[pd.DataFrame]:
        """Ambil data historis dari MT5"""
        try:
            # Jika num_candles tidak ditentukan, gunakan 2x sequence length
            if num_candles is None:
                num_candles = self.config.SEQUENCE_LENGTH * 2

            # Get data from MT5
            rates = mt5.copy_rates_from_pos(
                self.symbol, self._get_mt5_timeframe(), 0, num_candles
            )

            if rates is None:
                raise Exception(f"Failed to get data: {mt5.last_error()}")

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hitung indikator teknikal"""
        try:
            # RSI
            df["rsi_14"] = self._calculate_rsi(df["close"], period=14)

            # MACD
            macd_line, signal_line, hist = self._calculate_macd(df["close"])
            df["macd"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_hist"] = hist

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
            df["atr_14"] = self._calculate_atr(df, period=14)

            # ADX
            df["adx_14"] = self._calculate_adx(df, period=14)

            # CCI
            df["cci_20"] = self._calculate_cci(df, period=20)

            # Stochastic
            df["stoch_k"], df["stoch_d"] = self._calculate_stochastic(df)

            # Drop NaN values
            df.dropna(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            raise

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Hitung RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Hitung MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Hitung ATR"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Hitung ADX"""
        plus_dm = df["high"].diff()
        minus_dm = df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = self._calculate_atr(df, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Hitung CCI"""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        mean_dev = pd.Series(
            [
                abs(
                    typical_price[i : i + period].mean() - typical_price[i : i + period]
                ).mean()
                for i in range(len(typical_price) - period + 1)
            ]
        )
        return (typical_price - typical_price.rolling(window=period).mean()) / (
            0.015 * mean_dev
        )

    def _calculate_stochastic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Hitung Stochastic Oscillator"""
        low_min = df["low"].rolling(window=14).min()
        high_max = df["high"].rolling(window=14).max()

        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=3).mean()
        return k, d

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> Dict:
        """Prepare data untuk model"""
        try:
            # Calculate features
            df = self._calculate_technical_indicators(df)

            # Create sequences
            sequences = []
            targets = []

            for i in range(len(df) - self.config.SEQUENCE_LENGTH):
                sequence = df.iloc[i : i + self.config.SEQUENCE_LENGTH]

                if is_training:
                    target = (
                        1
                        if df["close"].iloc[i + self.config.SEQUENCE_LENGTH]
                        > df["close"].iloc[i + self.config.SEQUENCE_LENGTH - 1]
                        else 0
                    )
                    targets.append(target)

                sequences.append(sequence)

            # Prepare features
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
                "price_features": price_features,
                "price_features_cnn": price_features_cnn,
            }

            if is_training:
                data["target"] = np.array(targets)

            return data

        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train(self) -> bool:
        """Training model"""
        try:
            logger.info(f"Starting training for {self.symbol} {self.timeframe}")

            # Get historical data
            df = self._get_historical_data(5000)  # Get enough data for training
            if df is None:
                return False

            # Prepare data
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
            logger.info("Training XGBoost model...")
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
            logger.info("Training Deep Learning model...")
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
        """Membuat prediksi"""
        try:
            # Get latest data
            df = self._get_historical_data(self.config.SEQUENCE_LENGTH + 1)
            if df is None:
                return None

            # Prepare data
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

            # Get signals
            latest_data = df.iloc[-1]

            result = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "prediction": float(final_pred),
                "confidence": float(confidence),
                "signals": {
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
                },
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return result

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None

    def _save_models(self):
        """Save models to disk"""
        try:
            xgb_path, deep_path = self._get_model_paths()
            self.xgb_model.save_model(str(xgb_path))
            self.deep_model.save(str(deep_path))
            logger.info(f"Models saved to {self.model_dir}")

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
