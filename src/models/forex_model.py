import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional

from sklearn.model_selection import train_test_split
from config import (
    MODELS_DIR,
    SEQUENCE_LENGTH,
    FEATURE_COLUMNS,
    BATCH_SIZE,
    EPOCHS,
    TRAIN_TEST_SPLIT,
)
from src.data.data_handler import DataHandler

logger = logging.getLogger(__name__)


class ForexModel:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.model = None
        self.data_handler = DataHandler()
        self.model_path = self.get_model_path()
        self.load_or_create_model()

    def get_model_path(self) -> Path:
        """Path untuk file model"""
        symbol_dir = MODELS_DIR / self.symbol
        symbol_dir.mkdir(exist_ok=True)
        return symbol_dir / f"{self.symbol}_{self.timeframe}.h5"

    def create_model(self) -> None:
        """Buat model LSTM"""
        try:
            input_features = len(FEATURE_COLUMNS)

            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        100,
                        return_sequences=True,
                        input_shape=(SEQUENCE_LENGTH, input_features),
                    ),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(100, return_sequences=False),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(50, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            logger.info("Model berhasil dibuat")
        except Exception as e:
            logger.error(f"Terjadi error saat membuat model: {e}")
            raise

    def load_or_create_model(self) -> None:
        """Muat model yanh sudah ada atau buat baru"""
        try:
            if self.model_path.exists():
                logger.info(f"Muat model {self.model_path}")
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info("Model berhasil dimuat")

            else:
                logger.info("Tidak ada model yanh bisa dimuat, membuat model baru...")
                self.create_model()

        except Exception as e:
            logger.error(f"Gagal memuat model: {e}")
            self.create_model()

    def prepare_data(self, df) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data untuk training dan prediksi"""
        try:
            # Tambah indikator teknikal
            df = self.data_handler.add_technical_indicators(df)

            # Buat target
            df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
            df.dropna(inplace=True)

            # Konversi ke float32 untuk menghemat memori
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype("float32")

            # Proses data dalam batches
            batch_size = 1000
            X, y = [], []

            total_sequences = len(df) - SEQUENCE_LENGTH
            for i in range(0, total_sequences, batch_size):
                end_idx = min(i + batch_size, total_sequences)

                # Log progress
                logger.info(
                    f"Memproses sequences {i} sampai {end_idx} dari {total_sequences}"
                )

                batch_X = []
                batch_y = []

                for j in range(i, end_idx):
                    sequence = df[FEATURE_COLUMNS].iloc[j : j + SEQUENCE_LENGTH].values
                    target = df["target"].iloc[j + SEQUENCE_LENGTH]

                    batch_X.append(sequence)
                    batch_y.append(target)

                X.extend(batch_X)
                y.extend(batch_y)

            # Konversi ke numpy array dengan tipe data yang efisien
            X = np.array(X, dtype="float32")
            y = np.array(y, dtype="float32")

            # Normalisasi data
            X_reshaped = X.reshape(X.shape[0], -1)
            X_normalized = self.data_handler.scaler.fit_transform(X_reshaped)
            X = X_normalized.reshape(X.shape)

            return X, y

        except Exception as e:
            logger.error(f"Terjadi kesalahan saat prepare data: {e}")
            raise

    def train(self) -> bool:
        """Training model"""
        try:
            # Muat data
            df = self.data_handler.load_from_csv(self.symbol, self.timeframe)
            if df is None:
                logger.error("Data training tidak tersedia")
                return False

            # Prepare data
            X, y = self.prepare_data(df)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TRAIN_TEST_SPLIT, random_state=42
            )

            # Training model
            history = self.model.fit(
                X_train,
                y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test),
                verbose=1,
            )

            # Simpan model
            self.model_save(str(self.model_path))
            logger.info(f"Model disimpan ke: {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Terjadi error saat training model: {e}")
            return False

    def predict(self) -> Optional[Dict]:
        """Buat prediksi"""
        try:
            # Ambil dats terbaru
            df = self.data_handler.get_symbol_data(
                self.symbol, self.timeframe, SEQUENCE_LENGTH + 90
            )

            if df is None:
                logger.error("Gagal mengambil data market terbaru")
                return None

            # Preapare data untuk prediksi
            df = self.data_handler.add_technical_indicators(df)
            last_sequence = df[FEATURE_COLUMNS].iloc[-SEQUENCE_LENGTH:].values

            # Reshape dam normalisasi
            last_sequence_reshape = last_sequence.reshape(
                1, last_sequence.shape[0] * last_sequence.shape[1]
            )
            last_sequence_normalized = self.data_handler.scaler.transform(
                last_sequence_reshape
            )
            X = last_sequence_normalized.reshape(
                1, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)
            )

            # Buat prediksi
            prediction = self.model.predict(X)[0][0]

            # Ambil sinyal indikator
            last_row = df.iloc[-1]
            rsi_signal = (
                "Buy"
                if last_row["rsi"] < 30
                else "Sell"
                if last_row["rsi"] > 70
                else "Neutral"
            )

            macd_signal = (
                "Buy" if last_row["MACD_12_26_9"] > last_row["MACD_12_26_9"] else "Sell"
            )

            ma_signal = "Buy" if last_row["ema_9"] > last_row["ema_21"] else "Sell"

            # Kombinasi sinyal
            signals = [
                "Buy" if prediction > 0.5 else "Sell",
                rsi_signal,
                macd_signal,
                ma_signal,
            ]

            buy_count = signals.count("Buy")
            sell_count = signals.count("Sell")

            final_signal = "Buy" if buy_count > sell_count else "Sell"
            confidence = max(buy_count, sell_count) / len(signals)

            return {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "prediction": "Buy" if prediction > 0.5 else "Sell",
                "confidence": float(prediction)
                if prediction > 0.5
                else 1 - float(prediction),
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal,
                "ma_signal": ma_signal,
                "final_signal": final_signal,
                "final_confidence": confidence,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Gagal membuat prediksi: {e}")
            return None
