import numpy as np
import pandas as pd
import talib
from typing import List, Optional
from src.models.model_config import ModelConfig


class FeatureEngineer:
    def __init__(self, config: ModelConfig):
        self.config = config

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menambahkan indikator teknikal ke DataFrame"""
        try:
            # RSI
            df["rsi_14"] = talib.RSI(df["close"], timeperiod=14)

            # MACD
            macd, signal, hist = talib.MACD(df["close"])
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"])
            df["bb_upper"] = bb_upper
            df["bb_middle"] = bb_middle
            df["bb_lower"] = bb_lower

            # Moving Averages
            df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
            df["sma_50"] = talib.SMA(df["close"], timeperiod=50)
            df["ema_9"] = talib.EMA(df["close"], timeperiod=9)
            df["ema_21"] = talib.EMA(df["close"], timeperiod=21)

            # Additional Indicators
            df["atr_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["adx_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)
            df["cci_20"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod=20)

            # Stochastic
            df["stoch_k"], df["stoch_d"] = talib.STOCH(
                df["high"], df["low"], df["close"]
            )

            # Normalisasi volume
            df["volume"] = (df["volume"] - df["volume"].min()) / (
                df["volume"].max() - df["volume"].min()
            )

            return df

        except Exception as e:
            raise Exception(f"Error dalam penambahan indikator teknikal: {str(e)}")

    def create_sequences(
        self, data: np.ndarray, sequence_length: int, batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Membuat sequences untuk model sequence"""
        try:
            if batch_size is None:
                batch_size = len(data) - sequence_length

            sequences = []
            for i in range(0, batch_size):
                sequences.append(data[i : i + sequence_length])

            return np.array(sequences)

        except Exception as e:
            raise Exception(f"Error dalam pembuatan sequences: {str(e)}")

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Mempersiapkan data untuk semua model"""
        try:
            # Tambah indikator teknikal
            df = self.add_technical_indicators(df)

            # Pisahkan fitur
            technical_features = df[self.config.TECHNICAL_FEATURES].values
            price_features = df[self.config.PRICE_FEATURES].values

            # Buat target (1 jika harga naik, 0 jika turun)
            target = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]

            # Buat sequences
            technical_sequences = self.create_sequences(
                technical_features[:-1], self.config.SEQUENCE_LENGTH
            )

            price_sequences = self.create_sequences(
                price_features[:-1], self.config.SEQUENCE_LENGTH
            )

            # Reshape untuk CNN (tambah channel dimension)
            price_sequences_cnn = price_sequences.reshape(
                (
                    price_sequences.shape[0],
                    price_sequences.shape[1],
                    price_sequences.shape[2],
                    1,
                )
            )

            return {
                "technical_features": technical_sequences,
                "price_features": price_sequences,
                "price_features_cnn": price_sequences_cnn,
                "target": target[self.config.SEQUENCE_LENGTH - 1 :],
            }

        except Exception as e:
            raise Exception(f"Error dalam persiapan data: {str(e)}")

    def batch_generator(self, data: dict, batch_size: int):
        """Generator untuk memproses data dalam batch"""
        num_samples = len(data["target"])
        indices = np.arange(num_samples)

        while True:
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]

                batch_data = {
                    "technical_features": data["technical_features"][batch_indices],
                    "price_features": data["price_features"][batch_indices],
                    "price_features_cnn": data["price_features_cnn"][batch_indices],
                    "target": data["target"][batch_indices],
                }

                yield batch_data
