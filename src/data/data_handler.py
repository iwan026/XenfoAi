import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from pathlib import Path
from typing import Optional, Dict, Tuple
from sklearn.model_selection import train_test_split

from config import (
    DATASETS_DIR,
    TIMEFRAMES,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
)
from src.models.model_config import ModelConfig
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.initialize_mt5()

    def initialize_mt5(self) -> bool:
        """Inisiasi Koneksi MetaTrader5"""
        try:
            if not mt5.initialize():
                logger.error(f"Inisiasi koneksi MetaTrader5 gagal: {mt5.last_error()}")
                return False

            if not mt5.login(
                login=MT5_LOGIN,
                password=MT5_PASSWORD,
                server=MT5_SERVER,
            ):
                logger.error(f"Login ke MetaTrader5 gagal: {mt5.last_error()}")
                return False

            logger.info("Koneksi ke MetaTrader5 berhasil")
            return True

        except Exception as e:
            logger.error(f"Gagal menginisiasi MetaTrader5: {e}")
            return False

    def get_symbol_data(
        self, symbol: str, timeframe: str, num_candles: int = 5000
    ) -> Optional[pd.DataFrame]:
        """Ambil data OHLCV dari MetaTrader5"""
        try:
            mt5_timeframe = TIMEFRAMES.get(timeframe)
            if mt5_timeframe is None:
                raise ValueError(f"Timeframe tidak valid: {timeframe}")

            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.rename(
                columns={
                    "time": "datetime",
                    "tick_volume": "volume",
                },
                inplace=True,
            )

            # Tambahkan fitur teknikal
            df = self.feature_engineer.add_technical_indicators(df)

            return df

        except Exception as e:
            logger.error(f"Gagal mengambil data symbol: {e}")
            return None

    def prepare_training_data(
        self, df: pd.DataFrame, validation_split: float = 0.2
    ) -> Tuple[Dict, Dict]:
        """Mempersiapkan data untuk training"""
        try:
            # Persiapkan data menggunakan feature engineer
            prepared_data = self.feature_engineer.prepare_data(df)

            # Split data
            data_length = len(prepared_data["target"])
            split_idx = int(data_length * (1 - validation_split))

            train_data = {
                "technical_features": prepared_data["technical_features"][:split_idx],
                "price_features": prepared_data["price_features"][:split_idx],
                "price_features_cnn": prepared_data["price_features_cnn"][:split_idx],
                "target": prepared_data["target"][:split_idx],
            }

            val_data = {
                "technical_features": prepared_data["technical_features"][split_idx:],
                "price_features": prepared_data["price_features"][split_idx:],
                "price_features_cnn": prepared_data["price_features_cnn"][split_idx:],
                "target": prepared_data["target"][split_idx:],
            }

            return train_data, val_data

        except Exception as e:
            logger.error(f"Error dalam persiapan data training: {e}")
            raise

    def prepare_prediction_data(self, df: pd.DataFrame) -> Dict:
        """Mempersiapkan data untuk prediksi"""
        try:
            # Ambil data terakhir sesuai sequence length
            last_sequence = df.tail(self.config.SEQUENCE_LENGTH)

            # Persiapkan data menggunakan feature engineer
            prepared_data = self.feature_engineer.prepare_data(last_sequence)

            return {
                "technical_features": prepared_data["technical_features"][-1:],
                "price_features": prepared_data["price_features"][-1:],
                "price_features_cnn": prepared_data["price_features_cnn"][-1:],
            }

        except Exception as e:
            logger.error(f"Error dalam persiapan data prediksi: {e}")
            raise

    def save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Simpan dataframe ke csv"""
        try:
            symbol_dir = DATASETS_DIR / symbol.upper()
            symbol_dir.mkdir(exist_ok=True)

            file_path = symbol_dir / f"{symbol.upper()}_{timeframe.upper()}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Data disimpan ke: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Gagal menyimpan data ke csv: {e}")
            return False

    def load_from_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Muat data dari file CSV"""
        try:
            file_path = (
                DATASETS_DIR
                / symbol.upper()
                / f"{symbol.upper()}_{timeframe.upper()}.csv"
            )
            if not file_path.exists():
                logger.error(f"File tidak ada: {file_path}")
                return None

            # Baca file dalam chunks untuk menghemat memori
            chunks = []
            chunk_size = self.config.BATCH_SIZE * 100  # Sesuaikan dengan kebutuhan

            for chunk in pd.read_csv(
                file_path, parse_dates=["datetime"], chunksize=chunk_size
            ):
                # Konversi tipe data untuk menghemat memori
                for col in chunk.select_dtypes(include=["float64"]).columns:
                    chunk[col] = chunk[col].astype("float32")
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
            return df

        except Exception as e:
            logger.error(f"Gagal memuat data dari CSV: {e}")
            return None

    def update_dataset(self, symbol: str, timeframe: str) -> bool:
        """Update dataset dengan data terbaru"""
        try:
            new_data = self.get_symbol_data(symbol, timeframe)
            if new_data is None:
                return False

            return self.save_to_csv(new_data, symbol, timeframe)

        except Exception as e:
            logger.error(f"Error saat update dataset: {e}")
            return False

    def __del__(self):
        """Bersihkan koneksi MetaTrader5"""
        mt5.shutdown()
