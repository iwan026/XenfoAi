import logging
import pandas as pd
import MetaTrader5 as mt5
import pandas_ta as ta
from pathlib import Path
from config import (
    DATASETS_DIR,
    TIMEFRAMES,
    FEATURE_COLUMNS,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
)
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.initialize_mt5()

    def initialize_mt5(self):
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
    ) -> pd.DataFrame:
        """Ambil data OHLC dari MetaTrader5"""
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
            return df

        except Exception as e:
            logger.error(f"Gagal mengambil data symbol: {e}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan indikator teknikal"""
        try:
            df["rsi"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"])
            df = pd.concat([df, macd], axis=1)

            bbands = ta.bbands(df["close"])
            df = pd.concat([df, bbands], axis=1)

            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)
            df["ema_9"] = ta.ema(df["close"], length=9)
            df["ema_21"] = ta.ema(df["close"], length=21)

            stoch = ta.stoch(df["high"], df["low"], df["close"])
            df = pd.concat([df, stoch], axis=1)

            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)

            return df

        except Exception as e:
            logger.error(f"Terjadi error ketika menambahkan indikator teknikal: {e}")
            return df

    def save_to_csv(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> pd.DataFrame:
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

    def load_from_csv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Muat data dari file Csv"""
        try:
            file_path = (
                DATASETS_DIR
                / symbol.upper()
                / f"{symbol.upper()}_{timeframe.upper()}.csv"
            )
            if not file_path.exists():
                logger.error(f"File tidak ada: {file_path}")
                return None

            # Baca file dalam chunks
            chunk_size = 5000  # Ukuran chunk bisa disesuaikan
            chunks = []

            # Gunakan iterator untuk membaca file secara bertahap
            for chunk in pd.read_csv(
                file_path, parse_dates=["datetime"], dayfirst=True, chunksize=chunk_size
            ):
                # Konversi tipe data untuk menghemat memori
                for col in chunk.select_dtypes(include=["float64"]).columns:
                    chunk[col] = chunk[col].astype("float32")

                chunks.append(chunk)

                # Log progress
                logger.info(f"Membaca {len(chunks) * chunk_size} baris...")

            # Gabungkan semua chunks
            df = pd.concat(chunks, ignore_index=True)
            df["datetime"] = pd.to_datetime(df["datetime"])

            logger.info(f"Total baris yang dibaca: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Gagal memuat data dari file Csv: {e}")
            return None

    def update_dataset(self, symbol: str, timeframe: str) -> bool:
        """Update dataset dengan data terbaru"""
        try:
            new_data = self.get_symbol_data(symbol, timeframe)
            if new_data is None:
                return None

            new_data = self.add_technical_indicators(new_data)
            return self.save_to_csv(new_data, symbol, timeframe)

        except Exception as e:
            logger.error(f"Terjadi error saat update dataset: {e}")
            return False

    def __del__(self):
        """Bersihkan koneksi MetaTrader5"""
        try:
            mt5.shutdown()
            logger.info("Koneksi MetaTrader5 berhasil di tutup")
        except:
            pass
