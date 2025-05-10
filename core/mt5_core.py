import MetaTrader5 as mt5
import pandas as pd
from config import KonfigurasiMT5, KonfigurasiPath
import logging
import os

logger = logging.getLogger(__name__)


class MT5Core:
    def __init__(self):
        self.connected = False

    def initialize_mt5(self) -> bool:
        try:
            if not mt5.initialize():
                logger.error(f"Inisialisasi MetaTrader5 gagal: {mt5.last_error()}")
                return False

            if not mt5.login(
                login=KonfigurasiMT5.LOGIN,
                password=KonfigurasiMT5.PASSWORD,
                server=KonfigurasiMT5.SERVER,
            ):
                logger.error(f"Login ke MetaTrader5 gagal: {mt5.last_error()}")
                return False

            self.connected = True
            logger.info(f"Koneksi MetaTrader5 Berhasil")
            return True

        except Exception as e:
            logger.error(
                f"Terjadi kesalahan saat membuat koneksi dengan MetaTrader5: {str(e)}"
            )
            return False

    def get_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Ambil data historis untuk training"""
        try:
            if not self.connected and not self.initialize_mt5():
                logger.error(
                    f"Gagal mengambil data historis, terjadi kesalahan pada koneksi MetaTrader5"
                )
                return None

            timeframe_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
            }

            if timeframe not in timeframe_map:
                logger.error(f"Timeframe tidak valid: {timeframe}")
                return None

            count_map = {
                "M1": 300000,
                "M5": 150000,
                "M15": 100000,
                "M30": 60000,
                "H1": 40000,
                "H4": 20000,
                "D1": 7000,
                "W1": 1000,
            }

            count = count_map.get(timeframe)

            rates = mt5.copy_rates_from_pos(
                symbol, timeframe_map[timeframe], 0, count_map[count]
            )

            if rates is None or len(rates) == 0:
                logger.error(f"Gagal mengambil data histori untuk {symbol} {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            if not self.save_data_to_csv(df, symbol, timeframe):
                logger.error(f"Gagal menyimpan data {symbol} {timeframe}")
                return None

            return df

        except Exception as e:
            logger.error(f"Terjadi kesalahan saat mendapatkan data historis: {str(e)}")
            return None

    def save_data_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Simpan data historis ke direktori"""
        try:
            symbol_upper = symbol.upper()
            dir_path = os.path.join(KonfigurasiPath.DATA_DIR, symbol_upper)
            os.makedirs(dir_path, exist_ok=True)

            filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
            file_path = os.path.join(dir_path, filename)

            df.to_csv(file_path, index=False)
            logger.info(f"Data disimpan ke: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Terjadi kesalahan saat menyimpan data: {str(e)}")
            return None
