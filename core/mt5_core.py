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
