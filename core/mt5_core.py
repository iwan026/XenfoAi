import MetaTrader5 as mt5
import pandas as pd
import config
import logging
import os

logger = logging.getLogger(__name__)


class MT5Core:
    def __init__(self):
        self.connected = False

    def initialize_mt5(self):
        try:
            if not mt5.initialize():
                logger.error(f"Inisialisasi MetaTrader5 gagal: {mt5.last_error()}")
                return False

            if not mt5.login(
                login=config.MT5_LOGIN,
                password=config.MT5_PASSWORD,
                server=config.MT5_SERVER,
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
