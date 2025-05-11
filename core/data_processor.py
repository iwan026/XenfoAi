import logging
import os

from features.support_resistance import add_support_resistance
from config import KonfigurasiPath

logger = logging.getLogger(__name__)


def preprocessing_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Proses ulang data mentah untuk training model"""
    try:
        symbol_upper = symbol.upper()
        raw_symbol_dir = os.path.join(
            KonfigurasiPath.DATASET_FOREX_RAW_DIR, symbol_upper
        )

        filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
        raw_data_name = os.path.join(raw_symbol_dir, filename)

        data = df.read_csv(raw_data_name)

        add_support_resistance(df)
        return df

    except Exception as e:
        logger.error(f"Gagal preprocessing data: {str(e)}")
        return None


def save_processed_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Simpan processed data ke direktori"""
    try:
        symbol_upper = symbol.upper()
        dir_path = os.path.join(
            KonfigurasiPath.DATASET_FOREX_PROCESSED_DIR, symbol_upper
        )
        os.makedirs(dir_path, exist_ok=True)

        filename = f"{symbol.lower()}_{timeframe.lower()}.csv"
        file_path = os.path.join(dir_path, filename)
        df.to_csv(file_path, index=False)
        logger.info(f"Data processed berhasil disimpan ke: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Gagal menyimpan preprocessing data: {str(e)}")
        return False
