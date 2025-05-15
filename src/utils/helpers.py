import logging
from pathlib import Path
from typing import Dict, List
from config import LOGS_DIR


def setup_logging(name: str) -> logging.logger:
    """Setup konfigurasi logging"""
    log_file = LOGS_DIR / f"{name}.log"

    formatter = logging.formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def format_prediction_message(prediction: Dict) -> str:
    """Format prediction results for Telegram message"""
    return (
        f"📊 *Prediksi Forex untuk {prediction['symbol']} ({prediction['timeframe']})*\n\n"
        f"🔮 *Prediksi AI:* {prediction['prediction']}\n"
        f"⚖️ *Keyakinan:* {prediction['confidence']:.2f}\n\n"
        f"*Sinyal Indikator:*\n"
        f"📈 RSI: {prediction['rsi_signal']}\n"
        f"📉 MACD: {prediction['macd_signal']}\n"
        f"📊 MA: {prediction['ma_signal']}\n\n"
        f"*Keputusan Akhir:* {prediction['final_signal']} "
        f"dengan keyakinan {prediction['final_confidence']:.2f}\n\n"
        f"🕒 *Waktu:* {prediction['timestamp']}"
    )


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe input"""
    from config.config import TIMEFRAMES

    return timeframe.upper() in TIMEFRAMES


def get_available_symbols() -> List[str]:
    """Get list of available symbols from dataset directory"""
    from config.config import DATASET_DIR

    return [d.name for d in DATASET_DIR.iterdir() if d.is_dir()]


def get_available_timeframes(symbol: str) -> List[str]:
    """Get list of available timeframes for a symbol"""
    from config.config import DATASET_DIR

    symbol_dir = DATASET_DIR / symbol
    if not symbol_dir.exists():
        return []

    timeframes = []
    for file in symbol_dir.glob(f"{symbol}_*.csv"):
        timeframe = file.stem.split("_")[1]
        timeframes.append(timeframe)

    return timeframes
