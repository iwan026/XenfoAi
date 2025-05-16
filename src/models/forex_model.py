import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from src.models.model_config import ModelConfig
from src.models.hybrid_model import HybridForexModel
from src.data.data_handler import DataHandler
from src.utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class ForexModel:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.config = ModelConfig()
        self.data_handler = DataHandler(self.config)
        self.model = HybridForexModel(self.config)
        self.performance_monitor = PerformanceMonitor(self.symbol, self.timeframe)
        self.model_dir = self.get_model_dir()
        self.load_or_create_model()

    def get_model_dir(self) -> Path:
        """Mendapatkan direktori model"""
        model_dir = Path("models") / self.symbol
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_model_paths(self) -> tuple:
        """Mendapatkan path untuk file model"""
        xgb_path = self.model_dir / f"{self.symbol}_{self.timeframe}_xgb.json"
        deep_path = self.model_dir / f"{self.symbol}_{self.timeframe}_deep.h5"
        return xgb_path, deep_path

    def load_or_create_model(self) -> None:
        """Muat model yang sudah ada atau buat baru"""
        try:
            xgb_path, deep_path = self.get_model_paths()

            if xgb_path.exists() and deep_path.exists():
                logger.info(f"Memuat model dari {xgb_path} dan {deep_path}")
                self.model.load_models(str(xgb_path), str(deep_path))
            else:
                logger.info("Model tidak ditemukan, akan dibuat saat training")

        except Exception as e:
            logger.error(f"Error saat memuat model: {e}")
            raise

    def train(self) -> bool:
        """Training model"""
        try:
            logger.info(f"Memulai training untuk {self.symbol} {self.timeframe}")

            # Update dataset
            if not self.data_handler.update_dataset(self.symbol, self.timeframe):
                logger.error("Gagal mengupdate dataset")
                return False

            # Load data
            df = self.data_handler.load_from_csv(self.symbol, self.timeframe)
            if df is None:
                logger.error("Gagal memuat data dari CSV")
                return False

            # Prepare data
            train_data, val_data = self.data_handler.prepare_training_data(
                df, self.config.VALIDATION_SPLIT
            )

            # Train model
            logger.info("Memulai proses training model...")
            history = self.model.train(train_data, val_data)

            # Save model
            xgb_path, deep_path = self.get_model_paths()
            self.model.save_models(str(xgb_path), str(deep_path))

            # Record training metrics
            self.performance_monitor.add_training_metrics(history)

            logger.info(f"Model berhasil di-training dan disimpan")
            return True

        except Exception as e:
            logger.error(f"Error saat training model: {e}")
            return False

    def predict(self) -> Optional[Dict]:
        """Membuat prediksi"""
        try:
            # Get latest data
            df = self.data_handler.get_symbol_data(
                self.symbol, self.timeframe, num_candles=self.config.SEQUENCE_LENGTH + 1
            )

            if df is None:
                return None

            # Prepare data
            pred_data = self.data_handler.prepare_prediction_data(df)

            # Make prediction
            prediction = self.model.predict(pred_data)[0]

            # Calculate confidence and signals
            confidence = abs(prediction - 0.5) * 2  # Scale to 0-1

            # Get technical signals
            latest_data = df.iloc[-1]

            signals = {
                "rsi_signal": "Overbought"
                if latest_data["rsi_14"] > 70
                else "Oversold"
                if latest_data["rsi_14"] < 30
                else "Neutral",
                "macd_signal": "Buy"
                if latest_data["macd"] > latest_data["macd_signal"]
                else "Sell",
                "ma_signal": "Buy"
                if latest_data["sma_20"] > latest_data["sma_50"]
                else "Sell",
            }

            final_signal = "Buy" if prediction > 0.5 else "Sell"

            result = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "prediction": float(prediction),
                "confidence": float(confidence),
                "rsi_signal": signals["rsi_signal"],
                "macd_signal": signals["macd_signal"],
                "ma_signal": signals["ma_signal"],
                "final_signal": final_signal,
                "final_confidence": float(confidence),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Record prediction
            self.performance_monitor.add_prediction(result)

            return result

        except Exception as e:
            logger.error(f"Error saat membuat prediksi: {e}")
            return None

    def save_models(self, xgb_path: str, deep_path: str):
        """Menyimpan model"""
        try:
            self.model.save_models(xgb_path, deep_path)
        except Exception as e:
            logger.error(f"Error saat menyimpan model: {e}")
            raise
