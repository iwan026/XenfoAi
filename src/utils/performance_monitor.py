import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.metrics_path = Path("metrics") / f"{symbol}_{timeframe}_metrics.json"
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict = self.load_metrics()

    def load_metrics(self) -> Dict:
        """Load metrics dari file"""
        try:
            if self.metrics_path.exists():
                with open(self.metrics_path, "r") as f:
                    return json.load(f)
            return {
                "predictions": [],
                "accuracy": [],
                "training_history": [],
                "resource_usage": [],
            }
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            return {
                "predictions": [],
                "accuracy": [],
                "training_history": [],
                "resource_usage": [],
            }

    def save_metrics(self) -> None:
        """Simpan metrics ke file"""
        try:
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def add_prediction(self, prediction: Dict, actual_outcome: float = None) -> None:
        """Tambah hasil prediksi ke metrics"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        pred_data = {
            "timestamp": timestamp,
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "actual": actual_outcome,
            "signals": {
                "rsi": prediction["rsi_signal"],
                "macd": prediction["macd_signal"],
                "ma": prediction["ma_signal"],
            },
        }

        self.metrics["predictions"].append(pred_data)
        self.save_metrics()

    def add_training_metrics(self, history: Dict) -> None:
        """Tambah metrics training"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        training_data = {
            "timestamp": timestamp,
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history.get("val_loss", [0])[-1],
            "val_accuracy": history.history.get("val_accuracy", [0])[-1],
        }

        self.metrics["training_history"].append(training_data)
        self.save_metrics()

    def add_resource_usage(self, cpu_usage: float, memory_usage: float) -> None:
        """Tambah penggunaan resource"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        resource_data = {
            "timestamp": timestamp,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
        }

        self.metrics["resource_usage"].append(resource_data)
        self.save_metrics()

    def calculate_accuracy(self, window: int = 100) -> Dict:
        """Hitung akurasi prediksi"""
        try:
            predictions = self.metrics["predictions"][-window:]
            if not predictions:
                return {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                }

            correct = 0
            total = 0
            tp = fp = tn = fn = 0

            for pred in predictions:
                if pred.get("actual") is not None:
                    total += 1
                    pred_value = pred["prediction"] > 0.5
                    actual_value = pred["actual"] > 0

                    if pred_value == actual_value:
                        correct += 1

                    if pred_value and actual_value:
                        tp += 1
                    elif pred_value and not actual_value:
                        fp += 1
                    elif not pred_value and not actual_value:
                        tn += 1
                    else:
                        fn += 1

            accuracy = correct / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }

            self.metrics["accuracy"].append(
                {
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    **metrics,
                }
            )

            self.save_metrics()
            return metrics

        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    def get_performance_summary(self) -> Dict:
        """Dapatkan ringkasan performa"""
        try:
            recent_accuracy = self.calculate_accuracy()

            # Analisis trend
            predictions = self.metrics["predictions"][-1000:]
            timestamps = [pd.to_datetime(p["timestamp"]) for p in predictions]
            confidences = [p["confidence"] for p in predictions]

            # Hitung rata-rata confidence per hari
            df = pd.DataFrame({"timestamp": timestamps, "confidence": confidences})
            daily_confidence = df.set_index("timestamp").resample("D").mean()

            # Hitung trend confidence
            confidence_trend = (
                "Meningkat"
                if daily_confidence["confidence"].diff().mean() > 0
                else "Menurun"
            )

            # Analisis resource
            resource_usage = self.metrics["resource_usage"][-100:]
            avg_cpu = np.mean([r["cpu_usage"] for r in resource_usage])
            avg_memory = np.mean([r["memory_usage"] for r in resource_usage])

            return {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "accuracy_metrics": recent_accuracy,
                "confidence_trend": confidence_trend,
                "avg_confidence": daily_confidence["confidence"].mean(),
                "resource_metrics": {
                    "avg_cpu_usage": avg_cpu,
                    "avg_memory_usage": avg_memory,
                },
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
