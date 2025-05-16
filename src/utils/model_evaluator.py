import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.reports_dir = Path("reports") / f"{symbol}_{timeframe}"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray
    ) -> Dict:
        """Evaluasi prediksi model"""
        try:
            # Metrics dasar
            accuracy = np.mean(y_true == (y_pred > 0.5))

            # Confidence analysis
            high_conf_mask = confidences > 0.8
            high_conf_accuracy = np.mean(
                y_true[high_conf_mask] == (y_pred[high_conf_mask] > 0.5)
            )

            # Profit potential (sederhana)
            correct_ups = np.sum((y_pred > 0.5) & (y_true == 1))
            correct_downs = np.sum((y_pred <= 0.5) & (y_true == 0))
            potential_profit = (correct_ups + correct_downs) / len(y_true)

            return {
                "accuracy": accuracy,
                "high_confidence_accuracy": high_conf_accuracy,
                "potential_profit": potential_profit,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Error in prediction evaluation: {e}")
            return {}

    def generate_report(self, predictions: List[Dict], save_plots: bool = True) -> Dict:
        """Generate laporan evaluasi lengkap"""
        try:
            df = pd.DataFrame(predictions)

            # Performance metrics
            performance = {
                "total_predictions": len(df),
                "accuracy": np.mean(df["correct"]),
                "avg_confidence": df["confidence"].mean(),
                "profit_factor": len(df[df["profit"] > 0]) / len(df[df["profit"] < 0]),
            }

            if save_plots:
                # Plot accuracy over time
                plt.figure(figsize=(10, 6))
                plt.plot(df["timestamp"], df["correct"].rolling(100).mean())
                plt.title("Accuracy Rolling Average (100 predictions)")
                plt.xlabel("Time")
                plt.ylabel("Accuracy")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.reports_dir / "accuracy_trend.png")
                plt.close()

                # Plot confidence distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x="confidence", hue="correct")
                plt.title("Confidence Distribution by Outcome")
                plt.xlabel("Confidence")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(self.reports_dir / "confidence_dist.png")
                plt.close()

                # Plot confusion matrix
                conf_matrix = pd.crosstab(
                    df["predicted_direction"], df["actual_direction"]
                )
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.tight_layout()
                plt.savefig(self.reports_dir / "confusion_matrix.png")
                plt.close()

            # Save report
            report = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "performance_metrics": performance,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "plots_saved": save_plots,
                "plots_location": str(self.reports_dir) if save_plots else None,
            }

            # Save to file
            report_path = (
                self.reports_dir
                / f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {}
