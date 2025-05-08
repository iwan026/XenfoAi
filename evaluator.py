import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class SignalMetrics:
    """Container for signal performance metrics"""

    total_signals: int
    accuracy: float
    precision: Dict[str, float]  # Per signal type (BUY, SELL, HOLD)
    recall: Dict[str, float]  # Per signal type
    f1_score: Dict[str, float]  # Per signal type
    avg_confidence: float
    false_signals: int
    signal_distribution: Dict[str, int]  # Count of each signal type
    avg_time_between_signals: float  # In minutes


@dataclass
class Signal:
    """Container for signal information"""

    timestamp: datetime
    symbol: str
    timeframe: str
    signal_type: str  # BUY, SELL, or HOLD
    confidence: float
    market_regime: str
    trend_strength: float
    volatility: float
    validated: bool = False
    correct: Optional[bool] = None


class SignalEvaluator:
    """Enhanced evaluator for forex signal generation"""

    def __init__(self):
        self.signals: List[Signal] = []
        self.metrics: Optional[SignalMetrics] = None

    def evaluate_signals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_scores: np.ndarray,
        timestamps: List[datetime],
        market_regimes: List[str],
    ) -> Dict:
        """Comprehensive signal evaluation"""
        try:
            # Basic classification metrics
            class_metrics = self._calculate_classification_metrics(y_true, y_pred)

            # Signal quality metrics
            signal_metrics = self._calculate_signal_metrics(
                y_true, y_pred, confidence_scores, timestamps, market_regimes
            )

            # Combine metrics
            evaluation_results = {
                "classification_metrics": class_metrics,
                "signal_metrics": signal_metrics,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in signal evaluation: {str(e)}")
            raise

    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """Calculate classification performance metrics"""
        try:
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Get detailed classification report
            class_report = classification_report(
                y_true, y_pred, target_names=["SELL", "BUY", "HOLD"], output_dict=True
            )

            # Calculate per-class metrics
            class_metrics = {}
            for signal_type in ["SELL", "BUY", "HOLD"]:
                class_metrics[signal_type] = {
                    "precision": class_report[signal_type]["precision"],
                    "recall": class_report[signal_type]["recall"],
                    "f1-score": class_report[signal_type]["f1-score"],
                    "support": class_report[signal_type]["support"],
                }

            return {
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report,
                "per_class_metrics": class_metrics,
            }

        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            raise

    def _calculate_signal_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_scores: np.ndarray,
        timestamps: List[datetime],
        market_regimes: List[str],
    ) -> Dict:
        """Calculate comprehensive signal metrics"""
        try:
            # Count signals
            signal_counts = {
                "BUY": np.sum(y_pred == 1),
                "SELL": np.sum(y_pred == 0),
                "HOLD": np.sum(y_pred == 2),
            }

            # Calculate time between signals
            signal_times = [
                t for t, p in zip(timestamps, y_pred) if p != 2
            ]  # Exclude HOLD
            if len(signal_times) > 1:
                time_diffs = [
                    (t2 - t1).total_seconds() / 60
                    for t1, t2 in zip(signal_times[:-1], signal_times[1:])
                ]
                avg_time_between = np.mean(time_diffs) if time_diffs else 0
            else:
                avg_time_between = 0

            # Calculate accuracy metrics
            correct_signals = np.sum(y_true == y_pred)
            total_signals = len(y_pred)
            accuracy = correct_signals / total_signals if total_signals > 0 else 0

            # Calculate metrics per market regime
            regime_metrics = self._calculate_regime_metrics(
                y_true, y_pred, market_regimes
            )

            # Create metrics object
            self.metrics = SignalMetrics(
                total_signals=total_signals,
                accuracy=accuracy,
                precision={
                    "BUY": np.mean(y_true[y_pred == 1] == 1) if any(y_pred == 1) else 0,
                    "SELL": np.mean(y_true[y_pred == 0] == 0)
                    if any(y_pred == 0)
                    else 0,
                    "HOLD": np.mean(y_true[y_pred == 2] == 2)
                    if any(y_pred == 2)
                    else 0,
                },
                recall={
                    "BUY": np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
                    if any(y_true == 1)
                    else 0,
                    "SELL": np.sum((y_true == 0) & (y_pred == 0)) / np.sum(y_true == 0)
                    if any(y_true == 0)
                    else 0,
                    "HOLD": np.sum((y_true == 2) & (y_pred == 2)) / np.sum(y_true == 2)
                    if any(y_true == 2)
                    else 0,
                },
                f1_score={
                    signal: 2
                    * (self.metrics.precision[signal] * self.metrics.recall[signal])
                    / (self.metrics.precision[signal] + self.metrics.recall[signal])
                    if (self.metrics.precision[signal] + self.metrics.recall[signal])
                    > 0
                    else 0
                    for signal in ["BUY", "SELL", "HOLD"]
                },
                avg_confidence=np.mean(confidence_scores),
                false_signals=np.sum(y_true != y_pred),
                signal_distribution=signal_counts,
                avg_time_between_signals=avg_time_between,
            )

            return {"metrics": self.metrics.__dict__, "regime_metrics": regime_metrics}

        except Exception as e:
            logger.error(f"Error calculating signal metrics: {str(e)}")
            raise

    def _calculate_regime_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, market_regimes: List[str]
    ) -> Dict:
        """Calculate metrics per market regime"""
        try:
            regime_metrics = {}
            unique_regimes = set(market_regimes)

            for regime in unique_regimes:
                regime_mask = np.array(market_regimes) == regime
                if not any(regime_mask):
                    continue

                regime_true = y_true[regime_mask]
                regime_pred = y_pred[regime_mask]

                regime_metrics[regime] = {
                    "accuracy": np.mean(regime_true == regime_pred),
                    "signal_count": len(regime_pred),
                    "signal_distribution": {
                        "BUY": np.sum(regime_pred == 1),
                        "SELL": np.sum(regime_pred == 0),
                        "HOLD": np.sum(regime_pred == 2),
                    },
                }

            return regime_metrics

        except Exception as e:
            logger.error(f"Error calculating regime metrics: {str(e)}")
            raise

    def add_signal(self, signal: Signal) -> None:
        """Add a new signal for tracking"""
        try:
            self.signals.append(signal)
            logger.info(f"Added new signal for {signal.symbol} at {signal.timestamp}")
        except Exception as e:
            logger.error(f"Error adding signal: {str(e)}")

    def validate_signal(
        self, signal: Signal, current_price: float, target_price: float
    ) -> bool:
        """Validate a signal based on price movement"""
        try:
            if signal.validated:
                return signal.correct

            price_change = (target_price - current_price) / current_price

            if signal.signal_type == "BUY":
                signal.correct = price_change > 0.001  # 0.1% movement
            elif signal.signal_type == "SELL":
                signal.correct = price_change < -0.001  # -0.1% movement
            else:  # HOLD
                signal.correct = abs(price_change) <= 0.001

            signal.validated = True
            return signal.correct

        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            return False

    def get_signal_statistics(self, timeframe: str = "1D") -> Dict:
        """Get statistics for signals in specified timeframe"""
        try:
            current_time = datetime.utcnow()
            if timeframe == "1D":
                start_time = current_time - timedelta(days=1)
            elif timeframe == "1W":
                start_time = current_time - timedelta(weeks=1)
            elif timeframe == "1M":
                start_time = current_time - timedelta(days=30)
            else:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Filter signals
            recent_signals = [
                s for s in self.signals if s.timestamp >= start_time and s.validated
            ]

            if not recent_signals:
                return {}

            # Calculate statistics
            stats = {
                "total_signals": len(recent_signals),
                "successful_signals": len([s for s in recent_signals if s.correct]),
                "signal_types": {
                    "BUY": len([s for s in recent_signals if s.signal_type == "BUY"]),
                    "SELL": len([s for s in recent_signals if s.signal_type == "SELL"]),
                    "HOLD": len([s for s in recent_signals if s.signal_type == "HOLD"]),
                },
                "avg_confidence": np.mean([s.confidence for s in recent_signals]),
                "regime_distribution": {},
            }

            # Calculate success rate
            stats["success_rate"] = (
                stats["successful_signals"] / stats["total_signals"]
                if stats["total_signals"] > 0
                else 0
            )

            # Calculate regime distribution
            for regime in set(s.market_regime for s in recent_signals):
                regime_signals = [
                    s for s in recent_signals if s.market_regime == regime
                ]
                stats["regime_distribution"][regime] = {
                    "count": len(regime_signals),
                    "success_rate": len([s for s in regime_signals if s.correct])
                    / len(regime_signals)
                    if regime_signals
                    else 0,
                }

            return stats

        except Exception as e:
            logger.error(f"Error getting signal statistics: {str(e)}")
            return {}
