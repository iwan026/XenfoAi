import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalUtils:
    """Utility class for signal management and validation"""

    @staticmethod
    def setup_logging(log_level: int = logging.INFO) -> None:
        """Setup logging configuration"""
        try:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(
                        f"logs/signal_bot_{datetime.now().strftime('%Y%m%d')}.log"
                    ),
                ],
            )
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")

    @staticmethod
    def format_signal_message(signal_data: Dict) -> str:
        """Format signal data into a readable message"""
        try:
            emoji = (
                "🟢"
                if signal_data["signal"] == "BUY"
                else "🔴"
                if signal_data["signal"] == "SELL"
                else "⚪"
            )

            message = (
                f"{emoji} *FOREX SIGNAL*\n\n"
                f"*Symbol:* {signal_data['symbol']}\n"
                f"*Signal:* {signal_data['signal']}\n"
                f"*Timeframe:* {signal_data['timeframe']}\n"
                f"*Confidence:* {signal_data['confidence']:.2%}\n"
                f"*Time (UTC):* {signal_data['timestamp']}\n"
                f"*Market Regime:* {signal_data['market_regime']}\n\n"
                f"_This is not financial advice_"
            )

            return message
        except Exception as e:
            logger.error(f"Error formatting signal message: {str(e)}")
            return "Error formatting signal message"

    @staticmethod
    def save_signal_history(signal_data: Dict, base_path: str = "data/signals") -> bool:
        """Save signal to historical record"""
        try:
            # Create signal directory if it doesn't exist
            signal_dir = (
                Path(base_path) / signal_data["symbol"] / signal_data["timeframe"]
            )
            signal_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with current date
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            signal_file = signal_dir / f"signals_{current_date}.json"

            # Load existing signals or create new list
            if signal_file.exists():
                with open(signal_file, "r") as f:
                    signals = json.load(f)
            else:
                signals = []

            # Add new signal
            signals.append(
                {
                    **signal_data,
                    "saved_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )

            # Save updated signals
            with open(signal_file, "w") as f:
                json.dump(signals, f, indent=4)

            return True

        except Exception as e:
            logger.error(f"Error saving signal history: {str(e)}")
            return False

    @staticmethod
    def load_signal_history(
        symbol: str, timeframe: str, days: int = 7, base_path: str = "data/signals"
    ) -> List[Dict]:
        """Load historical signals for analysis"""
        try:
            signals = []
            signal_dir = Path(base_path) / symbol / timeframe

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Iterate through date range
            current_date = start_date
            while current_date <= end_date:
                signal_file = (
                    signal_dir / f"signals_{current_date.strftime('%Y%m%d')}.json"
                )

                if signal_file.exists():
                    with open(signal_file, "r") as f:
                        daily_signals = json.load(f)
                        signals.extend(daily_signals)

                current_date += timedelta(days=1)

            return signals

        except Exception as e:
            logger.error(f"Error loading signal history: {str(e)}")
            return []

    @staticmethod
    def calculate_signal_metrics(signals: List[Dict]) -> Dict:
        """Calculate metrics from historical signals"""
        try:
            if not signals:
                return {}

            total_signals = len(signals)
            signal_types = {
                "BUY": len([s for s in signals if s["signal"] == "BUY"]),
                "SELL": len([s for s in signals if s["signal"] == "SELL"]),
                "HOLD": len([s for s in signals if s["signal"] == "HOLD"]),
            }

            avg_confidence = sum(s["confidence"] for s in signals) / total_signals

            # Convert timestamps to datetime objects
            for signal in signals:
                signal["datetime"] = datetime.strptime(
                    signal["timestamp"], "%Y-%m-%d %H:%M:%S"
                )

            # Calculate time between signals
            signals.sort(key=lambda x: x["datetime"])
            time_diffs = []
            for i in range(1, len(signals)):
                diff = (
                    signals[i]["datetime"] - signals[i - 1]["datetime"]
                ).total_seconds() / 60
                time_diffs.append(diff)

            avg_time_between = sum(time_diffs) / len(time_diffs) if time_diffs else 0

            return {
                "total_signals": total_signals,
                "signal_distribution": signal_types,
                "average_confidence": avg_confidence,
                "average_time_between_signals": avg_time_between,
                "signals_per_day": total_signals
                / ((signals[-1]["datetime"] - signals[0]["datetime"]).days + 1),
            }

        except Exception as e:
            logger.error(f"Error calculating signal metrics: {str(e)}")
            return {}

    @staticmethod
    def validate_signal_timing(
        symbol: str, timeframe: str, last_signals: List[Dict], min_interval: int = 5
    ) -> bool:
        """Validate if enough time has passed since last signal"""
        try:
            if not last_signals:
                return True

            current_time = datetime.now(timezone.utc)
            last_signal_time = datetime.strptime(
                last_signals[-1]["timestamp"], "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)

            time_diff = (current_time - last_signal_time).total_seconds() / 60
            return time_diff >= min_interval

        except Exception as e:
            logger.error(f"Error validating signal timing: {str(e)}")
            return False

    @staticmethod
    def generate_signal_report(
        symbol: str, timeframe: str, signals: List[Dict], report_path: str = "reports"
    ) -> Optional[str]:
        """Generate a summary report of signal performance"""
        try:
            if not signals:
                return None

            # Create report directory
            report_dir = Path(report_path)
            report_dir.mkdir(parents=True, exist_ok=True)

            # Calculate metrics
            metrics = SignalUtils.calculate_signal_metrics(signals)

            # Generate report content
            report = (
                f"Signal Performance Report\n"
                f"========================\n"
                f"Symbol: {symbol}\n"
                f"Timeframe: {timeframe}\n"
                f"Period: {signals[0]['timestamp']} to {signals[-1]['timestamp']}\n\n"
                f"Signal Statistics\n"
                f"-----------------\n"
                f"Total Signals: {metrics['total_signals']}\n"
                f"Signals per Day: {metrics['signals_per_day']:.2f}\n"
                f"Average Confidence: {metrics['average_confidence']:.2%}\n"
                f"Average Time Between Signals: {metrics['average_time_between_signals']:.1f} minutes\n\n"
                f"Signal Distribution\n"
                f"------------------\n"
                f"BUY Signals: {metrics['signal_distribution']['BUY']}\n"
                f"SELL Signals: {metrics['signal_distribution']['SELL']}\n"
                f"HOLD Signals: {metrics['signal_distribution']['HOLD']}\n"
            )

            # Save report
            report_file = (
                report_dir
                / f"signal_report_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.txt"
            )
            with open(report_file, "w") as f:
                f.write(report)

            return str(report_file)

        except Exception as e:
            logger.error(f"Error generating signal report: {str(e)}")
            return None

    @staticmethod
    def cleanup_old_signals(
        base_path: str = "data/signals", days_to_keep: int = 30
    ) -> bool:
        """Clean up old signal files"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            base_dir = Path(base_path)

            for symbol_dir in base_dir.iterdir():
                if symbol_dir.is_dir():
                    for timeframe_dir in symbol_dir.iterdir():
                        if timeframe_dir.is_dir():
                            for signal_file in timeframe_dir.glob("signals_*.json"):
                                try:
                                    file_date = datetime.strptime(
                                        signal_file.stem.split("_")[1], "%Y%m%d"
                                    )
                                    if file_date < cutoff_date:
                                        signal_file.unlink()
                                except Exception as e:
                                    logger.warning(
                                        f"Error processing file {signal_file}: {str(e)}"
                                    )

            return True

        except Exception as e:
            logger.error(f"Error cleaning up old signals: {str(e)}")
            return False
