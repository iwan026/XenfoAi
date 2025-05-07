import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from config import TradingConfig

logger = logging.getLogger(__name__)


class TradingUtils:
    """Utility functions for forex trading application"""

    @staticmethod
    def setup_logging() -> None:
        """Setup logging configuration"""
        try:
            logging.basicConfig(
                filename=TradingConfig.LOG_FILE,
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Also log to console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            logging.getLogger("").addHandler(console_handler)

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")

    @staticmethod
    def create_trading_chart(
        df: pd.DataFrame,
        signals: Optional[List[Dict]] = None,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Create technical analysis chart with signals"""
        try:
            # Create figure with secondary y-axis
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1, 1]}
            )

            # Plot candlesticks
            width = 0.6
            width2 = 0.05

            up = df[df.close >= df.open]
            down = df[df.close < df.open]

            # Plot up candles
            ax1.bar(
                up.index,
                up.close - up.open,
                width,
                bottom=up.open,
                color="green",
                alpha=0.8,
            )
            ax1.bar(
                up.index,
                up.high - up.close,
                width2,
                bottom=up.close,
                color="green",
                alpha=0.8,
            )
            ax1.bar(
                up.index,
                up.low - up.open,
                width2,
                bottom=up.open,
                color="green",
                alpha=0.8,
            )

            # Plot down candles
            ax1.bar(
                down.index,
                down.close - down.open,
                width,
                bottom=down.open,
                color="red",
                alpha=0.8,
            )
            ax1.bar(
                down.index,
                down.high - down.open,
                width2,
                bottom=down.open,
                color="red",
                alpha=0.8,
            )
            ax1.bar(
                down.index,
                down.low - down.close,
                width2,
                bottom=down.close,
                color="red",
                alpha=0.8,
            )

            # Add technical indicators
            ax1.plot(df.index, df.sma_20, "b-", label="SMA 20", alpha=0.7)
            ax1.plot(df.index, df.ema_50, "y-", label="EMA 50", alpha=0.7)

            # Add Bollinger Bands
            ax1.plot(df.index, df["bb_middle"], "k--", alpha=0.3)
            ax1.fill_between(
                df.index, df["bb_upper"], df["bb_lower"], alpha=0.1, color="gray"
            )

            # Add RSI
            ax2.plot(df.index, df.rsi, "purple", label="RSI")
            ax2.axhline(y=70, color="r", linestyle="--", alpha=0.3)
            ax2.axhline(y=30, color="g", linestyle="--", alpha=0.3)
            ax2.set_ylim(0, 100)

            # Add MACD
            ax3.plot(df.index, df["macd"], "b-", label="MACD")
            ax3.plot(df.index, df["macd_signal"], "r-", label="Signal")
            ax3.bar(df.index, df["macd_hist"], 0.02, label="Histogram")

            # Add trading signals if provided
            if signals:
                for signal in signals:
                    signal_time = pd.to_datetime(signal["timestamp"])
                    if signal_time in df.index:
                        color = "green" if signal["signal"] == "BUY" else "red"
                        ax1.axvline(x=signal_time, color=color, alpha=0.5)

            # Customize appearance
            ax1.set_title(
                f"Trading Chart - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_title("RSI (14)")
            ax2.grid(True, alpha=0.3)

            ax3.set_title("MACD")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save or show chart
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return save_path
            else:
                plt.show()
                plt.close()
                return None

        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return None

    @staticmethod
    def calculate_trading_metrics(
        trades: List[Dict], initial_balance: float
    ) -> Dict[str, Union[float, int]]:
        """Calculate trading performance metrics"""
        try:
            if not trades:
                return {}

            # Extract profits/losses
            pnl_list = [trade["profit"] for trade in trades]

            # Basic metrics
            total_trades = len(trades)
            profitable_trades = len([p for p in pnl_list if p > 0])
            loss_trades = len([p for p in pnl_list if p < 0])

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            # Profit metrics
            total_profit = sum([p for p in pnl_list if p > 0])
            total_loss = abs(sum([p for p in pnl_list if p < 0]))
            net_profit = sum(pnl_list)

            profit_factor = (
                total_profit / total_loss if total_loss > 0 else float("inf")
            )

            # Calculate equity curve
            equity_curve = [initial_balance]
            for pnl in pnl_list:
                equity_curve.append(equity_curve[-1] + pnl)

            # Maximum drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (running_max - equity_curve) / running_max
            max_drawdown = np.max(drawdown) * 100  # as percentage

            # Risk-adjusted returns
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = (
                np.mean(returns) / np.std(returns) * np.sqrt(252)
                if len(returns) > 1
                else 0
            )

            return {
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "loss_trades": loss_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_profit": net_profit,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    @staticmethod
    def save_trading_results(results: Dict, filename: str) -> bool:
        """Save trading results to JSON file"""
        try:
            # Ensure directory exists
            Path("results").mkdir(exist_ok=True)

            filepath = Path("results") / filename

            # Convert datetime objects to string
            results_copy = results.copy()
            for key, value in results_copy.items():
                if isinstance(value, datetime):
                    results_copy[key] = value.strftime("%Y-%m-%d %H:%M:%S")

            with open(filepath, "w") as f:
                json.dump(results_copy, f, indent=4)

            logger.info(f"Results saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False

    @staticmethod
    def load_trading_results(filename: str) -> Optional[Dict]:
        """Load trading results from JSON file"""
        try:
            filepath = Path("results") / filename

            if not filepath.exists():
                logger.error(f"Results file not found: {filepath}")
                return None

            with open(filepath, "r") as f:
                results = json.load(f)

            # Convert string dates back to datetime
            for key, value in results.items():
                if isinstance(value, str) and "timestamp" in key.lower():
                    try:
                        results[key] = datetime.strptime(
                            value, "%Y-%m-%d %H:%M:%S"
                        ).replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass

            return results

        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return None

    @staticmethod
    def format_number(
        number: float, decimals: int = 2, include_sign: bool = False
    ) -> str:
        """Format number for display"""
        try:
            if abs(number) >= 1e6:
                return f"{'+' if include_sign and number > 0 else ''}{number / 1e6:.{decimals}f}M"
            elif abs(number) >= 1e3:
                return f"{'+' if include_sign and number > 0 else ''}{number / 1e3:.{decimals}f}K"
            else:
                return (
                    f"{'+' if include_sign and number > 0 else ''}{number:.{decimals}f}"
                )
        except Exception as e:
            logger.error(f"Error formatting number: {str(e)}")
            return str(number)
