import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, classification_report
import logging
from datetime import datetime, timedelta
import pyfolio as pf
from empyrical import (
    annual_return,
    annual_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    omega_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Container for trade performance metrics"""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    recovery_factor: float
    risk_reward_ratio: float
    expectancy: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float


@dataclass
class Trade:
    """Container for individual trade information"""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position: str  # 'long' or 'short'
    size: float
    pnl: float
    pip_movement: float
    hold_time: timedelta
    market_regime: str
    entry_signal_strength: float
    exit_signal_strength: float


class ForexModelEvaluator:
    """Enhanced evaluator for forex trading models"""

    def __init__(
        self,
        initial_balance: float = 10000,
        risk_per_trade: float = 0.02,
        pip_value: float = 0.0001,
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.pip_value = pip_value
        self.trades: List[Trade] = []
        self.metrics: Optional[TradeMetrics] = None

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        price_data: pd.DataFrame,
        signal_probabilities: np.ndarray,
        market_regimes: List[str],
    ) -> Dict:
        """Comprehensive model evaluation"""
        try:
            # Classification metrics
            class_metrics = self._calculate_classification_metrics(y_true, y_pred)

            # Trading simulation
            self._simulate_trading(
                y_pred, price_data, signal_probabilities, market_regimes
            )

            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics()

            # Risk metrics
            risk_metrics = self._calculate_risk_metrics()

            # Combine all metrics
            evaluation_results = {
                "classification_metrics": class_metrics,
                "trading_metrics": trading_metrics,
                "risk_metrics": risk_metrics,
            }

            return evaluation_results

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def _calculate_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """Calculate classification performance metrics"""
        try:
            # Basic classification metrics
            conf_matrix = confusion_matrix(y_true, y_pred)
            class_report = classification_report(y_true, y_pred, output_dict=True)

            # Calculate per-class metrics
            class_metrics = {}
            for class_label in np.unique(y_true):
                mask = y_true == class_label
                class_metrics[f"class_{class_label}"] = {
                    "precision": class_report[str(class_label)]["precision"],
                    "recall": class_report[str(class_label)]["recall"],
                    "f1-score": class_report[str(class_label)]["f1-score"],
                    "support": class_report[str(class_label)]["support"],
                }

            return {
                "confusion_matrix": conf_matrix,
                "classification_report": class_report,
                "per_class_metrics": class_metrics,
            }

        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            raise

    def _simulate_trading(
        self,
        signals: np.ndarray,
        price_data: pd.DataFrame,
        probabilities: np.ndarray,
        market_regimes: List[str],
    ) -> None:
        """Simulate trading with advanced position sizing and risk management"""
        try:
            balance = self.initial_balance
            position = None

            for i in range(1, len(signals)):
                current_time = price_data.index[i]
                current_price = price_data["close"].iloc[i]
                current_regime = market_regimes[i]

                # Close existing position if signal changes
                if position and signals[i] != signals[i - 1]:
                    exit_price = current_price
                    pnl = self._calculate_trade_pnl(position, exit_price)

                    # Record trade
                    trade = Trade(
                        entry_time=position["entry_time"],
                        exit_time=current_time,
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        position=position["type"],
                        size=position["size"],
                        pnl=pnl,
                        pip_movement=self._calculate_pip_movement(
                            position["entry_price"], exit_price
                        ),
                        hold_time=current_time - position["entry_time"],
                        market_regime=current_regime,
                        entry_signal_strength=position["entry_probability"],
                        exit_signal_strength=probabilities[i].max(),
                    )
                    self.trades.append(trade)

                    balance += pnl
                    position = None

                # Open new position if signal is not hold (4)
                if signals[i] != 4 and not position:
                    # Calculate position size based on risk
                    risk_amount = balance * self.risk_per_trade
                    # Adjust risk based on signal probability
                    signal_confidence = probabilities[i].max()
                    adjusted_risk = risk_amount * signal_confidence

                    position_size = self._calculate_position_size(
                        adjusted_risk, current_price
                    )

                    position = {
                        "type": "long" if signals[i] in [2, 3] else "short",
                        "entry_price": current_price,
                        "size": position_size,
                        "entry_time": current_time,
                        "entry_probability": signal_confidence,
                    }

        except Exception as e:
            logger.error(f"Error in trading simulation: {str(e)}")
            raise

    def _calculate_trading_metrics(self) -> TradeMetrics:
        """Calculate comprehensive trading metrics"""
        try:
            if not self.trades:
                raise ValueError("No trades to analyze")

            # Basic trade statistics
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]

            # Calculate metrics
            self.metrics = TradeMetrics(
                total_trades=len(self.trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=len(winning_trades) / len(self.trades),
                profit_factor=abs(
                    sum(t.pnl for t in winning_trades)
                    / sum(t.pnl for t in losing_trades)
                )
                if sum(t.pnl for t in losing_trades) != 0
                else float("inf"),
                avg_win=np.mean([t.pnl for t in winning_trades])
                if winning_trades
                else 0,
                avg_loss=np.mean([t.pnl for t in losing_trades])
                if losing_trades
                else 0,
                max_drawdown=self._calculate_max_drawdown(),
                recovery_factor=self._calculate_recovery_factor(),
                risk_reward_ratio=self._calculate_risk_reward_ratio(),
                expectancy=self._calculate_expectancy(),
                sharpe_ratio=self._calculate_sharpe_ratio(),
                sortino_ratio=self._calculate_sortino_ratio(),
                calmar_ratio=self._calculate_calmar_ratio(),
                omega_ratio=self._calculate_omega_ratio(),
            )

            return self.metrics

        except Exception as e:
            logger.error(f"Error calculating trading metrics: {str(e)}")
            raise

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate advanced risk metrics"""
        try:
            returns = pd.Series([t.pnl for t in self.trades])

            risk_metrics = {
                "var_95": self._calculate_var(returns, 0.95),
                "cvar_95": self._calculate_cvar(returns, 0.95),
                "downside_deviation": self._calculate_downside_deviation(returns),
                "volatility": returns.std(),
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
                "tail_ratio": self._calculate_tail_ratio(returns),
            }

            return risk_metrics

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            raise

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of results"""
        try:
            fig = plt.figure(figsize=(20, 15))

            # 1. Equity Curve
            ax1 = plt.subplot(321)
            self._plot_equity_curve(ax1)

            # 2. Drawdown Chart
            ax2 = plt.subplot(322)
            self._plot_drawdown(ax2)

            # 3. Trade Distribution
            ax3 = plt.subplot(323)
            self._plot_trade_distribution(ax3)

            # 4. Win Rate by Market Regime
            ax4 = plt.subplot(324)
            self._plot_regime_analysis(ax4)

            # 5. Monthly Returns Heatmap
            ax5 = plt.subplot(325)
            self._plot_monthly_returns(ax5)

            # 6. Risk Metrics
            ax6 = plt.subplot(326)
            self._plot_risk_metrics(ax6)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise

    def generate_report(self, output_path: str) -> None:
        """Generate comprehensive HTML report"""
        try:
            report = self._create_html_report()

            with open(output_path, "w") as f:
                f.write(report)

            logger.info(f"Report generated successfully at {output_path}")

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        return np.sqrt(np.mean(np.minimum(returns - returns.mean(), 0) ** 2))

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio"""
        return abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))

    def _create_html_report(self) -> str:
        """Create detailed HTML report"""
        template = """
        <html>
        <head>
            <title>Trading Model Evaluation Report</title>
            <style>
                /* Add your CSS styles here */
            </style>
        </head>
        <body>
            <h1>Trading Model Evaluation Report</h1>
            <div class="metrics">
                <h2>Performance Metrics</h2>
                <!-- Add metrics here -->
            </div>
            <div class="charts">
                <!-- Add base64 encoded charts here -->
            </div>
        </body>
        </html>
        """

        return template

    def _plot_equity_curve(self, ax: plt.Axes) -> None:
        """Plot equity curve with drawdown overlay"""
        equity_curve = self._calculate_equity_curve()
        ax.plot(equity_curve.index, equity_curve.values)
        ax.set_title("Equity Curve")
        ax.grid(True)

    def _plot_drawdown(self, ax: plt.Axes) -> None:
        """Plot drawdown chart"""
        drawdown = self._calculate_drawdown_series()
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        ax.set_title("Drawdown")
        ax.grid(True)

    def _plot_trade_distribution(self, ax: plt.Axes) -> None:
        """Plot trade profit/loss distribution"""
        pnls = [t.pnl for t in self.trades]
        ax.hist(pnls, bins=50, alpha=0.75)
        ax.set_title("Trade P&L Distribution")
        ax.grid(True)

    def _plot_regime_analysis(self, ax: plt.Axes) -> None:
        """Plot win rate by market regime"""
        regime_results = pd.DataFrame(
            [
                {"regime": t.market_regime, "win": 1 if t.pnl > 0 else 0}
                for t in self.trades
            ]
        )
        win_rates = regime_results.groupby("regime")["win"].mean()
        win_rates.plot(kind="bar", ax=ax)
        ax.set_title("Win Rate by Market Regime")
        ax.grid(True)

    def _plot_monthly_returns(self, ax: plt.Axes) -> None:
        """Plot monthly returns heatmap"""
        monthly_returns = self._calculate_monthly_returns()
        sns.heatmap(
            monthly_returns.pivot_table(
                index=monthly_returns.index.year,
                columns=monthly_returns.index.month,
                values="returns",
            ),
            ax=ax,
            cmap="RdYlGn",
            center=0,
        )
        ax.set_title("Monthly Returns")

    def _plot_risk_metrics(self, ax: plt.Axes) -> None:
        """Plot risk metrics"""
        risk_metrics = self._calculate_risk_metrics()
        y_pos = np.arange(len(risk_metrics))
        ax.barh(y_pos, list(risk_metrics.values()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(risk_metrics.keys()))
        ax.set_title("Risk Metrics")
