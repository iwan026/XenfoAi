import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timezone
import mplfinance as mpf
from scipy import stats
import talib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class TradingUtils:
    """Enhanced utility class for forex trading visualization and analysis"""

    def __init__(self):
        self.current_user = "iwan026"
        self.current_time = datetime.strptime(
            "2025-05-07 08:57:59", "%Y-%m-%d %H:%M:%S"
        )
        self.theme = "dark"
        plt.style.use("dark_background" if self.theme == "dark" else "default")

    def plot_trading_chart(
        self,
        df: pd.DataFrame,
        signals: Optional[List[Dict]] = None,
        save_path: Optional[str] = None,
        plot_type: str = "interactive",
    ) -> None:
        """Enhanced trading chart visualization"""
        try:
            if plot_type == "interactive":
                self._plot_interactive_chart(df, signals)
            else:
                self._plot_static_chart(df, signals)

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Chart saved to {save_path}")

        except Exception as e:
            logger.error(f"Error plotting trading chart: {str(e)}")
            raise

    def _plot_interactive_chart(
        self, df: pd.DataFrame, signals: Optional[List[Dict]] = None
    ) -> None:
        """Create interactive trading chart using Plotly"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Add Moving Averages
        for ma_period in [20, 50, 200]:
            ma = df["close"].rolling(window=ma_period).mean()
            fig.add_trace(
                go.Scatter(x=df.index, y=ma, name=f"MA{ma_period}", line=dict(width=1)),
                row=1,
                col=1,
            )

        # Add Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            df["close"], timeperiod=20, nbdevup=2, nbdevdn=2
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=bb_upper, name="BB Upper", line=dict(dash="dash")),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb_lower,
                name="BB Lower",
                line=dict(dash="dash"),
                fill="tonexty",
            ),
            row=1,
            col=1,
        )

        # Add RSI
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI"), row=2, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Add MACD
        fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD"), row=3, col=1)

        fig.add_trace(
            go.Scatter(x=df.index, y=df["macd_signal"], name="Signal"), row=3, col=1
        )

        # Add signals if provided
        if signals:
            for signal in signals:
                signal_time = pd.to_datetime(signal["timestamp"])
                if signal_time in df.index:
                    color = "green" if signal["signal"] == "BUY" else "red"
                    fig.add_vline(
                        x=signal_time, line_color=color, opacity=0.5, row=1, col=1
                    )

        # Update layout
        fig.update_layout(
            title=f"Trading Chart - {self.current_time}",
            yaxis_title="Price",
            yaxis2_title="RSI",
            yaxis3_title="MACD",
            xaxis_rangeslider_visible=False,
            height=800,
        )

        fig.show()

    def _plot_static_chart(
        self, df: pd.DataFrame, signals: Optional[List[Dict]] = None
    ) -> None:
        """Create static trading chart using mplfinance"""
        # Prepare data for mplfinance
        df_plot = df.copy()
        df_plot.index.name = "Date"

        # Define style
        mc = mpf.make_marketcolors(
            up="green",
            down="red",
            edge="inherit",
            wick="inherit",
            volume="in",
            ohlc="inherit",
        )

        s = mpf.make_mpf_style(marketcolors=mc, gridstyle="dotted", y_on_right=False)

        # Create subplots
        fig, axlist = mpf.plot(
            df_plot,
            type="candle",
            style=s,
            volume=True,
            figsize=(12, 8),
            panel_ratios=(6, 2, 2),
            addplot=self._create_indicator_subplots(df),
            returnfig=True,
        )

        # Add signals if provided
        if signals:
            ax = axlist[0]
            for signal in signals:
                signal_time = pd.to_datetime(signal["timestamp"])
                if signal_time in df.index:
                    color = "green" if signal["signal"] == "BUY" else "red"
                    ax.axvline(x=signal_time, color=color, alpha=0.5)

        plt.tight_layout()

    def _create_indicator_subplots(self, df: pd.DataFrame) -> List:
        """Create technical indicator subplots for mplfinance"""
        subplot_indicators = []

        # Add Moving Averages
        for ma_period in [20, 50, 200]:
            ma = df["close"].rolling(window=ma_period).mean()
            subplot_indicators.append(
                mpf.make_addplot(ma, panel=0, color=f"C{ma_period // 20}")
            )

        # Add RSI
        subplot_indicators.append(mpf.make_addplot(df["rsi"], panel=1, color="purple"))

        # Add MACD
        subplot_indicators.extend(
            [
                mpf.make_addplot(df["macd"], panel=2, color="blue"),
                mpf.make_addplot(df["macd_signal"], panel=2, color="orange"),
            ]
        )

        return subplot_indicators

    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analyze current market conditions"""
        try:
            analysis = {
                "volatility": self._analyze_volatility(df),
                "trend": self._analyze_trend(df),
                "momentum": self._analyze_momentum(df),
                "support_resistance": self._find_support_resistance(df),
                "correlation": self._analyze_correlation(df),
                "volume_analysis": self._analyze_volume(df)
                if "volume" in df.columns
                else None,
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            raise

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Analyze market volatility"""
        atr = df["atr"].iloc[-1]
        hist_vol = df["returns"].std() * np.sqrt(252)

        return {
            "current_atr": atr,
            "historical_volatility": hist_vol,
            "volatility_regime": "high"
            if hist_vol > df["returns"].std() * 2
            else "normal",
            "atr_percentile": stats.percentileofscore(df["atr"], atr),
        }

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Analyze market trend"""
        current_price = df["close"].iloc[-1]
        sma_200 = df["close"].rolling(200).mean().iloc[-1]
        adx = df["adx"].iloc[-1]

        return {
            "trend_direction": "uptrend" if current_price > sma_200 else "downtrend",
            "trend_strength": "strong" if adx > 25 else "weak",
            "price_location": {
                "above_200ma": current_price > sma_200,
                "above_50ma": current_price > df["close"].rolling(50).mean().iloc[-1],
                "above_20ma": current_price > df["close"].rolling(20).mean().iloc[-1],
            },
        }

    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze market momentum"""
        return {
            "rsi_condition": "overbought"
            if df["rsi"].iloc[-1] > 70
            else "oversold"
            if df["rsi"].iloc[-1] < 30
            else "neutral",
            "macd_signal": "bullish"
            if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
            else "bearish",
            "momentum_strength": abs(df["returns"].iloc[-20:].mean()) * 100,
        }

    def _find_support_resistance(
        self, df: pd.DataFrame, window: int = 20, threshold: float = 0.02
    ) -> Dict:
        """Find support and resistance levels"""
        highs = df["high"].rolling(window=window, center=True).max()
        lows = df["low"].rolling(window=window, center=True).min()

        resistance_levels = []
        support_levels = []

        for i in range(len(df) - window):
            if highs.iloc[i] == df["high"].iloc[i]:
                resistance_levels.append(df["high"].iloc[i])
            if lows.iloc[i] == df["low"].iloc[i]:
                support_levels.append(df["low"].iloc[i])

        return {
            "resistance_levels": sorted(set(resistance_levels[-5:])),
            "support_levels": sorted(set(support_levels[-5:])),
            "current_range": {
                "upper": max(resistance_levels[-5:]),
                "lower": min(support_levels[-5:]),
            },
        }

    def _analyze_correlation(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between different features"""
        corr_matrix = df[["close", "volume", "rsi", "macd"]].corr()

        return {
            "price_volume_corr": corr_matrix.loc["close", "volume"],
            "price_rsi_corr": corr_matrix.loc["close", "rsi"],
            "price_macd_corr": corr_matrix.loc["close", "macd"],
        }

    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        avg_volume = df["volume"].rolling(20).mean().iloc[-1]
        current_volume = df["volume"].iloc[-1]

        return {
            "volume_trend": "increasing"
            if current_volume > avg_volume
            else "decreasing",
            "volume_strength": current_volume / avg_volume,
            "volume_consistency": df["volume"].std() / df["volume"].mean(),
        }

    def calculate_position_size(
        self,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss: float,
        leverage: float = 100.0,
    ) -> Dict:
        """Calculate optimal position size"""
        try:
            # Calculate risk amount
            risk_amount = account_balance * (risk_percentage / 100)

            # Calculate pip value and position size
            pip_value = 0.0001  # Adjust for JPY pairs
            pip_risk = abs(entry_price - stop_loss) / pip_value

            # Calculate position size in lots
            position_size = risk_amount / (pip_risk * 10)  # Standard lot = 100,000
            position_size = min(position_size, account_balance * leverage / entry_price)

            return {
                "position_size_lots": position_size,
                "position_size_units": position_size * 100000,
                "risk_amount": risk_amount,
                "margin_required": (position_size * 100000 * entry_price) / leverage,
            }

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise

    def generate_trading_summary(
        self, trades: List[Dict], market_analysis: Dict
    ) -> str:
        """Generate human-readable trading summary"""
        try:
            summary = f"""
Trading Summary - {self.current_time}
Generated by: {self.current_user}

Market Analysis:
---------------
Trend: {market_analysis["trend"]["trend_direction"]} ({market_analysis["trend"]["trend_strength"]})
Volatility: {market_analysis["volatility"]["volatility_regime"]}
Momentum: {market_analysis["momentum"]["rsi_condition"]} (RSI), {market_analysis["momentum"]["macd_signal"]} (MACD)

Recent Trades:
-------------
"""
            for trade in trades[-5:]:  # Show last 5 trades
                summary += f"""
Time: {trade["timestamp"]}
Type: {trade["type"]}
Profit/Loss: {trade["pnl"]:.2f}
Duration: {trade["duration"]}
"""

            return summary

        except Exception as e:
            logger.error(f"Error generating trading summary: {str(e)}")
            raise
