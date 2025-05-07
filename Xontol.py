import os
import time
import logging
from datetime import datetime, timezone
import threading
from typing import List, Dict, Optional
import telebot
from pathlib import Path
from config import TradingConfig
from data_processor import ForexDataProcessor
from model import DeepForexModel
from trader import ForexTrader
from utils import TradingUtils

# Setup logging
TradingUtils.setup_logging()
logger = logging.getLogger(__name__)


class ForexTradingBot:
    """Main class for forex trading bot"""

    def __init__(self):
        self.trader = ForexTrader()
        self.data_processor = ForexDataProcessor()
        self.model = DeepForexModel()
        self.utils = TradingUtils()
        self.telegram_bot = telebot.TeleBot(TradingConfig.TELEGRAM_TOKEN)
        self.is_running = False
        self.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        self.current_user = "iwan026"  # Current authenticated user

    def setup_telegram_handlers(self):
        """Setup Telegram bot command handlers"""

        @self.telegram_bot.message_handler(commands=["start"])
        def handle_start(message):
            if message.from_user.id not in TradingConfig.ALLOWED_USERS:
                self.telegram_bot.reply_to(
                    message, "You are not authorized to use this bot."
                )
                return

            welcome_msg = (
                "Welcome to Forex AI Trading Bot!\n\n"
                "Available commands:\n"
                "/predict SYMBOL TIMEFRAME - Get trading signal\n"
                "/chart SYMBOL TIMEFRAME - Get technical analysis chart\n"
                "/backtest SYMBOL TIMEFRAME DAYS - Run backtest\n"
                "/stats - View trading statistics\n"
                "/status - Check bot status\n"
                "/start_trading - Start automated trading\n"
                "/stop_trading - Stop automated trading"
            )
            self.telegram_bot.reply_to(message, welcome_msg)

        @self.telegram_bot.message_handler(commands=["predict"])
        def handle_predict(message):
            if not self._is_authorized(message):
                return

            try:
                args = message.text.split()[1:]
                if len(args) != 2:
                    self.telegram_bot.reply_to(
                        message, "Usage: /predict SYMBOL TIMEFRAME"
                    )
                    return

                symbol = args[0].upper()
                timeframe = args[1].upper()

                # Send processing message
                processing_msg = self.telegram_bot.reply_to(
                    message, f"Analyzing {symbol} {timeframe}..."
                )

                # Get prediction
                signal = self.trader.get_trading_signal(symbol, timeframe)
                if not signal:
                    self.telegram_bot.edit_message_text(
                        "Failed to get prediction.",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                    )
                    return

                # Format response
                emoji = (
                    "🟢"
                    if signal["signal"] == "BUY"
                    else "🔴"
                    if signal["signal"] == "SELL"
                    else "⚪"
                )
                response = (
                    f"{emoji} *Trading Signal*\n\n"
                    f"*Symbol:* {signal['symbol']}\n"
                    f"*Timeframe:* {signal['timeframe']}\n"
                    f"*Signal:* {signal['signal']}\n"
                    f"*Confidence:* {signal['confidence']:.2%}\n"
                    f"*Price:* {signal['price']:.5f}\n"
                    f"*Spread:* {signal['spread']:.5f}\n"
                    f"*Time:* {signal['timestamp']}\n\n"
                    f"_This is not financial advice._"
                )

                # Update message
                self.telegram_bot.edit_message_text(
                    response,
                    chat_id=message.chat.id,
                    message_id=processing_msg.message_id,
                    parse_mode="Markdown",
                )

                # Generate and send chart
                df = self.data_processor.get_mt5_historical_data(symbol, timeframe, 100)
                if df is not None:
                    chart_path = Path(
                        f"charts/{symbol}_{timeframe}_{int(time.time())}.png"
                    )
                    if self.utils.create_trading_chart(df, [signal], str(chart_path)):
                        with open(chart_path, "rb") as chart:
                            self.telegram_bot.send_photo(message.chat.id, chart)
                        os.remove(chart_path)

            except Exception as e:
                logger.error(f"Error in predict command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while processing your request."
                )

        @self.telegram_bot.message_handler(commands=["chart"])
        def handle_chart(message):
            if not self._is_authorized(message):
                return

            try:
                args = message.text.split()[1:]
                if len(args) != 2:
                    self.telegram_bot.reply_to(
                        message, "Usage: /chart SYMBOL TIMEFRAME"
                    )
                    return

                symbol = args[0].upper()
                timeframe = args[1].upper()

                processing_msg = self.telegram_bot.reply_to(
                    message, f"Generating chart for {symbol} {timeframe}..."
                )

                df = self.data_processor.get_mt5_historical_data(symbol, timeframe, 100)
                if df is not None:
                    chart_path = Path(
                        f"charts/{symbol}_{timeframe}_{int(time.time())}.png"
                    )
                    if self.utils.create_trading_chart(df, save_path=str(chart_path)):
                        with open(chart_path, "rb") as chart:
                            self.telegram_bot.send_photo(message.chat.id, chart)
                        os.remove(chart_path)
                        self.telegram_bot.delete_message(
                            chat_id=message.chat.id,
                            message_id=processing_msg.message_id,
                        )
                    else:
                        self.telegram_bot.edit_message_text(
                            "Failed to generate chart.",
                            chat_id=message.chat.id,
                            message_id=processing_msg.message_id,
                        )
                else:
                    self.telegram_bot.edit_message_text(
                        "Failed to fetch market data.",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                    )

            except Exception as e:
                logger.error(f"Error in chart command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while generating the chart."
                )

        @self.telegram_bot.message_handler(commands=["stats"])
        def handle_stats(message):
            if not self._is_authorized(message):
                return

            try:
                stats = self._get_trading_stats()
                if not stats:
                    self.telegram_bot.reply_to(
                        message, "No trading statistics available."
                    )
                    return

                response = (
                    "📊 *Trading Statistics*\n\n"
                    f"Total Trades: {stats['total_trades']}\n"
                    f"Win Rate: {stats['win_rate']:.2%}\n"
                    f"Net Profit: {self.utils.format_number(stats['net_profit'], include_sign=True)}\n"
                    f"Profit Factor: {stats['profit_factor']:.2f}\n"
                    f"Max Drawdown: {stats['max_drawdown']:.2%}\n"
                    f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
                )

                self.telegram_bot.reply_to(message, response, parse_mode="Markdown")

            except Exception as e:
                logger.error(f"Error in stats command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while fetching statistics."
                )

        @self.telegram_bot.message_handler(commands=["start_trading"])
        def handle_start_trading(message):
            if not self._is_authorized(message):
                return

            if self.is_running:
                self.telegram_bot.reply_to(message, "Trading bot is already running.")
                return

            self.is_running = True
            threading.Thread(target=self._trading_loop).start()
            self.telegram_bot.reply_to(message, "Trading bot started successfully.")

        @self.telegram_bot.message_handler(commands=["stop_trading"])
        def handle_stop_trading(message):
            if not self._is_authorized(message):
                return

            if not self.is_running:
                self.telegram_bot.reply_to(message, "Trading bot is not running.")
                return

            self.is_running = False
            self.telegram_bot.reply_to(message, "Trading bot stopped successfully.")

    def _is_authorized(self, message) -> bool:
        """Check if user is authorized"""
        if message.from_user.id not in TradingConfig.ALLOWED_USERS:
            self.telegram_bot.reply_to(
                message, "You are not authorized to use this bot."
            )
            return False
        return True

    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop...")

        while self.is_running:
            try:
                self.trader.run_trading_cycle(self.trading_symbols, "H1")
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying

        logger.info("Trading loop stopped.")

    def _get_trading_stats(self) -> Optional[Dict]:
        """Get current trading statistics"""
        try:
            results_file = (
                f"results/trading_results_{datetime.now().strftime('%Y%m%d')}.json"
            )
            return self.utils.load_trading_results(results_file)
        except Exception as e:
            logger.error(f"Error getting trading stats: {str(e)}")
            return None

    def run(self):
        """Run the trading bot"""
        try:
            logger.info("Starting Forex Trading Bot...")

            # Setup Telegram handlers
            self.setup_telegram_handlers()

            # Start Telegram bot
            logger.info("Starting Telegram bot...")
            self.telegram_bot.infinity_polling()

        except Exception as e:
            logger.error(f"Error running trading bot: {str(e)}")
        finally:
            self.is_running = False


def main():
    """Main entry point"""
    try:
        # Create bot instance
        bot = ForexTradingBot()

        # Run bot
        bot.run()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
