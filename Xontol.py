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
from model import ForexSignalModel
from utils import SignalUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ForexSignalBot:
    """Main class for forex signal generation bot"""

    def __init__(self):
        self.data_processor = ForexDataProcessor()
        self.model = ForexSignalModel()
        self.utils = SignalUtils()
        self.telegram_bot = telebot.TeleBot(TradingConfig.TELEGRAM_TOKEN)
        self.trading_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

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
                "Welcome to Forex Signal Bot!\n\n"
                "Available commands:\n"
                "/predict SYMBOL TIMEFRAME - Get trading signal\n"
                "/train SYMBOL TIMEFRAME - Train model for symbol\n"
                "/backtest SYMBOL TIMEFRAME DAYS - Run backtest\n"
                "/stats - View model statistics\n"
                "/add_symbol SYMBOL - Add symbol to watch list\n"
                "/remove_symbol SYMBOL - Remove symbol from watch list\n"
                "/list_symbols - View watched symbols\n"
                "/help - Show this message"
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
                signal = self._get_trading_signal(symbol, timeframe)
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

            except Exception as e:
                logger.error(f"Error in predict command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while processing your request."
                )

        @self.telegram_bot.message_handler(commands=["train"])
        def handle_train(message):
            if not self._is_authorized(message):
                return

            try:
                args = message.text.split()[1:]
                if len(args) != 2:
                    self.telegram_bot.reply_to(
                        message, "Usage: /train SYMBOL TIMEFRAME"
                    )
                    return

                symbol = args[0].upper()
                timeframe = args[1].upper()

                processing_msg = self.telegram_bot.reply_to(
                    message, f"Starting training for {symbol} {timeframe}..."
                )

                success = self._train_model(symbol, timeframe)

                if success:
                    self.telegram_bot.edit_message_text(
                        f"Training completed successfully for {symbol} {timeframe}",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                    )
                else:
                    self.telegram_bot.edit_message_text(
                        f"Training failed for {symbol} {timeframe}",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                    )

            except Exception as e:
                logger.error(f"Error in train command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while training the model."
                )

        @self.telegram_bot.message_handler(commands=["backtest"])
        def handle_backtest(message):
            if not self._is_authorized(message):
                return

            try:
                args = message.text.split()[1:]
                if len(args) != 3:
                    self.telegram_bot.reply_to(
                        message, "Usage: /backtest SYMBOL TIMEFRAME DAYS"
                    )
                    return

                symbol = args[0].upper()
                timeframe = args[1].upper()
                days = int(args[2])

                processing_msg = self.telegram_bot.reply_to(
                    message, f"Running backtest for {symbol} {timeframe}..."
                )

                results = self._run_backtest(symbol, timeframe, days)
                if not results:
                    self.telegram_bot.edit_message_text(
                        "Backtest failed.",
                        chat_id=message.chat.id,
                        message_id=processing_msg.message_id,
                    )
                    return

                response = (
                    f"*Backtest Results*\n\n"
                    f"Symbol: {symbol}\n"
                    f"Timeframe: {timeframe}\n"
                    f"Period: {days} days\n\n"
                    f"Total Signals: {results['total_signals']}\n"
                    f"Accuracy: {results['accuracy']:.2%}\n"
                    f"Win Rate: {results['win_rate']:.2%}\n"
                )

                self.telegram_bot.edit_message_text(
                    response,
                    chat_id=message.chat.id,
                    message_id=processing_msg.message_id,
                    parse_mode="Markdown",
                )

            except Exception as e:
                logger.error(f"Error in backtest command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while running backtest."
                )

        @self.telegram_bot.message_handler(commands=["add_symbol"])
        def handle_add_symbol(message):
            if not self._is_authorized(message):
                return

            try:
                args = message.text.split()[1:]
                if len(args) != 1:
                    self.telegram_bot.reply_to(message, "Usage: /add_symbol SYMBOL")
                    return

                symbol = args[0].upper()
                if symbol not in self.trading_symbols:
                    self.trading_symbols.append(symbol)
                    self.telegram_bot.reply_to(
                        message, f"Added {symbol} to watch list."
                    )
                else:
                    self.telegram_bot.reply_to(
                        message, f"{symbol} is already in watch list."
                    )

            except Exception as e:
                logger.error(f"Error in add_symbol command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while adding symbol."
                )

        @self.telegram_bot.message_handler(commands=["remove_symbol"])
        def handle_remove_symbol(message):
            if not self._is_authorized(message):
                return

            try:
                args = message.text.split()[1:]
                if len(args) != 1:
                    self.telegram_bot.reply_to(message, "Usage: /remove_symbol SYMBOL")
                    return

                symbol = args[0].upper()
                if symbol in self.trading_symbols:
                    self.trading_symbols.remove(symbol)
                    self.telegram_bot.reply_to(
                        message, f"Removed {symbol} from watch list."
                    )
                else:
                    self.telegram_bot.reply_to(
                        message, f"{symbol} is not in watch list."
                    )

            except Exception as e:
                logger.error(f"Error in remove_symbol command: {str(e)}")
                self.telegram_bot.reply_to(
                    message, "An error occurred while removing symbol."
                )

        @self.telegram_bot.message_handler(commands=["list_symbols"])
        def handle_list_symbols(message):
            if not self._is_authorized(message):
                return

            symbols_list = "\n".join(self.trading_symbols)
            response = f"*Watched Symbols*\n\n{symbols_list}"
            self.telegram_bot.reply_to(message, response, parse_mode="Markdown")

    def _is_authorized(self, message) -> bool:
        """Check if user is authorized"""
        if message.from_user.id not in TradingConfig.ALLOWED_USERS:
            self.telegram_bot.reply_to(
                message, "You are not authorized to use this bot."
            )
            return False
        return True

    def _get_trading_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Generate trading signal for a symbol"""
        try:
            # Get market data
            df = self.data_processor.get_market_data(symbol, timeframe)
            if df is None:
                return None

            # Generate signal
            signal = self.model.generate_signal(df, symbol, timeframe)
            if signal is None:
                return None

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _train_model(self, symbol: str, timeframe: str) -> bool:
        """Train model for a specific symbol and timeframe"""
        try:
            # Get training data
            df = self.data_processor.get_training_data(symbol, timeframe)
            if df is None:
                return False

            # Train model
            success = self.model.train(df, symbol, timeframe)
            return success

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def _run_backtest(self, symbol: str, timeframe: str, days: int) -> Optional[Dict]:
        """Run backtest for a symbol"""
        try:
            # Get backtest data
            df = self.data_processor.get_backtest_data(symbol, timeframe, days)
            if df is None:
                return None

            # Run backtest
            results = self.model.backtest(df, symbol, timeframe)
            return results

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return None

    def run(self):
        """Run the signal bot"""
        try:
            logger.info("Starting Forex Signal Bot...")

            # Setup Telegram handlers
            self.setup_telegram_handlers()

            # Start Telegram bot
            logger.info("Starting Telegram bot...")
            self.telegram_bot.infinity_polling()

        except Exception as e:
            logger.error(f"Error running signal bot: {str(e)}")
        finally:
            logger.info("Signal bot stopped.")


def main():
    """Main entry point"""
    try:
        # Create bot instance
        bot = ForexSignalBot()

        # Run bot
        bot.run()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
