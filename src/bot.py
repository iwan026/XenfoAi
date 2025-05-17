import logging
from datetime import datetime
from typing import Dict, Optional
import asyncio
import pandas as pd
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import BOT_TOKEN, ADMIN_IDS, SYMBOLS
from src.model import ForexModel

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self):
        """Initialize bot and commands"""
        self.application = Application.builder().token(BOT_TOKEN).build()
        self.active_models: Dict[str, ForexModel] = {}
        self.start_time = datetime.utcnow()
        self._setup_handlers()
        logger.info("TelegramBot initialized")

    def _setup_handlers(self):
        """Setup command handlers"""
        # Basic commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))

        # Trading commands
        self.application.add_handler(CommandHandler("predict", self.predict_command))
        self.application.add_handler(CommandHandler("train", self.train_command))

        # Admin commands
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("models", self.models_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))

        # Error handler
        self.application.add_error_handler(self.error_handler)

    async def check_admin(self, update: Update) -> bool:
        """Check if user is admin"""
        user_id = update.effective_user.id
        if user_id not in ADMIN_IDS:
            await update.message.reply_text("❌ Unauthorized access.")
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "👋 Welcome to Forex Prediction Bot!\n"
            "Available commands:\n"
            "/predict [PAIR] - Get prediction\n"
            "/train [PAIR] - Train model\n"
            "/status - Bot status"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = (
            "🤖 *Bot Commands*\n\n"
            "*/predict [PAIR]* - Predict next candle\n"
            "Example: `/predict EURUSD`\n\n"
            "*/train [PAIR]* - Train model\n"
            "Example: `/train EURUSD`\n\n"
            "Available pairs: " + ", ".join(SYMBOLS)
        )
        await update.message.reply_text(help_msg, parse_mode="Markdown")

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predict command"""
        try:
            args = context.args
            if len(args) != 1:
                await update.message.reply_text(
                    "Usage: /predict [PAIR]\nExample: /predict EURUSD"
                )
                return

            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(
                    f"Invalid pair. Available: {', '.join(SYMBOLS)}"
                )
                return

            # Get or create model
            if symbol not in self.active_models:
                self.active_models[symbol] = ForexModel(symbol)

            processing_msg = await update.message.reply_text(
                f"⏳ Predicting {symbol}..."
            )

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.active_models[symbol].predict,
                    True,  # use_realtime
                )

                if result:
                    direction = "🟢 BUY" if result["direction"] > 0.5 else "🔴 SELL"
                    confidence = max(result["direction"], 1 - result["direction"])

                    with open(result["chart_path"], "rb") as chart_file:
                        await update.message.reply_photo(
                            photo=chart_file,
                            caption=(
                                f"📊 *{symbol} Prediction*\n\n"
                                f"*Signal:* {direction}\n"
                                f"*Confidence:* {confidence:.1%}\n"
                                f"*Current Price:* {result['current_close']:.5f}\n"
                                f"*Predicted Next Close:* {result['next_close']:.5f}\n\n"
                                f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                            ),
                            parse_mode="Markdown",
                        )
                else:
                    await update.message.reply_text("❌ Prediction failed")

                await processing_msg.delete()

            except Exception as e:
                await processing_msg.edit_text(f"❌ Prediction error: {str(e)}")
                logger.error(f"Prediction error: {e}")

        except Exception as e:
            logger.error(f"Command error: {e}")
            await update.message.reply_text(f"❌ Command error: {str(e)}")

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /train command"""
        try:
            args = context.args
            if len(args) != 1:
                await update.message.reply_text(
                    "Usage: /train [PAIR]\nExample: /train EURUSD"
                )
                return

            symbol = args[0].upper()
            if symbol not in SYMBOLS:
                await update.message.reply_text(
                    f"Invalid pair. Available: {', '.join(SYMBOLS)}"
                )
                return

            processing_msg = await update.message.reply_text(
                f"⏳ Training {symbol} model..."
            )

            # Get or create model
            if symbol not in self.active_models:
                self.active_models[symbol] = ForexModel(symbol)

            # Train model
            success = await asyncio.get_event_loop().run_in_executor(
                None, self.active_models[symbol].train
            )

            if success:
                await processing_msg.edit_text(f"✅ Training completed for {symbol}")
            else:
                await processing_msg.edit_text(f"❌ Training failed for {symbol}")

        except Exception as e:
            logger.error(f"Training error: {e}")
            await update.message.reply_text(f"❌ Training error: {str(e)}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        uptime = datetime.utcnow() - self.start_time
        await update.message.reply_text(
            f"⏱ Uptime: {uptime}\n"
            f"📊 Active models: {len(self.active_models)}\n"
            f"🔄 Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /models command"""
        if not await self.check_admin(update):
            return

        if not self.active_models:
            await update.message.reply_text("No active models")
            return

        msg = "📋 Active models:\n"
        msg += "\n".join([f"- {symbol}" for symbol in self.active_models.keys()])
        await update.message.reply_text(msg)

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        if not await self.check_admin(update):
            return

        self.active_models.clear()
        await update.message.reply_text("✅ All models cleared from memory")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Error: {context.error}")
        if update:
            await update.message.reply_text("❌ An error occurred")

    def run(self):
        """Run the bot"""
        logger.info("Starting bot...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
