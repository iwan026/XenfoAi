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

from config import BOT_TOKEN, ADMIN_IDS, TIMEFRAMES, SYMBOLS, ModelConfig
from model import ForexModel

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
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(CommandHandler("predict", self.predict_command))

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
            "👋 Welcome to Forex Prediction Bot!\nUse /help to see available commands."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = (
            "🤖 *Bot Commands*\n\n"
            "*/predict [PAIR] [TIMEFRAME]* - Make prediction\n"
            "Example: `/predict EURUSD H1`\n\n"
            "*/train [PAIR] [TIMEFRAME]* - Train model (admin)\n\n"
            "Available pairs: " + ", ".join(SYMBOLS) + "\n"
            "Available timeframes: " + ", ".join(TIMEFRAMES.keys())
        )
        await update.message.reply_text(help_msg, parse_mode="Markdown")

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /train command"""
        if not await self.check_admin(update):
            return

        try:
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "Usage: /train [PAIR] [TIMEFRAME]\nExample: /train EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            if symbol not in SYMBOLS:
                await update.message.reply_text(
                    f"Invalid pair. Available: {', '.join(SYMBOLS)}"
                )
                return

            if timeframe not in TIMEFRAMES:
                await update.message.reply_text(
                    f"Invalid timeframe. Available: {', '.join(TIMEFRAMES.keys())}"
                )
                return

            # Initialize model
            model_key = f"{symbol}_{timeframe}"
            self.active_models[model_key] = ForexModel(symbol, timeframe)

            # Start training
            msg = await update.message.reply_text(
                f"⏳ Training {symbol} {timeframe} model..."
            )

            success = await asyncio.get_event_loop().run_in_executor(
                None, self.active_models[model_key].train
            )

            if success:
                await msg.edit_text(f"✅ Training completed for {symbol} {timeframe}!")
            else:
                await msg.edit_text(f"❌ Training failed for {symbol} {timeframe}")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
            logger.error(f"Train error: {e}")

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predict command"""
        try:
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "Usage: /predict [PAIR] [TIMEFRAME]\nExample: /predict EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            if symbol not in SYMBOLS:
                await update.message.reply_text(
                    f"Invalid pair. Available: {', '.join(SYMBOLS)}"
                )
                return

            if timeframe not in TIMEFRAMES:
                await update.message.reply_text(
                    f"Invalid timeframe. Available: {', '.join(TIMEFRAMES.keys())}"
                )
                return

            # Get or create model
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.active_models:
                self.active_models[model_key] = ForexModel(symbol, timeframe)

            # Get latest data
            df = self.active_models[model_key]._load_dataset()
            prediction = self.active_models[model_key].predict(df)

            if prediction is not None:
                signal = "BUY" if prediction > 0.5 else "SELL"
                confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 range

                await update.message.reply_text(
                    f"📊 {symbol} {timeframe}\n"
                    f"Signal: {signal}\n"
                    f"Confidence: {confidence:.2%}\n"
                    f"Model Output: {prediction:.4f}"
                )
            else:
                await update.message.reply_text("❌ Prediction failed")

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
            logger.error(f"Predict error: {e}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        uptime = datetime.utcnow() - self.start_time
        await update.message.reply_text(
            f"⏱ Uptime: {uptime}\n📊 Active models: {len(self.active_models)}"
        )

    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /models command"""
        if not self.active_models:
            await update.message.reply_text("No active models")
            return

        msg = "Active models:\n"
        for model in self.active_models:
            msg += f"- {model}\n"

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
