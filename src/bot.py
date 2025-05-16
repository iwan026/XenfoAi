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
            "*/predict [PAIR]* - Make prediction for H1 timeframe\n"
            "Example: `/predict EURUSD`\n\n"
            "Available pairs: " + ", ".join(SYMBOLS)
        )
        await update.message.reply_text(help_msg, parse_mode="Markdown")

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predict command with realtime data"""
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
            model_key = symbol
            if model_key not in self.active_models:
                self.active_models[model_key] = ForexModel(symbol)

            # Start processing message
            processing_msg = await update.message.reply_text(
                f"⏳ Getting realtime data for {symbol} H1..."
            )

            try:
                # Get prediction with chart
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.active_models[model_key].predict,
                    True,  # use_realtime=True
                )

                if result is not None:
                    short_term_signal = (
                        "🟢 BUY" if result["short_term"] > 0.5 else "🔴 SELL"
                    )
                    long_term_signal = (
                        "🟢 BUY" if result["long_term"] > 0.5 else "🔴 SELL"
                    )

                    # Prepare indicators message
                    indicators = result["indicators"]
                    indicators_msg = (
                        "*Technical Indicators:*\n"
                        f"• EMA 9: {indicators['ema_9']:.4f}\n"
                        f"• EMA 21: {indicators['ema_21']:.4f}\n"
                        f"• EMA 50: {indicators['ema_50']:.4f}\n"
                        f"• RSI: {indicators['rsi']:.2f}\n"
                        f"• MACD: {indicators['macd']:.4f} (Signal: {indicators['macd_signal']:.4f})\n"
                        f"• Stoch: K={indicators['stoch_k']:.2f}, D={indicators['stoch_d']:.2f}\n"
                    )

                    # Send chart image
                    with open(result["chart_path"], "rb") as chart_file:
                        await update.message.reply_photo(
                            photo=chart_file,
                            caption=(
                                f"📊 *{symbol} H1*\n\n"
                                f"*Short Term (4h):* {short_term_signal} ({result['short_term']:.2%})\n"
                                f"*Long Term (24h):* {long_term_signal} ({result['long_term']:.2%})\n\n"
                                f"{indicators_msg}\n"
                                f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                            ),
                            parse_mode="Markdown",
                        )

                    # Delete processing message
                    await processing_msg.delete()
                else:
                    await processing_msg.edit_text(
                        "❌ Prediction failed - not enough data or model error"
                    )

            except Exception as e:
                await processing_msg.edit_text(f"❌ Error during prediction: {str(e)}")
                logger.error(f"Prediction error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Command error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Command error: {str(e)}")

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
