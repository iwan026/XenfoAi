import logging
from datetime import datetime
from typing import List, Optional
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import BOT_TOKEN, ADMIN_IDS, TIMEFRAMES
from src.model import ForexModel

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self):
        """Initialize bot dan commands"""
        self.application = Application.builder().token(BOT_TOKEN).build()
        self._setup_handlers()
        self.active_models: dict = {}  # Tracking active models

    def _setup_handlers(self):
        """Setup command handlers"""
        # Admin commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(CommandHandler("predict", self.predict_command))
        self.application.add_handler(CommandHandler("status", self.status_command))

        # Error handler
        self.application.add_error_handler(self.error_handler)

    async def check_admin(self, update: Update) -> bool:
        """Check if user is admin"""
        user_id = update.effective_user.id
        if user_id not in ADMIN_IDS:
            await update.message.reply_text(
                "❌ Anda tidak memiliki izin untuk menggunakan command ini."
            )
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = (
            "👋 Selamat datang di XenfoAI Bot!\n\n"
            "Bot ini menggunakan model Hybrid (XGBoost + Deep Learning) "
            "untuk memprediksi pergerakan forex.\n\n"
            "Gunakan /help untuk melihat daftar perintah yang tersedia."
        )
        await update.message.reply_text(welcome_msg)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = (
            "🤖 *XenfoAI Bot Commands*\n\n"
            "*Admin Commands:*\n"
            "/train PAIR TIMEFRAME - Train model baru\n"
            "Contoh: `/train EURUSD H1`\n\n"
            "/predict PAIR TIMEFRAME - Buat prediksi\n"
            "Contoh: `/predict EURUSD H1`\n\n"
            "/status - Cek status bot dan model\n\n"
            "*Timeframes yang tersedia:*\n"
            "M1, M5, M15, H1, H4, D1, W1\n\n"
            "*Format PAIR:*\n"
            "EURUSD, GBPUSD, USDJPY, XAUUSD, dll."
        )
        await update.message.reply_text(help_msg, parse_mode="Markdown")

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /train command"""
        if not await self.check_admin(update):
            return

        try:
            # Validate arguments
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "❌ Format salah.\n"
                    "Gunakan: /train [PAIR] [TIMEFRAME]\n"
                    "Contoh: /train EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            # Validate timeframe
            if timeframe not in TIMEFRAMES:
                available_timeframes = ", ".join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ Timeframe tidak valid.\nGunakan: {available_timeframes}"
                )
                return

            # Send processing message
            process_msg = await update.message.reply_text(
                f"⏳ Memulai training untuk {symbol} {timeframe}...\n"
                "Proses ini akan memakan waktu beberapa menit."
            )

            # Create and train model in separate thread
            model = ForexModel(symbol, timeframe)
            success = await asyncio.get_event_loop().run_in_executor(None, model.train)

            if success:
                # Save model to active models
                self.active_models[f"{symbol}_{timeframe}"] = model

                await process_msg.edit_text(
                    f"✅ Training selesai untuk {symbol} {timeframe}!\n"
                    "Model siap digunakan untuk prediksi."
                )
            else:
                await process_msg.edit_text(
                    f"❌ Training gagal untuk {symbol} {timeframe}.\n"
                    "Silakan cek log untuk detail error."
                )

        except Exception as e:
            logger.error(f"Error in train_command: {e}")
            await update.message.reply_text(f"❌ Terjadi error: {str(e)}")

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predict command"""
        try:
            # Validate arguments
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "❌ Format salah.\n"
                    "Gunakan: /predict [PAIR] [TIMEFRAME]\n"
                    "Contoh: /predict EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            # Validate timeframe
            if timeframe not in TIMEFRAMES:
                available_timeframes = ", ".join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ Timeframe tidak valid.\nGunakan: {available_timeframes}"
                )
                return

            # Get or create model
            model_key = f"{symbol}_{timeframe}"
            if model_key not in self.active_models:
                await update.message.reply_text(
                    f"⏳ Model untuk {symbol} {timeframe} belum ada.\n"
                    "Membuat model baru..."
                )
                self.active_models[model_key] = ForexModel(symbol, timeframe)

            # Make prediction
            prediction = await asyncio.get_event_loop().run_in_executor(
                None, self.active_models[model_key].predict
            )

            if prediction:
                # Format prediction message
                signal = "🟢 BELI" if prediction["prediction"] > 0.5 else "🔴 JUAL"
                confidence = prediction["confidence"] * 100

                msg = (
                    f"📊 *Prediksi untuk {symbol} {timeframe}*\n\n"
                    f"Signal: {signal}\n"
                    f"Confidence: {confidence:.2f}%\n\n"
                    "*Technical Indicators:*\n"
                    f"RSI: {prediction['signals']['rsi']}\n"
                    f"MACD: {prediction['signals']['macd']}\n"
                    f"MA: {prediction['signals']['ma']}\n\n"
                    f"🕒 {prediction['timestamp']} UTC"
                )

                await update.message.reply_text(msg, parse_mode="Markdown")
            else:
                await update.message.reply_text(
                    "❌ Gagal membuat prediksi.\nSilakan coba lagi nanti."
                )

        except Exception as e:
            logger.error(f"Error in predict_command: {e}")
            await update.message.reply_text(f"❌ Terjadi error: {str(e)}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not await self.check_admin(update):
            return

        try:
            active_models = len(self.active_models)
            current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            status_msg = (
                "📈 *XenfoAI Bot Status*\n\n"
                f"🕒 Current Time (UTC): {current_time}\n"
                f"📊 Active Models: {active_models}\n\n"
                "*Active Model List:*\n"
            )

            if self.active_models:
                for key in self.active_models:
                    symbol, timeframe = key.split("_")
                    status_msg += f"- {symbol} {timeframe}\n"
            else:
                status_msg += "No active models\n"

            await update.message.reply_text(status_msg, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text(f"❌ Terjadi error: {str(e)}")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        if update:
            await update.message.reply_text(
                "❌ Terjadi error internal.\n"
                "Silakan coba lagi nanti atau hubungi admin."
            )

    def run(self):
        """Run the bot"""
        logger.info("Starting bot...")
        self.application.run_polling()
