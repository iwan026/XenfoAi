import logging
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import BOT_TOKEN, ADMIN_IDS, TIMEFRAMES, SYMBOLS
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
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(CommandHandler("predict", self.predict_command))

        # Admin commands
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("models", self.models_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))

        # Error handler
        self.application.add_error_handler(self.error_handler)

        logger.info("Command handlers setup completed")

    async def check_admin(self, update: Update) -> bool:
        """Check if user is admin"""
        user_id = update.effective_user.id
        if user_id not in ADMIN_IDS:
            await update.message.reply_text(
                "❌ Anda tidak memiliki izin untuk menggunakan command ini."
            )
            logger.warning(f"Unauthorized access attempt by user {user_id}")
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        welcome_msg = (
            f"👋 Selamat datang di XenfoAI Bot, {user.first_name}!\n\n"
            "Bot ini menggunakan model Hybrid (XGBoost + Deep Learning) "
            "untuk memprediksi pergerakan forex.\n\n"
            "Gunakan /help untuk melihat daftar perintah yang tersedia."
        )
        await update.message.reply_text(welcome_msg)
        logger.info(f"New user started bot: {user.id} ({user.first_name})")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = (
            "🤖 *XenfoAI Bot Commands*\n\n"
            "*Trading Commands:*\n"
            "/predict PAIR TIMEFRAME - Buat prediksi\n"
            "Contoh: `/predict EURUSD H1`\n\n"
            "*Admin Commands:*\n"
            "/train PAIR TIMEFRAME - Train model baru\n"
            "/status - Cek status bot\n"
            "/models - Lihat daftar model aktif\n"
            "/clear - Hapus model dari memori\n\n"
            "*Pairs yang tersedia:*\n"
            f"{', '.join(SYMBOLS)}\n\n"
            "*Timeframes yang tersedia:*\n"
            f"{', '.join(TIMEFRAMES.keys())}"
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

            # Validate symbol
            if symbol not in SYMBOLS:
                available_symbols = ", ".join(SYMBOLS)
                await update.message.reply_text(
                    f"❌ Pair tidak valid.\nGunakan: {available_symbols}"
                )
                return

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
                model_key = f"{symbol}_{timeframe}"
                self.active_models[model_key] = model

                await process_msg.edit_text(
                    f"✅ Training selesai untuk {symbol} {timeframe}!\n"
                    "Model siap digunakan untuk prediksi."
                )
                logger.info(f"Training completed for {symbol} {timeframe}")
            else:
                await process_msg.edit_text(
                    f"❌ Training gagal untuk {symbol} {timeframe}.\n"
                    "Silakan cek log untuk detail error."
                )
                logger.error(f"Training failed for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error in train_command: {e}")
            await update.message.reply_text(
                f"❌ Terjadi error: {str(e)}\nSilakan hubungi admin untuk bantuan."
            )

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

            # Validate symbol and timeframe
            if symbol not in SYMBOLS:
                await update.message.reply_text(
                    f"❌ Pair tidak valid.\nGunakan: {', '.join(SYMBOLS)}"
                )
                return

            if timeframe not in TIMEFRAMES:
                await update.message.reply_text(
                    f"❌ Timeframe tidak valid.\nGunakan: {', '.join(TIMEFRAMES.keys())}"
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

                # Get emoji for technical indicators
                rsi_emoji = {"Overbought": "⚠️", "Oversold": "💫", "Neutral": "➖"}[
                    prediction["signals"]["rsi"]
                ]

                macd_emoji = "📈" if prediction["signals"]["macd"] == "Buy" else "📉"
                ma_emoji = "📈" if prediction["signals"]["ma"] == "Buy" else "📉"

                msg = (
                    f"📊 *Prediksi untuk {symbol} {timeframe}*\n\n"
                    f"Signal: {signal}\n"
                    f"Confidence: {confidence:.2f}%\n\n"
                    "*Technical Indicators:*\n"
                    f"RSI: {rsi_emoji} {prediction['signals']['rsi']}\n"
                    f"MACD: {macd_emoji} {prediction['signals']['macd']}\n"
                    f"MA: {ma_emoji} {prediction['signals']['ma']}\n\n"
                    f"🕒 {prediction['timestamp']} UTC"
                )

                await update.message.reply_text(msg, parse_mode="Markdown")
                logger.info(f"Prediction made for {symbol} {timeframe}")
            else:
                await update.message.reply_text(
                    "❌ Gagal membuat prediksi.\nSilakan coba lagi nanti."
                )
                logger.error(f"Prediction failed for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error in predict_command: {e}")
            await update.message.reply_text(
                f"❌ Terjadi error: {str(e)}\nSilakan hubungi admin untuk bantuan."
            )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not await self.check_admin(update):
            return

        try:
            uptime = datetime.utcnow() - self.start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)

            status_msg = (
                "📈 *XenfoAI Bot Status*\n\n"
                f"🕒 Current Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⏱ Uptime: {hours}h {minutes}m {seconds}s\n"
                f"📊 Active Models: {len(self.active_models)}\n\n"
                "*System Info:*\n"
                f"🤖 Bot Version: 1.0.0\n"
                f"👥 Admin Count: {len(ADMIN_IDS)}\n"
                f"💾 Supported Pairs: {len(SYMBOLS)}\n"
                f"⚙️ Timeframes: {len(TIMEFRAMES)}"
            )

            await update.message.reply_text(status_msg, parse_mode="Markdown")
            logger.info("Status command executed")

        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text(
                "❌ Terjadi error saat mengambil status bot."
            )

    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /models command"""
        if not await self.check_admin(update):
            return

        try:
            if not self.active_models:
                await update.message.reply_text("📊 Tidak ada model aktif saat ini.")
                return

            models_msg = "📊 *Model Aktif:*\n\n"
            for key in sorted(self.active_models.keys()):
                symbol, timeframe = key.split("_")
                models_msg += f"• {symbol} {timeframe}\n"

            await update.message.reply_text(models_msg, parse_mode="Markdown")
            logger.info("Models list displayed")

        except Exception as e:
            logger.error(f"Error in models_command: {e}")
            await update.message.reply_text(
                "❌ Terjadi error saat mengambil daftar model."
            )

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command - Clear models from memory"""
        if not await self.check_admin(update):
            return

        try:
            self.active_models.clear()
            await update.message.reply_text("✅ Semua model telah dihapus dari memori.")
            logger.info("All models cleared from memory")

        except Exception as e:
            logger.error(f"Error in clear_command: {e}")
            await update.message.reply_text("❌ Terjadi error saat menghapus model.")

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
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
