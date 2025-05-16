from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
import asyncio
from datetime import datetime

from config import BOT_TOKEN, TIMEFRAMES, ADMIN_ID
from src.models.forex_model import ForexModel
from src.utils.helpers import (
    setup_logging,
    format_prediction_message,
    validate_timeframe,
    get_available_symbols,
    get_available_timeframes,
)
from src.utils.resource_monitor import ResourceMonitor
from src.utils.cache_manager import CacheManager

logger = setup_logging(__name__)


class TelegramBot:
    def __init__(self):
        """Inisiasi bot telegram"""
        self.application = Application.builder().token(BOT_TOKEN).build()
        self.resource_monitor = ResourceMonitor()
        self.cache_manager = CacheManager()
        self.setup_handlers()

    def setup_handlers(self):
        """Setup perintah"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("prediksi", self.predict_command))
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(
            CommandHandler("symbols", self.list_symbols_command)
        )
        self.application.add_handler(
            CommandHandler("timeframes", self.list_timeframes_command)
        )
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    async def check_admin(self, update: Update) -> bool:
        """Memeriksa apakah pengguna adalah admin"""
        user_id = update.effective_user.id
        if user_id not in ADMIN_ID:
            await update.message.reply_text(
                "❌ Maaf, Anda tidak memiliki akses ke perintah ini."
            )
            return False
        return True

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Menampilkan status sistem"""
        if not await self.check_admin(update):
            return

        try:
            resources = self.resource_monitor.check_resources()

            status_message = (
                "📊 *Status Sistem*\n\n"
                f"🖥 *Penggunaan CPU:* {resources['cpu_usage']:.2f}%\n"
                f"💾 *Penggunaan Memori:* {resources['memory_usage']:.2f}GB\n"
                f"🕒 *Waktu:* {resources['timestamp']}\n\n"
                f"*Status:*\n"
                f"{'✅' if resources['cpu_ok'] else '⚠️'} CPU\n"
                f"{'✅' if resources['memory_ok'] else '⚠️'} Memori"
            )

            await update.message.reply_text(status_message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error di status_command: {e}")
            await update.message.reply_text(
                "❌ Terjadi kesalahan saat mengecek status sistem."
            )

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Membuat prediksi"""
        try:
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "❌ Format perintah salah.\n"
                    "Gunakan: /prediksi [PAIR] [TIMEFRAME]\n"
                    "Contoh: /prediksi EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            if not validate_timeframe(timeframe):
                available_timeframes = ", ".join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ Timeframe tidak valid.\n"
                    f"Gunakan salah satu dari: {available_timeframes}"
                )
                return

            # Cek resource sebelum memproses
            resources = self.resource_monitor.check_resources()
            if not resources["memory_ok"] or not resources["cpu_ok"]:
                await update.message.reply_text(
                    "⚠️ Sistem sedang sibuk. Mohon tunggu beberapa saat."
                )
                return

            # Cek cache
            cached_prediction = self.cache_manager.get(symbol, timeframe)
            if cached_prediction:
                result_message = format_prediction_message(cached_prediction)
                await update.message.reply_text(
                    result_message + "\n\n💾 *Dari cache*", parse_mode="Markdown"
                )
                return

            process_message = await update.message.reply_text(
                f"⏳ Memproses prediksi untuk {symbol} pada timeframe {timeframe}..."
            )

            # Buat prediksi dalam task terpisah
            model = ForexModel(symbol, timeframe)
            prediction = await asyncio.get_event_loop().run_in_executor(
                None, model.predict
            )

            if prediction is None:
                await process_message.edit_text(
                    f"❌ Gagal mendapatkan prediksi untuk {symbol} pada timeframe {timeframe}."
                )
                return

            # Simpan ke cache
            self.cache_manager.set(symbol, timeframe, prediction)

            result_message = format_prediction_message(prediction)
            await process_message.edit_text(result_message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error di predict_command: {e}")
            await update.message.reply_text(f"❌ Terjadi kesalahan: {str(e)}")

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Melatih model"""
        if not await self.check_admin(update):
            return

        try:
            args = context.args
            if len(args) != 2:
                await update.message.reply_text(
                    "❌ Format perintah salah.\n"
                    "Gunakan: /train [PAIR] [TIMEFRAME]\n"
                    "Contoh: /train EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            # Validasi timeframe
            if not validate_timeframe(timeframe):
                available_timeframes = ", ".join(TIMEFRAMES.keys())
                await update.message.reply_text(
                    f"❌ Timeframe tidak valid.\n"
                    f"Gunakan salah satu dari: {available_timeframes}"
                )
                return

            # Cek resource
            resources = self.resource_monitor.check_resources()
            if not resources["memory_ok"] or not resources["cpu_ok"]:
                await update.message.reply_text(
                    "⚠️ Sistem terlalu sibuk untuk melakukan training.\n"
                    "Mohon coba lagi nanti."
                )
                return

            process_message = await update.message.reply_text(
                f"⏳ Melatih model untuk {symbol} pada timeframe {timeframe}...\n"
                "Ini akan memakan waktu beberapa menit."
            )

            # Training dalam task terpisah
            model = ForexModel(symbol, timeframe)
            success = await asyncio.get_event_loop().run_in_executor(None, model.train)

            if success:
                # Bersihkan cache untuk pair ini
                self.cache_manager.clear()
                await process_message.edit_text(
                    "✅ Model berhasil dilatih dan disimpan."
                )
            else:
                await process_message.edit_text(
                    "❌ Gagal melatih model. Periksa log untuk informasi lebih lanjut."
                )

        except Exception as e:
            logger.error(f"Error di train_command: {e}")
            await update.message.reply_text(f"❌ Terjadi kesalahan: {str(e)}")

    def run(self):
        """Run bot"""
        logger.info("Starting bot...")
        self.application.run_polling()
