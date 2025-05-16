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

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_message = (
            "🤖 *Selamat datang di Bot Prediksi Forex!*\n\n"
            "*Perintah yang tersedia:*\n\n"
            "📊 */prediksi* [PAIR] [TIMEFRAME]\n"
            "   Contoh: /prediksi EURUSD H1\n\n"
            "🔄 */train* [PAIR] [TIMEFRAME]\n"
            "   Contoh: /train EURUSD H1\n\n"
            "📋 */symbols*\n"
            "   Menampilkan daftar pair yang tersedia\n\n"
            "⏱ */timeframes* [PAIR]\n"
            "   Menampilkan timeframe yang tersedia untuk pair tertentu\n\n"
            "❓ */help*\n"
            "   Menampilkan bantuan lengkap\n\n"
            "💡 *Tips:* Gunakan /symbols untuk melihat pair yang tersedia"
        )
        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_message = (
            "📚 *Panduan Penggunaan Bot Prediksi Forex*\n\n"
            "*1. Prediksi Forex*\n"
            "   Perintah: */prediksi* [PAIR] [TIMEFRAME]\n"
            "   Contoh: `/prediksi EURUSD H1`\n\n"
            "   Timeframe yang tersedia:\n"
            "   M1, M5, M15, H1, H4, D1, W1\n\n"
            "*2. Melatih Model*\n"
            "   Perintah: */train* [PAIR] [TIMEFRAME]\n"
            "   Contoh: `/train EURUSD H1`\n\n"
            "*3. Melihat Pair Tersedia*\n"
            "   Perintah: */symbols*\n\n"
            "*4. Melihat Timeframe Tersedia*\n"
            "   Perintah: */timeframes* [PAIR]\n"
            "   Contoh: `/timeframes EURUSD`\n\n"
            "❗ *Catatan:*\n"
            "- Pastikan pair dan timeframe tersedia\n"
            "- Proses pelatihan membutuhkan waktu beberapa menit\n"
            "- Data real-time diambil dari MetaTrader 5"
        )
        await update.message.reply_text(help_message, parse_mode="Markdown")

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

    async def list_symbols_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        try:
            available_symbols = get_available_symbols()
            if not available_symbols:
                await update.message.reply_text("❌ Tidak ada pair yang tersedia.")
                return

            message = "📊 *Pair yang tersedia:*\n\n"
            for symbol in available_symbols:
                message += f"• {symbol}\n"

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error di list_symbols_command: {str(e)}")
            await update.message.reply_text(
                "❌ Terjadi kesalahan saat mengambil daftar pair."
            )

    async def list_timeframes_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        try:
            args = context.args
            if len(args) != 1:
                await update.message.reply_text(
                    "❌ Format perintah salah.\n"
                    "Gunakan: /timeframes [PAIR]\n"
                    "Contoh: /timeframes EURUSD"
                )
                return

            symbol = args[0].upper()
            available_timeframes = get_available_timeframes(symbol)

            if not available_timeframes:
                await update.message.reply_text(
                    f"❌ Tidak ada timeframe tersedia untuk {symbol}."
                )
                return

            message = f"⏱ *Timeframe tersedia untuk {symbol}:*\n\n"
            for tf in available_timeframes:
                message += f"• {tf}\n"

            await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error di list_timeframes_command: {e}")
            await update.message.reply_text(f"❌ Terjadi kesalahan: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message_text = update.message.text.lower()

        if "prediksi" in message_text:
            words = message_text.split()
            if len(words) >= 3:
                for i, word in enumerate(words):
                    if word == "prediksi" and i + 2 < len(words):
                        context.args = [words[i + 1], words[i + 2]]
                        await self.predict_command(update, context)
                        return

            await update.message.reply_text(
                "❌ Format pesan salah.\n"
                "Gunakan: /prediksi [PAIR] [TIMEFRAME]\n"
                "Contoh: /prediksi EURUSD H1"
            )
        else:
            await update.message.reply_text(
                "❓ Saya tidak mengerti pesan Anda.\n"
                "Gunakan /help untuk melihat perintah yang tersedia."
            )

    def run(self):
        """Run bot"""
        logger.info("Starting bot...")
        self.application.run_polling()
