import os
import logging
import config
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from core.system import XenfoAi

logger = logging.getLogger(__name__)


class TelegramBot:
    """Kelas untuk handel operasi telegram"""

    def __init__(self):
        self.token = config.BOT_TOKEN
        self.admin = config.ADMIN_ID
        self.application = ApplicationBuilder().token(self.token).build()
        self.dataset_dir = config.DATASET_DIR
        self.system = XenfoAi
        self.timeframes = config.TIMEFRAMES
        self.setup_handlers()

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("prediksi", self.prediksi_command))
        self.application.add_handler(CommandHandler("train", self.train_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & -filters.COMMAND, self.handle_message)
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT):
        welcome_msg = (
            "*Selamat datang di Xenfo Ai!*\n\n"
            "Perintah yang tersedia:\n"
            "/prediksi [PAIR] [TIMEFRAME] - Melakukan prediksi untuk pair dan timeframe tertentu\n"
            "Contoh: /prediksi EURUSD H1\n\n"
            "/train [PAIR] [TIMEFRAME] - Melatih model dengan data dari file CSV\n"
            "Contoh: /train EURUSD H1\n\n"
            "/help - Menampilkan bantuan"
        )
        await update.message.reply_text(welcome_msg, parse_mode="Markdown")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT):
        help_msg = (
            "*Cara Menggunakan Bot Prediksi Forex:*\n\n"
            "*1. Prediksi Forex:*\n"
            "   /prediksi [PAIR] [TIMEFRAME]\n"
            "   Contoh: /prediksi EURUSD H1\n\n"
            "   Timeframe yang tersedia: M1, M5, M15, M30, H1, H4, D1\n\n"
            "*2. Melatih Model:*\n"
            "   /train [PAIR] [TIMEFRAME]\n"
            "   Contoh: /train EURUSD H1\n\n"
            "Pastikan MetaTrader 5 terhubung untuk mendapatkan data real-time."
        )
        await update.message.reply_text(help_msg, parse_mode="Markdown")

    async def prediksi_command(self, update: Update, context: ContextTypes.DEFAULT):
        try:
            args = context.args

            if len(args) != 2:
                await update.message.reply_text(
                    "Format perintah salah. Gunakan:\n"
                    "/prediksi [PAIR] [TIMEFRAME]\n"
                    "Contoh: /prediksi EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()

            if timeframe not in self.timeframes:
                await update.message.reply_text(
                    f"Timeframe tidak valid. Gunakan salah satu dari: {', '.join(self.timeframes.keys())}"
                )
                return

            proses_msg = await update.message.reply_text(
                f"Memproses prediksi untuk {symbol} pada timeframe {timeframe}..."
            )

            # Ambil prediksi
            prediction = self.system.predict(symbol, timeframe)

            if prediction is None:
                proses_msg.edit_text(
                    f"Gagal mendapatkan prediksi untuk {symbol} pada timeframe {timeframe}."
                )
                return

            # Format hasil
            result_msg = (
                f"📊 *Prediksi untuk {symbol} ({timeframe})*\n\n"
                f"🔮 *Prediksi AI:* {prediction['prediction']}\n"
                f"⚖️ *Keyakinan:* {prediction['confidence']:.2f}\n\n"
                f"*Sinyal Indikator:*\n"
                f"📈 RSI: {prediction['rsi_signal']}\n"
                f"📉 MACD: {prediction['macd_signal']}\n"
                f"📊 MA: {prediction['ma_signal']}\n\n"
                f"*Keputusan Akhir:* {prediction['final_signal']} dengan keyakinan {prediction['final_confidence']:.2f}\n\n"
                f"🕒 *Waktu:* {prediction['timestamp']}"
            )
            proses_msg.edit_text(result_msg, parse_mode="Markdown")

            # Log prediksi
            logger.info(
                f"Prediction for {symbol} ({timeframe}): {prediction['final_signal']}"
            )

        except Exception as e:
            logger.error(f"Error in predict_command: {e}")
            await update.message.reply_text(f"Terjadi kesalahan: {str(e)}")

    async def train_command(self, update: Update, context: ContextTypes.DEFAULT):
        try:
            args = context.args

            if len(args) != 2:
                await update.message.reply_text(
                    "Format perintah salah. Gunakan:\n"
                    "/train EURUSD H1\n"
                    "Contoh: /train EURUSD H1"
                )
                return

            symbol = args[0].upper()
            timeframe = args[1].upper()
            raw_data_dir = os.path.join(self.dataset_dir, symbol)
            filename = f"{symbol}_{timeframe}.csv"
            file_path = os.path.join(raw_data_dir, filename)

            # Cek file
            if not os.path.exists(file_path):
                await update.message.reply_text(f"File {file_path} tidak ditemukan.")
                return

            # Kirim pesan proses
            proses_msg = await update.message.reply_text(
                f"Melatih model dengan data dari {file_path}...\nIni akan memakan waktu beberapa menit."
            )

            succes = self.system.train_model(file_path=file_path)

            if succes:
                proses_msg.edit_text("Model berhasil dilatih dan disimpan.")
            else:
                proses_msg.edit_text(
                    "Gagal melatih model. Periksa log untuk informasi lebih lanjut."
                )
        except Exception as e:
            logger.error(f"Error in train_command: {e}")
            await update.message.reply_text(f"Terjadi kesalahan: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT):
        message_text = update.message.text.lower()

        if "prediksi" in message_text:
            words = message_text.split()
            if len(words) >= 3:
                for i, word in enumerate(words):
                    if word == "prediksi" and i + 2 < len(words):
                        pair = words[i + 1].upper()
                        timeframe = words[i + 2].upper()
                        if timeframe not in self.timeframes:
                            context.args = [pair, timeframe]
                            await self.predict_command(update, context)
                            return

            await update.message.reply_text(
                "Format pesan salah. Gunakan:\n"
                "/prediksi [PAIR] [TIMEFRAME]\n"
                "Contoh: /prediksi EURUSD H1"
            )

        else:
            await update.message.reply_text(
                "Saya tidak mengerti pesan Anda. Gunakan /help untuk melihat perintah yang tersedia."
            )
