import logging
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, ContextTypes, filters

from bot.rules.rule import get_response

logger = logging.getLogger(__name__)


def user_handler(app):
    async def rule_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_msg = update.message.text
        response = get_response(user_msg)
        await update.message.reply_text(response, parse_mode="Markdown")

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_msg = (
            "✨ *Halo! Selamat datang di XenfoAI* ✨\n\n"
            "Aku bakal bantu kamu dapatkan:\n"
            "✔️ Dapat sinyal tanpa perlu mantengin chart.\n"
            "✔️ Analisa pasar lebih mudah.\n"
            "✔️ Notifikasi berita/fundamental terkini.\n\n"
            "Ketik /menu buat mulai eksplor fiturnya. Yuk, trading jadi lebih santai! 😊\n\n"
            "*Semoga hari ini penuh green candles!* 📈"
        )
        await update.message.reply_text(welcome_msg, parse_mode="Markdown")

    async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
        menu_msg = (
            "💡 *Menu Utama* 💡\n\n"
            "Ini yang bisa aku lakukan buat kamu:\n"
            "• /prediksi [PAIR] [TIMEFRAME] - Analisa pair favoritmu\n"
            "• /stats - Performaku\n"
            "• /edu - Vitamin untuk pemula\n\n"
            "Pilih fitur yang kamu mau, yuk mulai trading cerdas hari ini! 💪"
        )
        await update.message.reply_text(menu_msg, parse_mode="Markdown")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), rule_response))
