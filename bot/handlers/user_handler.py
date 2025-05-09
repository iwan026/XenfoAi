import logging
from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, ContextTypes, filters

from bot.rules.user_rule import get_response

logger = logging.getLogger(__name__)


def user_handler(app):
    async def rule_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_msg = update.message.text
        response = get_response(user_msg)
        await update.message.reply_text(response)

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_msg = (
            "*Selamat datang di XenfoAi!*\n"
            "Siap bantu kamu dapetin sinyal trading terkini & akurat langsung dari kami.\n"
            "Dengan bot ini, kamu nggak perlu lagi mantengin chart tiap detik!\n\n"
            "Ketik /help buat lihat semua menu dan fitur yang tersedia — mulai dari sinyal harian, analisis pasar, sampai notifikasi real-time!\n"
            "*Stay cuan, stay santai!*\n"
            "📈🔥💸"
        )
        await update.message.reply_text(welcome_msg, parse_mode="Markdown")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), rule_response))
