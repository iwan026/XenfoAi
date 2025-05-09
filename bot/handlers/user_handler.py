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
        await update.message.reply_text("Halo, selamat datang di XenfoAi")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), rule_response))
