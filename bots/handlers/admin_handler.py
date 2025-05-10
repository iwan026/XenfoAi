from telegram import Update
from telegram.ext import CommandHandler, ContextTypes
import logging

from config import KonfigurasiTelegram

logger = logging.getLogger(__name__)


def admin_handler(app):
    async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message.chat_id not in KonfigurasiTelegram.ADMIN_ID:
            await update.message.reply_text("Akses ditolak!")
            return None

        admin_msg = (
            "💡 *Menu Admin* 💡\n\n"
            "Hormat kepada ichi boss:\n"
            "• /train [SYMBOL] [TIMEFRAME] - Training model\n"
        )
        await update.message.reply_text(admin_msg, parse_mode="Markdown")

    app.add_handler(CommandHandler("admin", admin))
