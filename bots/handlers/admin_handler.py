from telegram import Update
from telegram.ext import CommandHandler, ContextTypes
import logging

from config import KonfigurasiTelegram
from core.mt5_core import MT5Core

logger = logging.getLogger(__name__)


def admin_handler(app):
    async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message.chat_id not in KonfigurasiTelegram.ADMIN_ID:
            await update.message.reply_text("Akses ditolak!")
            return None

        admin_msg = (
            "💡 *Menu Admin* 💡\n\n"
            "Hormat kepada ichi boss:\n"
            "• /get [SYMBOL] [TIMEFRAME] - Ambil data historis\n"
        )
        await update.message.reply_text(admin_msg, parse_mode="Markdown")

    async def get(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message.chat_id not in KonfigurasiTelegram.ADMIN_ID:
            await update.message.reply_text("*Akses ditolak!*", parse_mode="Markdown")
            return None

        if len(context.args) < 2:
            await update.message.reply_text(
                "Format salah! gunakan: /get [SYMBOL] [TIMEFRAME]\n"
                "Contoh: /get EURUSD H1"
            )
            return None

        symbol = context.args[0].upper()
        timeframe = context.args[1].upper()

        await update.message.reply_text(
            f"🔄 Mengambil daya histori {symbol} {timeframe}"
        )

        df = MT5Core.get_historical_data(symbol, timeframe)

        if df is not None:
            await update.message.reply_text(
                f"✅ Data {symbol} {timeframe} berhasil diambil!\n"
                f"Jumlah baris: {len(df)}\n"
                f"Periode: {df['time'].iloc[0]} sampai {df['time'].iloc[-1]}"
            )
        else:
            await update.message.reply_text(
                f"❌ Gagal mengambil data {symbol} {timeframe}"
            )

    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("get", get))
