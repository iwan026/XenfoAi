import logging
from telegram.ext import ApplicationBuilder

from config import KonfigurasiTelegram
from bot.handlers.user_handler import user_handler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        app = ApplicationBuilder().token(KonfigurasiTelegram.BOT_TOKEN).build()
        user_handler(app)

        app.run_polling()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
