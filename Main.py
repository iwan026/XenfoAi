import logging
from telegram.ext import ApplicationBuilder

from config import KonfigurasiTelegram
from bots.handlers.user_handler import user_handler
from bots.handlers.admin_handler import admin_handler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        app = ApplicationBuilder().token(KonfigurasiTelegram.BOT_TOKEN).build()
        user_handler(app)
        admin_handler(app)

        app.run_polling()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
