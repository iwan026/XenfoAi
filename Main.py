import logging
from telegram.ext import ApplicationBuilder

import config
from bots.telegram_bot import TelegramBot

logger = logging.getLogger(__name__)


def main():
    try:
        bot = TelegramBot()
        bot.application.run_polling()

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()
