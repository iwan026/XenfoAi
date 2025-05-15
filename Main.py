import os
import logging
from src.bot.telegram_bot import TelegramBot
from src.utils.helpers import setup_logging
from config import BOT_TOKEN

logger = setup_logging(__name__)


def main():
    try:
        token = BOT_TOKEN
        if not token:
            logger.error("Token bot telegram belum di set")
            return

        bot = TelegramBot()
        logger.info("Bot running...")
        bot.run()

    except Exception as e:
        logger.error(f"Error di main: {str(e)}")


if __name__ == "__main__":
    main()
