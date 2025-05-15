import os
import logging
from src.bot.telegram_bot import TelegramBot
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


def main():
    try:
        bot = TelegramBot()
        logger.info("Bot running...")
        bot.run()

    except Exception as e:
        logger.error(f"Error di main: {str(e)}")


if __name__ == "__main__":
    main()
