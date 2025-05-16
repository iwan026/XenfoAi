import logging
from src.bot import TelegramBot


def main():
    """Main entry point"""
    try:
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
