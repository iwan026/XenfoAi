import logging
from pathlib import Path
from src.bot import TelegramBot
from config import BASE_DIR


def setup_logging():
    """Setup logging configuration"""
    log_file = BASE_DIR / "app.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_file))],
    )

    # Create logger
    logger = logging.getLogger(__name__)
    return logger


def main():
    """Main entry point"""
    logger = setup_logging()

    try:
        logger.info("Initializing XenfoAI Bot...")
        bot = TelegramBot()
        logger.info("Starting bot...")
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
