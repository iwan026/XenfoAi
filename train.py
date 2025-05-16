import logging
import argparse
from pathlib import Path
from src.model import ForexModel
from config import SYMBOLS, LOGS_DIR, PLOTS_DIR


def setup_logging():
    """Setup logging configuration"""
    log_file = LOGS_DIR / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(str(log_file))],
    )
    return logging.getLogger(__name__)


def main():
    """Main training function"""
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Train Forex prediction model")
    parser.add_argument("symbol", choices=SYMBOLS, help="Currency pair to train")
    args = parser.parse_args()

    try:
        logger.info(f"Starting training for {args.symbol}")
        model = ForexModel(args.symbol)
        success = model.train()

        if success:
            logger.info(f"Training completed successfully for {args.symbol}")
        else:
            logger.error(f"Training failed for {args.symbol}")
    except Exception as e:
        logger.error(f"Fatal error during training: {e}")
        raise


if __name__ == "__main__":
    main()
