import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import MetaTrader5 as mt5
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from config import TradingConfig
from data_processor import ForexDataProcessor
from model import DeepForexModel
from utils import TradingUtils

# Setup logging
logging.basicConfig(
    filename=TradingConfig.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_model(symbol="EURUSD", timeframe="H1", num_candles=10000):
    """Train model for specific symbol and timeframe"""
    try:
        print(f"\nTraining model for {symbol} {timeframe}")
        print("=" * 50)

        # Initialize components
        data_processor = ForexDataProcessor()
        model = DeepForexModel()

        # Get historical data
        print("\nFetching historical data...")
        df = data_processor.get_mt5_historical_data(symbol, timeframe, num_candles)
        if df is None:
            print("Failed to fetch historical data")
            return False

        # Add technical indicators
        print("Adding technical indicators...")
        df = data_processor.add_technical_indicators(df)
        if df is None:
            print("Failed to add technical indicators")
            return False

        # Prepare sequences
        print("Preparing training sequences...")
        X, y = data_processor.prepare_sequences(df, TradingConfig.LOOKBACK_PERIOD)
        if len(X) == 0 or len(y) == 0:
            print("Failed to prepare sequences")
            return False

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        # Create and train model
        print("\nCreating model architecture...")
        input_shape = (TradingConfig.LOOKBACK_PERIOD, X.shape[2])
        model.create_model_architecture(input_shape)

        print("\nStarting model training...")
        history = model.model.fit(
            X,
            y,
            epochs=TradingConfig.EPOCHS,
            batch_size=TradingConfig.BATCH_SIZE,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=TradingConfig.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=TradingConfig.REDUCE_LR_PATIENCE,
                ),
            ],
        )

        # Save model
        print("\nSaving model...")
        if model.save_model(symbol, timeframe):
            print(f"Model saved successfully!")
        else:
            print("Failed to save model")
            return False

        return True

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        print(f"Error occurred: {str(e)}")
        return False


if __name__ == "__main__":
    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        exit()

    try:
        # Train for specific pair
        symbol = "EURUSD"  # Bisa diganti dengan pair lain
        timeframe = "H1"  # Bisa diganti dengan timeframe lain

        success = train_model(symbol, timeframe)
        if success:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed!")

    finally:
        mt5.shutdown()
