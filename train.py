import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def train_model(symbol="EURUSD", timeframe="H1", num_candles=50000):
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
        if df is None or df.empty:
            print("Failed to fetch historical data")
            return False

        # Add technical indicators
        print("Adding technical indicators...")
        df = data_processor.add_technical_indicators(df)
        if df is None or df.empty:
            print("Failed to add technical indicators")
            return False

        # Prepare sequences
        print("Preparing training sequences...")
        X, y = data_processor.prepare_sequences(df, TradingConfig.LOOKBACK_PERIOD)

        # Validate data types and shapes
        print("\nValidating data...")
        print(f"X dtype: {X.dtype}, shape: {X.shape}")
        print(f"y dtype: {y.dtype}, shape: {y.shape}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            print("Empty training data")
            return False

        # Verify no NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            print("Training data contains NaN values")
            return False

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
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                ),
            ],
            verbose=1,
        )

        # Save model
        print("\nSaving model...")
        if model.save_model(symbol, timeframe):
            print("Model saved successfully!")

            # Plot training history
            plt.figure(figsize=(10, 5))
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"training_history_{symbol}_{timeframe}.png")
            plt.close()

            return True
        else:
            print("Failed to save model")
            return False

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        print(f"Error occurred: {str(e)}")
        return False


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        exit()

    try:
        symbol = "EURUSD"
        timeframe = "H1"
        success = train_model(symbol, timeframe)

        if success:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed!")

    finally:
        mt5.shutdown()
