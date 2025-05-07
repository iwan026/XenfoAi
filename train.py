import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
import json
import warnings

from config import TradingConfig, ModelParameters
from data_processor import ForexDataProcessor
from model import AdaptiveForexModel
from evaluator import ForexModelEvaluator
from utils import TradingUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Enhanced model training manager"""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.upper()
        self.timeframe = timeframe.upper()
        self.data_processor = ForexDataProcessor()
        self.model = AdaptiveForexModel()
        self.evaluator = ForexModelEvaluator()
        self.utils = TradingUtils()
        self.training_history = None
        self.market_analysis = None

    def prepare_training_data(self, num_candles: int = 50000) -> Tuple[np.ndarray, ...]:
        """Prepare data for model training with enhanced preprocessing"""
        try:
            logger.info(f"Preparing training data for {self.symbol} {self.timeframe}")

            # Fetch historical data
            df = self.data_processor.get_mt5_historical_data(
                self.symbol, self.timeframe, num_candles
            )

            if df is None or df.empty:
                raise ValueError("Failed to fetch historical data")

            # Analyze market conditions
            self.market_analysis = self.utils.analyze_market_conditions(df)

            # Add technical indicators with market regime awareness
            df = self.data_processor.add_technical_indicators(df)
            if df is None or df.empty:
                raise ValueError("Failed to add technical indicators")

            # Prepare sequences with advanced labeling
            X, y = self.data_processor.prepare_sequences(
                df, lookback_period=TradingConfig.MODEL_PARAMS.lookback_period
            )

            # Validate data
            if not self._validate_training_data(X, y):
                raise ValueError("Data validation failed")

            # Create train-test-validation split
            X_train, X_test, X_val, y_train, y_test, y_val = (
                self.data_processor.create_train_test_split(X, y)
            )

            logger.info("Data preparation completed successfully")
            return X_train, X_test, X_val, y_train, y_test, y_val, df

        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate training data quality"""
        try:
            if X.shape[0] == 0 or y.shape[0] == 0:
                logger.error("Empty training data")
                return False

            if np.isnan(X).any() or np.isnan(y).any():
                logger.error("Training data contains NaN values")
                return False

            if not np.isfinite(X).all() or not np.isfinite(y).all():
                logger.error("Training data contains infinite values")
                return False

            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique, counts))

            if len(unique) < 2:
                logger.error("Insufficient class variety in training data")
                return False

            # Check for minimum samples per class
            min_samples = 100
            if any(count < min_samples for count in counts):
                logger.warning("Some classes have very few samples")

            logger.info(f"Class distribution: {class_dist}")
            return True

        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return False

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> bool:
        """Train model with enhanced monitoring and adaptation"""
        try:
            logger.info("Starting model training")

            # Configure model based on market analysis
            self._configure_model_for_market_conditions()

            # Build model architecture
            input_shape = (TradingConfig.MODEL_PARAMS.lookback_period, X_train.shape[2])
            self.model.build_model(
                ModelParameters(
                    sequence_length=input_shape[0],
                    n_features=input_shape[1],
                    n_classes=len(np.unique(y_train)),
                )
            )

            # Train model with market regime awareness
            self.training_history = self.model.train(
                X_train,
                y_train,
                X_val,
                y_val,
                market_regime=self.market_analysis["trend"]["trend_direction"],
                epochs=TradingConfig.MODEL_PARAMS.epochs,
            )

            # Save training results
            self._save_training_results()

            logger.info("Model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return False

    def _configure_model_for_market_conditions(self) -> None:
        """Adjust model configuration based on market conditions"""
        try:
            # Adjust dropout based on volatility
            if self.market_analysis["volatility"]["volatility_regime"] == "high":
                TradingConfig.MODEL_PARAMS.dropout_rate *= 1.2

            # Adjust learning rate based on trend strength
            if self.market_analysis["trend"]["trend_strength"] == "weak":
                TradingConfig.MODEL_PARAMS.learning_rate *= 0.8

            # Adjust batch size based on market regime
            if self.market_analysis["trend"]["trend_direction"] == "ranging":
                TradingConfig.MODEL_PARAMS.batch_size = max(
                    16, TradingConfig.MODEL_PARAMS.batch_size // 2
                )

            logger.info("Model configuration adjusted for current market conditions")

        except Exception as e:
            logger.error(f"Error configuring model: {str(e)}")
            raise

    def evaluate_model(
        self, X_test: np.ndarray, y_test: np.ndarray, price_data: pd.DataFrame
    ) -> Dict:
        """Evaluate model with comprehensive metrics"""
        try:
            logger.info("Starting model evaluation")

            # Get predictions with probabilities
            predictions = self.model.predict(X_test)
            probabilities = self.model.predict(X_test, return_probabilities=True)

            # Get market regimes for test period
            market_regimes = [self.market_analysis["trend"]["trend_direction"]] * len(
                y_test
            )

            # Evaluate model
            evaluation_results = self.evaluator.evaluate_model(
                y_test,
                predictions,
                price_data.iloc[-len(y_test) :],
                probabilities,
                market_regimes,
            )

            # Save evaluation results
            self._save_evaluation_results(evaluation_results)

            # Generate and save visualizations
            self._generate_evaluation_plots(evaluation_results)

            logger.info("Model evaluation completed successfully")
            return evaluation_results

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def _save_training_results(self) -> None:
        """Save training results and model artifacts"""
        try:
            # Create results directory
            results_dir = os.path.join(
                TradingConfig.MODEL_DIR,
                f"{self.symbol}_{self.timeframe}",
                datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            )
            os.makedirs(results_dir, exist_ok=True)

            # Save model
            self.model.save_model(self.symbol, self.timeframe)

            # Save training history
            with open(os.path.join(results_dir, "training_history.json"), "w") as f:
                json.dump(self.training_history, f, indent=4)

            # Save market analysis
            with open(os.path.join(results_dir, "market_analysis.json"), "w") as f:
                json.dump(self.market_analysis, f, indent=4)

            # Plot and save training curves
            self._plot_training_curves(results_dir)

            logger.info(f"Training results saved to {results_dir}")

        except Exception as e:
            logger.error(f"Error saving training results: {str(e)}")
            raise

    def _plot_training_curves(self, save_dir: str) -> None:
        """Plot and save training curves"""
        try:
            plt.figure(figsize=(15, 5))

            # Loss curves
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history["loss"], label="Training Loss")
            plt.plot(self.training_history["val_loss"], label="Validation Loss")
            plt.title("Model Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            # Accuracy curves
            plt.subplot(1, 2, 2)
            plt.plot(self.training_history["accuracy"], label="Training Accuracy")
            plt.plot(self.training_history["val_accuracy"], label="Validation Accuracy")
            plt.title("Model Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "training_curves.png"))
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting training curves: {str(e)}")
            raise

    def _save_evaluation_results(self, results: Dict) -> None:
        """Save evaluation results"""
        try:
            eval_dir = os.path.join(
                TradingConfig.MODEL_DIR, f"{self.symbol}_{self.timeframe}", "evaluation"
            )
            os.makedirs(eval_dir, exist_ok=True)

            # Save results as JSON
            with open(os.path.join(eval_dir, "evaluation_results.json"), "w") as f:
                json.dump(results, f, indent=4)

            logger.info(f"Evaluation results saved to {eval_dir}")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise

    def _generate_evaluation_plots(self, results: Dict) -> None:
        """Generate and save evaluation plots"""
        try:
            self.evaluator.plot_results(
                save_path=os.path.join(
                    TradingConfig.MODEL_DIR,
                    f"{self.symbol}_{self.timeframe}",
                    "evaluation",
                    "evaluation_plots.png",
                )
            )

        except Exception as e:
            logger.error(f"Error generating evaluation plots: {str(e)}")
            raise


def main():
    """Main training execution"""
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Initialize MT5
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return

        # Training parameters
        symbol = "EURUSD"
        timeframe = "H1"

        # Create trainer instance
        trainer = ModelTrainer(symbol, timeframe)

        # Prepare data
        X_train, X_test, X_val, y_train, y_test, y_val, price_data = (
            trainer.prepare_training_data()
        )

        # Train model
        if trainer.train_model(X_train, y_train, X_val, y_val):
            # Evaluate model
            evaluation_results = trainer.evaluate_model(X_test, y_test, price_data)
            logger.info("Training and evaluation completed successfully")

            # Print summary
            print("\nTraining Summary:")
            print("=" * 50)
            print(f"Symbol: {symbol}")
            print(f"Timeframe: {timeframe}")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Test samples: {len(X_test)}")
            print("\nEvaluation Results:")
            print("=" * 50)
            print(
                f"Accuracy: {evaluation_results['classification_metrics']['accuracy']:.4f}"
            )
            print(
                f"Profit Factor: {evaluation_results['trading_metrics'].profit_factor:.2f}"
            )
            print(f"Win Rate: {evaluation_results['trading_metrics'].win_rate:.2%}")
            print(
                f"Sharpe Ratio: {evaluation_results['trading_metrics'].sharpe_ratio:.2f}"
            )
        else:
            logger.error("Training failed")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
