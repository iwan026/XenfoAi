import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    Conv1D,
    MultiHeadAttention,
    LayerNormalization,
    Bidirectional,
    GRU,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    BackupAndRestore,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from config import TradingConfig

logger = logging.getLogger(__name__)


class ForexSignalModel:
    """Enhanced neural network model for forex signal generation"""

    def __init__(self):
        self.model = None
        self.history = None
        self.current_market_regime = None
        self._best_weights = None
        self.last_signal_time = {}
        self.scaler = None

        # Model hyperparameters
        self.dropout_rate = 0.4
        self.l2_lambda = 0.02
        self.learning_rate = 0.001
        self.reduce_lr_factor = 0.6
        self.early_stopping_patience = 15
        self.reduce_lr_patience = 7

    def build_model(self, sequence_length: int, n_features: int) -> None:
        """Build enhanced signal generation model architecture"""
        try:
            # Input layer
            inputs = Input(shape=(sequence_length, n_features))

            # Convolutional feature extraction with increased regularization
            x = Conv1D(
                filters=32,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=l2(self.l2_lambda),
            )(inputs)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

            # Enhanced temporal attention mechanism
            attention = MultiHeadAttention(
                num_heads=8,  # Increased from 4
                key_dim=32,
            )(x, x)
            attention = LayerNormalization()(attention + x)

            # Bidirectional LSTM layers with increased regularization
            x = Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                    kernel_regularizer=l2(self.l2_lambda),
                    recurrent_regularizer=l2(self.l2_lambda),
                )
            )(attention)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

            # Additional GRU layer for better temporal patterns
            x = GRU(
                32,
                kernel_regularizer=l2(self.l2_lambda),
                recurrent_regularizer=l2(self.l2_lambda),
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

            # Output layer with stronger regularization
            outputs = Dense(
                3, activation="softmax", kernel_regularizer=l2(self.l2_lambda)
            )(x)

            # Create and compile model
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=[
                    "accuracy",
                    AUC(name="auc"),
                ],
            )

            logger.info(
                f"Model built successfully with {self.model.count_params()} parameters"
            )

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> bool:
        """Train model with enhanced features"""
        try:
            if self.model is None:
                self.build_model(X_train.shape[1], X_train.shape[2])

            # Calculate class weights
            class_counts = np.bincount(y_train)
            total = len(y_train)
            class_weights = {
                i: total / (len(class_counts) * count)
                for i, count in enumerate(class_counts)
            }

            # Create model directory
            model_dir = os.path.dirname(TradingConfig.get_model_path(symbol, timeframe))
            os.makedirs(model_dir, exist_ok=True)

            # Enhanced callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.reduce_lr_factor,
                    patience=self.reduce_lr_patience,
                    min_lr=1e-6,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=os.path.join(model_dir, "weights.weights.h5"),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=os.path.join(model_dir, "model.h5"),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1,
                ),
                BackupAndRestore(backup_dir=os.path.join(model_dir, "backups")),
                TensorBoard(log_dir=os.path.join(model_dir, "logs"), histogram_freq=1),
            ]

            # Train with enhanced parameters
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=TradingConfig.MODEL_PARAMS.epochs,
                batch_size=48,  # Increased batch size
                callbacks=callbacks,
                class_weight=class_weights,
                shuffle=True,
                verbose=1,
            )

            # Store best weights
            self._best_weights = self.model.get_weights()

            # Save model metadata and evaluate
            self._save_model_metadata(symbol, timeframe)
            self._evaluate_and_log(X_val, y_val, symbol, timeframe)

            return True

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

    def _evaluate_and_log(
        self, X_val: np.ndarray, y_val: np.ndarray, symbol: str, timeframe: str
    ) -> None:
        """Evaluate model and log results"""
        try:
            predictions = self.model.predict(X_val)
            y_pred = np.argmax(predictions, axis=1)

            # Calculate detailed metrics
            metrics = {
                "accuracy": float(np.mean(y_val == y_pred)),
                "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
                "classification_report": classification_report(
                    y_val,
                    y_pred,
                    target_names=["SELL", "BUY", "HOLD"],
                    output_dict=True,
                ),
            }

            # Log results
            logger.info(f"Model evaluation for {symbol}_{timeframe}:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Classification Report:\n{metrics['classification_report']}")

            # Save metrics
            metrics_path = os.path.join(
                os.path.dirname(TradingConfig.get_model_path(symbol, timeframe)),
                "evaluation_metrics.json",
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")

    def _save_model_metadata(self, symbol: str, timeframe: str) -> bool:
        """Save model metadata and configuration"""
        try:
            model_dir = os.path.dirname(TradingConfig.get_model_path(symbol, timeframe))

            # Save model architecture
            config_path = os.path.join(model_dir, "model_config.json")
            with open(config_path, "w") as f:
                f.write(str(self.model.get_config()))

            # Save training history
            history_path = os.path.join(model_dir, "training_history.json")
            with open(history_path, "w") as f:
                history_dict = {
                    k: [float(x) for x in v] for k, v in self.history.history.items()
                }
                json.dump(history_dict, f, indent=4)

            logger.info(f"Model metadata saved successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            return False

    def generate_signal(
        self,
        X: np.ndarray,
        symbol: str,
        timeframe: str,
        market_regime: str = "stable_trend",
    ) -> Optional[Dict]:
        """Generate trading signal"""
        try:
            # Validate input data
            if not isinstance(X, np.ndarray) or X.size == 0:
                logger.error(
                    f"Invalid input data for signal generation: X shape={X.shape if isinstance(X, np.ndarray) else 'None'}"
                )
                return None

            # Ensure input data type is float32
            X = X.astype(np.float32)

            # Check input shape
            if len(X.shape) != 3:
                logger.error(f"Invalid input shape: {X.shape}, expected 3 dimensions")
                return None

            expected_features = 16  # Number of features defined in data_processor
            if X.shape[2] != expected_features:
                logger.error(
                    f"Invalid number of features: {X.shape[2]}, expected {expected_features}"
                )
                return None

            # Load or build model if needed
            if self.model is None:
                if not self._load_model(symbol, timeframe):
                    logger.error(f"Failed to load model for {symbol}_{timeframe}")
                    # Try building new model
                    try:
                        self.build_model(X.shape[1], X.shape[2])
                    except Exception as e:
                        logger.error(f"Failed to build new model: {str(e)}")
                        return None

            if self.model is None:
                logger.error("Model initialization failed")
                return None

            # Generate prediction
            try:
                predictions = self.model.predict(X[-1:], verbose=0)
                signal_class = np.argmax(predictions[0])
                confidence = float(predictions[0][signal_class])
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return None

            # Apply confidence threshold
            if confidence < TradingConfig.SIGNAL_PARAMS.confidence_threshold:
                signal_type = "HOLD"
            else:
                signal_type = (
                    "BUY"
                    if signal_class == 1
                    else "SELL"
                    if signal_class == 0
                    else "HOLD"
                )

            # Create signal response
            current_time = datetime.utcnow()
            signal = {
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": signal_type,
                "confidence": confidence,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "market_regime": market_regime,
            }

            # Update last signal time
            self.last_signal_time[f"{symbol}_{timeframe}"] = current_time

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def backtest(
        self, X: np.ndarray, y_true: np.ndarray, symbol: str, timeframe: str
    ) -> Optional[Dict]:
        """Run backtest for signal generation"""
        try:
            if self.model is None:
                if not self._load_model(symbol, timeframe):
                    return None

            # Generate predictions
            predictions = self.model.predict(X)
            y_pred = np.argmax(predictions, axis=1)

            # Calculate metrics
            accuracy = np.mean(y_true == y_pred)

            # Signal distribution
            signal_dist = {
                "BUY": np.sum(y_pred == 1),
                "SELL": np.sum(y_pred == 0),
                "HOLD": np.sum(y_pred == 2),
            }

            # Calculate precision for each signal type
            precision = {}
            for signal_type, signal_value in [("BUY", 1), ("SELL", 0), ("HOLD", 2)]:
                true_positives = np.sum(
                    (y_pred == signal_value) & (y_true == signal_value)
                )
                predicted_positives = np.sum(y_pred == signal_value)
                precision[signal_type] = (
                    true_positives / predicted_positives
                    if predicted_positives > 0
                    else 0
                )

            return {
                "accuracy": accuracy,
                "total_signals": len(y_pred),
                "signal_distribution": signal_dist,
                "precision": precision,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return None

    def _save_model(self, symbol: str, timeframe: str) -> bool:
        """Save model and configuration"""
        try:
            model_path = TradingConfig.get_model_path(symbol, timeframe)

            # Save model weights
            self.model.save_weights(model_path)

            # Save model architecture
            model_config = self.model.get_config()
            config_path = model_path.replace(".h5", "_config.json")
            with open(config_path, "w") as f:
                f.write(str(model_config))

            # Save best weights
            weights_path = model_path.replace(".h5", "_best_weights.h5")
            np.save(weights_path, self._best_weights)

            logger.info(f"Model saved successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def _load_model(self, symbol: str, timeframe: str) -> bool:
        """Load model and configuration"""
        try:
            model_path = TradingConfig.get_model_path(symbol, timeframe)

            if not os.path.exists(model_path):
                logger.error(f"Model file not found for {symbol}_{timeframe}")
                return False

            # Load model architecture
            config_path = model_path.replace(".h5", "_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    model_config = eval(f.read())
                self.model = Model.from_config(model_config)
            else:
                # Build new model if config not found
                self.build_model(60, 16)  # Default values

            # Load weights
            self.model.load_weights(model_path)

            # Load best weights if available
            weights_path = model_path.replace(".h5", "_best_weights.h5")
            if os.path.exists(weights_path):
                self._best_weights = np.load(weights_path, allow_pickle=True)

            logger.info(f"Model loaded successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
