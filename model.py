import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
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
    GlobalAveragePooling1D,
    Concatenate,
    Add,
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
        self.metadata = {
            "created_at": "2025-05-08 12:50:03",
            "created_by": "iwan026",
            "version": "2.0.0",
        }

        # Enhanced hyperparameters
        self.dropout_rate = 0.4
        self.l2_lambda = 0.02
        self.learning_rate = 0.001
        self.reduce_lr_factor = 0.6
        self.early_stopping_patience = 15
        self.reduce_lr_patience = 7
        self.num_attention_heads = 8
        self.attention_key_dim = 32
        self.conv_filters = 32
        self.lstm_units = 64
        self.gru_units = 32

    def build_model(self, sequence_length: int, n_features: int) -> None:
        """Build enhanced signal generation model architecture"""
        try:
            # Input layers
            market_input = Input(
                shape=(sequence_length, n_features), name="market_data"
            )

            # Feature extraction branch 1: Convolutional
            conv_branch = self._build_conv_branch(market_input)

            # Feature extraction branch 2: Temporal
            temporal_branch = self._build_temporal_branch(market_input)

            # Combine branches
            combined = Concatenate()([conv_branch, temporal_branch])

            # Final processing
            x = self._build_output_layers(combined)

            # Create model
            self.model = Model(inputs=market_input, outputs=x)

            # Compile with enhanced metrics
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=[
                    "accuracy",
                    AUC(name="auc"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="recall"),
                ],
            )

            logger.info(
                f"Model built successfully with {self.model.count_params()} parameters"
            )

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def _build_conv_branch(self, inputs):
        """Build convolutional feature extraction branch"""
        try:
            # Multi-scale convolution blocks
            conv1 = Conv1D(
                filters=self.conv_filters,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=l2(self.l2_lambda),
            )(inputs)

            conv2 = Conv1D(
                filters=self.conv_filters * 2,
                kernel_size=5,
                padding="same",
                activation="relu",
                kernel_regularizer=l2(self.l2_lambda),
            )(inputs)

            # Combine different scales
            conv_combined = Add()([conv1, conv2])
            conv_combined = BatchNormalization()(conv_combined)
            conv_combined = Dropout(self.dropout_rate)(conv_combined)

            return conv_combined

        except Exception as e:
            logger.error(f"Error building conv branch: {str(e)}")
            raise

    def _build_temporal_branch(self, inputs):
        """Build temporal feature extraction branch"""
        try:
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=self.num_attention_heads, key_dim=self.attention_key_dim
            )(inputs, inputs)
            attention = LayerNormalization()(attention + inputs)

            # Bidirectional LSTM
            lstm = Bidirectional(
                LSTM(
                    self.lstm_units,
                    return_sequences=True,
                    kernel_regularizer=l2(self.l2_lambda),
                    recurrent_regularizer=l2(self.l2_lambda),
                )
            )(attention)
            lstm = BatchNormalization()(lstm)
            lstm = Dropout(self.dropout_rate)(lstm)

            # GRU layer
            gru = GRU(
                self.gru_units,
                kernel_regularizer=l2(self.l2_lambda),
                recurrent_regularizer=l2(self.l2_lambda),
            )(lstm)
            gru = BatchNormalization()(gru)

            return gru

        except Exception as e:
            logger.error(f"Error building temporal branch: {str(e)}")
            raise

    def _build_output_layers(self, inputs):
        """Build final output layers"""
        try:
            x = Dense(32, activation="relu", kernel_regularizer=l2(self.l2_lambda))(
                inputs
            )
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)

            # Output layer for 3 classes (Buy, Sell, Hold)
            outputs = Dense(
                3, activation="softmax", kernel_regularizer=l2(self.l2_lambda)
            )(x)

            return outputs

        except Exception as e:
            logger.error(f"Error building output layers: {str(e)}")
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
        """Train model with enhanced features and monitoring"""
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
            callbacks = self._setup_training_callbacks(model_dir)

            # Train with enhanced parameters
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=TradingConfig.MODEL_PARAMS.epochs,
                batch_size=48,
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

    def _setup_training_callbacks(self, model_dir: str) -> List:
        """Setup enhanced training callbacks"""
        try:
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
                    filepath=os.path.join(model_dir, "weights.h5"),
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
                TensorBoard(
                    log_dir=os.path.join(model_dir, "logs"),
                    histogram_freq=1,
                    update_freq="epoch",
                ),
            ]

            return callbacks

        except Exception as e:
            logger.error(f"Error setting up callbacks: {str(e)}")
            raise

    def generate_signal(
        self,
        X: np.ndarray,
        symbol: str,
        timeframe: str,
        market_regime: str = "stable_trend",
    ) -> Optional[Dict]:
        """Generate enhanced trading signal"""
        try:
            # Validate input data
            if not self._validate_input_data(X):
                return None

            # Load or build model if needed
            if not self._ensure_model_loaded(X, symbol, timeframe):
                return None

            # Generate prediction with confidence
            predictions = self.model.predict(X[-1:], verbose=0)
            signal_class = np.argmax(predictions[0])
            confidence = float(predictions[0][signal_class])

            # Apply dynamic confidence threshold based on market regime
            threshold = self._get_dynamic_threshold(market_regime)

            if confidence < threshold:
                signal_type = "HOLD"
            else:
                signal_type = (
                    "BUY"
                    if signal_class == 1
                    else "SELL"
                    if signal_class == 0
                    else "HOLD"
                )

            # Create enhanced signal response
            current_time = datetime.utcnow()
            signal = {
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": signal_type,
                "confidence": confidence,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "market_regime": market_regime,
                "threshold_used": threshold,
                "metadata": {
                    "model_version": self.metadata["version"],
                    "generated_by": self.metadata["created_by"],
                    "generated_at": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }

            # Update last signal time
            self.last_signal_time[f"{symbol}_{timeframe}"] = current_time

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _validate_input_data(self, X: np.ndarray) -> bool:
        """Validate input data for signal generation"""
        try:
            if not isinstance(X, np.ndarray) or X.size == 0:
                logger.error(
                    f"Invalid input data for signal generation: X shape={X.shape if isinstance(X, np.ndarray) else 'None'}"
                )
                return False

            if len(X.shape) != 3:
                logger.error(f"Invalid input shape: {X.shape}, expected 3 dimensions")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating input data: {str(e)}")
            return False

    def _get_dynamic_threshold(self, market_regime: str) -> float:
        """Get dynamic confidence threshold based on market regime"""
        try:
            base_threshold = TradingConfig.SIGNAL_PARAMS.confidence_threshold

            regime_multipliers = {
                "stable_trend": 1.0,
                "volatile_trend": 1.2,
                "stable_range": 1.3,
                "volatile_range": 1.5,
            }

            multiplier = regime_multipliers.get(market_regime, 1.0)
            return base_threshold * multiplier

        except Exception as e:
            logger.error(f"Error getting dynamic threshold: {str(e)}")
            return TradingConfig.SIGNAL_PARAMS.confidence_threshold

    def _ensure_model_loaded(self, X: np.ndarray, symbol: str, timeframe: str) -> bool:
        """Ensure model is loaded and ready for predictions"""
        try:
            if self.model is None:
                if not self._load_model(symbol, timeframe):
                    logger.error(f"Failed to load model for {symbol}_{timeframe}")
                    try:
                        self.build_model(X.shape[1], X.shape[2])
                    except Exception as e:
                        logger.error(f"Failed to build new model: {str(e)}")
                        return False

            if self.model is None:
                logger.error("Model initialization failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Error ensuring model loaded: {str(e)}")
            return False

    def _save_model_metadata(self, symbol: str, timeframe: str) -> bool:
        """Save enhanced model metadata and configuration"""
        try:
            model_dir = os.path.dirname(TradingConfig.get_model_path(symbol, timeframe))

            metadata = {
                "model_info": {
                    "created_at": self.metadata["created_at"],
                    "created_by": self.metadata["created_by"],
                    "version": self.metadata["version"],
                    "architecture": {
                        "conv_filters": self.conv_filters,
                        "lstm_units": self.lstm_units,
                        "gru_units": self.gru_units,
                        "attention_heads": self.num_attention_heads,
                    },
                },
                "training_params": {
                    "dropout_rate": self.dropout_rate,
                    "l2_lambda": self.l2_lambda,
                    "learning_rate": self.learning_rate,
                },
                "history": {
                    k: [float(x) for x in v] for k, v in self.history.history.items()
                },
            }

            # Save metadata
            metadata_path = os.path.join(model_dir, "model_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            return True

        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            return False
