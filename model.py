import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    LayerNormalization,
    Concatenate,
    Add,
    Activation,
    TimeDistributed,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler
import joblib
from config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture"""

    sequence_length: int
    n_features: int
    n_classes: int
    lstm_units: List[int]
    attention_heads: int
    dropout_rate: float
    learning_rate: float
    batch_size: int
    market_condition: Optional[str] = None


class AdaptiveForexModel:
    """Enhanced deep learning model for forex price prediction"""

    def __init__(self):
        self.model = None
        self.config = None
        self.market_regime = None
        self.history = None
        self._best_weights = None

    def build_model(self, config: ModelConfig) -> None:
        """Build adaptive model architecture"""
        try:
            self.config = config

            # Input layer
            inputs = Input(shape=(config.sequence_length, config.n_features))

            # 1. Price Processing Branch
            price_branch = self._create_price_branch(inputs)

            # 2. Technical Indicator Branch
            indicator_branch = self._create_indicator_branch(inputs)

            # 3. Market Regime Branch
            regime_branch = self._create_regime_branch(inputs)

            # Combine all branches
            combined = self._combine_branches(
                price_branch, indicator_branch, regime_branch
            )

            # Output layers based on market regime
            outputs = self._create_adaptive_outputs(combined)

            # Create and compile model
            self.model = Model(inputs=inputs, outputs=outputs)
            self._compile_model()

            logger.info("Model built successfully")

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def _create_price_branch(self, inputs: tf.Tensor) -> tf.Tensor:
        """Create price processing branch with temporal convolutions"""
        # Multi-scale convolutions
        conv_branches = []
        for kernel_size in [3, 5, 7]:
            conv = Conv1D(
                filters=32,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                kernel_regularizer=l2(0.01),
            )(inputs)
            conv = BatchNormalization()(conv)
            conv_branches.append(conv)

        # Concatenate different scales
        multi_scale_features = Concatenate()(conv_branches)

        # Residual LSTM
        lstm_out = self._residual_lstm_block(multi_scale_features, units=64)

        return lstm_out

    def _create_indicator_branch(self, inputs: tf.Tensor) -> tf.Tensor:
        """Process technical indicators with attention mechanism"""
        # Self-attention layers
        attention_layers = []
        for _ in range(self.config.attention_heads):
            attention = MultiHeadAttention(
                num_heads=8, key_dim=32, dropout=self.config.dropout_rate
            )(inputs, inputs)
            attention = LayerNormalization()(attention + inputs)
            attention_layers.append(attention)

        # Combine attention heads
        multi_head_attention = Concatenate()(attention_layers)

        # Process with GRU
        gru = GRU(units=64, return_sequences=True, dropout=self.config.dropout_rate)(
            multi_head_attention
        )

        return gru

    def _create_regime_branch(self, inputs: tf.Tensor) -> tf.Tensor:
        """Process market regime specific features"""
        # Temporal attention
        temporal_attention = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)

        # Convolutional feature extraction
        conv = Conv1D(filters=32, kernel_size=5, padding="same", activation="relu")(
            temporal_attention
        )

        # LSTM for temporal dependencies
        lstm = LSTM(units=32, return_sequences=True)(conv)

        return lstm

    def _residual_lstm_block(self, inputs: tf.Tensor, units: int) -> tf.Tensor:
        """Create residual LSTM block"""
        # Main path
        x = LSTM(units, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)

        # Skip connection (if input shape matches)
        if inputs.shape[-1] == units:
            return Add()([inputs, x])
        else:
            # If shapes don't match, transform input
            skip = Conv1D(units, 1)(inputs)
            return Add()([skip, x])

    def _combine_branches(
        self, price: tf.Tensor, indicator: tf.Tensor, regime: tf.Tensor
    ) -> tf.Tensor:
        """Combine different branches adaptively"""
        # Adaptive weighting based on market regime
        if self.market_regime == "volatile_trend":
            weights = [0.4, 0.3, 0.3]
        elif self.market_regime == "stable_trend":
            weights = [0.3, 0.4, 0.3]
        else:  # ranging market
            weights = [0.3, 0.3, 0.4]

        # Weighted combination
        combined = Concatenate()(
            [
                TimeDistributed(Dense(64))(price),
                TimeDistributed(Dense(64))(indicator),
                TimeDistributed(Dense(64))(regime),
            ]
        )

        # Final processing
        x = LSTM(128, return_sequences=True)(combined)
        x = BatchNormalization()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = LSTM(64)(x)

        return x

    def _create_adaptive_outputs(self, x: tf.Tensor) -> tf.Tensor:
        """Create adaptive output layers based on market regime"""
        # Dense layers with regime-specific dropout
        x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.get_adaptive_dropout())(x)

        x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.get_adaptive_dropout())(x)

        # Output layer
        outputs = Dense(
            self.config.n_classes, activation="softmax", kernel_regularizer=l2(0.01)
        )(x)

        return outputs

    def get_adaptive_dropout(self) -> float:
        """Get adaptive dropout rate based on market regime"""
        base_dropout = self.config.dropout_rate

        if self.market_regime == "volatile_trend":
            return base_dropout * 1.2
        elif self.market_regime == "stable_trend":
            return base_dropout * 0.8
        else:
            return base_dropout

    def _compile_model(self) -> None:
        """Compile model with adaptive learning rate and loss"""
        # Learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
        )

        # Optimizer
        optimizer = Adam(learning_rate=lr_schedule)

        # Compile with custom loss
        self.model.compile(
            optimizer=optimizer,
            loss=self._adaptive_loss,
            metrics=["accuracy", self._precision_m, self._recall_m, self._f1_m],
        )

    def _adaptive_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Adaptive loss function based on market regime and prediction confidence"""
        # Cast inputs to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Base categorical crossentropy
        epsilon = tf.constant(1e-7, dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate base loss
        base_loss = -tf.reduce_mean(
            tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        )

        # Add confidence penalty
        confidence_penalty = tf.reduce_mean(
            tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
        )

        # Add class weights based on market regime
        class_weights = self._get_class_weights()
        weighted_loss = base_loss * class_weights

        return weighted_loss + 0.1 * confidence_penalty

    def _get_class_weights(self) -> tf.Tensor:
        """Get adaptive class weights based on market regime"""
        if self.market_regime == "volatile_trend":
            return tf.constant(
                [1.5, 1.0, 1.5, 2.0, 0.5]
            )  # Higher weight for strong signals
        elif self.market_regime == "stable_trend":
            return tf.constant([1.2, 1.2, 1.2, 1.2, 0.8])  # Balanced weights
        else:  # ranging market
            return tf.constant([1.0, 1.0, 1.0, 1.0, 1.5])  # Higher weight for hold

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        market_regime: str = "stable_trend",
        epochs: int = 100,
    ) -> Dict:
        """Train the model with advanced callbacks and monitoring"""
        try:
            self.market_regime = market_regime

            # Create callbacks
            callbacks = self._create_callbacks()

            # Train model
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1,
            )

            # Store best weights
            self._best_weights = self.model.get_weights()

            return self.history.history

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _create_callbacks(self) -> List:
        """Create training callbacks"""
        return [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                "best_model.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
            TensorBoard(log_dir="./logs"),
        ]

    def _precision_m(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate precision metric"""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def _recall_m(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate recall metric"""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def _f1_m(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate F1 score"""
        precision = self._precision_m(y_true, y_pred)
        recall = self._recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def predict(self, X: np.ndarray, return_probabilities: bool = False) -> np.ndarray:
        """Make predictions with uncertainty estimation"""
        try:
            # Make predictions
            predictions = self.model.predict(X)

            if return_probabilities:
                return predictions

            # Get class predictions
            return np.argmax(predictions, axis=1)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, symbol: str, timeframe: str) -> bool:
        """Save model and configuration"""
        try:
            base_path = os.path.join(TradingConfig.MODEL_DIR, f"{symbol}_{timeframe}")
            os.makedirs(base_path, exist_ok=True)

            # Save model architecture and weights
            model_path = os.path.join(base_path, "model.h5")
            self.model.save(model_path)

            # Save best weights separately
            weights_path = os.path.join(base_path, "best_weights.h5")
            np.save(weights_path, self._best_weights)

            # Save configuration
            config_path = os.path.join(base_path, "config.pkl")
            joblib.dump(self.config, config_path)

            logger.info(f"Model saved successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, symbol: str, timeframe: str) -> bool:
        """Load model and configuration"""
        try:
            base_path = os.path.join(TradingConfig.MODEL_DIR, f"{symbol}_{timeframe}")

            # Load configuration
            config_path = os.path.join(base_path, "config.pkl")
            self.config = joblib.load(config_path)

            # Rebuild model
            self.build_model(self.config)

            # Load weights
            weights_path = os.path.join(base_path, "best_weights.h5")
            self._best_weights = np.load(weights_path, allow_pickle=True)
            self.model.set_weights(self._best_weights)

            logger.info(f"Model loaded successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
