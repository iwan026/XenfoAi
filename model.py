import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    MultiHeadAttention,
    Concatenate,
    Activation,
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
from sklearn.preprocessing import RobustScaler
import logging
import joblib
from config import TradingConfig

logger = logging.getLogger(__name__)


class DeepForexModel:
    """Deep Learning model for forex price prediction"""

    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.selected_features = None

    def create_model_architecture(self, input_shape):
        """Create hybrid CNN-LSTM model with attention mechanism"""
        try:
            # Input layer
            input_layer = Input(shape=input_shape, dtype=tf.float32)  # Specify dtype

            # CNN branch
            conv = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
            conv = BatchNormalization()(conv)
            conv = Activation("relu")(conv)
            conv = MaxPooling1D(pool_size=2)(conv)

            # LSTM branch
            lstm = LSTM(128, return_sequences=True)(input_layer)
            lstm = BatchNormalization()(lstm)
            lstm = Dropout(0.3)(lstm)
            lstm = LSTM(64)(lstm)

            # Self-attention mechanism
            attention = MultiHeadAttention(num_heads=4, key_dim=32)(
                input_layer, input_layer
            )
            attention = GlobalAveragePooling1D()(attention)

            # Combine branches
            combined = Concatenate()([GlobalMaxPooling1D()(conv), lstm, attention])

            # Dense layers with regularization
            x = Dense(64, kernel_regularizer=l2(0.01))(combined)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Dropout(0.4)(x)

            # Output layer
            output = Dense(1, activation="sigmoid")(x)

            # Create model
            model = Model(inputs=input_layer, outputs=output)

            # Compile model with updated loss function
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss=self._weighted_binary_crossentropy,
                metrics=["accuracy"],
            )

            self.model = model
            return model

        except Exception as e:
            logger.error(f"Error creating model architecture: {str(e)}")
            return None

    def _weighted_binary_crossentropy(self, y_true, y_pred):
        """Custom loss function with explicit type casting"""
        try:
            # Explicitly cast inputs to float32
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)

            # Weight for positive class
            pos_weight = 2.0

            # Calculate weighted binary crossentropy
            epsilon = tf.constant(1e-7, dtype=tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            loss = -tf.reduce_mean(
                pos_weight * y_true * tf.math.log(y_pred)
                + (1 - y_true) * tf.math.log(1 - y_pred)
            )

            return loss

        except Exception as e:
            logger.error(f"Error in loss function: {str(e)}")
            return None

    def _precision_m(self, y_true, y_pred):
        """Calculate precision metric"""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def _recall_m(self, y_true, y_pred):
        """Calculate recall metric"""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def _f1_m(self, y_true, y_pred):
        """Calculate F1 score"""
        precision = self._precision_m(y_true, y_pred)
        recall = self._recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def save_model(self, symbol, timeframe):
        """Save model and scaler"""
        try:
            model_path = TradingConfig.get_model_path(symbol, timeframe)
            scaler_path = TradingConfig.get_scaler_path(symbol, timeframe)

            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)

            logger.info(f"Model saved successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, symbol, timeframe):
        """Load saved model and scaler"""
        try:
            model_path = TradingConfig.get_model_path(symbol, timeframe)
            scaler_path = TradingConfig.get_scaler_path(symbol, timeframe)

            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                logger.error(f"Model files not found for {symbol}_{timeframe}")
                return False

            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "weighted_binary_crossentropy": self._weighted_binary_crossentropy,
                    "precision_m": self._precision_m,
                    "recall_m": self._recall_m,
                    "f1_m": self._f1_m,
                },
            )
            self.scaler = joblib.load(scaler_path)

            logger.info(f"Model loaded successfully for {symbol}_{timeframe}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
