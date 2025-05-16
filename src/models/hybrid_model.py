import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import xgboost as xgb
from typing import Dict, Optional
import gc

from src.models.model_config import ModelConfig
from src.data.feature_engineering import FeatureEngineer


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class HybridForexModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.xgb_model = None
        self.deep_model = None
        self.build_models()

    def build_models(self):
        """Membangun arsitektur model hybrid"""
        try:
            # XGBoost untuk fitur teknikal
            self.xgb_model = xgb.XGBClassifier(
                **self.config.XGB_PARAMS, n_jobs=self.config.PARALLEL_JOBS
            )

            # Deep Learning model (CNN + LSTM + Transformer)
            # Input layers
            technical_input = layers.Input(
                shape=(
                    self.config.SEQUENCE_LENGTH,
                    len(self.config.TECHNICAL_FEATURES),
                ),
                name="technical_input",
            )

            price_input = layers.Input(
                shape=(self.config.SEQUENCE_LENGTH, len(self.config.PRICE_FEATURES), 1),
                name="price_input",
            )

            # CNN branch
            cnn = price_input
            for i, (filters, kernel_size, pool_size) in enumerate(
                zip(
                    self.config.CNN_FILTERS,
                    self.config.CNN_KERNEL_SIZES,
                    self.config.CNN_POOL_SIZES,
                )
            ):
                cnn = layers.Conv2D(
                    filters,
                    kernel_size,
                    activation="relu",
                    padding="same",
                    name=f"conv2d_{i}",
                )(cnn)
                cnn = layers.MaxPooling2D(pool_size, name=f"maxpool_{i}")(cnn)
                cnn = layers.BatchNormalization(name=f"batchnorm_{i}")(cnn)

            cnn = layers.Flatten(name="flatten")(cnn)
            cnn = layers.Dense(64, activation="relu", name="cnn_dense")(cnn)
            cnn = layers.Dropout(self.config.CNN_DROPOUT, name="cnn_dropout")(cnn)

            # LSTM branch
            lstm = technical_input
            for i, units in enumerate(self.config.LSTM_UNITS):
                lstm = layers.LSTM(
                    units,
                    return_sequences=True
                    if i < len(self.config.LSTM_UNITS) - 1
                    else False,
                    dropout=self.config.LSTM_DROPOUT,
                    name=f"lstm_{i}",
                )(lstm)

            # Transformer branch
            transformer_block = TransformerBlock(
                self.config.TRANSFORMER_HEAD_SIZE,
                self.config.TRANSFORMER_NUM_HEADS,
                self.config.TRANSFORMER_FF_DIM,
                self.config.TRANSFORMER_DROPOUT,
            )

            transformer = transformer_block(technical_input)
            transformer = layers.GlobalAveragePooling1D(name="transformer_pool")(
                transformer
            )

            # Combine all branches
            combined = layers.concatenate([cnn, lstm, transformer], name="concatenate")
            combined = layers.Dense(32, activation="relu", name="combined_dense")(
                combined
            )
            combined = layers.Dropout(0.2, name="combined_dropout")(combined)
            output = layers.Dense(1, activation="sigmoid", name="output")(combined)

            # Create model
            self.deep_model = Model(
                inputs=[technical_input, price_input],
                outputs=output,
                name="hybrid_forex_model",
            )

            # Compile model
            self.deep_model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config.LEARNING_RATE),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

        except Exception as e:
            raise Exception(f"Error dalam pembuatan model: {str(e)}")

    def train(self, train_data: Dict, validation_data: Optional[Dict] = None):
        """Training model hybrid"""
        try:
            # Training XGBoost
            print("Training XGBoost model...")
            technical_features_reshaped = train_data["technical_features"].reshape(
                train_data["technical_features"].shape[0], -1
            )

            if validation_data:
                val_technical_features_reshaped = validation_data[
                    "technical_features"
                ].reshape(validation_data["technical_features"].shape[0], -1)
                eval_set = [
                    (val_technical_features_reshaped, validation_data["target"])
                ]
            else:
                eval_set = None

            self.xgb_model.fit(
                technical_features_reshaped,
                train_data["target"],
                eval_set=eval_set,
                verbose=True,
            )

            # Clear memory
            gc.collect()

            # Training Deep Learning model
            print("Training Deep Learning model...")
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_data else "loss",
                    patience=self.config.PATIENCE,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if validation_data else "loss",
                    factor=0.5,
                    patience=3,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    "best_model.h5",
                    monitor="val_loss" if validation_data else "loss",
                    save_best_only=True,
                ),
            ]

            history = self.deep_model.fit(
                [train_data["technical_features"], train_data["price_features_cnn"]],
                train_data["target"],
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_data=(
                    [
                        validation_data["technical_features"],
                        validation_data["price_features_cnn"],
                    ],
                    validation_data["target"],
                )
                if validation_data
                else None,
                callbacks=callbacks,
                verbose=1,
            )

            return history

        except Exception as e:
            raise Exception(f"Error dalam training model: {str(e)}")

    def predict(self, data: Dict) -> np.ndarray:
        """Membuat prediksi menggunakan model hybrid"""
        try:
            # XGBoost prediction
            xgb_pred = self.xgb_model.predict_proba(
                data["technical_features"].reshape(
                    data["technical_features"].shape[0], -1
                )
            )[:, 1]

            # Deep Learning prediction
            deep_pred = self.deep_model.predict(
                [data["technical_features"], data["price_features_cnn"]],
                batch_size=self.config.BATCH_SIZE,
                verbose=0,
            ).flatten()

            # Combine predictions (simple average)
            final_pred = (xgb_pred + deep_pred) / 2

            return final_pred

        except Exception as e:
            raise Exception(f"Error dalam pembuatan prediksi: {str(e)}")
