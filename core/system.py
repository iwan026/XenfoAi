import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from core.mt5_core import MT5Core
import os
import logging
import config

logger = logging.getLogger(__name__)


class XenfoAi:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.mt5_core = MT5Core()
        self.model_dir = config.MODEL_DIR
        
    def create_model(self):
        try: 
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(
                        100,
                        return_sequences=True,
                        input_shapes=(config.SEQUENCE_LENGTH, config.INPUT_FEATURES)
                    ),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.LSTM(100, return_sequences=False),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(50, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                    
                ]
            )
        
    
