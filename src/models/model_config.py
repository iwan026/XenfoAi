from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    # Konfigurasi umum
    SEQUENCE_LENGTH: int = 60
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    RANDOM_STATE: int = 42

    # XGBoost parameters
    XGB_PARAMS: Dict = field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.01,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "tree_method": "hist",  # Untuk efisiensi memori
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": ["logloss", "auc"],
            "early_stopping_rounds": 10,
        }
    )

    # CNN parameters
    CNN_FILTERS: List[int] = field(default_factory=lambda: [32, 64, 128])
    CNN_KERNEL_SIZES: List[int] = field(default_factory=lambda: [3, 3, 3])
    CNN_POOL_SIZES: List[int] = field(default_factory=lambda: [2, 2, 2])
    CNN_DROPOUT: float = 0.2

    # LSTM parameters
    LSTM_UNITS: List[int] = field(default_factory=lambda: [100, 50])
    LSTM_DROPOUT: float = 0.2

    # Transformer parameters
    TRANSFORMER_HEAD_SIZE: int = 256
    TRANSFORMER_NUM_HEADS: int = 4
    TRANSFORMER_FF_DIM: int = 4
    TRANSFORMER_DROPOUT: float = 0.2

    # Feature Engineering
    TECHNICAL_FEATURES: List[str] = field(
        default_factory=lambda: [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "sma_20",
            "sma_50",
            "ema_9",
            "ema_21",
            "atr_14",
            "adx_14",
            "cci_20",
            "stoch_k",
            "stoch_d",
        ]
    )

    PRICE_FEATURES: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    # Training
    EPOCHS: int = 50
    PATIENCE: int = 5
    LEARNING_RATE: float = 0.001

    # Resource Management
    MAX_MEMORY_GB: float = 6.0  # Batas penggunaan memori
    PARALLEL_JOBS: int = 2  # Jumlah core yang digunakan
