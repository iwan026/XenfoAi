import optuna
import numpy as np
from typing import Dict, Tuple
import logging
from datetime import datetime
from src.models.model_config import ModelConfig
from src.utils.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.resource_monitor = ResourceMonitor()
        self.best_params = None

    def objective(self, trial: optuna.Trial, train_data: Dict, val_data: Dict) -> float:
        """Objective function untuk optimasi"""
        try:
            # Cek resource
            resources = self.resource_monitor.check_resources()
            if not resources["memory_ok"] or not resources["cpu_ok"]:
                raise optuna.TrialPruned()

            # Parameter XGBoost
            xgb_params = {
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "xgb_learning_rate", 1e-4, 1e-1, log=True
                ),
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 200),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "xgb_colsample_bytree", 0.5, 1.0
                ),
            }

            # Parameter CNN
            cnn_params = {
                "filters": trial.suggest_int("cnn_filters", 16, 128),
                "kernel_size": trial.suggest_int("cnn_kernel_size", 2, 5),
                "pool_size": trial.suggest_int("cnn_pool_size", 2, 4),
                "dropout": trial.suggest_float("cnn_dropout", 0.1, 0.5),
            }

            # Parameter LSTM
            lstm_params = {
                "units": trial.suggest_int("lstm_units", 32, 256),
                "dropout": trial.suggest_float("lstm_dropout", 0.1, 0.5),
            }

            # Parameter Transformer
            transformer_params = {
                "head_size": trial.suggest_int("transformer_head_size", 32, 256),
                "num_heads": trial.suggest_int("transformer_num_heads", 2, 8),
                "ff_dim": trial.suggest_int("transformer_ff_dim", 2, 8),
                "dropout": trial.suggest_float("transformer_dropout", 0.1, 0.5),
            }

            # Training parameters
            batch_size = trial.suggest_int("batch_size", 16, 128)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

            # Update config
            config = ModelConfig()
            config.XGB_PARAMS.update(xgb_params)
            config.CNN_FILTERS = [cnn_params["filters"]]
            config.CNN_KERNEL_SIZES = [cnn_params["kernel_size"]]
            config.CNN_POOL_SIZES = [cnn_params["pool_size"]]
            config.CNN_DROPOUT = cnn_params["dropout"]
            config.LSTM_UNITS = [lstm_params["units"]]
            config.LSTM_DROPOUT = lstm_params["dropout"]
            config.TRANSFORMER_HEAD_SIZE = transformer_params["head_size"]
            config.TRANSFORMER_NUM_HEADS = transformer_params["num_heads"]
            config.TRANSFORMER_FF_DIM = transformer_params["ff_dim"]
            config.TRANSFORMER_DROPOUT = transformer_params["dropout"]
            config.BATCH_SIZE = batch_size
            config.LEARNING_RATE = learning_rate

            # Train model dengan parameter baru
            from src.models.hybrid_model import HybridForexModel

            model = HybridForexModel(config)
            history = model.train(train_data, val_data)

            # Evaluasi
            val_accuracy = history.history["val_accuracy"][-1]

            return val_accuracy

        except Exception as e:
            logger.error(f"Error in optimization objective: {e}")
            raise optuna.TrialPruned()

    def optimize(
        self, train_data: Dict, val_data: Dict, n_trials: int = 50
    ) -> Tuple[Dict, float]:
        """Jalankan optimasi hyperparameter"""
        try:
            study = optuna.create_study(
                direction="maximize", pruner=optuna.pruners.MedianPruner()
            )

            study.optimize(
                lambda trial: self.objective(trial, train_data, val_data),
                n_trials=n_trials,
                timeout=3600,  # 1 jam
            )

            self.best_params = study.best_params
            best_value = study.best_value

            # Simpan hasil optimasi
            self.save_optimization_results(study)

            return self.best_params, best_value

        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return None, 0.0

    def save_optimization_results(self, study: optuna.Study) -> None:
        """Simpan hasil optimasi"""
        try:
            results = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "best_params": study.best_params,
                "best_value": study.best_value,
                "n_trials": len(study.trials),
                "optimization_history": [
                    {"trial": t.number, "value": t.value, "params": t.params}
                    for t in study.trials
                    if t.value is not None
                ],
            }

            save_path = f"optimization_results_{self.symbol}_{self.timeframe}.json"
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
