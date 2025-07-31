# adapters/output/mlflow_logger.py

import mlflow
import mlflow.sklearn
from typing import Dict, Any

class MLflowLogger:
    def __init__(self, experiment_name: str = "IngresoUNI"):
        mlflow.set_experiment(experiment_name)

    def log_model(self, model, model_name: str = "DecisionTreeModel"):
        mlflow.sklearn.log_model(model, "model")
        print(f"📦 Modelo registrado en MLflow bajo el nombre: {model_name}")

    def log_metrics(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        print(f"📈 Métricas registradas: {metrics}")

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"⚙️ Parámetros registrados: {params}")
