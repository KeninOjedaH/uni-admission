# adapters/output/model_storage.py

import joblib
from typing import Any

class ModelStorage:
    @staticmethod
    def guardar_modelo(modelo: Any, ruta: str):
        joblib.dump(modelo, ruta)
        print(f"💾 Modelo guardado en: {ruta}")

    @staticmethod
    def cargar_modelo(ruta: str) -> Any:
        modelo = joblib.load(ruta)
        print(f"📂 Modelo cargado desde: {ruta}")
        return modelo
