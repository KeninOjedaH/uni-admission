import pandas as pd
import numpy as np
import joblib
from typing import Dict, Union
from ports.input_port import AdmissionPredictionInputPort
from src.config import MODEL_OUT, DATA_PATH

def mapear_categorias_desconocidas(df: pd.DataFrame, categorias_validas: Dict[str, list]) -> pd.DataFrame:
    for col, valores_validos in categorias_validas.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in valores_validos else "OTROS")
    return df

class AdmissionPredictor(AdmissionPredictionInputPort):
    def __init__(self):
        self.model = joblib.load(MODEL_OUT)

        df = pd.read_csv(DATA_PATH)
        self.columnas_modelo = df.drop(columns="INGRESO").columns.tolist()

        # Extraer categorías válidas desde los datos de entrenamiento
        self.categorias_validas = {
            "MODALIDAD": df["MODALIDAD"].dropna().unique().tolist() if "MODALIDAD" in df else [],
            "COLEGIO_DEPA": df["COLEGIO_DEPA"].dropna().unique().tolist() if "COLEGIO_DEPA" in df else [],
            "DOMICILIO_DEPA": df["DOMICILIO_DEPA"].dropna().unique().tolist() if "DOMICILIO_DEPA" in df else [],
            "SEXO": df["SEXO"].dropna().unique().tolist() if "SEXO" in df else [],
            "ESPECIALIDAD": df["ESPECIALIDAD"].dropna().unique().tolist() if "ESPECIALIDAD" in df else []
        }

    def predecir_ingreso(self, datos: Union[Dict, pd.DataFrame]) -> Dict:
        if isinstance(datos, dict):
            df = pd.DataFrame([datos])
        elif isinstance(datos, pd.DataFrame):
            df = datos.copy()
        else:
            raise ValueError("El input debe ser un diccionario o un DataFrame")

        for col in self.columnas_modelo:
            if col not in df.columns:
                df[col] = 0

        df = mapear_categorias_desconocidas(df, self.categorias_validas)
        df = df[self.columnas_modelo]

        pred = self.model.predict(df)[0]
        return {"prediccion": "Sí" if pred == 1 else "No"}

    def obtener_opciones(self) -> Dict[str, list]:
        df = pd.read_csv(DATA_PATH)
        return {
            "COLEGIO_DIST": sorted(df["COLEGIO_DIST"].dropna().unique().tolist()) if "COLEGIO_DIST" in df else [],
            "COLEGIO_PROV": sorted(df["COLEGIO_PROV"].dropna().unique().tolist()) if "COLEGIO_PROV" in df else [],
            "COLEGIO_DEPA": sorted(df["COLEGIO_DEPA"].dropna().unique().tolist()) if "COLEGIO_DEPA" in df else [],
            "ESPECIALIDAD": sorted(df["ESPECIALIDAD"].dropna().unique().tolist()) if "ESPECIALIDAD" in df else [],
            "MODALIDAD": sorted(df["MODALIDAD"].dropna().unique().tolist()) if "MODALIDAD" in df else [],
            "SEXO": sorted(df["SEXO"].dropna().unique().tolist()) if "SEXO" in df else []
        }
