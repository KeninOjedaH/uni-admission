# adapters/input/fastapi_adapter.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from domain.use_cases import AdmissionPredictor
from src.config import FEATURES  # solo se usan las finales
import pandas as pd
import traceback

app = FastAPI()
predictor = AdmissionPredictor()

# Recibe todo lo que el formulario manda, aunque no se use todo
class EstudianteInput(BaseModel):
    COLEGIO: str
    COLEGIO_DIST: str
    COLEGIO_PROV: str
    COLEGIO_DEPA: str
    COLEGIO_ANIO_EGRESO: int
    DOMICILIO_DEPA: str
    ESPECIALIDAD: str
    MODALIDAD: str
    SEXO: str
    ANIO_POSTULA: int
    ANIO_NACIMIENTO: int
    h_e_Matemática: float
    h_e_fisica_quimica: float
    h_e_Aptitud: float
    h_total_semana: float
    calificacion: float  # ✅ Esta sí la usas

@app.post("/predecir")
def predecir_estudiante(data: EstudianteInput) -> Dict:
    try:
        # Convertimos todo a DataFrame
        df_input = pd.DataFrame([data.dict()])

        # Nos quedamos SOLO con las variables usadas por el modelo
        df_filtrado = df_input[FEATURES]

        # Predictor usa solo esas columnas
        resultado = predictor.predecir_ingreso(df_filtrado)

        return resultado

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
