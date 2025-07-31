# src/config.py

# Ruta de datos y modelo
DATA_PATH = "data/processed/datos_procesados_ingreso_uni.csv"
MODEL_OUT = "models/arbol_decision_model.pkl"

# Variable objetivo
TARGET = "INGRESO"

# Variables numéricas y categóricas seleccionadas
NUMERIC_COLS = ["h_e_Aptitud", "calificacion"]
CATEGORICAL_COLS = [
    "SEXO",
    "MODALIDAD",
    "ESPECIALIDAD",
    "COLEGIO_DEPA",
    "DOMICILIO_DEPA"
]

# 🔥 Todas las features finales (en el orden correcto)
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
