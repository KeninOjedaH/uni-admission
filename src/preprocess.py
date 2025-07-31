# src/preprocess.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import NUMERIC_COLS, CATEGORICAL_COLS

def cargar_y_preparar_datos(path: str):
    df = pd.read_excel(path)

    columnas_irrelevantes = [
        'IDHASH', 'COLEGIO', 'COLEGIO_PROV', 'COLEGIO_DIST', 'COLEGIO_PAIS',
        'DOMICILIO_PROV', 'DOMICILIO_DIST', 'NACIMIENTO_PAIS',
        'NACIMIENTO_DEPA', 'NACIMIENTO_PROV', 'NACIMIENTO_DIST'
    ]
    df = df.drop(columns=[col for col in columnas_irrelevantes if col in df.columns])

    columnas_importantes = CATEGORICAL_COLS + ['INGRESO']
    df = df.dropna(subset=columnas_importantes)

    # Agrupar categorías poco frecuentes en 'OTROS'
    umbral = 0.01 * len(df)
    for col in ['MODALIDAD', 'COLEGIO_DEPA', 'DOMICILIO_DEPA']:
        frecs = df[col].value_counts()
        categorias_menores = frecs[frecs < umbral].index
        df[col] = df[col].replace(categorias_menores, 'OTROS')

    return df

def construir_pipeline():
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS)
    ])

    return preprocessor
