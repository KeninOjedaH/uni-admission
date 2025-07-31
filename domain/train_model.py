# domain/train_model.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import mlflow

from src.config import MODEL_OUT, DATA_PATH
from src.preprocess import cargar_y_preparar_datos, construir_pipeline
from adapters.output.model_storage import ModelStorage
from adapters.output.mlflow_logger import MLflowLogger

def entrenar_modelo():
    print("🚧 Iniciando entrenamiento del modelo...")

    df = cargar_y_preparar_datos("data/raw/Ingresantes_UNI_BAL_60_40.xlsx")
    df["INGRESO"] = df["INGRESO"].map({"NO": 0, "SI": 1})
    df.to_csv(DATA_PATH, index=False)

    X = df.drop(columns="INGRESO")
    y = df["INGRESO"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", construir_pipeline()),
        ("classifier", DecisionTreeClassifier(
            criterion="entropy",
            max_depth=None,
            min_samples_split=5,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("📊 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Ingresó (0)", "Ingresó (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    etiquetas = ["No Ingresó (0)", "Ingresó (1)"]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=etiquetas, yticklabels=etiquetas)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("📊 Matriz de Confusión")
    plt.tight_layout()
    plt.show()

    ModelStorage.guardar_modelo(pipeline, MODEL_OUT)

    with mlflow.start_run(run_name="DecisionTreeModel"):
        logger = MLflowLogger()
        logger.log_params(pipeline.named_steps["classifier"].get_params())
        logger.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        })
        logger.log_model(pipeline, model_name="DecisionTreeModel")
