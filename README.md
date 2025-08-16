# 🎓 uni-admission

Sistema de predicción del ingreso a la Universidad Nacional de Ingeniería (UNI) utilizando Machine Learning, con arquitectura hexagonal y principios de MLOps.

---

## 📌 Descripción

Este proyecto permite predecir si un postulante ingresará a la UNI, usando características como especialidad, modalidad de ingreso, calificaciones y más. El modelo está implementado con XGBoost, empaquetado como una API REST mediante FastAPI, y gestionado con MLflow para el seguimiento de experimentos.

---

## 🧱 Arquitectura

- 🔷 **Arquitectura Hexagonal**: Separación entre lógica de negocio, entradas y salidas.
- ⚙️ **MLOps Integrado**: MLflow para gestionar modelos, parámetros y métricas.
- 🐳 **Docker Ready**: Contenedorizado para despliegue local o en la nube.

```
src/
├── domain/               # Lógica principal y entrenamiento del modelo
├── adapters/             # Interfaces de entrada (CLI, API) y salida (MLflow logger)
├── ports/                # Puertos (interfaces) para entrada/salida
├── config/               # Rutas y configuración general
├── app/                  # Punto de entrada del sistema
```

---

## 🚀 Instrucciones Rápidas

### 1. Clonar el repositorio

```bash
git clone https://github.com/IngresoUNI/uni-admission.git
cd uni-admission
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate     # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Entrenar el modelo

```bash
python app/main.py train
```

Se entrenará el modelo XGBoost y se guardará en `models/arbol_decision_model.pkl`. Además, se registrará en MLflow.

### 4. Ejecutar predicción por CLI

```bash
python app/main.py cli --data estudiante.json
```

Ejemplo de `estudiante.json`:

```json
{
  "COLEGIO": "COLEGIO NACIONAL",
  "COLEGIO_DIST": "MIRAFLORES",
  "COLEGIO_PROV": "LIMA",
  "COLEGIO_DEPA": "LIMA",
  "COLEGIO_ANIO_EGRESO": 2023,
  "DOMICILIO_DEPA": "LIMA",
  "ESPECIALIDAD": "INGENIERIA DE SISTEMAS",
  "MODALIDAD": "ORDINARIO",
  "SEXO": "MASCULINO",
  "ANIO_POSTULA": 2024,
  "ANIO_NACIMIENTO": 2006,
  "h_e_Matemática": 16.5,
  "h_e_fisica_quimica": 14.2,
  "h_e_Aptitud": 15.0,
  "calificacion": 15.8
}
```

---

## 📈 MLflow Tracking

Para visualizar los experimentos y métricas registrados:

```bash
mlflow ui
```

Luego visita: [http://localhost:5000](http://localhost:5000)

---

## 🐳 Despliegue con Docker

(Una vez tengas Docker Desktop listo)

```bash
docker compose up --build
```

Visita: [http://localhost:8000/docs](http://localhost:8000/docs) para interactuar con la API REST.

---

## 🧪 Stack tecnológico

- 🧠 XGBoost (modelo de predicción)
- ⚙️ scikit-learn + pandas + numpy
- 🚀 FastAPI
- 📊 MLflow
- 🐳 Docker & docker-compose
- 🧩 Arquitectura hexagonal

---


## 💡 Autor

Kenin Ojeda 

## 📄 Licencia

Este proyecto está bajo la licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
