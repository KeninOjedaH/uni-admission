# adapters/input/cli_adapter.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import json
from domain.use_cases import AdmissionPredictor

def main():
    predictor = AdmissionPredictor()

    parser = argparse.ArgumentParser(description="Predicción de ingreso a la UNI")
    parser.add_argument("--data", type=str, required=True,
                        help="Ruta al archivo JSON con los datos del estudiante")
    args = parser.parse_args()

    # Leer datos desde el archivo JSON
    with open(args.data, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Realizar predicción
    resultado = predictor.predecir_ingreso(input_data)

    print("📊 Resultado de la predicción:")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
