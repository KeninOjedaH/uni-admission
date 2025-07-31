import sys
import os
import argparse
import subprocess

# Agregar la ruta raíz al path para importar correctamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar funciones
from domain.train_model import entrenar_modelo

def run_api():
    print("🚀 Iniciando API en http://localhost:8000")
    os.system("python -m uvicorn adapters.input.fastapi_adapter:app --reload")

def run_cli(extra_args):
    print("ℹ️ Ejecutando CLI...")
    subprocess.run(["python", "adapters/input/cli_adapter.py"] + extra_args)

def run_train():
    print("🚧 Entrenando modelo...")
    entrenar_modelo()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema de predicción de ingreso a la UNI")
    parser.add_argument("modo", choices=["api", "cli", "train"], help="Modo de ejecución")
    args, extra_args = parser.parse_known_args()

    if args.modo == "api":
        run_api()
    elif args.modo == "cli":
        run_cli(extra_args)
    elif args.modo == "train":
        run_train()
