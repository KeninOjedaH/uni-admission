# ports/input_port.py

from abc import ABC, abstractmethod
from typing import Dict

class AdmissionPredictionInputPort(ABC):
    @abstractmethod
    def predecir_ingreso(self, datos: Dict) -> Dict:
        """
        Recibe un diccionario con los datos del estudiante
        y devuelve un diccionario con:
        - probabilidad_ingreso (%)
        - prediccion (0 o 1)
        - riesgo ("Low", "Medium", "High")
        """
        pass

    @abstractmethod
    def obtener_opciones(self) -> Dict[str, list]:
        """
        Devuelve un diccionario con listas únicas
        para variables categóricas como:
        - COLEGIO_DEPA
        - MODALIDAD
        - ESPECIALIDAD
        - SEXO
        etc.
        """
        pass
