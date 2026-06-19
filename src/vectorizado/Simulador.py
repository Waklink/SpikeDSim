from .Neurona import Neurona
from .Red_de_neuronas import Red_de_neuronas
import time
import numpy as np
import cupy as cp

class Simulador:
    """
    """

    def __init__(self, paso_temporal: float, tiempo_total: int):
        """
        """
        self.dt = paso_temporal
        self.tiempo_total = tiempo_total
        self.pasos_totales = self.dt * self.tiempo_total
        self.registro = []

    def cargar_red(self, red: Red_de_neuronas) -> None:
        """
        """
        self.es_red = True
        self.red = red
        self.registro.append(self.red.get_datos())
    
    def cargar_neurona(self, neurona: Neurona) -> None:
        """
        """
        self.es_red = False
        self.neurona = neurona
        self.registro.append(neurona.get_estado())
