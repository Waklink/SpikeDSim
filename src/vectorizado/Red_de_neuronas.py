import numpy as np
import cupy as cp
from .Neurona import Neurona

class Red_de_neuronas:
    """
    Representación de una red de neuronas, formada por un conjunto de neuronas y sus conexiones.

    Atributos:
        neuronas (ndarray): Lista de instancias de la clase Neurona.
        conexiones (ndarray): Lista de conexiones entre las neuronas, donde cada conexión es una tupla (pre, post, peso).
    """

    def __init__(self, neuronas: dict, conexiones: np.ndarray | int, backend: str = "numpy"):
        """
        Inicializa una instancia de la clase Red_de_Neuronas, con un conjunto de neuronas y sus conexiones.

        Args:
            neuronas (dict): Diccionario de neuronas a crear, donde la clave es una instancia de la clase Neurona y el valor
                             es la cantidad de neuronas de ese tipo.
            conexiones (np.ndarray | int): Lista de conexiones entre las neuronas, donde cada conexión es una lista on los pesos de la
                                     conexión equivalente a la neurona actual; o entero que indica el número de conexiones aleatorias
                                     a crear entre las neuronas.
            backend (str, optional): Nombre del backend a utilizar, puede ser "numpy" o "cupy".
        """
        if backend == "numpy":
            self.xp = np
        elif backend == "cupy":
            self.xp = cp
        else:
            raise ValueError("El parámetro 'backend' debe ser 'numpy' o 'cupy'.")

        self.a = self.xp.zeros(sum(neuronas.values()))
        self.b = self.xp.zeros(sum(neuronas.values()))
        self.c = self.xp.zeros(sum(neuronas.values()))
        self.d = self.xp.zeros(sum(neuronas.values()))
        self.v = self.xp.zeros(sum(neuronas.values()))
        self.u = self.xp.zeros(sum(neuronas.values()))
        indice_actual = 0
        for neurona, cantidad in neuronas.items():
            a, b, c, d = neurona.get_parametros()
            v, u = neurona.get_estado()
            self.a[indice_actual:indice_actual + cantidad] = a
            self.b[indice_actual:indice_actual + cantidad] = b
            self.c[indice_actual:indice_actual + cantidad] = c
            self.d[indice_actual:indice_actual + cantidad] = d
            self.v[indice_actual:indice_actual + cantidad] = v
            self.u[indice_actual:indice_actual + cantidad] = u
            indice_actual += cantidad
        if isinstance(conexiones, int):
            n = sum(neuronas.values())
            self.conexiones = self.xp.random.randn(n, n) * (1.0 / n)
            self.xp.fill_diagonal(self.conexiones, 0)
        elif isinstance(conexiones, self.xp.ndarray):
            self.conexiones = conexiones
        else:
            raise ValueError("El parámetro 'conexiones' debe ser un entero o un array de " + self.xp.__name__ + ".")
    
    def get_datos(self) -> dict:
        return {
            "v": self.v,
            "u": self.u
        }
    
    def get_parametros(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d
        }
    
    def get_conexiones(self):
        return self.conexiones
