import cupy as cp
import numpy as np
from .Neurona import Neurona
from typing import TypeAlias, Literal
import numpy.typing as npt

Array: TypeAlias = np.ndarray | cp.ndarray

class RedDeNeuronas:
    """
    Representación de una red de neuronas, formada por un conjunto de neuronas basadas en el modelo de Izhikevich y sus conexiones.

    Attributes:
        xp (Module): Librería a usar para realizar los cálculos, puede ser numpy o cupy, dependiendo de si se quiere usar la cpu o la gpu.
            neuronas (dict[Neurona, int]): diccionario con el número de cada tipo de neurona.
        a (Array): Array con el parámetro que regula la velocidad de recuperación de cada neurona.
        b (Array): Array con el parámetro que regula la sensibilidad de cada neurona al estímulo.
        c (Array): Array con el parámetro que regula el potencial de membrana de recuperación de cada neurona.
        d (Array): Array con el parámetro que regula la velocidad de recuperación de la variable de recuperación de cada neurona.
        v (Array): Array con el potencial de membrana de cada neurona.
        u (Array): Array con la variable de recuperación de cada neurona.
        conexiones (Array): Array con los pesos de las conexiones de cada neurona. Un 0 representa que no hay conexión, cada columna representa
                    las neuronas presinápticas, mientras que las filas son las neuronas postsinápticas.
    """

    def __init__(self, neuronas: dict[Neurona, int], conexiones: Array| list[list[float]] | int, backend: Literal["numpy", "cupy"] = "numpy"):
        """
        Inicializa una instancia de la clase Red_de_Neuronas, con un conjunto de neuronas y sus conexiones.

        Args:
            neuronas (dict[Neurona, int]): Diccionario de neuronas a crear:
                - Clave (Neurona): Instancia de la clase Neurona a usar para obtener los parámetros a usar.
                - Valor (int): Número de neuronas a crear de cada tipo.
            conexiones (Array | list[list[float]] | int): Matriz de conexiones entre las neuronas, donde cada fila es una lista con los pesos de las
                        conexiones de la neurona actual; o entero que indica el número de conexiones aleatorias
                        a crear entre las neuronas.
            backend (Literal["numpy", "cupy"], optional): Nombre del backend a utilizar, puede ser "numpy" o "cupy".
        
        Raises:
            ValueError: Si no se ha pasado correctamente algún parámetro, las dimensiones de las conexiones no son correctas o su diagonal no es 0.
        """

        if backend == "numpy":
            self.__xp = np
        elif backend == "cupy":
            self.__xp = cp
        else:
            raise ValueError("El parámetro 'backend' debe ser 'numpy' o 'cupy'.")
        
        self.__neuronas = neuronas

        self.__num_neuronas = sum(neuronas.values())

        self.__a = self.__xp.zeros(self.__num_neuronas)
        self.__b = self.__xp.zeros(self.__num_neuronas)
        self.__c = self.__xp.zeros(self.__num_neuronas)
        self.__d = self.__xp.zeros(self.__num_neuronas)
        self.__v = self.__xp.zeros(self.__num_neuronas)
        self.__u = self.__xp.zeros(self.__num_neuronas)
        self.__tipo = self.__xp.zeros(self.__num_neuronas, bool)
        indice_actual = 0
        for neurona, cantidad in neuronas.items():
            a, b, c, d = neurona.get_parametros()
            v, u = neurona.get_estado()
            self.__a[indice_actual:indice_actual + cantidad] = a
            self.__b[indice_actual:indice_actual + cantidad] = b
            self.__c[indice_actual:indice_actual + cantidad] = c
            self.__d[indice_actual:indice_actual + cantidad] = d
            self.__v[indice_actual:indice_actual + cantidad] = v
            self.__u[indice_actual:indice_actual + cantidad] = u
            self.__tipo[indice_actual:indice_actual + cantidad] = neurona.es_excitatoria()
            indice_actual += cantidad

        if isinstance(conexiones, int):
            self.crear_conexiones_aleatorias(conexiones)
        elif isinstance(conexiones, list):
            self.__conexiones = self.__xp.asarray(conexiones)
        elif isinstance(conexiones, Array):
            self.__conexiones = conexiones
        else:
            raise ValueError("El parámetro 'conexiones' debe ser un entero, una matriz formada por listas, o un array en forma de matriz cuadrada.")
        
        if self.__conexiones.shape != (self.__num_neuronas, self.__num_neuronas):
            raise ValueError(
                "Dimensiones incorrectas."
            )

        if self.__xp.any(self.__xp.diag(self.__conexiones)):
            raise ValueError(
                "La diagonal debe ser cero."
            )

    def crear_conexiones_aleatorias(self, num:int) -> None:
        """
        Crear un número especificado de conexiones con pesos aleatorios.

        Args:
            num (int): Número de conexiones a crear.
        
        Raises:
            ValueError: Si las conexiones a crear son mayores que el número máximo posible de conexiones.
        """
        
        xp = self.__xp

        # Inicializar matriz
        self.__conexiones = xp.zeros((self.__num_neuronas, self.__num_neuronas))

        # Máximo número de conexiones posibles (sin diagonal)
        max_conexiones = self.__num_neuronas * (self.__num_neuronas - 1)
        if num > max_conexiones:
            raise ValueError(
                f"No se pueden crear {num} conexiones. "
                f"Máximo permitido: {max_conexiones}"
            )
        
        # Seleccionar posiciones aleatorias fuera de la diagonal
        posibles = xp.arange(max_conexiones)

        seleccion = xp.random.choice(posibles, size=num, replace=False)

        # Convertir índices lineales → (fila, columna)
        filas = seleccion // (self.__num_neuronas - 1)
        columnas = seleccion % (self.__num_neuronas - 1)

        # Ajustar para saltar diagonal
        columnas = xp.where(columnas >= filas, columnas + 1, columnas)

        # Generar pesos según tipo de neurona presináptica (columnas)
        pesos = xp.random.random(num)

        pesos = xp.where(
            self.__tipo[columnas],   # excitatoria
            pesos,                   # [0,1)
            -pesos                   # (-1,0]
        )

        # Asignar conexiones
        self.__conexiones[filas, columnas] = pesos


    def actualizar(self, I: Array, dt: float = 0.5) -> Array:
        """
        Actualizar el estado de las neuronas.

        Args:
            I (Array): El input de corriente a introducir en las neuronas.
            dt (float): El paso temporal a usar.
        
        Returns:
            Array: Un vector con los spikes que han habido en el paso anterior.
        """

        es_spike = (self.__v >= 30)

        I_total = I + self.__xp.dot(self.__conexiones, es_spike)

        self.__v += dt * ((0.04 * self.__v**2 + 5 * self.__v + 140 - self.__u + I_total))
        self.__u += dt * (self.__a * (self.__b * self.__v - self.__u))

        if self.__xp.any(es_spike):
            self.__v[es_spike] = self.__c[es_spike]
            self.__u[es_spike] += self.__d[es_spike]
        return es_spike


    def get_datos(self) -> dict[str, Array]:
        """
        Recuperar los datos del estado de las neuronas.

        Returns:
            dict[str, Array]: Un diccionario con el potencial de membrana y la variable de recuperación de cada neurona de la red.
        """
        return {
            "v": self.__v.copy(),
            "u": self.__u.copy()
        }


    def get_parametros(self) -> dict[str, Array]:
        """
        Recuperar los parámetros de las neuronas

        Returns:
            dict[str, Array]: Un diccionario con los cuatro parámetros del modelo de Izhikevich de cada neurona de la red.
        """
        return {
            "a": self.__a.copy(),
            "b": self.__b.copy(),
            "c": self.__c.copy(),
            "d": self.__d.copy()
        }


    @property
    def neuronas(self) -> dict[Neurona, int]:
        """
        Las neuronas y el número de cada neurona que hay en la red.
        """
        return self.__neuronas.copy()


    @property
    def conexiones(self) -> Array:
        """
        Matriz de conexiones de la red.

        Cada columna representa una neurona presináptica y cada fila una postsináptica.
        La diagonal contiene siempre ceros.
        """
        return self.__conexiones.copy()
