import cupy as cp
import numpy as np
import scipy.sparse as sp
import cupyx.scipy.sparse as cpsp
from .Neurona import Neurona
from typing import TypeAlias, Literal

Array: TypeAlias = np.ndarray | cp.ndarray
SparseArray: TypeAlias = sp.csr_matrix | cpsp.csr_matrix

class RedDeNeuronas:
    """
    Red de neuronas basada en el modelo de Izhikevich con soporte para ejecución en CPU o GPU.

    Esta clase representa una red de neuronas compuesta por varios vectores de parámetros de
    neuronas y una matriz de conexiones que define la influencia sináptica entre ellas.
    El backend de cálculo se puede seleccionar entre NumPy para CPU y CuPy para GPU.

    Attributes
    ----------
    backend : str
        Cadena que describe el backend usado para los cálculos.
    uso_gpu: bool
        Indica de forma directa si se utiliza la GPU como backend o, por el contrario, la CPU.
    neuronas : dict[Neurona, int]
        Tipos de neuronas y su cantidad en la red.
    conexiones : Array | SparseArray
        Matriz de pesos sinápticos con diagonal cero. Puede almacenarse como una matriz densa o
        mediante una representación dispersa CSR.
    num_neuronas : int
        Número total de neuronas en la red.
    num_conexiones : int
        Número total de conexiones diferentes de cero.
    estado : dict[str, Array]
        Estado actual de la red con v y u.
    parametros : dict[str, Array]
        Parámetros del modelo Izhikevich (a, b, c, d) para todas las neuronas.
    dtype : np.dtype | cp.dtype
        Tipo de dato utilizado internamente por los arrays de la red. Puede ser float32 o
        float64 según la precisión seleccionada en el constructor.
    sparse: bool
        Indica si la matriz de conexiones se almacena utilizando una representación dispersa CSR.
    """

    def __init__(self, neuronas: dict[Neurona, int], conexiones: Array | list[list[float]] | int,
                 backend: Literal["numpy", "cupy"] = "numpy", precision: Literal[32, 64] = 32,
                 sparse: bool = True):
        """
        Inicializa una instancia de la clase RedDeNeuronas con un conjunto de neuronas y sus conexiones.

        Parameters
        ----------
        neuronas : dict[Neurona, int]
            Diccionario de neuronas a crear. La clave es una instancia de Neurona y el valor es el número
            de neuronas de ese tipo.
        conexiones : Array | list[list[float]] | int
            Matriz de conexiones entre las neuronas, donde cada fila es una lista con los pesos de las
            conexiones de la neurona actual; o un entero que indica el número de conexiones aleatorias a crear.
        backend : Literal["numpy", "cupy"], optional
            Nombre del backend a utilizar. Utilizar "numpy" para CPU o "cupy" para GPU.
        precision : Literal[32, 64], optional
            Tamaño, en bits, en el que se guardan los valores de los arrays. La precisión por defecto es 32 bits
            (float32).
        sparse : bool, optional
            Indica si la matriz de conexiones se almacena utilizando una representación CSR dispersa. Por defecto es True.

        Raises
        ------
        ValueError
            Si algún parámetro no es válido, las dimensiones de las conexiones son incorrectas o la diagonal
            no es cero.
        """

        if backend == "numpy":
            self.__xp = np
            self.__sp = sp
        elif backend == "cupy":
            self.__xp = cp
            self.__sp = cpsp
        else:
            raise ValueError("El parámetro 'backend' debe ser 'numpy' o 'cupy'.")
        
        self.__uso_gpu = backend == "cupy"
        self.__sparse = sparse

        if precision == 32:
            self.__dtype = self.__xp.float32
        elif precision == 64:
            self.__dtype = self.__xp.float64
        else:
            raise ValueError("El dtype debe ser 32 o 64.")
                
        for cantidad in neuronas.values():
            if not isinstance(cantidad, int) or cantidad < 0:
                raise ValueError("Las cantidades de las neuronas deben ser enteros positivos.")
        
        self.__neuronas = neuronas.copy()

        self.__num_neuronas = sum(neuronas.values())

        self.__a = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__b = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__c = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__d = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__v = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__u = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__tipo = self.__xp.empty(self.__num_neuronas, bool)
        indice_actual = 0
        for neurona, cantidad in neuronas.items():
            a, b, c, d = neurona.parametros
            v, u = neurona.estado
            self.__a[indice_actual:indice_actual + cantidad] = a
            self.__b[indice_actual:indice_actual + cantidad] = b
            self.__c[indice_actual:indice_actual + cantidad] = c
            self.__d[indice_actual:indice_actual + cantidad] = d
            self.__v[indice_actual:indice_actual + cantidad] = v
            self.__u[indice_actual:indice_actual + cantidad] = u
            self.__tipo[indice_actual:indice_actual + cantidad] = neurona.es_excitatoria
            indice_actual += cantidad

        self.__v_inicial = self.__v.copy()
        self.__u_inicial = self.__u.copy()

        if isinstance(conexiones, int):
            self.__crear_conexiones_aleatorias(conexiones)
        elif isinstance(conexiones, list):
            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix(conexiones, dtype=self.__dtype)
            else:
                self.__conexiones = self.__xp.asarray(conexiones, dtype=self.__dtype)

        elif isinstance(conexiones, (np.ndarray, cp.ndarray)):
            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix(conexiones, dtype=self.__dtype)
            else:
                self.__conexiones = self.__xp.asarray(conexiones, dtype=self.__dtype)
        else:
            raise ValueError("El parámetro 'conexiones' debe ser un entero, una matriz formada por listas \
                             o un array en forma de matriz cuadrada.")
        
        if self.__conexiones.shape != (self.__num_neuronas, self.__num_neuronas):
            raise ValueError("Dimensiones incorrectas.")

        if self.__sparse:
            if self.__conexiones.diagonal().any():
                raise ValueError("La diagonal debe ser cero.")
        else:
            if self.__xp.any(self.__xp.diag(self.__conexiones)):
                raise ValueError("La diagonal debe ser cero.")

        self.__num_conexiones = self.__conexiones.nnz if self.__sparse else int(self.__xp.count_nonzero(self.__conexiones))


    def __crear_conexiones_aleatorias(self, num:int) -> None:
        """
        Crear un número especificado de conexiones aleatorias con pesos aleatorios.

        Parameters
        ----------
        num : int
            Número de conexiones a crear.

        Raises
        ------
        ValueError
            Si el número de conexiones supera el máximo posible según el número de neuronas existentes.
        """
        
        xp = self.__xp

        # Máximo número de conexiones posibles (sin diagonal)
        max_conexiones = self.__num_neuronas * (self.__num_neuronas - 1)

        if num < 0:
            raise ValueError("La cantidad de conexiones a crear debe ser un entero positivo.")
        elif num > max_conexiones:
            raise ValueError(
                f"No se pueden crear {num} conexiones. "
                f"Máximo permitido: {max_conexiones}"
            )
        
        # Caso sin conexiones
        if num == 0:
            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix((self.__num_neuronas, self.__num_neuronas), dtype=self.__dtype)
            else:
                self.__conexiones = xp.zeros((self.__num_neuronas, self.__num_neuronas), dtype=self.__dtype)
            return
        
        # Seleccionar posiciones aleatorias fuera de la diagonal
        posibles = xp.arange(max_conexiones)

        seleccion = xp.random.choice(
            posibles,
            size=num,
            replace=False
        )

        # Convertir índices lineales → (fila, columna)
        filas = seleccion // (self.__num_neuronas - 1)
        columnas = seleccion % (self.__num_neuronas - 1)

        # Ajustar para saltar diagonal
        columnas = xp.where(
            columnas >= filas,
            columnas + 1,
            columnas
        )

        # Generar pesos según tipo de neurona presináptica (columnas)
        pesos = xp.random.random(num).astype(self.__dtype)

        pesos = xp.where(
            self.__tipo[columnas],
            pesos,
            -pesos
        )

        if self.__sparse:
            # Crear directamente CSR sin pasar por matriz densa
            self.__conexiones = self.__sp.csr_matrix((pesos, (filas, columnas)),
                                                     shape=(self.__num_neuronas, self.__num_neuronas), 
                                                     dtype=self.__dtype)
        
        else:
            # Crear matriz densa
            self.__conexiones = xp.zeros((self.__num_neuronas, self.__num_neuronas), dtype=self.__dtype)
            self.__conexiones[filas, columnas] = pesos


    def actualizar(self, I: Array | float | int, dt: float = 0.5) -> Array:
        """
        Avanza un paso temporal de la simulación y actualiza el estado de todas las neuronas.

        Parameters
        ----------
        I : Array | float | int
            Corriente de entrada aplicada a cada neurona. Puede ser un escalar para aplicar la misma
            corriente a todas las neuronas o un vector de longitud igual al número de neuronas.
        dt : float
            Tamaño del paso temporal.

        Returns
        -------
        Array
            Vector booleano de longitud num_neuronas indicando qué neuronas se han disparado.

        Raises
        ------
        ValueError
            Si el tamaño de I no coincide con el número de neuronas o si dt no es positivo.
        
        Notes
        -----
        Las conexiones se representan mediante una matriz donde las columnas corresponden a neuronas
        presinápticas y las filas a neuronas postsinápticas.

        La corriente total aplicada se calcula como:

            I_total = conexiones · spikes + I
        """

        es_spike = (self.__v >= 30)

        if isinstance(I, (np.ndarray, cp.ndarray)):
            if I.shape != (self.__num_neuronas,):
                raise ValueError("El input de corriente debe ser un vector de longitud N, donde N es el \
                                 número total de neuronas, o ser un número a usar para aplicar el mismo \
                                 input a todas las neuronas.")
            I = self.__xp.asarray(I, dtype=self.__dtype)
            
        elif not isinstance(I, (float, int)):
            raise ValueError("El input de corriente debe ser un vector de longitud N, donde N es el \
                             número total de neuronas, o ser un número a usar para aplicar el mismo \
                             input a todas las neuronas.")
        elif isinstance(I, (float, int)):
            I = self.__dtype.type(I)
        
        if dt <= 0:
            raise ValueError("El paso temporal debe ser positivo.")

        # cambiar dt al dtype interno, para evitar que se haga promoción dentro de los arrays de v y de u a float64.
        dt = self.__dtype.type(dt)

        I_total = self.__conexiones.dot(es_spike.astype(self.__dtype)) + I

        # Evitar asignaciones intermedias de elevar al cuadrado haciendo la multiplicación directamente
        self.__v += dt * ((0.04 * self.__v * self.__v + 5 * self.__v + 140 - self.__u + I_total))
        self.__u += dt * (self.__a * (self.__b * self.__v - self.__u))

        if self.__xp.any(es_spike):
            self.__v[es_spike] = self.__c[es_spike]
            self.__u[es_spike] += self.__d[es_spike]
        return es_spike
    

    def reiniciar(self) -> None:
        """
        Restaura el estado de la red al estado inicial almacenado en la creación.

        Esto restablece los vectores v y u a los valores iniciales usados para construir la red.
        """
        self.__v[:] = self.__v_inicial
        self.__u[:] = self.__u_inicial
    

    def establecer_estado(self, v: Array | None = None, u: Array | None = None) -> None:
        """
        Establece el estado interno v y/o u de la red en nuevos valores.

        Parameters
        ----------
        v : Array | None, optional
            Nuevo vector de potenciales de membrana. Si es None, no se modifica v.
        u : Array | None, optional
            Nuevo vector de variables de recuperación. Si es None, no se modifica u.

        Raises
        ------
        ValueError
            Si alguno de los vectores no tiene longitud igual a num_neuronas.
        """
        if v is not None:
            if v.shape != (self.__num_neuronas,):
                raise ValueError(f"El vector de potenciales de membrana tiene que tener una longitud \
                                 de {self.__num_neuronas} elementos.")
            self.__v[:] = self.__xp.asarray(v, dtype=self.__dtype)
        
        if u is not None:
            if u.shape != (self.__num_neuronas,):
                raise ValueError(f"El vector de variables de recuperación tiene que tener una longitud \
                                 de {self.__num_neuronas} elementos.")
            self.__u[:] = self.__xp.asarray(u, dtype=self.__dtype)
    

    def _estado(self) -> tuple[Array, Array]:
        """
        Devuelve referencias directas al estado interno de la red.

        A diferencia de la propiedad estado, este método no crea copias de los datos y está destinado
        exclusivamente para uso interno del simulador.

        Modificar los objetos devueltos modifica directamente el estado interno de la red.

        Returns
        -------
        tuple[Array, Array]
            Tupla con los vectores de potencial de membrana (v) y de recuperación (u) de todas las neuronas.
        """
        return self.__v, self.__u


    @property
    def estado(self) -> dict[str, Array]:
        """
        Obtiene el estado actual de la red en los vectores v y u.

        Returns
        -------
        dict[str, Array]
            Diccionario con claves v y u, donde cada valor es un vector del estado de cada neurona.
        """
        return {
            "v": self.__v.copy(),
            "u": self.__u.copy()
        }

    @property
    def parametros(self) -> dict[str, Array]:
        """
        Obtiene los parámetros (a, b, c, d) de todas las neuronas.

        Returns
        -------
        dict[str, Array]
            Diccionario con los cuatro parámetros del modelo de Izhikevich (a, b, c, d) para cada neurona.
        """
        return {
            "a": self.__a.copy(),
            "b": self.__b.copy(),
            "c": self.__c.copy(),
            "d": self.__d.copy()
        }
    
    @property
    def uso_gpu(self) -> bool:
        """
        Indica si la red utiliza la CPU o la GPU como backend de cálculo.

        Returns
        -------
        bool
            True si la red se ejecuta en GPU con CuPy, False si utiliza CPU con NumPy.
        """
        return self.__uso_gpu

    @property
    def backend(self) -> str:
        """
        Indica el backend de cálculo elegido para la red.

        Returns
        -------
        str
            "CPU (NumPy)" si se usa NumPy, o "GPU (CuPy)" si se usa CuPy.
        """
        return "GPU (CuPy)" if self.__uso_gpu else "CPU (NumPy)"

    @property
    def neuronas(self) -> dict[Neurona, int]:
        """
        Devuelve los tipos de neuronas y su cantidad en la red.

        Returns
        -------
        dict[Neurona, int]
            Copia del diccionario original que asocia cada objeto Neurona con el número de instancias
            de ese tipo.
        """
        return self.__neuronas.copy()

    @property
    def conexiones(self) -> Array | SparseArray:
        """
        Obtiene la matriz de pesos sinápticos de la red.

        Returns
        -------
        Array | SparseArray
            Matriz cuadrada de tamaño num_neuronas x num_neuronas donde cada valor representa
            el peso de la conexión de una neurona presináptica (columnas) a una postsináptica (filas).
            La diagonal es siempre cero.
        """
        return self.__conexiones.copy()

    @property
    def num_neuronas(self) -> int:
        """
        Número total de neuronas en la red.

        Returns
        -------
        int
            Suma de todas las neuronas definidas en el constructor.
        """
        return self.__num_neuronas
    
    @property
    def num_conexiones(self) -> int:
        """
        Número total de conexiones activas en la red.

        Returns
        -------
        int
            Cantidad de elementos distintos de cero en la matriz de conexiones.
        """
        return self.__num_conexiones
    
    @property
    def dtype(self) -> type[np.floating] | type[cp.floating]:
        """
        Tipo de dato utilizado internamente por la red.

        Returns
        -------
        type[np.floating] | type[cp.floating]
            Tipo de dato interno utilizado por los arrays de la red (por ejemplo np.float32 o cp.float32).
        """
        return self.__dtype
    
    @property
    def sparse(self) -> bool:
        """
        Indica si las conexiones utilizan una representación dispersa CSR.

        Returns
        -------
        bool
            True si la matriz de conexiones está almacenada como sparse CSR.
        """
        return self.__sparse
