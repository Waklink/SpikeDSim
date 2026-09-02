from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from typing import TYPE_CHECKING, TypeAlias, Literal, TypedDict
from numbers import Real
from collections.abc import Sequence


from .Neurona import Neurona
from ..backend import cp, cpsp, CUPY_DISPONIBLE

if TYPE_CHECKING:
    if CUPY_DISPONIBLE:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

    Array: TypeAlias = np.ndarray | cp.ndarray
    SparseArray: TypeAlias = sp.csr_matrix | cpsp.csr_matrix
else:
    if CUPY_DISPONIBLE:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
    else:
        cp = None
        cpsp = None

    Array: TypeAlias = np.ndarray
    SparseMatrix: TypeAlias = sp.csr_matrix


# --------------------------------------------------
# TIPOS AUXILIARES
# --------------------------------------------------

Vector4D: TypeAlias = tuple[float | None, float | None, float | None, float | None]

class SubParametros(TypedDict, total=False):
    """
    Parámetros individuales de aleatorización relativa para una neurona.

    Cada valor indica la amplitud máxima de variación aleatoria del parámetro correspondiente. Un
    valor de None equivale a no aplicar aleatorización.
    """
    a: float | None
    b: float | None
    c: float | None
    d: float | None

class AleatParamTuplas(TypedDict):
    """
    Configuración de aleatorización de parámetros mediante tuplas.

    Cada tupla contiene los factores de aleatorización correspondientes a los parámetros (a, b, c, d)
    para neuronas excitatorias e inhibitorias.
    """
    excitatoria: Vector4D
    inhibitoria: Vector4D

class AleatParam(TypedDict):
    """
    Configuración de aleatorización de parámetros mediante diccionarios.

    Permite especificar únicamente los parámetros que se desean aleatorizar para cada tipo de neurona.
    """
    excitatoria: SubParametros
    inhibitoria: SubParametros

class DictAleatorizacion(TypedDict):
    """
    Información de la aleatorización utilizada al crear una red.

    Incluye la configuración de aleatorización de parámetros, la de conexiones y la semilla utilizada
    por el generador de números aleatorios.
    """
    aleat_param: AleatParam
    aleat_conex: tuple[float, float] | None
    semilla: int | None


# --------------------------------------------------
# CLASE
# --------------------------------------------------

class RedDeNeuronas:
    """
    Red de neuronas basada en el modelo de Izhikevich con soporte para ejecución en CPU o GPU.

    Esta clase representa una red de neuronas compuesta por varios vectores de parámetros de
    neuronas y una matriz de conexiones que define la influencia sináptica entre ellas.
    El backend de cálculo se puede seleccionar entre NumPy para CPU y CuPy para GPU.

    Attributes
    ----------
    backend : str
        Backend utilizado para los cálculos ("numpy" o "cupy").

    uso_gpu: bool
        Indica de forma directa si se utiliza la GPU como backend o, por el contrario, la CPU.

    neuronas : dict[Neurona, int]
        Tipos de neuronas y su cantidad en la red. Estos tipos pueden especificarse con instancias
        de Neurona o nombres de tipos predefinidos, que se convierten a instancias al construirse la
        red.

    nombre : list[str]
        Nombres de las neuronas, en el orden en el que están en la red.

    es_excitatoria : list[bool]
        Lista que indica si las neuronas de la red son excitatorias o inhibitorias.

    conexiones : list[list[float]]
        Matriz de pesos sinápticos con diagonal cero. Puede almacenarse como una matriz densa o
        mediante una representación dispersa CSR.

        Los elementos de la matriz representan conexiones desde la neurona de la columna (presináptica)
        hacia la neurona de la fila (postsináptica). Los pesos positivos corresponden a conexiones
        excitatorias y los negativos a inhibitorias.

    aleatorizacion : DictAleatorizacion
        Información utilizada durante la generación aleatoria de la red. Incluye los valores de
        aleatorización de los parámetros neuronales y de las conexiones y la semilla utilizada para
        el generador de números aleatorios.

    num_neuronas : int
        Número total de neuronas en la red.

    num_conexiones : int
        Número total de conexiones diferentes de cero.

    estado : dict[str, list]
        Estado actual de la red con v y u.

    parametros : dict[str, list[float]]
        Parámetros (a, b, c, d) de todas las neuronas.

    precision : Literal[32, 64]
        Precisión utilizada para almacenar los datos internos de la red, expresada en bits. Puede
        ser 32 o 64 según el valor pasado en el constructor.

    dtype : np.float32 | cp.float32 | np.float64 | cp.float64
        Tipo de dato utilizado internamente por los arrays de la red. Puede ser float32 o
        float64 según la precisión seleccionada en el constructor.

    sparse : bool
        Indica si la matriz de conexiones se almacena utilizando una representación dispersa CSR.

    estadisticas : dict[str, int | float]
        Diccionario con estadísticas generales de la red. Incluye el número de neuronas totales,
        cantidad de neuronas excitatorias e inhibitorias, número de conexiones activas, densidad de
        conexiones y cantidad de conexiones excitatorias e inhibitorias.

    Examples
    --------
    Crear una red de 800 neuronas excitatorias y 200 inhibitorias:

    >>> red = RedDeNeuronas({"rs": 800, "fs": 200}, conexiones=5000)

    Ejecutar un paso de simulación:

    >>> spikes = red.actualizar(5)
    >>> spikes.shape
    (1000,)
    """


    # --------------------------------------------------
    # CONSTRUCTOR
    # --------------------------------------------------

    def __init__(self, neuronas: dict[Neurona | str, int], conexiones: int | list[list[float]] | Array | SparseArray = 0,
                 backend: Literal["numpy", "cupy"] = "numpy", precision: Literal[32, 64] = 32,
                 sparse: bool = True, semilla: int | None = None,
                 aleat_param: AleatParamTuplas | AleatParam | None = None,
                 aleat_conex: tuple[float, float] | None = None):
        """
        Inicializa una instancia de la clase RedDeNeuronas con un conjunto de neuronas y sus conexiones.

        Parameters
        ----------
        neuronas : dict[Neurona | str, int]
            Diccionario de neuronas a crear. La clave es una instancia de Neurona, o el nombre de un
            tipo predefinido, y el valor es el número de neuronas de ese tipo. Las cadenas de texto
            se convierten automáticamente a instancias de Neurona.

        conexiones : int | list[list[float]] | Array | SparseArray
            Matriz de conexiones entre las neuronas, donde cada fila es una lista con los pesos de
            las conexiones de la neurona que representa; o un entero que indica el número de conexiones
            aleatorias a crear. También puede ser una matriz dispersa csr. En el caso de ser un array
            o una matriz dispersa, deben coincidir con el backend seleccionado (NumPy/SciPy para CPU
            o CuPy/CuPyX para GPU). Por defecto el número de conexiones es 0.

        backend : Literal["numpy", "cupy"], optional
            Nombre del backend a utilizar. Utilizar "numpy" para CPU o "cupy" para GPU, ignorándose
            mayúsculas y espacios al principio y final. Por defecto se usa NumPy.

        precision : Literal[32, 64], optional
            Tamaño, en bits, en el que se guardan los valores de los arrays. La precisión por defecto
            es 32 bits (float32).

        sparse : bool, optional
            Indica si la matriz de conexiones se almacena utilizando una representación CSR dispersa.
            Por defecto es True.

        semilla : int | None, optional
            La semilla a utilizar para generar las conexiones aleatorias en el caso de que conexiones
            sea un entero. Por defecto vale None.

        aleat_param : AleatParamTuplas | AleatParam | None, optional
            Diccionario con la amplitud máxima de la aleatorización de los parámetros de
            las neuronas.

            Puede especificarse mediante tuplas:

            {
                "excitatoria": (a, b, c, d),
                "inhibitoria": (a, b, c, d)
            }

            o mediante diccionarios:

            {
                "excitatoria": {"a": ..., "b": ..., "c": ..., "d": ...},
                "inhibitoria": {"a": ..., "b": ..., "c": ..., "d": ...}
            }

            Pudiendo no pasar todos los parámetros en la segunda forma.

            Un 0 o None en cualquier valor significa que ese parámetro no se aleatorizará. En el
            caso de que aleat_param sea None, todos los valores serán 0, sin aleatorizar ningún
            parámetro.

            Los valores determinan las máximas modificaciones posibles de los parámetros correspondientes.

        aleat_conex : tuple[float, float] | None, optional
            Factores máximos para los pesos de las conexiones aleatorias.

            Solo se utiliza cuando conexiones es un entero.

            El primer elemento corresponde a conexiones excitatorias y el segundo a inhibitorias.

            Si es None, se utiliza (1, 1) cuando se generan conexiones aleatorias y None en cualquier
            otro caso.

            Los valores deben estar en el intervalo (0, 1]. Un valor de 1 permite generar pesos en
            todo el rango posible de la conexión correspondiente, mientras que valores menores reducen
            su magnitud máxima, no pudiendo ser 0, ya que eso implicaría que no existen conexiones.

        Raises
        ------
        TypeError
            Si alguno de los parámetros no es del tipo de dato correcto, neuronas tiene que ser un
            diccionario formado por instancias de Neurona o cadenas de texto y enteros, conexiones
            debe ser un entero mayor o igual a 0, una lista de listas, o un Array o csr_matrix que coincida con el backend
            seleccionado.

        ValueError
            Si algún parámetro no es válido, las dimensiones de las conexiones son incorrectas o su
            diagonal no es cero.
        """
        # Comprobación y asignación de tipo de backend, de sparse y de precision
        self._validar_parametros_generales(backend, precision, sparse)

        # Comprobaciones de tipo y valores del diccionario con las neuronas
        # Conversión de cadenas de texto a instancias de Neurona
        self._validar_neuronas(neuronas)

        # Número total de neuronas
        self.__num_neuronas = sum(self.__neuronas.values())

        # Semilla y RNG (Random Number Generator)
        # Comprobación de semilla
        if semilla is not None and not isinstance(semilla, int):
            raise TypeError("La semilla debe ser un entero.")

        self.__semilla = semilla
        self.__rng = self.__xp.random.RandomState(semilla)

        # Aleatorización de los parámetros
        self.__aleat_param = self._normalizar_aleat_param(aleat_param)

        # Inicializar y llenar vectores de parámetros y estado inicial
        self._crear_vectores()

        # Aleatorización de los pesos de las conexiones
        self.__aleat_conex = None

        # Creación y llenado de la matriz de conexiones entre neuronas
        self._crear_matriz_conexiones(conexiones, aleat_conex)

        # Comprobaciones de la matriz de conexiones
        self._validar_conexiones()

        # Guardar número de conexiones que existan.
        self.__num_conexiones = self.__conexiones.nnz if self.__sparse else int(self.__xp.count_nonzero(self.__conexiones))

        # Liberar memoria de VRAM reservada que no se está utilizando
        if self.__uso_gpu:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


    # --------------------------------------------------
    # MÉTODOS PÚBLICOS
    # --------------------------------------------------

    def actualizar(self, I: Array | list[float] | float | int, dt: float = 1) -> Array:
        """
        Avanza un paso temporal de la simulación y actualiza el estado de todas las neuronas.

        Parameters
        ----------
        I : Array | list[float] | float | int
            Corriente de entrada aplicada a cada neurona. Puede ser un escalar para aplicar la misma
            corriente a todas las neuronas o un vector de longitud igual al número de neuronas.

        dt : float
            Paso temporal de simulación en milisegundos. Debe ser mayor que 0.

        Returns
        -------
        Array
            Vector booleano de longitud num_neuronas indicando qué neuronas se han disparado.

        Raises
        ------
        TypeError
            Si I no es un vector ni un número real, o si dt no es un número real.

        ValueError
            Si el tamaño de I no coincide con el número de neuronas o si dt no es mayor que 0.

        Notes
        -----
        Las conexiones se representan mediante una matriz donde las columnas corresponden a neuronas
        presinápticas y las filas a neuronas postsinápticas.

        La corriente total aplicada se calcula como:

            I_total = conexiones · spikes + I
        """
        if isinstance(I, (np.ndarray, cp.ndarray) if CUPY_DISPONIBLE else (np.ndarray)):
            if I.shape != (self.__num_neuronas,):
                raise ValueError("La entrada de corriente debe ser un vector de longitud N, donde N "
                                 "es el número total de neuronas, o ser un número a usar para aplicar"
                                 " la misma corriente a todas las neuronas.")

            I = self.__xp.asarray(I, dtype=self.__dtype)
        elif isinstance(I, list):
            if len(I) != self.__num_neuronas:
                raise ValueError("La entrada de corriente debe ser un vector de longitud N, donde N "
                                 "es el número total de neuronas, o ser un número a usar para aplicar"
                                 " la misma corriente a todas las neuronas.")

            I = self.__xp.asarray(I, dtype=self.__dtype)
        elif isinstance(I, Real):
            # Convertir dt al dtype interno para evitar promociones innecesarias durante los cálculos.
            I = self.__dtype(I)
        else:
            raise TypeError("La entrada de corriente debe ser un vector de longitud N, donde N es el"
                            " número total de neuronas, o ser un número a usar para aplicar la misma"
                            " corriente a todas las neuronas.")

        if not isinstance(dt, Real):
            raise TypeError("El paso temporal debe ser un número real.")

        if dt <= 0:
            raise ValueError("El paso temporal debe ser mayor que 0.")

        # cambiar dt al dtype interno, para evitar que se haga promoción dentro de los arrays de v
        # y de u a float64.
        dt = self.__dtype(dt)

        return self._actualizar(I, dt)

    def reiniciar(self) -> None:
        """
        Restaura el estado de la red al estado inicial almacenado en la creación.

        Esto restablece los vectores v y u a los valores iniciales usados para construir la red.
        """
        self.__v[:] = self.__v_inicial
        self.__u[:] = self.__u_inicial

    def establecer_estado(self, v: Array | list[float] | None = None,
                          u: Array | list[float] | None = None) -> None:
        """
        Establece el estado interno v y/o u de la red en nuevos valores. Pudiendo actualizar solo
        uno de los dos estados o ninguno.

        Parameters
        ----------
        v : Array | list[float] | None, optional
            Nuevo vector de potenciales de membrana. Si es None, no se modifica v.

        u : Array | list[float] | None, optional
            Nuevo vector de variables de recuperación. Si es None, no se modifica u.

        Raises
        ------
        TypeError
            Si alguno de los vectores no es un array de NumPy o de CuPy ni una lista.

        ValueError
            Si alguno de los vectores no tiene longitud igual a num_neuronas.
        """
        if v is not None:
            if isinstance(v, (np.ndarray, cp.ndarray) if CUPY_DISPONIBLE else (np.ndarray)):
                if v.shape != (self.__num_neuronas,):
                    raise ValueError("El vector de potenciales de membrana tiene que tener una longitud"
                                    f" de {self.__num_neuronas} elementos.")
            elif isinstance(v, list):
                if len(v) != self.__num_neuronas:
                    raise ValueError("El vector de potenciales de membrana tiene que tener una longitud"
                                    f" de {self.__num_neuronas} elementos.")
            else:
                raise TypeError("El vector de potenciales de membrana nuevo debe ser un vector de "
                                "NumPy o de CuPy, o una lista de números.")

            self.__v[:] = self.__xp.asarray(v, dtype=self.__dtype)

        if u is not None:
            if isinstance(u, (np.ndarray, cp.ndarray) if CUPY_DISPONIBLE else (np.ndarray)):
                if u.shape != (self.__num_neuronas,):
                    raise ValueError("El vector de variables de recuperación tiene que tener una longitud"
                                    f" de {self.__num_neuronas} elementos.")
            elif isinstance(u, list):
                if len(u) != self.__num_neuronas:
                    raise ValueError("El vector de variables de recuperación tiene que tener una longitud"
                                    f" de {self.__num_neuronas} elementos.")
            else:
                raise TypeError("El vector de variables de recuperación nuevo debe ser un vector de"
                                " NumPy o de CuPy, o una lista de números.")

            self.__u[:] = self.__xp.asarray(u, dtype=self.__dtype)

    def convertir_backend(self, backend: Literal["numpy", "cupy"]) -> RedDeNeuronas:
        """
        Crear una copia de la red actual con el backend especificado, conservando también su estado
        actual.

        Parameters
        ----------
        backend : Literal["numpy", "cupy"]
            El backend a utilizar en la copia de la red, se aceptan variantes con mayúsculas como
            "NuMPy" o "Cupy ".

        Returns
        -------
        RedDeNeuronas
            Una copia de la red actual con el nuevo backend especificado.
        """
        nuevo_backend = backend.lower().strip()
        conexiones = 0

        if nuevo_backend == self.backend:
            conexiones = self.__conexiones.copy()
        elif nuevo_backend == "numpy":
            # Backend actual CuPy a NumPy
            conexiones = self.__conexiones.get()
        elif nuevo_backend == "cupy":
            # Backend actual NumPy a CuPy
            if self.__sparse:
                conexiones = cpsp.csr_matrix(self.__conexiones)
            else:
                conexiones = cp.asarray(self.__conexiones)

        # En el caso de que backend no sea numpy ni cupy, se detectará al principio del constructor
        nueva_red = RedDeNeuronas(self.__neuronas.copy(), conexiones, nuevo_backend, self.__precision,
                                  self.__sparse, self.__semilla, self.__aleat_param, self.__aleat_conex)

        nueva_red.establecer_estado(**self.estado)
        return nueva_red

    def convertir_formato(self, nuevo_sparse: bool) -> RedDeNeuronas:
        """
        Crear una copia de la red actual, con el nuevo formato de la matriz de conexiones especificado,
        conservando también su estado actual.

        Parameters
        ----------
        nuevo_sparse : bool
            Nuevo formato de la matriz, puede ser una matriz dispersa csr o una matriz
            densa.

        Returns
        -------
        RedDeNeuronas
            Copia de la red, con el nuevo formato de la matriz de conexiones.
        """
        nueva_red = RedDeNeuronas(self.__neuronas.copy(), self.__conexiones.copy(), self.backend,
                                  self.__precision, nuevo_sparse, self.__semilla, self.__aleat_param,
                                  self.__aleat_conex)

        nueva_red.establecer_estado(self.__v.copy(), self.__u.copy())
        return nueva_red

    def cambiar_precision(self, nueva_precision: int) -> RedDeNeuronas:
        """
        Crear una copia de la red actual, con la nueva precisión para los datos guardados, conservando
        también su estado actual.

        Parameters
        ----------
        nueva_precision : int
            Nueva precisión de los parámetros y el estado guardados.

        Returns
        -------
        RedDeNeuronas
            Copia de la red, con la nueva precisión.
        """
        nueva_red = RedDeNeuronas(self.__neuronas.copy(), self.__conexiones.copy(), self.backend,
                                  nueva_precision, self.__sparse, self.__semilla, self.__aleat_param,
                                  self.__aleat_conex)

        nueva_red.establecer_estado(self.__v.copy(), self.__u.copy())
        return nueva_red

    def copy(self) -> RedDeNeuronas:
        """
        Devuelve una copia de esta red.

        Returns
        -------
        RedDeNeuronas
            Copia de la red de neuronas actual.
        """
        red = RedDeNeuronas(self.__neuronas.copy(), self.__conexiones.copy(), self.backend,
                             self.__precision, self.__sparse, self.__semilla, self.__aleat_param,
                             self.__aleat_conex)

        red.establecer_estado(v=self.__v, u=self.__u)
        return red

    def informacion(self, indice: int | slice | Sequence[int] | None
                    ) -> dict[int, dict[str, str | bool | dict[str, float]]]:
        """
        Obtiene información de una o varias neuronas de la red.

        Parameters
        ----------
        indice : int | slice | Sequence[int] | None
            Índice, rango o secuencia de índices de las neuronas cuya información se desea obtener.
            Si None, se devuelven todas las neuronas.

        Returns
        -------
        dict[int, dict[str, int | str | bool | dict[str, float]]]
            Diccionario cuyas claves son los índices de las neuronas solicitadas y cuyos valores
            contienen su nombre, tipo, parámetros y estado actual.

        Raises
        ------
        TypeError
            Si el índice no es un entero, un slice o una secuencia de enteros.

        ValueError
            Si alguno de los índices está fuera del rango permitido.
        """
        if indice is None:
            indices = np.arange(self.__num_neuronas)
        elif isinstance(indice, int):
            indices = np.asarray([indice], dtype=int)
        elif isinstance(indice, slice):
            indices = np.arange(self.__num_neuronas)[indice]
        elif isinstance(indice, Sequence) and not isinstance(indice, (str, bytes)):
            if not all(isinstance(i, int) and i >= 0 for i in indice):
                raise TypeError("Los índices deben ser enteros mayores o iguales a 0.")

            if len(indice) == 0:
                raise ValueError("No se ha pasado ningún índice.")

            indices = np.asarray(indice, dtype=int)
        else:
            raise TypeError("El índice debe ser un entero, un slice, una secuencia de enteros o None.")

        if np.any(indices >= self.__num_neuronas):
            raise IndexError("Hay índices fuera de rango.")

        return {int(i): self._informacion_neurona(int(i)) for i in indices}


    # --------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------

    def _validar_parametros_generales(self, backend: Literal["numpy", "cupy"], precision: Literal[32, 64],
                                      sparse: bool) -> None:
        """
        Comprobar y asignar el backend, la precisión de los datos a guardar y si las conexiones se guardan
        como una sparse matrix.

        Parameters
        ----------
        backend : Literal["numpy", "cupy"]
            El backend a validar.

        precision : Literal[32, 64]
            La precisión a validar.

        sparse : bool
            El sparse a validar.

        Raises
        ------
        TypeError
            Si alguno de los parámetros pasados no son del tipo correcto.

        ImportError
            Si no se tiene instalado el paquete cupy y se quiere usarlo como backend.

        ValueError
            Si el backend o la precisión no son valores válidos.
        """
        if not isinstance(backend, str):
            raise TypeError("El backend debe ser una cadena de texto.")

        if not isinstance(sparse, bool):
            raise TypeError("El parámetro sparse debe ser un booleano.")

        if not isinstance(precision, int):
            raise TypeError("La precisión debe ser un entero.")

        # Asignación del backend y del módulo de sparse a utilizar
        backend = backend.lower().strip()
        if backend == "numpy":
            self.__xp = np
            self.__sp = sp
        elif backend == "cupy":
            if not CUPY_DISPONIBLE:
                raise ImportError("El backend cupy requiere tener instalado el paquete de cupy.")

            self.__xp = cp
            self.__sp = cpsp
        else:
            raise ValueError("El parámetro 'backend' debe ser 'numpy' o 'cupy'.")

        # Guardar el valor de los parámetros backend y sparse
        self.__uso_gpu = backend == "cupy"
        self.__sparse = sparse

        # Asignación del tipo de dato para los vectores
        if precision == 32:
            self.__dtype = self.__xp.float32
        elif precision == 64:
            self.__dtype = self.__xp.float64
        else:
            raise ValueError("La precisión debe ser 32 o 64.")
 
        self.__precision = precision

    def _validar_neuronas(self, neuronas: dict[Neurona | str, int]) -> None:
        """
        Validar las neuronas, convirtiendo las cadenas de texto a instancias de Neurona, con los valores
        predefinidos para esos tipos.

        Parameters
        ----------
        neuronas : dict[Neurona | str, int]
            Diccionario con las neuronas y sus cantidades a validar y/o convertir.

        Raises
        ------
        TypeError
            Si el parámetro neuronas no es un diccionario, alguna de las neuronas no son instancias
            de Neurona o cadenas de texto o alguna cantidad no es un entero.

        ValueError
            Si alguna cantidad es negativa.
        """
        if not isinstance(neuronas, dict):
            raise TypeError("Las neuronas deben pasarse como un diccionario con las neuronas y la "
                            "cantidad a crear.")

        self.__neuronas = {}
        self.__nombre = []

        for neurona, cantidad in neuronas.items():
            if not isinstance(neurona, (Neurona, str)):
                raise TypeError("Las claves del diccionario de neuronas deben ser instancias de la"
                                " clase Neurona o nombres de tipos predefinidos.")

            if not isinstance(cantidad, int):
                raise TypeError("Las cantidades de las neuronas deben ser números enteros.")

            if cantidad < 0:
                raise ValueError("Las cantidades de las neuronas deben ser positivas.")

            # Convertir cadena a instancia de Neurona
            if isinstance(neurona, str):
                neurona = Neurona.predefinida(neurona)

            # Crear copia de la neurona, para evitar que se pueda acceder desde el diccionario de
            # la red
            neurona = neurona.copy()

            self.__neuronas[neurona] = cantidad
            self.__nombre.extend([neurona.nombre] * cantidad)

    def _normalizar_aleat_param(self, aleat_param: AleatParamTuplas | AleatParam | None) -> AleatParam:
        """
        Validar y normalizar aleat_param.

        Parameters
        ----------
        aleat_param : AleatParamTuplas | AleatParam | None
            diccionario con los límites de aleatorización de los parámetros de las neuronas, pudiéndose
            personalizar valores distintos para neuronas excitatorias e inhibitorias.

        Returns
        -------
        AleatParam
            aleat_param normalizado.

        Raises
        ------
        TypeError
            - Si aleat_param no es None ni un diccionario.
            - Si alguno de los valores no es None ni un número.

        ValueError
            - Si aleat_param tiene claves distintas de excitatoria e inhibitoria, o si falta alguna de ellas.
            - Si los valores se pasan en tuplas, y no se pasan todos los 4 valores.
        """
        valores_por_defecto = {"a": 0, "b": 0, "c": 0, "d": 0}
        aleat_param_normalizado = {"excitatoria": valores_por_defecto.copy(),
                                   "inhibitoria": valores_por_defecto.copy()}

        if aleat_param is not None:
            if not isinstance(aleat_param, dict):
                raise TypeError("aleat_param tiene que ser None o un diccionario.")

            if len(aleat_param) != 2 or any(clave not in aleat_param_normalizado.keys()
                                            for clave in aleat_param):
                raise ValueError("aleat_param solo tiene que tener excitatoria e inhibitoria como "
                                 "claves.")

            if all(isinstance(elem, tuple) for elem in aleat_param.values()):
                if any(len(elem) != 4 for elem in aleat_param.values()):
                    raise ValueError("aleat_param tiene que tener tuplas de 4 valores, con un 0 o "
                                     "None en los valores de los parámetros que no se quieran aleatorizar.")

                if not all(valor is None or isinstance(valor, Real) for params in aleat_param.values()
                           for valor in params):
                    raise TypeError("Los valores pasados deben ser None o un número.")

                for clave, valor in aleat_param.items():
                    for i, param in enumerate(("a", "b", "c", "d")):
                        valor_param = valor[i]
                        if valor_param is not None:
                            aleat_param_normalizado[clave][param] = valor_param

            elif all(isinstance(elem, dict) for elem in aleat_param.values()):
                if not all(valor is None or isinstance(valor, Real) for params in aleat_param.values()
                           for valor in params.values()):
                    raise TypeError("Los valores pasados deben ser None o un número.")

                for param in valores_por_defecto:
                    valor_exc = aleat_param["excitatoria"].get(param)
                    valor_inh = aleat_param["inhibitoria"].get(param)
                    if valor_exc is not None:
                        aleat_param_normalizado["excitatoria"][param] = valor_exc

                    if valor_inh is not None:
                        aleat_param_normalizado["inhibitoria"][param] = valor_inh
            else:
                raise TypeError("Los valores de aleat_param deben ser todos tuplas o todos diccionarios.")

        return aleat_param_normalizado

    def _crear_vectores(self) -> None:
        """
        Inicializa y llena los vectores de parámetros y el estado inicial de todas las neuronas.
        """
        xp = self.__xp
        rng = self.__rng
        dtype = self.__dtype

        # Creación de vectores con los parámetros de las neuronas
        self.__a = xp.empty(self.__num_neuronas, dtype=dtype)
        self.__b = xp.empty(self.__num_neuronas, dtype=dtype)
        self.__c = xp.empty(self.__num_neuronas, dtype=dtype)
        self.__d = xp.empty(self.__num_neuronas, dtype=dtype)
        self.__v = xp.empty(self.__num_neuronas, dtype=dtype)
        self.__u = xp.empty(self.__num_neuronas, dtype=dtype)
        self.__es_excitatoria = xp.empty(self.__num_neuronas, dtype=bool)

        # Llenar los vectores de parámetros
        indice_actual = 0

        for neurona, cantidad in self.__neuronas.items():
            a, b, c, d = neurona.parametros
            v, u = neurona.estado
            es_excitatoria = "excitatoria" if neurona.es_excitatoria else "inhibitoria"

            aleat = self.__aleat_param[es_excitatoria]
            sumandos = xp.asarray(rng.random_sample((cantidad, 4)), dtype=dtype) * xp.asarray([aleat["a"],
                       aleat["b"], aleat["c"], aleat["d"]], dtype=dtype)

            self.__a[indice_actual:indice_actual + cantidad] = a + sumandos[:, 0]
            self.__b[indice_actual:indice_actual + cantidad] = b + sumandos[:, 1]
            self.__c[indice_actual:indice_actual + cantidad] = c + sumandos[:, 2]
            self.__d[indice_actual:indice_actual + cantidad] = d + sumandos[:, 3]
            self.__v[indice_actual:indice_actual + cantidad] = v
            self.__u[indice_actual:indice_actual + cantidad] = u
            self.__es_excitatoria[indice_actual:indice_actual + cantidad] = neurona.es_excitatoria
            indice_actual += cantidad

        # Guardar el estado inicial de la red
        self.__v_inicial = self.__v.copy()
        self.__u_inicial = self.__u.copy()

    def _crear_matriz_conexiones(self, conexiones: int | list[list[float]] | Array | SparseArray,
                                 aleat_conex: tuple[float, float] | None) -> None:
        """
        Crear la matriz de conexiones, basándose en si es dispersa o no y si se proporciona una matriz
        completa o se quiere crear valores aleatorios.

        Parameters
        ----------
        conexiones : int | list[list[float]] | Array | SparseArray
            El número de conexiones aleatorias a crear, o una matriz con las conexiones ya establecidas.
            En el caso de que sea un Array o un SparseArray, este tiene que ser del backend utilizado.

        aleat_conex : tuple[float, float] | None
            El valor máximo, o mínimo en el caso de neuronas inhibitorias, de los pesos de las conexiones,
            solo se tiene en cuenta si conexiones es un entero, en este caso, se convertirá a (1, 1)
            por defecto si se pasa None.

        Raises
        ------
        TypeError
            Si conexiones no es ningún tipo aceptado.

        ValueError
            Si aleat_conex no tiene dos elementos, o alguno de ellos no está en el intervalo (0, 1].
        """
        # Matriz aleatoria con número de conexiones especificado
        if isinstance(conexiones, int):
            if aleat_conex is None:
                aleat_conex = (1, 1)

            if not isinstance(aleat_conex, tuple) or not all(isinstance(elem, Real) for elem in aleat_conex):
                raise TypeError("aleat_conex tiene que ser una tupla de números.")

            if len(aleat_conex) != 2:
                raise ValueError("aleat_conex tiene que tener dos elementos.")

            if not all(0 < elem <= 1 for elem in aleat_conex):
                raise ValueError("Los elementos de aleat_conex tienen que estar en el intervalo (0, 1].")

            self.__aleat_conex = aleat_conex

            self._crear_conexiones_aleatorias(conexiones)

        # Matriz a partir de una lista de listas
        elif isinstance(conexiones, list):
            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix(conexiones, dtype=self.__dtype)
            else:
                self.__conexiones = self.__xp.asarray(conexiones, dtype=self.__dtype)

        # Matriz a partir de un array de NumPy o de CuPy
        elif isinstance(conexiones, self.__xp.ndarray):
            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix(conexiones, dtype=self.__dtype)
            else:
                self.__conexiones = self.__xp.asarray(conexiones, dtype=self.__dtype)

        # Matriz a partir de una matriz dispersa CSR
        elif isinstance(conexiones, self.__sp.csr_matrix):
            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix(conexiones, dtype=self.__dtype)
            else:
                self.__conexiones = self.__xp.asarray(conexiones.toarray(), dtype=self.__dtype)

        else:
            raise TypeError("El parámetro 'conexiones' debe ser un entero, una matriz formada por "
                            "listas o un array, ambos en forma de matriz cuadrada; o una matriz dispersa,"
                            " todo ello del mismo backend indicado.")

    def _crear_conexiones_aleatorias(self, num_conexiones: int) -> None:
        """
        Crear un número especificado de conexiones aleatorias con pesos aleatorios.

        Parameters
        ----------
        num_conexiones : int
            Número de conexiones a crear.

        Raises
        ------
        ValueError
            Si el número de conexiones es negativo o si supera el máximo posible según el número de
            neuronas existentes.

        Notes
        -----
        Los índices de conexión se generan considerando que las columnas representan neuronas
        presinápticas y las filas postsinápticas.
        """
        xp = self.__xp
        rng = self.__rng
        n = self.__num_neuronas

        # Máximo número de conexiones posibles (sin diagonal)
        max_conexiones = n * (n - 1)

        if num_conexiones < 0:
            raise ValueError("La cantidad de conexiones a crear debe ser un entero mayor o igual a 0.")

        if num_conexiones > max_conexiones:
            raise ValueError(f"No se pueden crear {num_conexiones} conexiones. Máximo permitido: {max_conexiones}")

        # Caso sin conexiones
        if num_conexiones == 0:
            self.__aleat_conex = None

            if self.__sparse:
                self.__conexiones = self.__sp.csr_matrix((n, n), dtype=self.__dtype)
            else:
                self.__conexiones = xp.zeros((n, n), dtype=self.__dtype)

            return

        # --------------------------------------------------
        # Selección de posiciones
        # --------------------------------------------------

        if num_conexiones <= max_conexiones // 2:
            # Seleccionar directamente las conexiones existentes.
            seleccion = rng.choice(max_conexiones, size=num_conexiones, replace=False)

            # Ordenar para que las posiciones queden agrupadas por fila.
            seleccion.sort()

        elif num_conexiones == max_conexiones:
            # Todas las posiciones posibles.
            seleccion = None

        else:
            # Seleccionar únicamente las posiciones que NO existirán.
            num_faltantes = max_conexiones - num_conexiones

            faltantes = rng.choice(max_conexiones, size=num_faltantes, replace=False)
            faltantes.sort()

            mascara = xp.ones(max_conexiones, dtype=bool)
            mascara[faltantes] = False

            seleccion = xp.nonzero(mascara)[0]

            del faltantes, mascara

        # --------------------------------------------------
        # Convertir posiciones lineales a fila/columna
        # --------------------------------------------------

        if seleccion is None:
            # Caso de densidad máxima.
            #
            # No existe ninguna selección aleatoria que realizar:
            # todas las posiciones fuera de la diagonal existen.
            filas = xp.repeat(xp.arange(n, dtype=int), n - 1)
            columnas = xp.tile(xp.arange(n - 1, dtype=int), n)
            columnas = xp.where(columnas >= filas, columnas + 1, columnas)
        else:
            filas = seleccion // (n - 1)
            columnas = seleccion % (n - 1)
            columnas = xp.where(columnas >= filas, columnas + 1, columnas)


        # --------------------------------------------------
        # Generar pesos
        # --------------------------------------------------

        pesos = rng.random_sample(num_conexiones).astype(self.__dtype)

        signos= self.__es_excitatoria[columnas]
        factor_exc = self.__dtype(self.__aleat_conex[0])
        factor_inh = -self.__dtype(self.__aleat_conex[1])
        pesos *= xp.where(signos, factor_exc, factor_inh)

        # --------------------------------------------------
        # Crear conexiones
        # --------------------------------------------------

        if self.__sparse:
            # Como seleccion está ordenada por posición lineal, las conexiones ya están ordenadas
            # por filas.
            #
            # Por tanto podemos construir directamente la CSR, sin pasar por COO.
            indices = columnas

            if seleccion is None:
                # Todas las posiciones posibles están ocupadas:
                # cada fila tiene exactamente n - 1 conexiones.
                indptr = xp.arange(0, max_conexiones + 1, n - 1, dtype=int)
            else:
                # seleccion está ordenada, por lo que las posiciones están agrupadas por filas.
                # searchsorted obtiene directamente cuántas conexiones hay acumuladas hasta cada fila.
                indptr = xp.searchsorted(seleccion, xp.arange(n + 1, dtype=int) * (n - 1), side="left")

            self.__conexiones = self.__sp.csr_matrix((pesos, indices, indptr), shape=(n, n),
                                                     dtype=self.__dtype)
        else:
            self.__conexiones = xp.zeros((n, n), dtype=self.__dtype)
            self.__conexiones[filas, columnas] = pesos

    def _validar_conexiones(self) -> None:
        """
        Validar que la matriz de conexiones se haya creado correctamente, y que no tenga valores poco
        realistas.

        Raises
        ------
        ValueError
            - Si las dimensiones de la matriz son incorrectas.
            - Si la diagonal principal de la matriz no es 0.
            - Si alguno de los pesos está fuera del intervalo [-1, 1]
        """
        if self.__conexiones.shape != (self.__num_neuronas, self.__num_neuronas):
            raise ValueError("Dimensiones incorrectas de la matriz de conexiones.")

        if self.__xp.any(self.__conexiones.diagonal()):
            raise ValueError("La diagonal de la matriz de conexiones debe ser cero.")

        if self.__sparse:
            datos = self.__conexiones.data
        else:
            datos = self.__conexiones

        if self.__xp.any((datos < -1) | (datos > 1)):
            raise ValueError("Los pesos de las conexiones deben estar en el intervalo [-1, 1].")

    def _actualizar(self, I: Array | float, dt: float) -> Array:
        """
        Avanza un paso temporal de la simulación y actualiza el estado de todas las neuronas sin
        validar los parámetros de entrada.

        Parameters
        ----------
        I : Array | float
            Corriente de entrada aplicada a cada neurona. Puede ser un escalar para aplicar la misma
            corriente a todas las neuronas o un vector de longitud igual al número de neuronas.

        dt : float
            Paso temporal de simulación en milisegundos. Debe ser mayor que 0

        Returns
        -------
        Array
            Vector booleano de longitud num_neuronas indicando qué neuronas se han disparado.

        Notes
        -----
        Las conexiones se representan mediante una matriz donde las columnas corresponden a neuronas
        presinápticas y las filas a neuronas postsinápticas.

        La corriente total aplicada se calcula como:

            I_total = conexiones · spikes + I

        Este método está destinado al uso interno de la librería cuando los parámetros ya han sido
        validados previamente.
        """
        spikes_previos = (self.__v >= 30)
        v = self.__v
        u = self.__u

        if spikes_previos.any():
            v[spikes_previos] = self.__c[spikes_previos]
            u[spikes_previos] += self.__d[spikes_previos]

        if self.__num_conexiones == 0:
            I_total = I
        else:
            I_total = self.__conexiones.dot(spikes_previos.astype(self.__dtype)) + I

        # Evitar posibles asignaciones intermedias de elevar al cuadrado haciendo la multiplicación
        # directamente
        # Calcular v en dos pasos para estabilidad numérica
        v += 0.5 * dt * ((0.04 * v * v + 5 * v + 140 - u + I_total))
        v += 0.5 * dt * ((0.04 * v * v + 5 * v + 140 - u + I_total))
        u += dt * (self.__a * (self.__b * v - u))

        spikes_actuales = (v >= 30)
        v[spikes_actuales] = self.__dtype(30)

        return spikes_actuales

    def _informacion_neurona(self, indice: int) -> dict[str, str | bool | dict[str, float]]:
        """
        Obtiene la información completa de una neurona concreta de la red.

        Parameters
        ----------
        indice : int
            Índice de la neurona dentro de la red.

        Returns
        -------
        dict[str, str | bool | dict[str, float]]
            Diccionario con el índice, nombre, tipo, parámetros y estado actual de la neurona.

        Raises
        ------
        TypeError
            Si el índice no es un entero.

        IndexError
            Si el índice está fuera del rango de neuronas existentes.
        """
        if not isinstance(indice, int):
            raise TypeError("El índice debe ser un entero.")

        if indice < 0 or indice >= self.__num_neuronas:
            raise IndexError("El índice de la neurona está fuera del rango permitido.")

        return {"indice": indice,
                "nombre": self.__nombre[indice],
                "es_excitatoria": bool(self.__es_excitatoria[indice]),
                "parametros": {"a": float(self.__a[indice]),
                               "b": float(self.__b[indice]),
                               "c": float(self.__c[indice]),
                               "d": float(self.__d[indice])},
                "estado": {"v": float(self.__v[indice]),
                           "u": float(self.__u[indice])}}

    def _estado(self) -> tuple[Array, Array]:
        """
        Referencias directas al estado interno de la red.

        A diferencia de la propiedad estado, este método no crea copias de los datos y está destinado
        exclusivamente para uso interno del simulador.

        Modificar los objetos devueltos modifica directamente el estado interno de la red.

        Returns
        -------
        tuple[Array, Array]
            Tupla con los vectores de potencial de membrana (v) y de recuperación (u) de todas las
            neuronas.
        """
        return self.__v, self.__u


    # --------------------------------------------------
    # PROPIEDADES
    # --------------------------------------------------

    @property
    def estado(self) -> dict[str, list]:
        """
        Obtiene el estado actual de la red en los vectores v y u.

        Returns
        -------
        dict[str, list]
            Diccionario con claves v y u, donde cada valor es una lista con el estado de cada neurona.
        """
        return {"v": self.__v.tolist(),
                "u": self.__u.tolist()}

    @property
    def parametros(self) -> dict[str, list[float]]:
        """
        Obtiene los parámetros (a, b, c, d) de todas las neuronas.

        Returns
        -------
        dict[str, list[float]]
            Diccionario con los cuatro parámetros (a, b, c, d) de cada neurona.
        """
        return {"a": self.__a.tolist(),
                "b": self.__b.tolist(),
                "c": self.__c.tolist(),
                "d": self.__d.tolist()}

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
            "numpy" si se usa NumPy, o "cupy" si se usa CuPy.
        """
        return "cupy" if self.__uso_gpu else "numpy"

    @property
    def neuronas(self) -> dict[Neurona, int]:
        """
        Los tipos de neuronas y su cantidad en la red.

        Returns
        -------
        dict[Neurona, int]
            Copia del diccionario original que asocia cada objeto Neurona con el número de instancias
            de ese tipo. Las cadenas de tipos se han convertido a instancias de Neurona al crear la
            red.
        """
        return self.__neuronas.copy()

    @property
    def nombre(self) -> list[str]:
        """
        Los nombres de las neuronas de la red, en el orden en el que están en la misma.

        Returns
        -------
        list[str]
            Lista con los nombres ordenados de las neuronas en la red.
        """
        return self.__nombre.copy()

    @property
    def es_excitatoria(self) -> list[bool]:
        """
        Indica si cada neurona de la red es excitatoria o inhibitoria.

        Returns
        -------
        list[bool]
            Lista donde True indica que es excitatoria y False inhibitoria.
        """
        return self.__es_excitatoria.tolist()

    @property
    def conexiones(self) -> list[list[float]]:
        """
        Obtiene la matriz de pesos sinápticos de la red, devolviéndola siempre en forma de matriz densa
        formada por listas de listas.

        La matriz devuelta es una copia y modificarla no altera la red interna.

        Returns
        -------
        list[list[float]]
            Matriz cuadrada de tamaño num_neuronas x num_neuronas donde cada valor representa el peso
            de la conexión de una neurona presináptica (columnas) a una postsináptica (filas). La
            diagonal es siempre cero.
        """
        if self.__sparse:
            return self.__conexiones.toarray().tolist()
        else:
            return self.__conexiones.tolist()

    @property
    def aleatorizacion(self) -> DictAleatorizacion:
        """
        Valores de aleatorización de los parámetros y las conexiones, junto con la semilla usada
        para el generador de números aleatorios.

        Si la matriz de conexiones no se ha generado aleatoriamente, aleat_conex será None.

        Los valores de aleat_conex están en el intervalo (0, 1].

        Returns
        -------
        DictAleatorizacion
            Diccionario con los valores de aleatorización de los parámetros y de los pesos
            de las conexiones, junto con la semilla usada.
        """
        aleat_param = {clave: valor.copy() for clave, valor in self.__aleat_param.items()}
        return {"aleat_param": aleat_param, "aleat_conex": self.__aleat_conex, "semilla": self.__semilla}

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
    def precision(self) -> Literal[32, 64]:
        """
        Precisión utilizada internamente por la red.

        Returns
        -------
        Literal[32, 64]
            Número de bits utilizados para almacenar los datos.
        """
        return self.__precision

    @property
    def dtype(self) -> np.float32 | cp.float32 | np.float64 | cp.float64:
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
        Indica si las conexiones utilizan una representación dispersa por filas CSR.

        Returns
        -------
        bool
            True si la matriz de conexiones está almacenada como una matriz CSR.
        """
        return self.__sparse

    @property
    def estadisticas(self) -> dict[str, int | float]:
        """
        Obtiene estadísticas generales de la red.

        Returns
        -------
        dict[str, int | float]
            Diccionario con información sobre neuronas y conexiones.
        """
        excitatorias = int(self.__xp.count_nonzero(self.__es_excitatoria))
        inhibitorias = self.__num_neuronas - excitatorias

        if self.__num_neuronas > 1:
            densidad = self.__num_conexiones / (self.__num_neuronas * (self.__num_neuronas - 1))
        else:
            densidad = 0

        if self.__sparse:
            pesos = self.__conexiones.data
        else:
            pesos = self.__conexiones[self.__conexiones != 0]

        conexiones_exc = int(self.__xp.count_nonzero(pesos > 0))
        conexiones_inh = int(self.__xp.count_nonzero(pesos < 0))

        return {"num_neuronas": self.__num_neuronas,
                "excitatorias": excitatorias,
                "inhibitorias": inhibitorias,
                "num_conexiones": self.__num_conexiones,
                "densidad": densidad,
                "conexiones_excitatorias": conexiones_exc,
                "conexiones_inhibitorias": conexiones_inh}
