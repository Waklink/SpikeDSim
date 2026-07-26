import cupy as cp
import numpy as np
import scipy.sparse as sp
import cupyx.scipy.sparse as cpsp
from .Neurona import Neurona
from typing import TypeAlias, Literal
from numbers import Real

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
        Tipos de neuronas y su cantidad en la red. Estos tipos pueden especificarse con instancias
        de Neurona o nombres de tipos predefinidos, que se convierten a instancias al construirse la
        red.

    conexiones : list
        Matriz de pesos sinápticos con diagonal cero. Puede almacenarse como una matriz densa o
        mediante una representación dispersa CSR.

        Los elementos de la matriz representan conexiones desde la neurona de la columna (presináptica)
        hacia la neurona de la fila (postsináptica). Los pesos positivos corresponden a conexiones
        excitatorias y los negativos a inhibitorias. 

    num_neuronas : int
        Número total de neuronas en la red.

    num_conexiones : int
        Número total de conexiones diferentes de cero.

    estado : dict[str, list]
        Estado actual de la red con v y u.

    parametros : dict[str, list]
        Parámetros (a, b, c, d) de todas las neuronas.

    dtype : type[np.floating] | type[cp.floating]:
        Tipo de dato utilizado internamente por los arrays de la red. Puede ser float32 o
        float64 según la precisión seleccionada en el constructor.

    sparse: bool
        Indica si la matriz de conexiones se almacena utilizando una representación dispersa CSR.
    """

    def __init__(self, neuronas: dict[Neurona | str, int], conexiones: int | list[list[float]] | Array | SparseArray = 0,
                 backend: Literal["numpy", "cupy"] = "numpy", precision: Literal[32, 64] = 32,
                 sparse: bool = True, semilla: int | None = None):
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

        Raises
        ------
        TypeError
            Si alguno de los parámetros no es del tipo de dato correcto, neuronas tiene que ser un
            diccionario formado por instancias de Neurona o cadenas de texto y enteros, conexiones debe ser un entero
            positivo, una lista, o un Array o csr_matriz que coincida con el backend seleccionado.

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

        # Inicializar y llenar vectores de parámetros y estado inicial
        self._crear_vectores()

        # Creación y llenado de la matriz de conexiones entre neuronas
        self._crear_matriz_conexiones(conexiones, semilla)
        
        # Comprobaciones de la matriz de conexiones
        self._validar_conexiones()

        # Guardar número de conexiones que existan.
        self.__num_conexiones = self.__conexiones.nnz if self.__sparse else int(self.__xp.count_nonzero(self.__conexiones))
    

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
            try:
                cp.zeros(1)
            except Exception as e:
                raise RuntimeError("Su equipo no puede realizar operaciones en la gpu, por favor "
                                   "asegúrese de tener una tarjeta gráfica compatible con CuPy, y"
                                   " de tener el driver adecuado instalado.") from e
            
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


    def _crear_vectores(self) -> None:
        """
        Inicializa y llena los vectores de parámetros y el estado inicial de todas las neuronas.
        """

        # Creación de vectores con los parámetros de las neuronas
        self.__a = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__b = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__c = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__d = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__v = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__u = self.__xp.empty(self.__num_neuronas, dtype=self.__dtype)
        self.__tipo = self.__xp.empty(self.__num_neuronas, dtype=bool)
        
        # LLenar los vectores de parámetros
        indice_actual = 0

        for neurona, cantidad in self.__neuronas.items():
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

        # Guardar el estado inicial de la red
        self.__v_inicial = self.__v.copy()
        self.__u_inicial = self.__u.copy()


    def _crear_matriz_conexiones(self, conexiones: int | list[list[float]] | Array | SparseArray,
                                 semilla: int | None) -> None:
        """
        Crear la matriz de conexiones, basándose en si es dispersa o no y si se proporciona una matriz
        completa o se quiere crear valores aleatorios.

        Parameters
        ----------
        conexiones : int | list[list[float]] | Array | SparseArray
            El número de conexiones aleatorias a crear, o una matriz con las conexiones ya establecidas.
            En el caso de que sea un Array o un SparseArray, este tiene que ser del backend utilizado.

        semilla : int | None
            Semilla a usar para generar valores aleatorios si conexiones es un entero.

        Raises
        ------
        TypeError
            Si semilla no es un entero o conexiones no es ningún tipo aceptado. 
        """
        # Matriz aleatoria con número de conexiones especificado
        if isinstance(conexiones, int):
            # Comprobación de semilla
            if semilla is not None and not isinstance(semilla, int):
                raise TypeError("La semilla debe ser un entero.")
            
            self._crear_conexiones_aleatorias(conexiones, semilla)
        # Matriz a partir de una lista
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


    def _crear_conexiones_aleatorias(self, num: int, semilla: int | None = None) -> None:
        """
        Crear un número especificado de conexiones aleatorias con pesos aleatorios.

        Parameters
        ----------
        num : int
            Número de conexiones a crear.
        
        semilla : int | None, optional
            Semilla a utilizar en la generación aleatoria. Por defecto es None.

        Raises
        ------
        ValueError
            Si el número de conexiones es negativo o si supera el máximo posible según el número de
            neuronas existentes.
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
                self.__conexiones = self.__sp.csr_matrix((self.__num_neuronas, self.__num_neuronas),
                                                         dtype=self.__dtype)
            else:
                self.__conexiones = xp.zeros((self.__num_neuronas, self.__num_neuronas), dtype=self.__dtype)
            return
        
        # Seleccionar posiciones aleatorias fuera de la diagonal
        posibles = xp.arange(max_conexiones)

        if semilla is not None:
            rng = xp.random.RandomState(semilla)
        else:
            rng = xp.random

        seleccion = rng.choice(posibles, size=num, replace=False)

        # Convertir índices lineales a fila y columna
        filas = seleccion // (self.__num_neuronas - 1)
        columnas = seleccion % (self.__num_neuronas - 1)

        # Ajustar para saltar diagonal
        columnas = xp.where(columnas >= filas, columnas + 1, columnas)

        # Generar pesos según tipo de neurona presináptica (columnas)
        pesos = rng.random(num).astype(self.__dtype)

        pesos = xp.where(self.__tipo[columnas], pesos, -pesos)

        if self.__sparse:
            # Crear directamente CSR sin pasar por matriz densa
            self.__conexiones = self.__sp.csr_matrix((pesos, (filas, columnas)),
                                                     shape=(self.__num_neuronas, self.__num_neuronas), 
                                                     dtype=self.__dtype)
        else:
            # Crear matriz densa
            self.__conexiones = xp.zeros((self.__num_neuronas, self.__num_neuronas), dtype=self.__dtype)
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
            - Si alguno de los pesos está fuera del intervalo (-1, 1)
        """
        if self.__conexiones.shape != (self.__num_neuronas, self.__num_neuronas):
            raise ValueError("Dimensiones incorrectas de la matriz de conexiones.")
        
        if self.__conexiones.diagonal().any():
            raise ValueError("La diagonal de la matriz de conexiones debe ser cero.")

        if self.__sparse:
            datos = self.__conexiones.data
        else:
            datos = self.__conexiones

        if self.__xp.any((datos <= -1) | (datos >= 1)):
            raise ValueError("Los pesos de las conexiones deben estar en el intervalo (-1, 1).")



    def actualizar(self, I: Array | list[float] | float | int, dt: float = 0.5) -> Array:
        """
        Avanza un paso temporal de la simulación y actualiza el estado de todas las neuronas.

        Parameters
        ----------
        I : Array | list[float] | float | int
            Corriente de entrada aplicada a cada neurona. Puede ser un escalar para aplicar la misma
            corriente a todas las neuronas o un vector de longitud igual al número de neuronas.

        dt : float
            Tamaño del paso temporal en milisegundos.

        Returns
        -------
        Array
            Vector booleano de longitud num_neuronas indicando qué neuronas se han disparado.

        Raises
        ------
        TypeError
            Si I no es un vector ni un número real, o si dt no es un número real.

        ValueError
            Si el tamaño de I no coincide con el número de neuronas o si dt no es positivo.
        
        Notes
        -----
        Las conexiones se representan mediante una matriz donde las columnas corresponden a neuronas
        presinápticas y las filas a neuronas postsinápticas.

        La corriente total aplicada se calcula como:

            I_total = conexiones · spikes + I
        """

        if isinstance(I, (np.ndarray, cp.ndarray)):
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
        elif not isinstance(I, Real):
            raise TypeError("La entrada de corriente debe ser un vector de longitud N, donde N es el "
                            "número total de neuronas, o ser un número a usar para aplicar el mismo"
                            " la misma corriente a todas las neuronas.")
        elif isinstance(I, Real):
            # cambiar I al dtype interno, para evitar que se haga promoción dentro de los arrays de
            # v y de u a float64.
            I = self.__dtype(I)
        
        if not isinstance(dt, Real):
            raise TypeError("El paso temporal debe ser un número real.")
        
        if dt <= 0:
            raise ValueError("El paso temporal debe ser positivo.")
        
        spikes_previos = (self.__v >= 30)

        if spikes_previos.any():
            self.__v[spikes_previos] = self.__c[spikes_previos]
            self.__u[spikes_previos] += self.__d[spikes_previos]

        # cambiar dt al dtype interno, para evitar que se haga promoción dentro de los arrays de v
        # y de u a float64.
        dt = self.__dtype(dt)

        I_total = self.__conexiones.dot(spikes_previos.astype(self.__dtype)) + I

        # Evitar posibles asignaciones intermedias de elevar al cuadrado haciendo la multiplicación
        # directamente
        self.__v += dt * ((0.04 * self.__v * self.__v + 5 * self.__v + 140 - self.__u + I_total))
        self.__u += dt * (self.__a * (self.__b * self.__v - self.__u))
        
        spikes_actuales = (self.__v >= 30)

        return spikes_actuales
    

    def reiniciar(self) -> None:
        """
        Restaura el estado de la red al estado inicial almacenado en la creación.

        Esto restablece los vectores v y u a los valores iniciales usados para construir la red.
        """
        self.__v[:] = self.__v_inicial
        self.__u[:] = self.__u_inicial
    

    def establecer_estado(self, v: Array | list[float] | None = None, u: Array | list[float] | None = None) -> None:
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
            if isinstance(v, (np.ndarray, cp.ndarray)):
                if v.shape != (self.__num_neuronas,):
                    raise ValueError("El vector de potenciales de membrana tiene que tener una longitud"
                                    f" de {self.__num_neuronas} elementos.")
            elif isinstance(v, list):
                if len(v) != self.__num_neuronas:
                    raise ValueError("El vector de potenciales de mebrana tiene que tener una longitud"
                                    f" de {self.__num_neuronas} elementos.")
            else:
                raise TypeError("El vector de potenciales de membrana nuevo debe ser un vector de "
                                "NumPy o de CuPy, o una lista de números.")

            self.__v[:] = self.__xp.asarray(v, dtype=self.__dtype)

        if u is not None:
            if isinstance(u, (np.ndarray, cp.ndarray)):
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
                                  self.__sparse)

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
                                  self.__precision, nuevo_sparse)

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
                                  nueva_precision, self.__sparse)

        nueva_red.establecer_estado(self.__v.copy(), self.__u.copy())

        return nueva_red

    def copy(self) -> RedDeNeuronas:
        """
        Devuelve una copia de esta red. Es equivalente a red.convertir_backend(red.backend) o
        red.convertir_formato(red.sparse)

        Returns
        -------
        RedDeNeuronas
            Copia de la red de neuronas actual
        """
        return RedDeNeuronas(self.__neuronas.copy(), self.__conexiones.copy(), self.backend,
                             self.__precision, self.__sparse)
    

    def _estado(self) -> tuple[Array, Array]:
        """
        Devuelve referencias directas al estado interno de la red.

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


    @property
    def estado(self) -> dict[str, list]:
        """
        Obtiene el estado actual de la red en los vectores v y u.

        Returns
        -------
        dict[str, list]
            Diccionario con claves v y u, donde cada valor es una lista con el estado de cada neurona.
        """
        return {
            "v": self.__v.tolist(),
            "u": self.__u.tolist()
        }

    @property
    def parametros(self) -> dict[str, list]:
        """
        Obtiene los parámetros (a, b, c, d) de todas las neuronas.

        Returns
        -------
        dict[str, list]
            Diccionario con los cuatro parámetros (a, b, c, d) de cada neurona.
        """
        return {
            "a": self.__a.tolist(),
            "b": self.__b.tolist(),
            "c": self.__c.tolist(),
            "d": self.__d.tolist()
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
            "numpy" si se usa NumPy, o "cupy" si se usa CuPy.
        """
        return "cupy" if self.__uso_gpu else "numpy"

    @property
    def neuronas(self) -> dict[Neurona, int]:
        """
        Devuelve los tipos de neuronas y su cantidad en la red.

        Returns
        -------
        dict[Neurona, int]
            Copia del diccionario original que asocia cada objeto Neurona con el número de instancias
            de ese tipo. Las cadenas de tipos se han convertido a instancias de Neurona al crear la
            red.
        """
        return self.__neuronas.copy()

    @property
    def conexiones(self) -> list[list[float]]:
        """
        Obtiene la matriz de pesos sinápticos de la red, devolviendola siempre en forma de matriz densa
        formada por listas de listas.

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
        Indica si las conexiones utilizan una representación dispersa por filas CSR.

        Returns
        -------
        bool
            True si la matriz de conexiones está almacenada como una matriz CSR.
        """
        return self.__sparse
