class Neurona:
    """
    Representación de una neurona del modelo de Izhikevich.

    Permite definir neuronas con parámetros personalizados, o utilizar configuraciones predefinidas, según el artículo de Izhikevich.
    Mantiene el estado dinámico (v, u) y permite simular la evolución temporal de la neurona de forma individual.
    
    Attributes:
        estado (tuple[float, float]): Estado actual de la neurona, formado por:
            - v (float): El potencial de membrana (en mV) de la neurona.
            - u (float): La variable de recuperación de la neurona.
        parametros (tuple[float, float, float, float]): Parámetros del modelo de Izhikevich, formado por:
            - a (float): Parámetro que regula la velocidad de recuperación de la neurona.
            - b (float): Parámetro que regula la sensibilidad de la neurona al estímulo.
            - c (float): Parámetro que regula el potencial de membrana de recuperación.
            - d (float): Parámetro que regula la velocidad de recuperación de la variable de recuperación.
        tipo (str): tipo de la neurona, puede ser un tipo precargado (RS, IB, CH, FS, LTS, TC, RZ), o personalizado.
        es_excitatoria (bool): Representación de si la neurona es excitatoria o inhibitoria.
    """

    def __init__(self, a: float = 0.02, b: float = 0.2, c: int | float = -65, d: int | float = 2, v_inicial: int | float = -65,
                 u_inicial: int | float | None = None, tipo: str = "Personalizado", es_excitatoria: bool = True):
        """
        Inicializa una instancia de la clase Neurona, con parámetros personalizados.

        Args:
            a (float): Parámetro que regula la velocidad de recuperación de la neurona.
            b (float): Parámetro que regula la sensibilidad de la neurona al estímulo.
            c (int | float): Parámetro que regula el potencial de membrana de recuperación.
            d (int | float): Parámetro que regula la velocidad de recuperación de la variable de recuperación.
            v_inicial (int | float, optional): Potencial de membrana inicial de la neurona. Por defecto se le asigna un valor de -65 mV.
            u_inicial (int | float, optional): Variable de recuperación inicial de la neurona. Por defecto se calcula automáticamente
                                         como b * v.
            tipo (str): tipo de la neurona, puede ser un tipo precargado (RS, IB, CH, FS, LTS, TC, RZ), o personalizado.
            es_excitatoria (bool): Tipo de la neurona según su sinapsis, en caso de ser False, es inhibitoria.
        """

        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

        self.__tipo = tipo
        self.__es_excitatoria = es_excitatoria

        self.__v = v_inicial
        
        if u_inicial is None:
            self.__u = self.__b * self.__v
        else:
            self.__u = u_inicial


    _ALIAS = {
        "rs": "rs",
        "regular spiking": "rs",
        "regular-spiking": "rs",

        "ib": "ib",
        "intrinsically bursting": "ib",
        "intrinsically-bursting": "ib",

        "ch": "ch",
        "chattering": "ch",

        "fs": "fs",
        "fast spiking": "fs",
        "fast-spiking": "fs",

        "lts": "lts",
        "low threshold spiking": "lts",
        "low-threshold spiking": "lts",
        "low-threshold-spiking": "lts",

        "tc": "tc",
        "thalamocortical": "tc",
        "thalamo cortical": "tc",
        "thalamo-cortical": "tc",

        "rz": "rz",
        "resonator": "rz"
    }

    _TIPOS = {
        "rs": (0.02, 0.2, -65, 8, "RS", True),
        "ib": (0.02, 0.2, -55, 4, "IB", True),
        "ch": (0.02, 0.2, -50, 2, "CH", True),
        "fs": (0.1, 0.2, -65, 2, "FS", False),
        "lts": (0.02, 0.25, -65, 2, "LTS", False),
        "tc": (0.02, 0.25, -65, 0.05, "TC", True),
        "rz": (0.1, 0.26, -65, 2, "RZ", False)
    }

    @classmethod
    def predefinida(cls, tipo: str, v_inicial: float = -65, u_inicial: float | None = None) -> Neurona:
        """
        Crea una instancia de la clase Neurona, con parámetros precargados según el tipo de la neurona.

        Args:
            tipo (str): tipo de la neurona, debe ser un tipo predefinido. Los tipos disponibles son:
                - RS (Regular Spiking)
                - IB (Intrinsically Bursting)
                - CH (Chattering)
                - FS (Fast Spiking)
                - LTS (Low Threshold Spiking)
                - TC (Thalamocortical)
                - RZ (Resonator)

                Aceptándose variantes del nombre en minúsculas y con "-" como separador de palabras en vez del espacio " ".
            v_inicial (float, optional): Potencial de membrana inicial de la neurona. Por defecto es -65 mV.
            u_inicial (float, optional): Variable de recuperación inicial de la neurona. Por defecto se calcula automáticamente como b*v.
        
        Returns:
            Neurona: Una instancia de la clase Neurona con los parámetros típicos según el tipo pasado.
        
        Raises:
            ValueError: Si el tipo de la neurona no es reconocido.
        """

        try:
            tipo = tipo.strip().lower()
            if tipo not in cls._ALIAS.keys():
                raise ValueError(f"El tipo {tipo} no existe, por favor usa un valor predefinido, o cree un tipo personalizado con el constructor")
            tipo = cls._ALIAS[tipo]
            a, b, c, d, nombre, es_excitatoria = cls._TIPOS[tipo]
        except KeyError as e:
            raise ValueError(f"El tipo {tipo} no existe, por favor, use un valor predefinido o cree un tipo personalizado con el constructor.") from e
        
        v = v_inicial
        u = b * v if u_inicial is None else u_inicial

        return cls(a, b, c, d, v, u, nombre, es_excitatoria)


    def actualizar(self, I: float, dt: float = 0.5) -> bool:
        """
        Actualiza las variables de estado (v, u) de la neurona, simulando la ejecución de un paso temporal (dt).
        
        En el caso de que se produzca un disparo (v >=30 mV), se reinicia el estado según las ecuaciones de Izhikevich y se devuelve True.

        Args:
            I (float): Corriente de entrada.
            dt (float): Paso temporal de simulación, debe ser positivo.

        Returns:
            bool: True si la neurona se ha disparado, False si no.

        Raises:
            ValueError: Si el paso temporal, dt, es negativo o igual a 0.
        """

        if dt <= 0:
            raise ValueError("El paso temporal tiene que ser positivo.")

        self.__v += dt * (0.04 * self.__v**2 + 5 * self.__v + 140 - self.__u + I)
        self.__u += dt * (self.__a * (self.__b * self.__v - self.__u))

        if self.__v >= 30:
            self.__v = self.__c
            self.__u += self.__d
            return True
        
        return False


    @classmethod
    def alias(cls) -> dict[str, str]:
        """
        Diccionario con todos los alias existentes para cada tipo.
        """
        return cls._ALIAS.copy()
    

    @classmethod
    def tipos(cls) -> dict[str, tuple[float, float, float, float, str, bool]]:
        """
        Diccionario con los parámetros para cada tipo de neurona predefinidas.
        """
        return cls._TIPOS.copy()
    

    @classmethod
    def tipos_disponibles(cls) -> tuple[str, ...]:
        """
        Listado de los tipos de neuronas predefinidos.
        """
        return tuple(cls._TIPOS.keys())


    @property
    def estado(self) -> tuple[float, float]:
        """
        Estado actual (v, u) de la neurona.
        """
        return self.__v, self.__u
    
    @property
    def tipo(self) -> str:
        """
        Tipo de neurona.
        """
        return self.__tipo
    
    @property
    def es_excitatoria(self) -> bool:
        """
        Si la neurona es excitatoria (True) o inhibitoria (False).
        """
        return self.__es_excitatoria
    
    @property
    def parametros(self) -> tuple[float, float, float, float]:
        """
        Los parámetros de la neurona.
        """
        return self.__a, self.__b, self.__c, self.__d
