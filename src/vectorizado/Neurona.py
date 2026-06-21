class Neurona:
    """
    Representación de una neurona del modelo de Izhikevich.
    
    Atributos:
        a (float): Parámetro que regula la velocidad de recuperación de la neurona.
        b (float): Parámetro que regula la sensibilidad de la neurona al estímulo.
        c (float): Parámetro que regula el potencial de membrana de recuperación.
        d (float): Parámetro que regula la velocidad de recuperación de la variable de recuperación.
        v (float): Potencial de membrana de la neurona (en mV).
        u (float): Variable de recuperación de la neurona.
        tipo (str): tipo de la neurona, puede ser un tipo precargado (RS, IB, CH, FS, LTS, TC, RZ), o personalizado.
        es_excitatoria (bol): Representación de si la neurona es excitatoria o inhibitoria.
    """

    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = -65, d: float = 2, v_inicial: float = -65,
                 u_inicial: float | None = None, tipo: str = "Personalizado", es_excitatoria: bool = True):
        """
        Inicializa una instancia de la clase Neurona, con parámetros personalizados.

        Args:
            a (float): Parámetro que regula la velocidad de recuperación de la neurona.
            b (float): Parámetro que regula la sensibilidad de la neurona al estímulo.
            c (float): Parámetro que regula el potencial de membrana de recuperación.
            d (float): Parámetro que regula la velocidad de recuperación de la variable de recuperación.
            v_inicial (float, optional): Potencial de membrana inicial de la neurona. Si no se proporciona, se usa un valor por defecto.
            u_inicial (float, optional): Variable de recuperación inicial de la neurona. Si no se proporciona, se calcula automáticamente.
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


    @classmethod
    def predefinida(cls, tipo: str, v_inicial: float = -65, u_inicial: float | None = None) -> Neurona:
        """
        Crea una instancia de la clase Neurona, con parámetros precargados según el tipo de la neurona.

        Args:
            tipo (str): tipo de la neurona, debe ser un tipo precargado (RS, IB, CH, FS, LTS, TC, RZ).
            v_inicial (float, optional): Potencial de membrana inicial de la neurona. Si no se proporciona, se usa un valor por defecto.
            u_inicial (float, optional): Variable de recuperación inicial de la neurona. Si no se proporciona, se calcula automáticamente.
        
        Returns:
            Neurona: Una instancia de la clase Neurona con los parámetros típicos según el tipo pasado.
        
        Raises:
            ValueError: Si el tipo de la neurona no es reconocido.
        """
        ALIAS = {
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

        TIPOS = {
            "rs": (0.02, 0.2, -65, 8, "RS", True),
            "ib": (0.02, 0.2, -55, 4, "IB", True),
            "ch": (0.02, 0.2, -50, 2, "CH", True),
            "fs": (0.1, 0.2, -65, 2, "FS", False),
            "lts": (0.02, 0.25, -65, 2, "LTS", False),
            "tc": (0.02, 0.25, -65, 0.05, "TC", False),
            "rz": (0.1, 0.26, -65, 2, "RZ", False)
        }

        try:
            tipo = ALIAS[tipo.lower()]
            a, b, c, d, nombre, es_excitatoria = TIPOS[tipo]
        except KeyError:
            raise ValueError(f"El tipo {tipo} no existe en los predefinidos, por favor, use un valor precargado \
                             o cree un tipo personalizado con el constructor.")
        
        v = v_inicial
        u = b * v if u_inicial is None else u_inicial

        return cls(a, b, c, d, v, u, nombre, es_excitatoria)


    def actualizar(self, I: float, dt: float) -> bool:
        """
        Actualiza las variables de estado de la neurona.

        Args:
            I (float): Corriente de entrada.
            dt (float): Paso de tiempo.

        Returns:
            bool: True si la neurona se ha disparado, False si no.
        """

        self.__v += dt * (0.04 * self.__v**2 + 5 * self.__v + 140 - self.__u + I)
        self.__u += dt * (self.__a * (self.__b * self.__v - self.__u))

        if self.__v >= 30:
            self.__v = self.__c
            self.__u += self.__d
            return True
        
        return False


    @property
    def estado(self) -> tuple[float, float]:
        """
        Obtiene el estado actual de la neurona.
        """
        return self.__v, self.__u
    
    @property
    def tipo(self) -> str:
        """
        Obtiene el tipo de la neurona.
        """
        return self.__tipo
    
    @property
    def es_excitatoria(self) -> bool:
        """
        Obtiene si la neurona es excitatoria o no, en cuyo caso se considera inhibitoria.
        """
        return self.__es_excitatoria
    
    @property
    def parametros(self) -> tuple[float, float, float, float]:
        """
        Obtiene los parámetros de la neurona.
        """
        return self.__a, self.__b, self.__c, self.__d
