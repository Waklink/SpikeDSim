from numbers import Real

class Neurona:
    """
    Representación de una neurona del modelo de Izhikevich.

    Permite definir neuronas con parámetros personalizados o utilizar configuraciones
    predefinidas. Mantiene el estado dinámico ('v', 'u') y permite simular la evolución
    temporal de manera individual.

    Attributes
    ----------
    estado : tuple[float, float]
        Estado actual de la neurona como (v, u), donde v es el potencial de membrana y u es la
        variable de recuperación.

    parametros : tuple[float, float, float, float]
        Parámetros del modelo Izhikevich como (a, b, c, d), donde a representa la velocidad de
        recuperación, b la sensibilidad al estímulo, c el potencial de recuperación después de un
        disparo y d el incremento de la variable de recuperación después de un disparo.

    tipo : str
        Tipo de la neurona, puede ser un tipo predefinido (RS, IB, CH, FS, LTS, TC o RZ) o personalizado.

    es_excitatoria : bool
        Indica si la neurona es excitatoria (True) o inhibitoria (False).
    """

    def __init__(self, a: float = 0.02, b: float = 0.2, c: int | float = -65, d: int | float = 2,
                 v_inicial: int | float = -65, u_inicial: int | float | None = None,
                 tipo: str = "Personalizado", es_excitatoria: bool = True):
        """
        Inicializa una instancia de la clase Neurona con parámetros personalizados.

        Parameters
        ----------
        a : float
            Parámetro que regula la velocidad de recuperación de la neurona.

        b : float
            Parámetro que regula la sensibilidad de la neurona al estímulo.

        c : int | float
            Parámetro que regula el potencial de membrana de recuperación.

        d : int | float
            Parámetro que regula la velocidad de recuperación de la variable de recuperación.

        v_inicial : int | float, optional
            Potencial de membrana inicial de la neurona. Por defecto es -65 mV.

        u_inicial : int | float, optional
            Variable de recuperación inicial de la neurona. Por defecto se calcula como b * v_inicial.

        tipo : str
            Tipo de la neurona.

        es_excitatoria : bool
            Indica si la neurona es excitatoria (True) o inhibitoria (False).
        
        Raises
        ------
        TypeError
            Si alguno de los parámetros pasados no son del tipo correcto, los números deben ser reales,
            el tipo un string y es_excitatoria un booleano.
        """
        if not (isinstance(a, Real) and isinstance(b, Real) and isinstance(c, Real) and isinstance(d, Real)):
            raise TypeError("Los parámetros deben ser números reales.")

        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

        if not isinstance(tipo, str):
            raise TypeError("El tipo de neurona debe ser un string con un nombre que represente a la neurona.")
        
        if not isinstance(es_excitatoria, bool):
            raise TypeError("El parámetro es_excitatoria debe ser un booleano que representa si la neurona \
                            es excitatoria o inhibitoria.")

        self.__tipo = tipo
        self.__es_excitatoria = es_excitatoria

        if not isinstance(v_inicial, Real):
            raise TypeError("El potencial de membrana inicial, v_inicial, debe ser un número real.")
        
        if u_inicial is not None and not isinstance(u_inicial, Real):
            raise TypeError("La variable de recuperación inicial, u_inicial, debe ser un número real o None, \
                            para que se calcule automáticamente.")

        self.__v = v_inicial
        
        if u_inicial is None:
            self.__u = self.__b * self.__v
        else:
            self.__u = u_inicial
        
        self.__v_inicial = self.__v
        self.__u_inicial = self.__u


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
    def predefinida(cls, tipo: str, v_inicial: int | float = -65,
                    u_inicial: int | float | None = None) -> Neurona:
        """
        Crea una instancia de Neurona con parámetros precargados según el tipo.

        Parameters
        ----------
        tipo : str
            Tipo de la neurona. Los tipos disponibles son:
            RS (Regular Spiking), IB (Intrinsically Bursting), CH (Chattering), FS (Fast Spiking),
            LTS (Low Threshold Spiking), TC (Thalamo Cortical), RZ (Resonator).
            Se aceptan variantes en minúsculas y con '-' en lugar de espacios.

        v_inicial : int | float, optional
            Potencial de membrana inicial de la neurona. Por defecto es -65 mV.

        u_inicial : int | float, optional
            Variable de recuperación inicial de la neurona. Por defecto se calcula como b * v_inicial.

        Returns
        -------
        Neurona
            Una instancia de Neurona con parámetros típicos según el tipo solicitado.

        Raises
        ------
        TypeError
            Si el tipo no es un string o los valores de estado iniciales no son números reales, y,
            en el caso de u_inicial, tampoco es None.

        ValueError
            Si el tipo de neurona no es reconocido.
        """

        if not isinstance(tipo, str):
            raise TypeError("El tipo de neurona debe ser un string con un nombre que represente a la neurona.")
        
        if not isinstance(v_inicial, Real):
            raise TypeError("El potencial de membrana inicial, v_inicial, debe ser un número real.")
        
        if u_inicial is not None and not isinstance(u_inicial, Real):
            raise TypeError("La variable de recuperación inicial, u_inicial, debe ser un número real o None, \
                            para que se calcule automáticamente.")

        tipo = tipo.strip().lower()
        
        if tipo not in cls._ALIAS:
            raise ValueError(f"El tipo {tipo} no existe, por favor use un valor predefinido o cree \
                                una neurona personalizada con el constructor.")
        
        tipo = cls._ALIAS[tipo]
        a, b, c, d, nombre, es_excitatoria = cls._TIPOS[tipo]
        
        v = v_inicial
        u = b * v if u_inicial is None else u_inicial

        return cls(a, b, c, d, v, u, nombre, es_excitatoria)


    def actualizar(self, I: float, dt: float = 0.5) -> bool:
        """
        Actualiza el estado de la neurona para un paso temporal.

        En caso de disparo (v >= 30), reinicia v y actualiza u según las ecuaciones de Izhikevich.

        Parameters
        ----------
        I : float
            Corriente de entrada.

        dt : float
            Paso temporal de simulación. Debe ser positivo.

        Returns
        -------
        bool
            True si la neurona se dispara en este paso, False en caso contrario.

        Raises
        ------
        TypeError
            Si I o dt no son números reales.

        ValueError
            Si dt es menor o igual a 0.
        """

        if not isinstance(I, Real):
            raise TypeError("La corriente de entrada debe ser un número real.")
        
        if not isinstance(dt, Real):
            raise TypeError("El paso temporal tiene que ser un número real.")
        

        if dt <= 0:
            raise ValueError("El paso temporal tiene que ser positivo.")

        # Evitar asignaciones intermedias de elevar al cuadrado haciendo la multiplicación directamente
        self.__v += dt * (0.04 * self.__v * self.__v + 5 * self.__v + 140 - self.__u + I)
        self.__u += dt * (self.__a * (self.__b * self.__v - self.__u))

        if self.__v >= 30:
            self.__v = self.__c
            self.__u += self.__d
            return True
        
        return False
    

    def reiniciar(self) -> None:
        """
        Restaura el estado de la neurona al estado inicial de la creación.

        Esto restablece los valores de v y u a los iniciales usados al crear la neurona.
        """
        self.__v = self.__v_inicial
        self.__u = self.__u_inicial
    

    def establecer_estado(self, v: float | int | None = None, u: float | int | None = None) -> None:
        """
        Establece el estado interno v y/o u de la neurona en nuevos valores.

        Parameters
        ----------
        v : float | int | None, optional
            Nuevo valor del potencial de membrana. Si es None, no se modifica v.

        u : float | int | None, optional
            Nuevo valor de la variable de recuperación. Si es None, no se modifica u.

        Raises
        ------
        TypeError
            Si alguno de los valores pasados no es un número real.
        """
        if v is not None:
            if not isinstance(v, Real):
                raise TypeError("El potencial de membrana, v, pasado debe ser un número real.")
            
            self.__v = v
        
        if u is not None:
            if not isinstance(u, Real):
                raise TypeError("La variable de recuperación, u, pasada debe ser un número real.")
            
            self.__u = u


    @classmethod
    def alias(cls) -> dict[str, str]:
        """
        Obtiene el diccionario de alias para los tipos de neurona.

        Returns
        -------
        dict[str, str]
            Mapa de alias a nombres canónicos de tipo.
        """
        return cls._ALIAS.copy()
    

    @classmethod
    def tipos(cls) -> dict[str, tuple[float, float, float, float, str, bool]]:
        """
        Obtiene los parámetros de los tipos de neurona predefinidos.

        Returns
        -------
        dict[str, tuple[float, float, float, float, str, bool]]
            Diccionario que enlaza cada tipo canónico con los valores (a, b, c, d, nombre, es_excitatoria).
        """
        return cls._TIPOS.copy()
    

    @classmethod
    def tipos_disponibles(cls) -> tuple[str, ...]:
        """
        Obtiene los códigos de los tipos de neurona predefinidos.

        Returns
        -------
        tuple[str, ...]
            Tupla con los códigos de tipo válidos.
        """
        return tuple(cls._TIPOS.keys())
    

    def _estado(self) -> tuple[float, float]:
        """
        Devuelve referencias directas al estado interno de la neurona.

        A diferencia de la propiedad estado, este método está destinado exclusivamente
        para uso interno, donde se requiere acceder al estado con el menor coste posible.

        Returns
        -------
        tuple[float, float]
            Tupla con los valores actuales de v y u.
        """
        return self.__v, self.__u


    @property
    def estado(self) -> tuple[float, float]:
        """
        Devuelve una copia del estado actual de la neurona.

        Returns
        -------
        tuple[float, float]
            Tupla (v, u) con el potencial de membrana y la variable de recuperación.
        """
        return float(self.__v), float(self.__u)
    
    @property
    def tipo(self) -> str:
        """
        Devuelve el nombre del tipo de neurona.

        Returns
        -------
        str
            Tipo de la neurona.
        """
        return self.__tipo
    
    @property
    def es_excitatoria(self) -> bool:
        """
        Indica si la neurona es excitatoria o inhibitoria.

        Returns
        -------
        bool
            True si la neurona es excitatoria, False si es inhibitoria.
        """
        return self.__es_excitatoria
    
    @property
    def parametros(self) -> tuple[float, float, float, float]:
        """
        Devuelve los parámetros del modelo de Izhikevich.

        Returns
        -------
        tuple[float, float, float, float]
            Parámetros (a, b, c, d) de la neurona.
        """
        return self.__a, self.__b, self.__c, self.__d

    @property
    def uso_gpu(self) -> bool:
        """
        Indica si la neurona utiliza GPU como backend de cálculo.

        Returns
        -------
        bool
            False, ya que la simulación individual de una neurona se realiza en CPU.
        """
        return False
