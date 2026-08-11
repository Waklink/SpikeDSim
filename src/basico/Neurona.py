from numbers import Real


class Neurona:
    """
    Representación de una neurona del modelo de Izhikevich.

    Permite definir neuronas con parámetros personalizados o utilizar configuraciones predefinidas.
    Mantiene el estado dinámico del modelo de Izhikevich (v, u) y permite actualizarlo mediante pasos
    individuales.

    Attributes
    ----------
    estado : tuple[float, float]
        Estado actual de la neurona como (v, u), donde v es el potencial de membrana y u es la variable
        de recuperación.

    parametros : tuple[float, float, float, float]
        Parámetros del modelo Izhikevich como (a, b, c, d), donde a representa la escala temporal de
        la variable de recuperación, b la sensibilidad de la variable de recuperación al potencial
        de membrana, c el potencial de membrana al que se reinicia la neurona después de un disparo
        y d el incremento de la variable de recuperación después de un disparo.

    nombre : str
        Nombre identificativo de la neurona, puede ser un tipo predefinido o un nombre personalizado.

    es_excitatoria : bool
        Indica si la neurona es excitatoria (True) o inhibitoria (False).

    Notes
    -----
    La integración temporal sigue el método propuesto por Izhikevich, realizando dos semipasos para
    la ecuación del potencial de membrana con el fin de mejorar la estabilidad numérica.
    """

    # --------------------------------------------------
    # CONSTANTES
    # --------------------------------------------------

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
        "rs": {"parametros": (0.02, 0.2, -65, 8),
               "nombre": "Regular Spiking",
               "es_excitatoria": True},

        "ib": {"parametros": (0.02, 0.2, -55, 4),
               "nombre": "Intrinsically Bursting",
               "es_excitatoria": True},

        "ch": {"parametros": (0.02, 0.2, -50, 2),
               "nombre": "Chattering",
               "es_excitatoria": True},

        "fs": {"parametros": (0.1, 0.2, -65, 2),
               "nombre": "Fast Spiking",
               "es_excitatoria": False},

        "lts": {"parametros": (0.02, 0.25, -65, 2),
               "nombre": "Low Threshold Spiking",
               "es_excitatoria": False},

        "tc": {"parametros": (0.02, 0.25, -65, 0.05),
               "nombre": "Thalamocortical",
               "es_excitatoria": True},

        "rz": {"parametros": (0.1, 0.26, -65, 2),
               "nombre": "Resonator",
               "es_excitatoria": False}
    }


    # --------------------------------------------------
    # CONSTRUCTOR
    # --------------------------------------------------

    def __init__(self, a: float, b: float, c: float, d: float, v_inicial: float = -65,
                 u_inicial: float | None = None, nombre: str = "Personalizado",
                 es_excitatoria: bool = True):
        """
        Inicializa una instancia de la clase Neurona con parámetros personalizados.

        Parameters
        ----------
        a : float
            Escala temporal de la variable de recuperación.

        b : float
            Sensibilidad de la variable de recuperación respecto al potencial de membrana.

        c : float
            Potencial de membrana al que se reinicia la neurona tras un disparo.

        d : float
            Incremento de la variable de recuperación tras un disparo.

        v_inicial : float, optional
            Potencial de membrana inicial de la neurona. Por defecto es -65 mV.

        u_inicial : float | None, optional
            Variable de recuperación inicial de la neurona. Por defecto, si no se proporciona, se
            calcula como b * v_inicial.

        nombre : str, optional
            Nombre de la neurona. Por defecto es "Personalizado".

        es_excitatoria : bool, optional
            Indica si la neurona es excitatoria (True) o inhibitoria (False). Por defecto es True.

        Raises
        ------
        TypeError
            Si alguno de los parámetros pasados no es del tipo correcto, los números deben ser reales,
            el nombre una cadena de texto y es_excitatoria un booleano.
        """
        # Validaciones
        self._validar_numero_real(a, "a")
        self._validar_numero_real(b, "b")
        self._validar_numero_real(c, "c")
        self._validar_numero_real(d, "d")

        if not isinstance(nombre, str):
            raise TypeError("El nombre de la neurona debe ser una cadena de texto.")

        if not isinstance(es_excitatoria, bool):
            raise TypeError("El parámetro es_excitatoria debe ser un booleano.")

        self._validar_numero_real(v_inicial, "v_inicial")

        if u_inicial is not None:
            self._validar_numero_real(u_inicial, "u_inicial", " o None para calcularlo automáticamente")

        # Asignaciones
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

        self.__nombre = nombre
        self.__es_excitatoria = es_excitatoria

        self.__v = v_inicial

        if u_inicial is None:
            self.__u = self.__b * self.__v
        else:
            self.__u = u_inicial

        self.__v_inicial = self.__v
        self.__u_inicial = self.__u


    # --------------------------------------------------
    # CONSTRUCTOR ALTERNATIVO
    # --------------------------------------------------

    @classmethod
    def predefinida(cls, tipo: str, v_inicial: float = -65,
                    u_inicial: float | None = None) -> "Neurona":
        """
        Crea una instancia de Neurona con parámetros predefinidos según el tipo.

        Parameters
        ----------
        tipo : str
            Tipo de neurona. Los tipos disponibles son:

            RS (Regular Spiking), IB (Intrinsically Bursting), CH (Chattering), FS (Fast Spiking),
            LTS (Low Threshold Spiking), TC (Thalamocortical), RZ (Resonator).

            Se aceptan variantes en mayúsculas o minúsculas y con guiones en lugar de espacios.

        v_inicial : float, optional
            Potencial de membrana inicial de la neurona. Por defecto es -65 mV.

        u_inicial : float | None, optional
            Variable de recuperación inicial de la neurona. Por defecto se calcula como b * v_inicial.

        Returns
        -------
        Neurona
            Una instancia de Neurona con parámetros típicos según el tipo solicitado.

        Raises
        ------
        TypeError
            Si el tipo no es una cadena de texto o los valores de estado iniciales no son números
            reales, y, en el caso de u_inicial, tampoco es None.

        ValueError
            Si el tipo de neurona no es reconocido.

        Examples
        --------
        >>> n = Neurona.predefinida("RS")
        >>> n.nombre
        "Regular Spiking"
        >>> n2 = Neurona.predefinida("regular spiking")
        >>> n2.nombre
        "Regular Spiking"
        """
        if not isinstance(tipo, str):
            raise TypeError("El tipo de neurona debe ser una cadena de texto.")

        cls._validar_numero_real(v_inicial, "v_inicial")

        if u_inicial is not None:
            cls._validar_numero_real(u_inicial, "u_inicial", " o None para calcularlo automáticamente")

        tipo = tipo.strip().lower()

        if tipo not in cls._ALIAS:
            raise ValueError(f"El tipo de neurona '{tipo}' no existe. Use un valor predefinido o cree"
                             " una neurona personalizada con el constructor.")

        tipo = cls._ALIAS[tipo]
        datos = cls._TIPOS[tipo]

        a, b, c, d = datos["parametros"]
        nombre = datos["nombre"]
        es_excitatoria = datos["es_excitatoria"]

        u = b * v_inicial if u_inicial is None else u_inicial

        return cls(a, b, c, d, v_inicial, u, nombre, es_excitatoria)


    # --------------------------------------------------
    # MÉTODOS PÚBLICOS
    # --------------------------------------------------

    def actualizar(self, I: float, dt: float = 0.5) -> bool:
        """
        Actualiza el estado de la neurona para un paso temporal.

        Parameters
        ----------
        I : float
            Corriente de entrada.

        dt : float
            Paso temporal de simulación en milisegundos. Debe ser mayor que 0.

        Returns
        -------
        bool
            True si se ha disparado la neurona, False en caso contrario.

        Raises
        ------
        TypeError
            Si I o dt no son números reales.

        ValueError
            Si dt no es mayor que 0.

        Notes
        -----
        La integración se realiza mediante dos semipasos de Euler para la ecuación del potencial de
        membrana y un único paso para la variable de recuperación, siguiendo la implementación propuesta
        por Izhikevich.
        """

        self._validar_numero_real(I, "I")
        self._validar_numero_real(dt, "dt")

        if dt <= 0:
            raise ValueError("El paso temporal tiene que ser mayor a 0.")

        return self._actualizar(I, dt)

    def reiniciar(self) -> None:
        """
        Restaura el estado de la neurona al estado inicial de la creación.

        Esto restablece los valores de v y u a los iniciales usados al crear la neurona.
        """
        self.__v = self.__v_inicial
        self.__u = self.__u_inicial

    def establecer_estado(self, v: float| None = None, u: float | None = None) -> None:
        """
        Establece el estado interno v y/o u de la neurona en nuevos valores.

        Parameters
        ----------
        v : float | None, optional
            Nuevo valor del potencial de membrana. Si es None, no se modifica v.

        u : float | None, optional
            Nuevo valor de la variable de recuperación. Si es None, no se modifica u.

        Raises
        ------
        TypeError
            Si alguno de los valores pasados no es un número real.
        """
        if v is not None:
            self._validar_numero_real(v, "v")
            self.__v = v

        if u is not None:
            self._validar_numero_real(u, "u")
            self.__u = u

    def copy(self) -> "Neurona":
        """
        Realiza una copia independiente de la neurona actual.

        Returns
        -------
        Neurona
            Copia de la instancia de neurona actual, con los mismos parámetros, estado y nombre.
        """
        return Neurona(self.__a, self.__b, self.__c, self.__d, self.__v, self.__u, self.__nombre,
                       self.__es_excitatoria)


    # --------------------------------------------------
    # MÈTODOS DE CLASE
    # --------------------------------------------------

    @classmethod
    def alias(cls) -> dict[str, str]:
        """
        Obtiene el diccionario de alias para los tipos de neurona.

        Returns
        -------
        dict[str, str]
            Copia del diccionario de alias a nombres canónicos de tipo.
        """
        return cls._ALIAS.copy()

    @classmethod
    def tipos(cls) -> dict[str, dict[str, tuple[float, float, float, float] | str | bool]]:
        """
        Obtiene los parámetros, el nombre y si la neurona es excitatoria de los tipos de neurona
        predefinidos.

        Returns
        -------
        dict[str, dict[str, tuple[float, float, float, float] | str | bool]]
            Diccionario que relaciona cada tipo con un diccionario con los parámetros, el nombre y
            si la neurona es excitatoria.
        """
        return {k: v.copy() for k, v in cls._TIPOS.items()}

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


    # --------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------

    @staticmethod
    def _validar_numero_real(valor: object, nombre: str, mensaje: str | None = None) -> None:
        """
        Comprueba que un valor sea un número real.

        Parameters
        ----------
        valor : object
            Valor a comprobar.

        nombre : str
            Nombre del parámetro utilizado en el mensaje de error.

        mensaje : str | None, optional
            Mensaje de error extra, añadido al final del mensaje por defecto.

        Raises
        ------
        TypeError
            Si el valor no es un número real.
        """
        if not isinstance(valor, Real):
            raise TypeError(f"{nombre} debe ser un número real{mensaje if mensaje is not None else ''}.")

    def _actualizar(self, I: float, dt: float) -> bool:
        """
        Actualiza el estado de la neurona para un paso temporal sin validar los parámetros de entrada.

        Parameters
        ----------
        I : float
            Corriente de entrada.

        dt : float
            Paso temporal de simulación en milisegundos. Debe ser mayor que 0.

        Returns
        -------
        bool
            True si se ha disparado la neurona, False en caso contrario.

        Notes
        -----
        La integración se realiza mediante dos semipasos de Euler para la ecuación del potencial de
        membrana y un único paso para la variable de recuperación, siguiendo la implementación propuesta
        por Izhikevich.

        Este método está destinado al uso interno de la librería cuando los parámetros ya han sido
        validados previamente.
        """
        spike = False
        v = self.__v
        u = self.__u

        if v >= 30:
            v = self.__c
            u += self.__d

        # Evitar asignaciones intermedias de elevar al cuadrado haciendo la multiplicación directamente
        # Calcular v en dos pasos para estabilidad numérica
        v += 0.5 * dt * (0.04 * v * v + 5 * v + 140 - u + I)
        v += 0.5 * dt * (0.04 * v * v + 5 * v + 140 - u + I)
        u += dt * (self.__a * (self.__b * v - u))

        if v >= 30:
            spike = True
            v = 30

        self.__v = v
        self.__u = u

        return spike

    def _estado(self) -> tuple[float, float]:
        """
        Devuelve el estado interno de la neurona.

        Este método está destinado al uso interno de la librería para proporcionar una interfaz
        homogénea con RedDeNeuronas.

        Returns
        -------
        tuple[float, float]
            Tupla con los valores actuales de v y u.
        """
        return self.__v, self.__u


    # --------------------------------------------------
    # PROPIEDADES
    # --------------------------------------------------

    @property
    def estado(self) -> tuple[float, float]:
        """
        Devuelve el estado actual de la neurona.

        Returns
        -------
        tuple[float, float]
            Tupla (v, u) con el potencial de membrana y la variable de recuperación.
        """
        return self.__v, self.__u

    @property
    def nombre(self) -> str:
        """
        Devuelve el nombre de la neurona.

        Returns
        -------
        str
            Nombre de la neurona.
        """
        return self.__nombre

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

        Esta propiedad existe para mantener una interfaz homogénea con RedDeNeuronas.

        Returns
        -------
        bool
            False, ya que la simulación individual de una neurona se realiza en CPU.
        """
        return False


    # --------------------------------------------------
    # REPRESENTACIÓN
    # --------------------------------------------------

    def __repr__(self) -> str:
        """
        Devuelve una representación informal de la neurona.

        Returns
        -------
        str
            Cadena con el nombre, los parámetros y el estado actual de la neurona.
        """
        return (f"Neurona(nombre='{self.__nombre}', es_excitatoria={self.__es_excitatoria}, "
                f"a={self.__a}, b={self.__b}, c={self.__c}, d={self.__d}, "
                f"v={self.__v}, u={self.__u})")
