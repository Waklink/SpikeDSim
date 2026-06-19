

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
        nombre (str): Nombre de la neurona, puede ser un nombre precargado (RS, IB, CH, FS, LTS, TC, RZ), o personalizado.
        tipo (str): Tipo de la neurona según su sinapsis (Excitatoria o Inhibitoria).
        es_excitatoria (bol): Representación de si la neurona es excitatoria o inhibitoria.
    """

    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = -65, d: float = 2, v_inicial: float = -65,
                 u_inicial: None | float = None, nombre: str = "Personalizado", es_excitatoria: bool = True):
        """
        Inicializa una instancia de la clase Neurona, con parámetros personalizados.

        Args:
            a (float): Parámetro que regula la velocidad de recuperación de la neurona.
            b (float): Parámetro que regula la sensibilidad de la neurona al estímulo.
            c (float): Parámetro que regula el potencial de membrana de recuperación.
            d (float): Parámetro que regula la velocidad de recuperación de la variable de recuperación.
            v_inicial (float, optional): Potencial de membrana inicial de la neurona. Si no se proporciona, se calcula automáticamente.
            u_inicial (float, optional): Variable de recuperación inicial de la neurona. Si no se proporciona, se calcula automáticamente.
            nombre (str): Nombre de la neurona, puede ser un nombre precargado (RS, IB, CH, FS, LTS, TC, RZ), o personalizado.
            tipo (str): Tipo de la neurona según su sinapsis (Excitatoria o Inhibitoria).
        """

        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d

        self.__nombre = nombre
        self.__es_excitatoria = es_excitatoria
        self.__tipo = "Excitatoria" if self.__es_excitatoria else "Inhibitoria"

        self.__v = v_inicial
        
        if u_inicial is None:
            self.__u = self.__b * self.__v
        else:
            self.__u = u_inicial


    def crear_precargada(self, nombre: str):
        """
        Crea una instancia de la clase Neurona, con parámetros precargados según el nombre de la neurona.

        Args:
            nombre (str): Nombre de la neurona, debe ser un nombre precargado (RS, IB, CH, FS, LTS, TC, RZ).
        
        Raises:
            ValueError: Si el nombre de la neurona no es reconocido.
        """
        if nombre.lower() in ("rs", "regular spiking", "regular-spiking"):
            self.__nombre = "RS"
            self.__tipo = "Excitatoria"
            self.__a = 0.02
            self.__b = 0.2
            self.__c = -65
            self.__d = 8
        elif nombre.lower() in ("ib", "intrinsically bursting", "intrinsically-bursting"):
            self.__nombre = "IB"
            self.__tipo = "Excitatoria"
            self.__a = 0.02
            self.__b = 0.2
            self.__c = -55
            self.__d = 4
        elif nombre.lower() in ("ch", "chattering"):
            self.__nombre = "CH"
            self.__tipo = "Excitatoria"
            self.__a = 0.02
            self.__b = 0.2
            self.__c = -50
            self.__d = 2
        elif nombre.lower() in ("fs", "fast spiking", "fast-spiking"):
            self.__nombre = "FS"
            self.__tipo = "Inhibitoria"
            self.__a = 0.1
            self.__b = 0.2
            self.__c = -65
            self.__d = 2
        elif nombre.lower() in ("lts", "low-threshold spiking"):
            self.__nombre = "LTS"
            self.__tipo = "Inhibitoria"
            self.__a = 0.02
            self.__b = 0.25
            self.__c = -65
            self.__d = 2
        elif nombre.lower() in ("tc", "thalamocortical", "thalamo cortical", "thalamo-cortical"):
            self.__nombre = "TC"
            self.__tipo = "Excitatoria"
            self.__a = 0.02
            self.__b = 0.25
            self.__c = -65
            self.__d = 0.05
        elif nombre.lower() in ("rz", "resonator"):
            self.__nombre = "RZ"
            self.__tipo = "Inhibitoria"
            self.__a = 0.1
            self.__b = 0.26
            self.__c = -65
            self.__d = 2
        else:
            raise ValueError("Nombre de neurona no reconocida. Las opciones son: RS, IB, CH, FS, LTS, TC, RZ.\n" \
            "En caso de no existir la nombre deseada, se puede crear una personalizada utilizando el constructor con parámetros a, b, c, d.")
        
        self.__v = -65
        self.__u = self.__b * self.__v

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

    def get_estado(self) -> tuple[float, float]:
        """
        Obtiene el estado actual de la neurona.

        Returns:
            tuple[float, float]: Los valores de v y u.
        """
        return self.__v, self.__u
    
    def get_nombre(self) -> str:
        """
        Obtiene el nombre de la neurona.

        Returns:
            str: El nombre de la neurona.
        """
        return self.__nombre
    
    def get_tipo(self) -> str:
        """
        Obtiene el tipo de la neurona según el tipo de sinapsis.

        Returns:
            str: El tipo de la neurona según la sinapsis.
        """
        return self.__tipo
    
    def es_excitatoria(self) -> bool:
        """
        Obtiene si la neurona es excitatoria o no, en cuyo caso se considera inhibitoria.

        Returns:
            bool: Si la neurona es excitatoria.
        """
        return self.__es_excitatoria
    
    def get_parametros(self) -> tuple[float, float, float, float]:
        """
        Obtiene los parámetros de la neurona.

        Returns:
            tuple[float, float, float, float]: Los parámetros a, b, c y d.
        """
        return self.__a, self.__b, self.__c, self.__d

    def copy(self):
        return Neurona(self.__a, self.__b, self.__c, self.__d, self.__v, self.__u, self.__nombre, self.__es_excitatoria)
