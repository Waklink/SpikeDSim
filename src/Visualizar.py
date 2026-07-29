from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, TypeAlias

Historial: TypeAlias = dict[str, Any] | None

class Visualizar:
    """
    Visualizador de datos de una ismulación en forma de diferentes gráficos.

    Attributes
    ----------
    - Todos los atributos devolverán None si no hay ningún historial cargado.
    - Se puede cargar un nuevo historial usando cargar_historial, lo que sobreescribirá el historial
      que esté ya cargado.
    - En el caso de que la red usada haya tenido solo una neurona, o de que se haya simulado una
      sola neurona, entonces los historiales estarán aplanados a 1 sola dimensión, es decir, pasarán
      de tener una forma (pasos, num_neuronas) a (pasos,).
    - Si algún parámetro habitual del historial no existe, esté valdrá None.

    historial : Historial
        Copia del historial cargado. Con claves: spikes, v, u, I, nombre, es_excitatoria y dt.

    spikes : np.ndarray | None
        Matriz con los disparos de la red a lo largo del tiempo.

    v : np.ndarray | None
        Matriz con la evolución de los potenciales de membrana de las neuronas de la red a lo largo
        del tiempo.

    u : np.ndarray | None
        Matriz con la evolución de las variables de recuperación de las neuronas de la red a lo largo
        del tiempo.

    I : np.ndarray | None
        Matriz con las corrientes que se han introducido a la red a lo largo del tiempo.

    nombre : list[str] | None
        Lista ordenada de los nombres de las neuronas de la red.

    es_excitatoria : list[bool] | None
        Lista con si las neuronas de la red son excitatorias o inhibitorias.

    dt : float | None
        El paso temporal usado durante la simulación.
    """

    def __init__(self, path_historial: str | Path | None = None, historial: Historial = None):
        """
        Inicializa una instancia de Visualizar con un posible historial.

        En el caso de que se pasen ambos parámetros, se cargará el historial pasado.

        Parameters
        ----------
        path_historial : str | Path | None, optional
            Path al archivo con el historial.

        historial : Historial | None = None
            Historial a cargar.
        """
        self.__historial = None

        if historial is not None:
            self.cargar_historial(historial=historial)
        elif path_historial is not None:
            self.cargar_historial(path_historial)


    def cargar_historial(self, path: str | Path | None = None, historial: Historial = None) -> None:
        """
        Carga un historial en la clase, sobreescribiendo el posible historial que ya estuviese cargado,
        pudiendo cargarlo desde un archivo o desde un historial de Simulador.

        Debe pasarse al menos un path o un historial, en el caso de que se pasen ambos, se cargará
        únicamente el historial.

        Parameters
        ----------
        path : str | Path | None, optional
            Path al archivo que se quiera cargar, puede ser una cadena de texto con el path absoluto
            o relativo, o una instancia de la clase Path.
        
        historial : Historial | None, optional
            Historial creado desde Simulador, en forma de diccionario con spikes, v, u e I y datos
            de nombre, es_excitatoria de las neuronas y dt de la simualción.
        
        Raises
        ------
        ValueError
            Si no se pasa ningún argumento a la función.
        """
        if historial is not None:
            self.__historial = self.__cargar_desde_dict(historial)
        elif path is not None:
            self.__historial = self.__cargar_desde_archivo(path)
        else:
            raise ValueError("Pase un historial, o un path a un archivo con el historial.")


    def __obtener_historial(self, path: str | Path | None = None, historial: Historial = None) -> Historial:
        """
        Obtiene un historial normalizado desde un diccionario o desde un archivo dado el path al mismo.
        En el caso de recibir ambos parámetros, se devolverá el diccionario normalizado, sin acceder
        al archivo.

        Si alguna clave no existe en el diccionario pasado o en el archivo, su valor será None en el
        historial normalizado.

        Parameters
        ----------
        path : str | Path | None, optional
            Path al archivo que se quiera cargar, puede ser una cadena de texto con el path absoluto
            o relativo, o una instancia de la clase Path.
        
        historial : Historial | None, optional
            Historial creado desde Simulador, en forma de diccionario con spikes, v, u e I y datos
            de nombre y es_excitatoria de las neuronas y dt de la simualción.

        Returns
        -------
        Historial
            Historial normalizado.
        """
        hist = None
        if historial is not None:
            hist = self.__cargar_desde_dict(historial)
        elif path is not None:
            hist = self.__cargar_desde_archivo(path)
        return hist


    def __cargar_desde_dict(self, historial: Historial) -> Historial:
        """
        Cargar un historial desde un diccionario.

        Parameters
        ----------
        historial : Historial
            Historial de un Simulador.

        Returns
        -------
        Historial
            Historial normalizado.
        
        Raises
        ------
        TypeError
            Si el historial pasado no es un diccionario.
        
        ValueError
            Si el diccionario no tiene todas las claves necesarias.
        """
        if not isinstance(historial, dict):
            raise TypeError("Pase un diccionario como historial.")
        return self.__normalizar_historial(historial)

    def __cargar_desde_archivo(self, path: str | Path) -> Historial:
        """
        Cargar un historial desde uno o varios archivos dado el path donde se encuentra.

        En el caso de que no exista el archivo, historial será None, si el formato es csv o txt y
        alguno de los archivos necesarios no existen, solo el valo0r de la clave correspondiente
        será None.

        Parameters
        ----------
        path : str | Path
            Path donde se encuentra el archivo, en el caso de que la extensión sea csv o txt, se
            usará para construir el nombre de los archivos en base a cada clave del historial.

            Las extensiones de archivo soportadas son npz, json, csv y txt.

        Returns
        -------
        Historial
            Historial normalizado.
        
        Raises
        ------
        TypeError
            Si el path pasado no es una cadena de texto o una instancia de Path.
        
        ValueError
            Si el archivo es de formato jsno y no contiene ningún diccionario.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError("El path pasado debe ser una cadena con el path, o una instancia de la "
                            "clase Path")

        path = Path(path)
        formato = path.suffix.lower().replace(".", "")
        if formato not in ("npz", "json", "csv", "txt"):
            raise ValueError("El archivo no tiene una extensión soportada.")

        if formato == "npz":
            if not path.exists():
                return None
            with np.load(path) as datos:
                spikes = datos["spikes"] if "spikes" in datos.files else None
                v = datos["v"] if "v" in datos.files else None
                u = datos["u"] if "u" in datos.files else None
                i = datos["I"] if "I" in datos.files else None
                nombre = datos["nombre"] if "nombre" in datos.files else None
                es_excitatoria = datos["es_excitatoria"] if "es_excitatoria" in datos.files else None
                dt = float(datos["dt"]) if "dt" in datos.files else None
            historial = {"spikes": spikes,
                         "v": v,
                         "u": u,
                         "I": i,
                         "nombre": nombre,
                         "es_excitatoria": es_excitatoria,
                         "dt": dt}
            historial = self.__normalizar_historial(historial)

        elif formato == "json":
            if not path.exists():
                return None
            with open(path, "r") as f:
                datos = json.loads(f.read())
                if not isinstance(datos, dict):
                    raise ValueError("EL archivo json tiene que contener un diccionario.")

            historial = {"spikes": datos.get("spikes"),
                         "v": datos.get("v"),
                         "u": datos.get("u"),
                         "I": datos.get("I"),
                         "nombre": datos.get("nombre"),
                         "es_excitatoria": datos.get("es_excitatoria"),
                         "dt": datos.get("dt")}
            historial = self.__normalizar_historial(historial)

        elif formato == "csv":
            historial = self.__cargar_desde_varios_archivos(path, "csv", ",")
        elif formato == "txt":
            historial = self.__cargar_desde_varios_archivos(path, "txt")

        return historial

    def __cargar_desde_varios_archivos(self, path: Path, extension: str| None = None, delim: str | None = None) -> Historial:
        """
        Cargar un historial desde variso archivos txt o csv, el resto def ormatos no está soportado
        ni probado directamente.

        Parameters
        ----------
        path : Path
            Path raiz para deducir los archivos, añadiendo _clave al final del nombre del archivo.

        extension : str | None, optional
            Extensión del archivo, si no se pasa, se deduce del path del archivo pasado, si se pasa,
            tiene que coincidir con la extensión del path del archivo pasado.

        delim : str | None
            Delimitador usado para separar los valores en los archivos. en el caso de ser None, para
            nombre y es_excitatoria se usara \\t.

        Returns
        -------
        Historial
            Historial normalizado.

        Raises
        ------
        ValueError
            Si la extensión del archivo y la pasada no coinciden.
        """
        if extension is None:
            extension = path.suffix.replace(".", "")
        elif extension != path.suffix.replace(".", ""):
            raise ValueError("La extensión del archivo original debe coincidir con la extensión pasada.")

        path_spikes = path.with_name(f"{path.stem}_spikes.{extension}")
        path_v = path.with_name(f"{path.stem}_v.{extension}")
        path_u = path.with_name(f"{path.stem}_u.{extension}")
        path_I = path.with_name(f"{path.stem}_I.{extension}")
        path_nombre = path.with_name(f"{path.stem}_nombre.{extension}")
        path_excitatorias = path.with_name(f"{path.stem}_es_excitatoria.{extension}")
        path_dt = path.with_name(f"{path.stem}_dt.{extension}")
        spikes = np.loadtxt(path_spikes, delimiter=delim) if path_spikes.exists() else None
        v = np.loadtxt(path_v, delimiter=delim) if path_v.exists() else None
        u = np.loadtxt(path_u, delimiter=delim) if path_u.exists() else None
        i = np.loadtxt(path_I, delimiter=delim) if path_I.exists() else None
        nombre = np.loadtxt(path_nombre, dtype=str, delimiter="\t" if delim is None else delim).tolist() if path_nombre.exists() else None
        es_excitatoria = np.loadtxt(path_excitatorias, dtype=bool, delimiter="\t" if delim is None else delim).tolist() if path_excitatorias.exists() else None
        dt = np.loadtxt(path_dt, delimiter=delim) if path_dt.exists() else None
        historial = {"spikes": spikes,
                     "v": v,
                     "u": u,
                     "I": i,
                     "nombre": nombre,
                     "es_excitatoria": es_excitatoria,
                     "dt": dt}

        return self.__normalizar_historial(historial)


    def __normalizar_historial(self, hist: Historial) -> Historial:
        """
        Normalizar el historial pasado, en el caso de que no tenga ninguna de las claves necesarias,
        o todas sean None, se guardará el historial como None. Asimismo, si alguna de las claves no
        existe, o su valor es None, se guardará con el valor None.

        La normalización hecha es:
        - spikes -> numpy.ndarray
        - v -> numpy.ndarray
        - u -> numpy.ndarray
        - I -> numpy.ndarray
        - nombre -> list[str]
        - es_excitatoria -> list[bool]
        - dt -> float

        Parameters
        ----------
        hist : Historial
            Historial a normalizar.

        Returns
        -------
        Historial
            Historial normalizado.
        
        Raises
        ------
        TypeError
            Si el historial pasado no es un diccionario.
        """
        if not isinstance(hist, dict):
            raise TypeError("El historial pasado debe ser un diccionario.")
        
        shape = None
        spikes, shape = self.__validar_historial_array(hist.get("spikes"),  shape)
        v, shape = self.__validar_historial_array(hist.get("v"), shape)
        u, shape = self.__validar_historial_array(hist.get("u"), shape)
        i, shape = self.__validar_historial_array(hist.get("I"), shape)

        if shape is None:
            num_neuronas = None
        elif len(shape) == 1:
            num_neuronas = 1
        else:
            num_neuronas = shape[1]

        nombre = hist.get("nombre")
        if isinstance(nombre, str):
            nombre = [nombre]
        elif isinstance(nombre, np.ndarray):
            nombre = nombre.tolist()
        if nombre is not None:
            if num_neuronas is None:
                num_neuronas = len(nombre)
            elif num_neuronas != len(nombre):
                raise ValueError("El número de nombres debe coincidir con el número de neuronas con"
                                 " las que se haya realizado la simulación.")

        es_excitatoria = hist.get("es_excitatoria")
        if isinstance(es_excitatoria, bool):
            es_excitatoria = [es_excitatoria]
        elif isinstance(es_excitatoria, np.ndarray):
            es_excitatoria = es_excitatoria.tolist()
        if es_excitatoria is not None and num_neuronas is not None:
            if num_neuronas != len(es_excitatoria):
                raise ValueError("La longitud de es_excitatoria debe coincidir con el número de neuronas"
                                 " con las que se haya realizado la simulación.")

        dt = hist.get("dt")
        if dt is not None:
            dt = float(dt)
            if dt <= 0:
                raise ValueError("El paso temporal debe ser positivo.")

        if all(valor is None for valor in (spikes, v, u, i, nombre, es_excitatoria, dt)):
            return None
        else:
            return {"spikes": spikes,
                    "v": v,
                    "u": u,
                    "I": i,
                    "nombre": nombre,
                    "es_excitatoria": es_excitatoria,
                    "dt": dt}

    def __validar_historial_array(self, dato: Any, shape: tuple[int, ...] | None
                                 ) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
        """
        Validar que el dato pasado tenga la forma correcta.

        Parameters
        ----------
        dato : any
            Dato pasado, puede ser como una matriz con forma (pasos, num_neuronas), un vector de
            longitud pasos o None.

        shape : tuple[int, ...] | None
            Forma que tiene que tener el dato pasado, si es None, se obtiene la forma del dato tras
            convertirlo a un numpy.ndarray.

        Returns
        -------
        tuple[np.ndarray | None, tuple[int, ...] | None]
            Tupla con el dato en forma de numpy.ndarray y la forma que tiene.

        Raises
        ------
        ValueError
            Si el dato pasado no es un vector ni una matriz de dos dimensiones.
            En el caso de que se pase un shape distinto de None, si el dato pasado no tiene la forma correcta.
        """
        if dato is None:
            return None, shape

        dato = np.asarray(dato)

        if dato.ndim not in (1, 2):
            raise ValueError("El dato pasado tiene que ser el equivalente a un vector o una matriz"
                             " de 2 dimensiones.")

        if shape is None:
            return dato, dato.shape

        if dato.shape != shape:
            raise ValueError("Los historiales de spikes, v, u e I deben tener la misma forma.")

        return dato, shape


    def __obtener(self, clave: str) -> np.ndarray | list[str] | list[bool] | float | None:
        """
        Obtener cualquier variable del historial.

        Parameters
        ----------
        clave : str
            Clave del valor del historial que se queire obtener.

        Returns
        -------
        np.ndarray | list[str] | list[bool] | float | None
            Valor de una clave del historial, puede ser un array de numpy, para spikes, v, u o I,
            una lista de nombres, o de booleanos para es_excitatoria, un float para dt o None si no
            tienen valor o no hay ningún historial cargado.
        """
        if self.__historial is None:
            return None
        valor = self.__historial.get(clave)
        if hasattr(valor, "copy"):
            return valor.copy()
        return valor


    @property
    def historial(self) -> Historial:
        """
        Devuelve una copia completa del historial que haya cargado.

        Returns
        -------
        Historial | None
            Historial cargado. Si no hay ningún historial cargado, se devolerá None.
        """
        if self.__historial is None:
            return None
        return {"spikes": self.__obtener("spikes"),
                "v": self.__obtener("v"),
                "u": self.__obtener("u"),
                "I": self.__obtener("I"),
                "nombre": self.__obtener("nombre"),
                "es_excitatoria": self.__obtener("es_excitatoria"),
                "dt": self.__obtener("dt")}

    @property
    def spikes(self) -> np.ndarray | None:
        """
        Copia del historial de disparos de la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona que se ha disparado. Si
            no hay ningún historial cargado, se devolerá None.
        """
        return self.__obtener("spikes")

    @property
    def v(self) -> np.ndarray | None:
        """
        Copia del historial de potenciales de membrana de la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona a la que representa. Si
            no hay ningún historial cargado, se devolerá None.
        """
        return self.__obtener("v")

    @property
    def u(self) -> np.ndarray | None:
        """
        Copia del historial de variables de recuperación de la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona a la que representa. Si
            no hay ningún historial cargado, se devolerá None.
        """
        return self.__obtener("u")

    @property
    def I(self) -> np.ndarray | None:
        """
        Copia del historial de corrientes introducidas a la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona a la que se le ha aplicado
            la corriente. Si no hay ningún historial cargado, se devolerá None.
        """
        return self.__obtener("I")

    @property
    def nombre(self) -> list[str] | None:
        """
        Copia de la lista ordenada de nombres de las neuronas que de la red de la simulación.

        Returns
        -------
        list[str] | None
            Lista ordenada de los nombres de las neuronas. Si no hay ningún historial cargado, se
            devolerá None.
        """
        return self.__obtener("nombre")

    @property
    def es_excitatoria(self) -> list[bool] | None:
        """
        Copia de la lista que determina si las neuronas de la red de la simulación son excitatorias
        o inhibitorias.

        Returns
        -------
        list[bool] | None
            Lista con si las neuronas son excitatorias o inhibitorias. Si no hay ningún historial
            cargado, se devolerá None.
        """
        return self.__obtener("es_excitatoria")

    @property
    def dt(self) -> float | None:
        """
        El paso temporal usado en el historial cargado.

        Returns
        -------
        float | None
            El paso temporal. Si no hay ningún historial cargado, se devolerá None.
        """
        return self.__obtener("dt")
