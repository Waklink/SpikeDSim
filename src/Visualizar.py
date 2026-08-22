from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike
from typing import TypedDict
from numbers import Real
from collections.abc import Sequence



# --------------------------------------------------
# TIPO AUXILIAR
# --------------------------------------------------

class Historial(TypedDict, total=False):
    """
    Tipo de datos utilizado para representar el historial de una simulación.

    Las claves son opcionales y pueden no estar en el diccionario. Cuando una clave existe, su valor
    también puede ser None si su dato no está disponible.

    Attributes
    ----------
    spikes : np.ndarray | None
        Historial de disparos de las neuronas.

    v : np.ndarray | None
        Historial de los potenciales de membrana.

    u : np.ndarray | None
        Historial de las variables de recuperación.

    I : np.ndarray | None
        Historial de las corrientes de entrada.

    nombre : list[str] | None
        Lista ordenada con los nombres de las neuronas.

    es_excitatoria : list[bool] | None
        Lista ordenada que indica si cada neurona es excitatoria (True) o inhibitoria (False).

    dt : float | None
        Paso temporal utilizado durante la simulación, en milisegundos.
    """
    spikes: np.ndarray | None
    v: np.ndarray | None
    u: np.ndarray | None
    I: np.ndarray | None
    nombre: list[str] | None
    es_excitatoria: list[bool] | None
    dt: float | None


# --------------------------------------------------
# CLASE
# --------------------------------------------------

class Visualizar:
    """
    Visualizador de datos de una simulación en forma de diferentes gráficos.

    Attributes
    ----------
    historial : Historial | None
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

    Notes
    -----
    - Todos los atributos devolverán None si no hay ningún historial cargado.
    - En el caso de que la red usada haya tenido solo una neurona, o de que se haya simulado una
      sola neurona, entonces los historiales de spikes, v, u e I estarán aplanados a una sola dimensión,
      es decir, pasarán de tener una forma (pasos, num_neuronas) a (pasos,).
    - Si algún parámetro habitual del historial no existe, esté valdrá None.
    """

    # --------------------------------------------------
    # CONSTRUCTOR
    # --------------------------------------------------

    def __init__(self, path: str | Path | None = None, historial: Historial | None = None):
        """
        Inicializa una instancia de Visualizar con un posible historial.

        En el caso de que se pasen ambos parámetros, se cargará el historial pasado.

        Parameters
        ----------
        path : str | Path | None, optional
            Path al archivo con el historial.

        historial : Historial | None, optional
            Historial a cargar.
        """
        self.__historial = None

        if historial is not None:
            self.cargar_historial(historial=historial)
        elif path is not None:
            self.cargar_historial(path)


    # --------------------------------------------------
    # MÉTODOS PÚBLICOS
    # --------------------------------------------------

    def cargar_historial(self, path: str | Path | None = None, historial: Historial | None = None
                         ) -> None:
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
            de nombre, es_excitatoria de las neuronas y dt de la simulación.

        Raises
        ------
        ValueError
            Si no se pasa ningún argumento a la función.
        """
        if historial is not None:
            self.__historial = self._cargar_desde_dict(historial)
        elif path is not None:
            self.__historial = self._cargar_desde_archivo(path)
        else:
            raise ValueError("Pase un historial, o un path a un archivo con el historial.")


    def raster_plot(self, path: str | Path | None = None, historial: Historial | None = None,
                    neuronas: int | slice | Sequence[int] | None = None, separar_tipo: bool = True,
                    figsize: tuple[float, float] = (10, 6), color: str = "black",
                    markersize: float = 2, max_labels: int = 20, titulo: str = "Raster plot",
                    mostrar: bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar un raster plot de los disparos de la simulación.

        Cada punto representa un spike de una neurona en un instante de tiempo.

        Parameters
        ----------
        Se puede usar un historial desde un archivo con el path, desde un diccionario pasándolo en
        historial, o usar el historial cargado en la clase si no se pasa nada. En el caso de que se
        pasen ambos parámetros, se usará el historial y se ignorará el path.

        path : str | Path | None, optional
            Archivo desde el que cargar el historial.

        historial : Historial | None, optional
            Historial pasado directamente.

        Datos de la gráfica que se mostrará:

        neuronas : int | slice | Sequence[int] | None, optional
            Subconjunto de neuronas que se representarán.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

        separar_tipo : bool
            Si es True, las neuronas se separarán, colocando las inhibitorias
            en la parte superior y las excitatorias en la inferior. Se dibujará una línea para separar
            ambos grupos.

            El historial tiene que tener datos de es_excitatoria.

        figsize : tuple[float, float]
            Tamaño de la figura.

        color : str
            Color de los puntos.

        markersize : float
            Tamaño de los puntos.

        max_labels : int
            Número máximo de labels que se mostrará en el eje del número de neurona (eje Y).

        titulo : str
            Título del gráfico.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show(). Si False, se cierra la
            figura sin mostrarla.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.

        Raises
        ------
        TypeError
            Si neuronas no es un int, un slice, una secuencia de índices o None.

        ValueError
            Si no existe historial de spikes o se pasa separar_tipo=True y no existe datos de
            es_excitatoria.

        IndexError
            Si algún índice de neurona está fuera de rango.

        Notes
        -----
        Si el historial no contiene el paso temporal dt se asumirá un valor de 1 ms.
        """
        if not isinstance(max_labels, int):
            raise TypeError("max_labels debe ser un entero.")
        if max_labels <= 0:
            raise ValueError("max_labels debe ser mayor que 0.")

        if not isinstance(markersize, Real):
            raise TypeError("markersize debe ser un número real.")
        if markersize <= 0:
            raise ValueError("markersize debe ser mayor que 0.")

        if (not isinstance(figsize, tuple) or len(figsize) != 2 or not all(isinstance(valor, Real)
                                                                           for valor in figsize)):
            raise TypeError("figsize debe ser una tupla de dos números reales.")
        if any(valor <= 0 for valor in figsize):
            raise ValueError("Los valores de figsize deben ser mayores que 0.")

        hist = self._obtener_historial(path, historial)

        if hist is None or hist.get("spikes") is None:
            raise ValueError("No hay historial de spikes disponible.")

        spikes = hist["spikes"]
        dt = hist.get("dt") if hist.get("dt") is not None else 1.0

        # Caso de una sola neurona
        if spikes.ndim == 1:
            spikes = spikes[:, np.newaxis]

        indices = self._obtener_indices_neuronas(neuronas, spikes.shape[1])

        if separar_tipo:
            tipos = hist.get("es_excitatoria")

            if tipos is None:
                raise ValueError("No existe información sobre el tipo de neuronas.")

            indices_ordenados, separacion, etiquetas_tipo = self._separar_raster_por_tipo(indices, tipos)
        else:
            indices_ordenados = indices
            separacion = None
            etiquetas_tipo = {}

        spikes = spikes[:, indices_ordenados]
        tiempos, columnas = np.where(spikes)
        # Posiciones verticales del raster
        neuronas_plot = columnas

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(tiempos * dt, neuronas_plot, s=markersize, c=color, marker=".")

        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Número de neurona")
        ax.set_title(titulo)
        ax.set_xlim(0, spikes.shape[0] * dt)

        if len(indices_ordenados) <= max_labels:
            ticks = np.arange(len(indices_ordenados))
        else:
            ticks = np.linspace(0, len(indices_ordenados) - 1, max_labels, dtype=int)

        ax.set_yticks(ticks)
        ax.set_yticklabels(indices_ordenados[ticks])

        if separar_tipo: 
            if separacion is not None:
                ax.axhline(separacion, linestyle="--", linewidth=1)

            for etiqueta, posicion in etiquetas_tipo.items():
                ax.text(-0.15, posicion, etiqueta, transform=ax.get_yaxis_transform(),
                               rotation=90, va="center", ha="right")

        fig.tight_layout()
        if mostrar:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


    def potencial_membrana(self, path: str | Path | None = None, historial: Historial | None = None,
                           neuronas: int | slice | Sequence[int] | None = 0,
                           figsize: tuple[float, float] = (10, 6),
                           titulo: str = "Potencial de membrana",
                           max_etiquetas_leyenda: int = 10,
                           mostrar: bool = True, tolerancia_similitud: float = 0
                           ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar la evolución temporal del potencial de membrana de una o varias neuronas.

        Todas las neuronas seleccionadas se representan en la misma gráfica, utilizando una línea
        diferente para cada neurona.

        Parameters
        ----------
        Se puede usar un historial desde un archivo con el path, desde un diccionario pasándolo en
        historial, o usar el historial cargado en la clase si no se pasa nada. En el caso de que se
        pasen ambos parámetros, se usará el historial y se ignorará el path.

        path : str | Path | None, optional
            Archivo desde el que cargar el historial.

        historial : Historial | None, optional
            Historial pasado directamente.

        neuronas : int | slice | Sequence[int] | None
            Subconjunto de neuronas que se representarán.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

            Por defecto se representa la neurona 0.

        figsize : tuple[float, float]
            Tamaño de la figura cuando se crea una nueva.

        titulo : str
            Título de la gráfica.

        max_etiquetas_leyenda : int
            Número máximo de etiquetas a partir del cual no se muestra la leyenda.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show(). Si False, se cierra la
            figura sin mostrarla.

        tolerancia_similitud : float
            Diferencia media absoluta máxima entre las evoluciones de dos neuronas con el mismo nombre
            para considerarlas iguales y agruparlas. Si es 0, se exige igualdad exacta. Si es menor
            que 0, no se agruparán neuronas.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.

        Raises
        ------
        ValueError
            Si no existe historial del potencial de membrana.

        TypeError
            Si neuronas no es un int, un slice o una lista de enteros.

        IndexError
            Si algún índice de neurona está fuera de rango.

        Notes
        -----
        Si el historial no contiene el paso temporal dt se asumirá un valor de 1 ms.
        """
        hist = self._obtener_historial(path, historial)

        if hist is None or hist.get("v") is None:
            raise ValueError("No hay historial del potencial de membrana disponible.")

        v = hist["v"]
        dt = hist.get("dt") if hist.get("dt") is not None else 1.0

        datos, indices, etiquetas = self._separar_neuronas(v, neuronas, hist.get("nombre"))
        datos, etiquetas = self._agrupar_neuronas_similares(datos, indices, etiquetas, tolerancia_similitud)

        return self._mostrar_grafica(datos, etiquetas, dt, "v (mV)", figsize, titulo, max_etiquetas_leyenda, mostrar)


    def variable_recuperacion(self, path: str | Path | None = None, historial: Historial | None = None, 
                              neuronas: int | slice | Sequence[int] | None = 0,
                              figsize: tuple[float, float] = (10, 6),
                              titulo: str = "Variable de recuperación",
                              max_etiquetas_leyenda: int = 10,
                              mostrar: bool = True, tolerancia_similitud: float = 0
                              ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar la evolución temporal de la variable de recuperación de una o varias neuronas.

        Todas las neuronas seleccionadas se representan en la misma gráfica, utilizando una línea
        diferente para cada neurona.

        Parameters
        ----------
        Se puede usar un historial desde un archivo con el path, desde un diccionario pasándolo en
        historial, o usar el historial cargado en la clase si no se pasa nada. En el caso de que se
        pasen ambos parámetros, se usará el historial y se ignorará el path.

        path : str | Path | None, optional
            Archivo desde el que cargar el historial.

        historial : Historial | None, optional
            Historial pasado directamente.

        neuronas : int | slice | Sequence[int] | None
            Subconjunto de neuronas que se representarán.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

            Por defecto se representa la neurona 0.

        figsize : tuple[float, float]
            Tamaño de la figura cuando se crea una nueva.

        titulo : str
            Título de la gráfica.

        max_etiquetas_leyenda : int
            Número máximo de etiquetas a partir del cual no se muestra la leyenda.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show(). Si False, se cierra la
            figura sin mostrarla.

        tolerancia_similitud : float
            Diferencia media absoluta máxima entre las evoluciones de dos neuronas con el mismo nombre
            para considerarlas iguales y agruparlas. Si es 0, se exige igualdad exacta. Si es menor
            que 0, no se agruparán neuronas.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.

        Raises
        ------
        ValueError
            Si no existe historial de la variable de recuperación.

        TypeError
            Si neuronas no es un int, un slice, una secuencia de índices o None.

        IndexError
            Si algún índice de neurona está fuera de rango.

        Notes
        -----
        Si el historial no contiene el paso temporal dt se asumirá un valor de 1 ms.
        """
        hist = self._obtener_historial(path, historial)

        if hist is None or hist.get("u") is None:
            raise ValueError("No hay historial de variable de recuperación disponible.")

        u = hist["u"]
        dt = hist.get("dt") if hist.get("dt") is not None else 1.0

        datos, indices, etiquetas = self._separar_neuronas(u, neuronas, hist.get("nombre"))
        datos, etiquetas = self._agrupar_neuronas_similares(datos, indices, etiquetas, tolerancia_similitud)

        return self._mostrar_grafica(datos, etiquetas, dt, "u", figsize, titulo, max_etiquetas_leyenda, mostrar)


    def corriente(self, path: str | Path | None = None, historial: Historial | None = None,
                  neuronas: int | slice | Sequence[int] | None = 0,
                  figsize: tuple[float, float] = (10, 6), titulo: str = "Corriente de entrada",
                  max_etiquetas_leyenda: int = 10,
                  mostrar: bool = True, tolerancia_similitud: float = 0
                  ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar la evolución temporal de la corriente aplicada.

        Todas las neuronas seleccionadas se representan en la misma gráfica, utilizando una línea
        diferente para cada neurona.

        Parameters
        ----------
        Se puede usar un historial desde un archivo con el path, desde un diccionario pasándolo en
        historial, o usar el historial cargado en la clase si no se pasa nada. En el caso de que se
        pasen ambos parámetros, se usará el historial y se ignorará el path.

        path : str | Path | None, optional
            Archivo desde el que cargar el historial.

        historial : Historial | None, optional
            Historial pasado directamente.

        neuronas : int | slice | Sequence[int] | None
            Subconjunto de neuronas que se representarán.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

            Por defecto se representa la neurona 0.

        figsize : tuple[float, float]
            Tamaño de la figura cuando se crea una nueva.

        titulo : str
            Título de la gráfica.

        max_etiquetas_leyenda : int
            Número máximo de etiquetas a partir del cual no se muestra la leyenda.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show(). Si False, se cierra la
            figura sin mostrarla.

        tolerancia_similitud : float
            Diferencia media absoluta máxima entre las evoluciones de dos neuronas con el mismo nombre
            para considerarlas iguales y agruparlas. Si es 0, se exige igualdad exacta. Si es menor
            que 0, no se agruparán neuronas.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.

        Raises
        ------
        ValueError
            Si no existe historial de corriente.

        TypeError
            Si neuronas no es un int, un slice, una secuencia de índices o None.

        IndexError
            Si algún índice de neurona está fuera de rango.

        Notes
        -----
        Si el historial no contiene el paso temporal dt se asumirá un valor de 1 ms.
        """
        hist = self._obtener_historial(path, historial)

        if hist is None or hist.get("I") is None:
            raise ValueError("No hay historial de corriente disponible.")

        corriente = hist["I"]
        dt = hist.get("dt") if hist.get("dt") is not None else 1.0

        datos, indices, etiquetas = self._separar_neuronas(corriente, neuronas, hist.get("nombre"))
        datos, etiquetas = self._agrupar_neuronas_similares(datos, indices, etiquetas, tolerancia_similitud)

        return self._mostrar_grafica(datos, etiquetas, dt, "I", figsize, titulo, max_etiquetas_leyenda, mostrar)


    def espacio_fase(self, path: str | Path | None = None, historial: Historial | None = None,
                     neuronas: int | slice | Sequence[int] | None = 0,
                     figsize: tuple[float, float] = (7, 7), titulo: str = "Espacio de fase",
                     max_etiquetas_leyenda: int = 10,
                     mostrar: bool = True, tolerancia_similitud: float = 0
                     ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar el espacio de fase de una o varias neuronas.

        El espacio de fase representa la evolución de la variable de recuperación frente al potencial
        de membrana, es decir, cada trayectoria corresponde a la evolución temporal del par (v, u)
        de una neurona.

        Parameters
        ----------
        Se puede usar un historial desde un archivo con el path, desde un diccionario pasándolo en
        historial, o usar el historial cargado en la clase si no se pasa nada. En el caso de que se
        pasen ambos parámetros, se usará el historial y se ignorará el path.

        path : str | Path | None, optional
            Archivo desde el que cargar el historial.

        historial : Historial | None, optional
            Historial pasado directamente.

        neuronas : int | slice | Sequence[int] | None
            Subconjunto de neuronas que se representarán.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

            Por defecto se representa la neurona 0.

        figsize : tuple[float, float]
            Tamaño de la figura.

        titulo : str
            Título de la gráfica.

        max_etiquetas_leyenda : int
            Número máximo de etiquetas a partir del cual no se muestra la leyenda.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show(). Si False, se cierra la
            figura sin mostrarla.

        tolerancia_similitud : float
            Diferencia media absoluta máxima entre las evoluciones de dos neuronas con el mismo nombre
            para considerarlas iguales y agruparlas. Si es 0, se exige igualdad exacta. Si es menor
            que 0, no se agruparán neuronas.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.

        Raises
        ------
        ValueError
            Si no existe historial del potencial de membrana o de la variable de recuperación.

        TypeError
            Si neuronas no es un int, un slice, una secuencia de índices o None.

        IndexError
            Si algún índice de neurona está fuera de rango.
        """
        hist = self._obtener_historial(path, historial)

        if hist is None or hist.get("v") is None:
            raise ValueError("No hay historial del potencial de membrana disponible.")

        if hist.get("u") is None:
            raise ValueError("No hay historial de la variable de recuperación disponible.")

        v, indices, etiquetas = self._separar_neuronas(hist["v"], neuronas, hist.get("nombre"))
        u, _, _ = self._separar_neuronas(hist["u"], neuronas, hist.get("nombre"))

        v, u, etiquetas = self._agrupar_neuronas_similares_dos_variables(v, u, indices, etiquetas, tolerancia_similitud)

        fig, ax = plt.subplots(figsize=figsize)

        for i, etiqueta in enumerate(etiquetas):
            v_plot = v[:, i].copy()
            u_plot = u[:, i].copy()
            resets = np.where(v_plot[:-1] >= 30)[0]

            for r in resets:
                v_plot[r + 1] = np.nan
                u_plot[r + 1] = np.nan

            ax.plot(v_plot, u_plot, label=etiqueta)
            # Marcar el inicio de la trayectoria
            ax.scatter(v[0, i], u[0, i], color="green", marker="o", s=20, zorder=5)
            # Marcar el final de la trayectoria
            ax.scatter(v[-1, i], u[-1, i], color="red", marker="s", s=20, zorder=5)

        ax.set_xlabel("Potencial de membrana, v (mV)")
        ax.set_ylabel("Variable de recuperación, u")
        ax.set_title(titulo)

        if len(etiquetas) <= max_etiquetas_leyenda:
            ax.legend()

        fig.tight_layout()

        if mostrar:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


    def frecuencia_disparos(self, path: str | Path |None = None, historial: Historial | None = None,
                           neuronas: int | slice | Sequence[int] | None = 0,
                           figsize: tuple[float, float] = (10, 6),
                           titulo: str = "Frecuencia de disparos",
                           max_etiquetas_leyenda: int = 10,
                           mostrar: bool = True) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar la frecuencia de disparos de una o varias neuronas.

        La frecuencia de disparo se calcula como el número total de spikes dividido por la duración de la
        simulación, expresándose en Hz.

        Cuando varias neuronas con el mismo nombre presentan exactamente el mismo historial de disparos
        (o uno suficientemente parecido según la tolerancia), se representan mediante un único punto.

        Parameters
        ----------
        Se puede usar un historial desde un archivo con el path, desde un diccionario pasándolo en
        historial, o usar el historial cargado en la clase si no se pasa nada. En el caso de que se
        pasen ambos parámetros, se usará el historial y se ignorará el path.

        path : str | Path | None, optional
            Archivo desde el que cargar el historial.

        historial : Historial | None, optional
            Historial pasado directamente.

        neuronas : int | slice | Sequence[int] | None
            Subconjunto de neuronas que se representarán.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

            Por defecto se representa la neurona 0.

        figsize : tuple[float, float]
            Tamaño de la figura.

        titulo : str
            Título de la gráfica.

        max_etiquetas_leyenda : int
            Número máximo de etiquetas a partir del cual no se muestra la leyenda.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show(). Si False, se cierra la
            figura sin mostrarla.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.

        Raises
        ------
        ValueError
            Si no existe historial de spikes.

        TypeError
            Si neuronas no es un int, un slice, una secuencia de índices o None.

        IndexError
            Si algún índice de neurona está fuera de rango.

        Notes
        -----
        Si el historial no contiene el paso temporal dt se asumirá un valor de 1 ms.
        """
        hist = self._obtener_historial(path, historial)

        if hist is None or hist.get("spikes") is None:
            raise ValueError("No hay historial de spikes disponible.")

        spikes = hist["spikes"]
        dt = hist.get("dt") if hist.get("dt") is not None else 1.0

        datos, indices, etiquetas = self._separar_neuronas(spikes, neuronas, hist.get("nombre"))

        duracion = (datos.shape[0] - 1) * dt / 1000
        tasas = np.sum(datos, axis=0) / duracion if duracion > 0 else np.zeros(1)

        fig, ax = plt.subplots(figsize=figsize)

        for i, etiqueta in enumerate(etiquetas):
            ax.scatter(i, tasas[i], label=etiqueta, c="blue")

        ax.set_xlabel("Neurona")
        ax.set_ylabel("Frecuencia (disparos por s)")
        ax.set_title(titulo)

        if len(indices) <= max_etiquetas_leyenda:
            ticks = np.arange(len(indices))
        else:
            ticks = np.linspace(0, len(indices) - 1, max_etiquetas_leyenda, dtype=int)

        ax.set_xticks(ticks)
        ax.set_xticklabels(indices[ticks])

        if len(etiquetas) <= max_etiquetas_leyenda:
            ax.legend()

        fig.tight_layout()
        if mostrar:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


    # --------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------

    # --------------------------------------------------
    # Carga y normalización del historial
    # --------------------------------------------------

    def _obtener_historial(self, path: str | Path | None = None, historial: Historial | None = None
                           ) -> Historial | None:
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
            de nombre y es_excitatoria de las neuronas y dt de la simulación.

        Returns
        -------
        Historial | None
            Historial normalizado.
        """
        if historial is not None:
            return self._cargar_desde_dict(historial)
        elif path is not None:
            return self._cargar_desde_archivo(path)
        else:
            return self.historial


    def _cargar_desde_dict(self, historial: Historial | None) -> Historial | None:
        """
        Cargar un historial desde un diccionario.

        Parameters
        ----------
        historial : Historial | None
            Historial de un Simulador.

        Returns
        -------
        Historial | None
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

        return self._normalizar_historial(historial)

    def _cargar_desde_archivo(self, path: str | Path) -> Historial | None:
        """
        Cargar un historial desde uno o varios archivos dado el path donde se encuentra.

        En el caso de que no exista el archivo, historial será None, si el formato es CSV o TXT y
        alguno de los archivos necesarios no existen, solo el valor de la clave correspondiente
        será None.

        Parameters
        ----------
        path : str | Path
            Path donde se encuentra el archivo, en el caso de que la extensión sea CSV o TXT, se
            usará para construir el nombre de los archivos en base a cada clave del historial.

            Las extensiones de archivo soportadas son NPZ, JSON, CSV y TXT.

        Returns
        -------
        Historial | None
            Historial normalizado.

        Raises
        ------
        TypeError
            Si el path pasado no es una cadena de texto o una instancia de Path.

        ValueError
            Si el archivo es de formato JSON y no contiene ningún diccionario.
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
            historial = self._normalizar_historial(historial)

        elif formato == "json":
            if not path.exists():
                return None

            with open(path, "r") as f:
                datos = json.load(f)
                if not isinstance(datos, dict):
                    raise ValueError("El archivo JSON tiene que contener un diccionario.")

            historial = {"spikes": datos.get("spikes"),
                         "v": datos.get("v"),
                         "u": datos.get("u"),
                         "I": datos.get("I"),
                         "nombre": datos.get("nombre"),
                         "es_excitatoria": datos.get("es_excitatoria"),
                         "dt": datos.get("dt")}
            historial = self._normalizar_historial(historial)

        elif formato == "csv":
            historial = self._cargar_desde_varios_archivos(path, "csv", ",")
        elif formato == "txt":
            historial = self._cargar_desde_varios_archivos(path, "txt")

        return historial

    def _cargar_desde_varios_archivos(self, path: Path, extension: str | None = None,
                                      delim: str | None = None) -> Historial | None:
        """
        Cargar un historial desde varios archivos TXT o CSV.

        Parameters
        ----------
        path : Path
            Path raíz para deducir los archivos, añadiendo _clave al final del nombre del archivo.

        extension : str | None, optional
            Extensión del archivo, si no se pasa, se deduce del path del archivo pasado, si se pasa,
            tiene que coincidir con la extensión del path del archivo pasado.

        delim : str | None
            Delimitador usado para separar los valores en los archivos. En el caso de ser None, para
            nombre y es_excitatoria se usara \\t.

        Returns
        -------
        Historial | None
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

        nombre = np.loadtxt(path_nombre, dtype=str, delimiter="\t" if delim is None else delim
                            ).tolist() if path_nombre.exists() else None
        es_excitatoria = np.loadtxt(path_excitatorias, dtype=bool, delimiter="\t" if delim is None
                                    else delim).tolist() if path_excitatorias.exists() else None
        dt = np.loadtxt(path_dt, delimiter=delim) if path_dt.exists() else None

        historial = {"spikes": spikes,
                     "v": v,
                     "u": u,
                     "I": i,
                     "nombre": nombre,
                     "es_excitatoria": es_excitatoria,
                     "dt": dt}

        return self._normalizar_historial(historial)

    def _normalizar_historial(self, hist: Historial | None) -> Historial | None:
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
        hist : Historial | None
            Historial a normalizar.

        Returns
        -------
        Historial | None
            Historial normalizado.

        Raises
        ------
        TypeError
            Si el historial pasado no es un diccionario, o alguno de los valores no sea adecuado.
        """
        if not isinstance(hist, dict):
            raise TypeError("El historial pasado debe ser un diccionario.")

        shape = None
        spikes, shape = self._validar_historial_array(hist.get("spikes"), shape)
        v, shape = self._validar_historial_array(hist.get("v"), shape)
        u, shape = self._validar_historial_array(hist.get("u"), shape)
        i, shape = self._validar_historial_array(hist.get("I"), shape)

        if shape is None:
            num_neuronas = None
        elif len(shape) == 1:
            num_neuronas = 1
        else:
            num_neuronas = shape[1]

        nombre = hist.get("nombre")
        if nombre is not None:
            if isinstance(nombre, str):
                nombre = [nombre]
            elif isinstance(nombre, np.ndarray):
                nombre = nombre.tolist()
            elif not isinstance(nombre, list):
                raise TypeError("Los nombres tienen que estar en una lista o un vector.")

            if not all(isinstance(elem, str) for elem in nombre):
                raise TypeError("La lista de nombres tiene que contener solo cadenas de texto.")

            if num_neuronas is None:
                num_neuronas = len(nombre)
            elif num_neuronas != len(nombre):
                raise ValueError("El número de nombres debe coincidir con el número de neuronas con"
                                 " las que se haya realizado la simulación.")

        es_excitatoria = hist.get("es_excitatoria")
        if es_excitatoria is not None:
            if isinstance(es_excitatoria, bool):
                es_excitatoria = [es_excitatoria]
            elif isinstance(es_excitatoria, np.ndarray):
                es_excitatoria = es_excitatoria.tolist()
            elif not isinstance(es_excitatoria, list):
                raise TypeError("es_excitatoria tiene que ser una lista o un vector.")

            if not all(isinstance(elem, bool) or (isinstance(elem, int) and elem in (0, 1)) for elem in es_excitatoria):
                raise TypeError("La lista de es_excitatoria tiene que contener solo booleanos.")

            if num_neuronas is not None and num_neuronas != len(es_excitatoria):
                raise ValueError("La longitud de es_excitatoria debe coincidir con el número de "
                                 "neuronas con las que se haya realizado la simulación.")

        dt = hist.get("dt")
        if dt is not None:
            dt = float(dt)
            if dt <= 0:
                raise ValueError("El paso temporal debe ser mayor que 0.")

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

    def _validar_historial_array(self, dato: ArrayLike | None, shape: tuple[int, ...] | None
                                 ) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
        """
        Validar que el dato pasado tenga la forma correcta.

        Parameters
        ----------
        dato : ArrayLike | None
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

            En el caso de que se pase un shape distinto de None, si el dato pasado no tiene la forma
            correcta.
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


    # --------------------------------------------------
    # Creación de gráficas
    # --------------------------------------------------

    def _mostrar_grafica(self, datos: np.ndarray, etiquetas: list[str], dt: float,
                         ylabel: str, figsize: tuple[float, float], titulo: str,
                         max_etiquetas_leyenda : int, mostrar: bool
                         ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """
        Mostrar una gráfica temporal de una o varias neuronas.

        Parameters
        ----------
        datos : np.ndarray
            Datos temporales con forma (pasos, num_neuronas).

        etiquetas : list[str]
            Etiquetas de las neuronas que se mostrarán en la leyenda.

        dt : float
            Paso temporal utilizado durante la simulación, expresado en milisegundos.

        ylabel : str
            Etiqueta del eje Y.

        figsize : tuple[float, float]
            Tamaño de la figura.

        titulo : str
            Título de la gráfica.

        max_etiquetas_leyenda : int
            Número máximo de etiquetas a partir del cual no se muestra la leyenda.

        mostrar : bool
            Si True, se muestra la figura mediante matplotlib.pyplot.show().
            Si False, se cierra la figura sin mostrarla.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            Figura y ejes creados para el gráfico.
        """
        tiempo = np.arange(datos.shape[0]) * dt

        fig, ax = plt.subplots(figsize=figsize)

        for columna, etiqueta in enumerate(etiquetas):
            ax.plot(tiempo, datos[:, columna], label=etiqueta)

        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)

        if datos.shape[1] <= max_etiquetas_leyenda:
            ax.legend()

        fig.tight_layout()

        if mostrar:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax


    # --------------------------------------------------
    # Selección y agrupación de neuronas
    # --------------------------------------------------

    def _obtener_indices_neuronas(self, neuronas: int | slice | Sequence[int] | None,
                                   num_neuronas: int) -> np.ndarray:
        """
        Obtener los índices de las neuronas que se desean representar.

        Parameters
        ----------
        neuronas : int | slice | Sequence[int] | None
            Neurona o conjunto de neuronas seleccionado.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

        num_neuronas : int
            Número total de neuronas disponibles.

        Returns
        -------
        np.ndarray
            Vector con los índices de las neuronas seleccionadas.

        Raises
        ------
        TypeError
            - Si algún índice no es un entero.
            - Si neuronas no es un int, un slice, una secuencia de índices o None.

        ValueError
            - Si no se pasa ningún índice dentro de una secuencia.
        """
        if neuronas is None:
            return np.arange(num_neuronas)
        elif isinstance(neuronas, int):
            indices = np.asarray([neuronas], dtype=int)
        elif isinstance(neuronas, slice):
            indices = np.arange(num_neuronas)[neuronas]
        elif isinstance(neuronas, (Sequence, np.ndarray)) and not isinstance(neuronas, (str, bytes)):
            if not all(isinstance(n, int) for n in neuronas):
                raise TypeError("Los índices deben ser enteros.")

            if len(neuronas) == 0:
                raise ValueError("No se ha pasado ningún índice.")

            indices = np.asarray(neuronas, dtype=int)
        else:
            raise TypeError("neuronas debe ser int, slice, una secuencia de enteros o None.")

        if np.any(indices >= num_neuronas) or np.any(indices < -num_neuronas):
            raise IndexError("Hay índices de neuronas fuera de rango.")

        return indices

    def _separar_neuronas(self, datos: np.ndarray, neuronas: int | slice | Sequence[int] | None,
                           nombres: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Separar los datos de las neuronas seleccionadas.

        Convierte un historial de una única neurona con forma (pasos,) a una matriz
        con forma (pasos, 1), obtiene los índices seleccionados y devuelve los datos
        correspondientes junto con sus etiquetas.

        Parameters
        ----------
        datos : np.ndarray
            Historial temporal de una variable neuronal. Puede tener forma
            (pasos,) o (pasos, num_neuronas).

        neuronas : int | slice | Sequence[int] | None
            Neuronas que se quieren representar.
            - None: todas las neuronas.
            - int: una única neurona.
            - slice: rango de neuronas.
            - Sequence[int]: secuencia de índices concretos (por ejemplo, una lista o una tupla).

        nombres : list[str] | None
            Nombres de las neuronas. Si es None, se usarán etiquetas genéricas.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Tupla con:
            - Datos separados con forma (pasos, neuronas_seleccionadas).
            - Índices originales de las neuronas seleccionadas.
            - Etiquetas para la leyenda.

        Raises
        ------
        TypeError
            Si neuronas no tiene un formato válido.

        IndexError
            Si algún índice está fuera de rango.
        """
        if datos.ndim == 1:
            datos = datos[:, np.newaxis]

        indices = self._obtener_indices_neuronas(neuronas, datos.shape[1])
        datos = datos[:, indices]

        if nombres is not None:
            etiquetas = [nombres[indice] for indice in indices]
        else:
            etiquetas = [f"Neurona {indice}" for indice in indices]

        return datos, indices, etiquetas

    def _separar_raster_por_tipo(self, indices: np.ndarray, es_excitatoria: list[bool]
                                 ) -> tuple[np.ndarray, float | None, dict[str, float]]:
        """
        Reordenar posiciones verticales del raster separando neuronas
        excitatorias e inhibitorias.

        Las inhibitorias se colocan arriba y las excitatorias abajo.

        Parameters
        ----------
        indices : np.ndarray
            Índices originales de las neuronas representadas.

        es_excitatoria : list[bool]
            Lista ordenada de si las neuronas son excitatorias o inhibitorias.

        Returns
        -------
        tuple[np.ndarray, float | None, dict[str, float]]
            - Nuevas posiciones verticales de cada neurona siguiendo el orden de indices.
            - Posición de la separación entre grupos.
            - Posición vertical de las etiquetas de excitatorias e inhibitorias.
        """
        excitatorias = [indice for indice in indices if es_excitatoria[indice]]
        inhibitorias = [indice for indice in indices if not es_excitatoria[indice]]
        posiciones = np.asarray(excitatorias + inhibitorias)

        etiquetas = {}

        if excitatorias:
            etiquetas["Excitatorias"] = (len(excitatorias) - 1) / 2

        if inhibitorias:
            etiquetas["Inhibitorias"] = len(excitatorias) +  (len(inhibitorias) - 1) / 2

        separacion = len(excitatorias) - 0.5 if inhibitorias and excitatorias else None

        return posiciones, separacion, etiquetas

    def _agrupar_neuronas_similares(self, datos: np.ndarray, indices: np.ndarray, etiquetas: list[str],
                                    tolerancia: float = 0) -> tuple[np.ndarray, list[str]]:
        """
        Agrupar neuronas del mismo nombre con evoluciones temporales prácticamente iguales.

        Parameters
        ----------
        datos : np.ndarray
            Datos temporales con forma (pasos, num_neuronas).

        indices : np.ndarray
            Índices originales de las neuronas representadas.

        etiquetas : list[str]
            Etiquetas originales de las neuronas.

        tolerancia : float
            Diferencia media absoluta máxima para considerar dos neuronas iguales.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            Tupla con:
            - Datos reducidos con una única curva por grupo.
            - Etiquetas nuevas para cada grupo.
        """
        if datos.shape[1] <= 1:
            etiqueta = f"{etiquetas[0]}: {self._formatear_indices_neuronas(indices)}"
            return datos, [etiqueta]

        grupos = []
        usados = set()

        for i in range(datos.shape[1]):
            if i not in usados:
                grupo = [i]
                usados.add(i)

                for j in range(i + 1, datos.shape[1]):
                    # Solo comparar neuronas con el mismo nombre
                    if j not in usados and etiquetas[i] == etiquetas[j]:
                        if tolerancia == 0:
                            if np.array_equal(datos[:, i], datos[:, j]):
                                grupo.append(j)
                                usados.add(j)
                        else:
                            diferencia = np.mean(np.abs(datos[:, i] - datos[:, j]))
                            if diferencia <= tolerancia:
                                grupo.append(j)
                                usados.add(j)

                grupos.append(grupo)

        representantes = []
        etiquetas_agrupadas = []
        for grupo in grupos:
            # Se conserva la primera neurona del grupo como representante
            representantes.append(grupo[0])

            indices_grupo = indices[grupo]
            etiqueta_base = etiquetas[grupo[0]]

            etiqueta = f"{etiqueta_base}: {self._formatear_indices_neuronas(indices_grupo)}"
            etiquetas_agrupadas.append(etiqueta)

        return datos[:, representantes], etiquetas_agrupadas

    def _agrupar_neuronas_similares_dos_variables(self, datos_1: np.ndarray, datos_2: np.ndarray,
                                                  indices: np.ndarray, etiquetas: list[str],
                                                  tolerancia: float = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Agrupar neuronas con evoluciones temporales prácticamente iguales en dos variables.

        Una neurona se agrupará con otra cuando tengan el mismo nombre y sus evoluciones temporales
        sean iguales o suficientemente similares en ambas variables.

        Parameters
        ----------
        datos_1 : np.ndarray
            Primer conjunto de datos temporales con forma (pasos, num_neuronas).

        datos_2 : np.ndarray
            Segundo conjunto de datos temporales con forma (pasos, num_neuronas).

        indices : np.ndarray
            Índices originales de las neuronas representadas.

        etiquetas : list[str]
            Etiquetas originales de las neuronas.

        tolerancia : float
            Diferencia media absoluta máxima permitida en cada una de las dos variables para considerar
            dos neuronas iguales. Si es 0, las dos evoluciones deben ser exactamente iguales.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[str]]
            Tupla con:
            - Primer conjunto de datos reducido con una única columna por grupo.
            - Segundo conjunto de datos reducido con una única columna por grupo.
            - Etiquetas nuevas para cada grupo.

        Raises
        ------
        ValueError
            Si los dos conjuntos de datos no tienen el mismo número de pasos o de neuronas.
        """
        if datos_1.shape != datos_2.shape:
            raise ValueError("Los dos conjuntos de datos deben tener la misma forma.")

        if datos_1.shape[1] <= 1:
            etiqueta = f"{etiquetas[0]}: {self._formatear_indices_neuronas(indices)}"
            return datos_1, datos_2, [etiqueta]

        grupos = []
        usados = set()

        for i in range(datos_1.shape[1]):
            if i not in usados:
                grupo = [i]
                usados.add(i)

                for j in range(i + 1, datos_1.shape[1]):
                    if j not in usados and etiquetas[i] == etiquetas[j]:
                        if tolerancia == 0:
                            iguales = (np.array_equal(datos_1[:, i], datos_1[:, j]) and
                                       np.array_equal(datos_2[:, i], datos_2[:, j]))
                        else:
                            diferencia_1 = np.mean(np.abs(datos_1[:, i] - datos_1[:, j]))
                            diferencia_2 = np.mean(np.abs(datos_2[:, i] - datos_2[:, j]))

                            iguales = (diferencia_1 <= tolerancia and diferencia_2 <= tolerancia)

                        if iguales:
                            grupo.append(j)
                            usados.add(j)

                grupos.append(grupo)

        representantes = []
        etiquetas_agrupadas = []

        for grupo in grupos:
            # Se conserva la primera neurona del grupo como representante
            representantes.append(grupo[0])

            indices_grupo = indices[grupo]
            etiqueta_base = etiquetas[grupo[0]]

            etiqueta = (f"{etiqueta_base}: {self._formatear_indices_neuronas(indices_grupo)}")
            etiquetas_agrupadas.append(etiqueta)

        return datos_1[:, representantes], datos_2[:, representantes], etiquetas_agrupadas    

    def _formatear_indices_neuronas(self, indices: np.ndarray) -> str:
        """
        Convertir índices de neuronas en una representación compacta.

        Parameters
        ----------
        indices : np.ndarray
            Índices a convertir.

        Returns
        -------
        str
            Representación de los índices en forma de texto.

        Examples
        --------
        [0,1,2,3] -> "0 a 3"

        [0,2,5] -> "0, 2, 5"

        [0, 1, 2, 5, 6, 7] -> "0 a 2, 5 a 7"
        """
        indices = np.sort(indices)

        if len(indices) == 0:
            return ""

        partes = []
        inicio = indices[0]
        anterior = indices[0]

        for actual in indices[1:]:
            if actual == anterior + 1:
                anterior = actual
            else:
                if inicio == anterior:
                    partes.append(str(inicio))
                else:
                    partes.append(f"{inicio} a {anterior}")

                inicio = anterior = actual

        if inicio == anterior:
            partes.append(str(inicio))
        else:
            partes.append(f"{inicio} a {anterior}")

        return ", ".join(partes)

    # --------------------------------------------------
    # Acceso al historial
    # --------------------------------------------------

    def _obtener(self, clave: str) -> np.ndarray | list[str] | list[bool] | float | None:
        """
        Obtener cualquier variable del historial.

        Parameters
        ----------
        clave : str
            Clave del valor del historial que se quiere obtener.

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


    # --------------------------------------------------
    # PROPIEDADES
    # --------------------------------------------------

    @property
    def historial(self) -> Historial | None:
        """
        Devuelve una copia completa del historial que haya cargado.

        Returns
        -------
        Historial | None
            Historial cargado. Si no hay ningún historial cargado, se devolverá None.
        """
        if self.__historial is None:
            return None

        return {"spikes": self._obtener("spikes"),
                "v": self._obtener("v"),
                "u": self._obtener("u"),
                "I": self._obtener("I"),
                "nombre": self._obtener("nombre"),
                "es_excitatoria": self._obtener("es_excitatoria"),
                "dt": self._obtener("dt")}

    @property
    def spikes(self) -> np.ndarray | None:
        """
        Copia del historial de disparos de la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz con los disparos de la red a lo largo del tiempo. Si
            no hay ningún historial cargado, se devolverá None.
        """
        return self._obtener("spikes")

    @property
    def v(self) -> np.ndarray | None:
        """
        Copia del historial de potenciales de membrana de la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona a la que representa. Si
            no hay ningún historial cargado, se devolverá None.
        """
        return self._obtener("v")

    @property
    def u(self) -> np.ndarray | None:
        """
        Copia del historial de variables de recuperación de la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona a la que representa. Si
            no hay ningún historial cargado, se devolverá None.
        """
        return self._obtener("u")

    @property
    def I(self) -> np.ndarray | None:
        """
        Copia del historial de corrientes introducidas a la red durante la simulación.

        Returns
        -------
        np.ndarray | None
            Matriz donde cada fila es un paso y cada columna es la neurona a la que se le ha aplicado
            la corriente. Si no hay ningún historial cargado, se devolverá None.
        """
        return self._obtener("I")

    @property
    def nombre(self) -> list[str] | None:
        """
        Copia de la lista ordenada de nombres de las neuronas de la red de la simulación.

        Returns
        -------
        list[str] | None
            Lista ordenada de los nombres de las neuronas. Si no hay ningún historial cargado, se
            devolverá None.
        """
        return self._obtener("nombre")

    @property
    def es_excitatoria(self) -> list[bool] | None:
        """
        Copia de la lista que determina si las neuronas de la red de la simulación son excitatorias
        o inhibitorias.

        Returns
        -------
        list[bool] | None
            Lista con si las neuronas son excitatorias o inhibitorias. Si no hay ningún historial
            cargado, se devolverá None.
        """
        return self._obtener("es_excitatoria")

    @property
    def dt(self) -> float | None:
        """
        El paso temporal usado en el historial cargado.

        Returns
        -------
        float | None
            El paso temporal. Si no hay ningún historial cargado, se devolverá None.
        """
        return self._obtener("dt")
