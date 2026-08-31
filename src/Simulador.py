import time
import numpy as np
import cupy as cp

import psutil
import pynvml

from typing import Any, Literal
from pathlib import Path
from numbers import Real
from tqdm import tqdm

from .Neurona import Neurona
from .RedDeNeuronas import RedDeNeuronas, Array


class Simulador:
    """
    Simulador de una red de neuronas basada en el modelo de Izhikevich.

    Attributes
    ----------
    dt : float
        Paso temporal de integración en milisegundos.

    red : Neurona | RedDeNeuronas | None
        Modelo actualmente cargado en el simulador.
        Puede ser una instancia de Neurona o RedDeNeuronas.

    num_neuronas : int
        Número de neuronas del sistema actual.

    paso_actual : int
        Índice del próximo paso a simular.

    historial : dict[str, np.ndarray | list[str] | list[bool] | float] | None
        Historial de spikes, variables de estado (v y u) y corriente de entrada en cada paso, siendo
        estos None si aún no se ha ejecutado ninguna simulación. Además, también contiene información
        adicional para facilitar la interpretación de los datos, como una lista ordenada de los nombres
        de las neuronas cargadas, si son excitatorias o inhibitorias y el paso temporal de la simulación.

    rendimiento : dict[str, float | None]
        Valores de rendimiento de la última simulación ejecutada.

    configuracion : dict[str, bool | str | int]
        Configuración por defecto de parámetros de la simulación.

    Notes
    -----
    Internamente se almacenan buffers separados para spikes, v y u en memoria CPU para evitar saturación
    de VRAM en ejecuciones con backend GPU, almacenándose buffers temporales de menor tamaño para
    agrupar transferencias.

    Estos historiales siempre tienen precisión float32, en el caso de v y de u, independientemente
    del dtype interno utilizado por la red.

    Las corrientes de entrada pueden proporcionarse como:
    - Un escalar, aplicándose la misma corriente a todas las neuronas durante toda la simulación.
    - Un vector de longitud N, aplicando una corriente fija a cada neurona.
    - Una matriz de dimensiones (M, N), donde cada fila representa la corriente aplicada en un paso temporal.
    Si M es menor que el número de pasos simulados, la matriz se recorre cíclicamente.

    En el caso de una única neurona, un vector de longitud M se interpreta como una corriente variable
    temporalmente.
    """


    # --------------------------------------------------
    # CONSTRUCTOR
    # --------------------------------------------------

    def __init__(self, paso_temporal: float = 1, guardar_resultados: bool = False,
                 path_guardado: str = "./historial.npz", mostrar_progreso: bool = False,
                 medir_rendimiento: bool = False, intervalo_rendimiento: int = 100,
                 tamano_batch: int = 100):
        """
        Inicializa una instancia de Simulador con el paso temporal indicado.

        Parameters
        ----------
        paso_temporal : float
            Paso temporal de integración expresado en milisegundos.

        guardar_resultados : bool, optional
            Decidir si guardar los resultados automáticamente al final de la simulación o no.

        path_guardado : str, optional
            El path por defecto donde guardar los resultados en el caso de que se decida guardarlos.

        mostrar_progreso : bool, optional
            Decidir si mostrar el progreso de la simulación en forma de una barra de progreso, si True,
            reduce el rendimiento de la simulación.

        medir_rendimiento : bool, optional
            Decidir si medir el rendimiento durante la simulación, si True, reduce el rendimiento
            según el intervalo de medición del rendimiento.

        intervalo_rendimiento : int, optional
            Decidir cada cuántos pasos se mide el rendimiento, a menor intervalo, más se reduce el
            rendimiento de la simulación.

        tamano_batch : int, optional
            Tamaño de los batches que se guardan en la gpu antes de transferirse a memoria. Solo
            afecta si la red cargada tiene de backend cupy.

        Raises
        ------
        TypeError
            Si el paso temporal no es un número real.

        ValueError
            Si el paso temporal proporcionado es negativo o igual a 0.
        """
        if not isinstance(paso_temporal, Real):
            raise TypeError("El paso temporal debe ser un número Real.")

        if paso_temporal <= 0:
            raise ValueError("El paso temporal debe ser positivo.")

        self.__dt = paso_temporal
        self.__red = None

        # Valores por defecto de parámetros de las simulaciones
        self.__guardar_resultados = None
        self.__path_guardado = None
        self.__mostrar_progreso = None
        self.__medir_rendimiento = None
        self.__intervalo_rendimiento = None
        self.__tamano_batch = None
        self.configurar_simulacion(guardar_resultados, path_guardado, mostrar_progreso,
                                   medir_rendimiento, intervalo_rendimiento, tamano_batch)

        # Historiales en RAM (NumPy) para evitar saturar la VRAM de la GPU
        self.__historial_spikes = None
        self.__historial_v = None
        self.__historial_u = None
        self.__historial_I = None

        # Paso actual de la simulación.
        self.__paso_actual = 0

        # Número de neuronas cargadas
        self.__num_neuronas = 0

        # Valores de rendimiento
        self.__tiempo_ejecucion = 0

        self.__cpu_media = None
        self.__cpu_maxima = None

        self.__ram_media = None
        self.__ram_maxima = None

        self.__gpu_media = None
        self.__gpu_maxima = None

        self.__vram_media = None
        self.__vram_maxima = None


    # --------------------------------------------------
    # MÉTODOS PÚBLICOS
    # --------------------------------------------------

    def cargar_red(self, red: RedDeNeuronas) -> None:
        """
        Cargar una red de neuronas en el simulador.

        Parameters
        ----------
        red : RedDeNeuronas
            Red a cargar.

        Raises
        ------
        TypeError
            Si la red a cargar no es una instancia de RedDeNeuronas.

        ValueError
            Si ya hay otra red o una neurona cargada en el simulador.
        """
        if not isinstance(red, RedDeNeuronas):
            raise TypeError("Solo se puede cargar una RedDeNeuronas con este método.")

        if self.__red is not None:
            raise ValueError("Ya hay una neurona o red cargada. Limpia el simulador primero.")

        self.__red = red
        self.__num_neuronas = self.__red.num_neuronas

    def cargar_neurona(self, neurona: Neurona) -> None:
        """
        Cargar una neurona en el simulador.

        Parameters
        ----------
        neurona : Neurona
            Neurona a cargar.

        Raises
        ------
        TypeError
            Si la neurona a cargar no es una instancia de Neurona.

        ValueError
            Si ya hay otra neurona o una red de neuronas cargada en el simulador.
        """
        if not isinstance(neurona, Neurona):
            raise TypeError("Solo se puede cargar una Neurona con este método.")

        if self.__red is not None:
            raise ValueError("Ya hay una neurona o red cargada. Limpia el simulador primero.")

        self.__red = neurona
        self.__num_neuronas = 1

    def configurar_simulacion(self, guardar_resultados: bool | None = None, path_guardado: str | None = None,
                              mostrar_progreso: bool | None = None, medir_rendimiento: bool | None = None,
                              intervalo_rendimiento: int | None = None, tamano_batch: int | None = None) -> None:
        """
        Configurar el comportamiento por defecto al simular.

        Parameters
        ----------
        guardar_resultados : bool | None, optional
            Decidir si guardar los resultados automáticamente al final de la simulación o no.

        path_guardado : str | None, optional
            El path por defecto donde guardar los resultados en el caso de que se decida guardarlos.

        mostrar_progreso : bool | None, optional
            Decidir si mostrar el progreso de la simulación en forma de una barra de progreso, si True,
            reduce el rendimiento de la simulación.

        medir_rendimiento : bool | None, optional
            Decidir si medir el rendimiento durante la simulación, si True, reduce el rendimiento
            según el intervalo de medición del rendimiento.

        intervalo_rendimiento : int | None, optional
            Decidir cada cuántos pasos se mide el rendimiento, a menor intervalo, más se reduce el
            rendimiento de la simulación.

        tamano_batch : int | None, optional
            Tamaño de los batches que se guardan en la gpu antes de transferirse a memoria. Solo
            afecta si la red cargada tiene de backend cupy.
        """
        config = {"guardar_resultados": self.__guardar_resultados,
                  "path_guardado": self.__path_guardado,
                  "mostrar_progreso": self.__mostrar_progreso,
                  "medir_rendimiento": self.__medir_rendimiento,
                  "intervalo_rendimiento": self.__intervalo_rendimiento,
                  "tamano_batch": self.__tamano_batch}

        if guardar_resultados is not None:
            config["guardar_resultados"] = guardar_resultados
        if path_guardado is not None:
            config["path_guardado"] = path_guardado
        if mostrar_progreso is not None:
            config["mostrar_progreso"] = mostrar_progreso
        if medir_rendimiento is not None:
            config["medir_rendimiento"] = medir_rendimiento
        if intervalo_rendimiento is not None:
            config["intervalo_rendimiento"] = intervalo_rendimiento
        if tamano_batch is not None:
            config["tamano_batch"] = tamano_batch

        self._validar_parametros_simulacion(**config)

        self.__guardar_resultados = config["guardar_resultados"]
        self.__path_guardado = config["path_guardado"]
        self.__mostrar_progreso = config["mostrar_progreso"]
        self.__medir_rendimiento = config["medir_rendimiento"]
        self.__intervalo_rendimiento = config["intervalo_rendimiento"]
        self.__tamano_batch = config["tamano_batch"]

    def simular(self, pasos: int, I: float | list[float] | list[list[float]] | Array = 0,
                guardar_resultados: bool | None = None, path_guardado: str | None = None,
                mostrar_progreso: bool | None = None, medir_rendimiento: bool | None = None,
                intervalo_rendimiento: int | None = None, tamano_batch: int | None = None) -> float:
        """
        Realizar un cierto número de pasos de simulación. Pudiendo decidir si guardar los historiales
        al finalizar, así como si mostrar el progreso de forma dinámica o si medir el rendimiento.
        Se tardará menos en simular si no se muestra el progreso ni se mide el rendimiento.

        Parameters
        ----------
        pasos : int
            Número de pasos a simular. En el caso de ser 0, si no hay nada guardado en el historial,
            se guardará el estado actual como estado inicial, si ya hay estados guardados en el historial,
            no se guardará nada.

        I : float | list[float] | Array, optional
            Corriente de entrada para las neuronas. Por defecto, no se aplica corriente a las neuronas.

            Puede proporcionarse como:
            - Escalar: misma corriente para todas las neuronas y pasos.
            - Vector de longitud N: corriente constante para cada neurona. En el caso de una única
              neurona, un vector de longitud N se interpreta como una secuencia temporal de corrientes.
            - Matriz (M, N): corriente variable temporalmente, donde M es el número
            de muestras disponibles y N el número de neuronas.

            Si la matriz tiene menos filas que pasos simulados, se reutiliza
            cíclicamente.

            En caso de una única neurona, un vector de longitud M se interpreta
            como una corriente temporal.

        Los siguientes parámetros son para configuraciones personalizadas, no modifican los valores
        por defecto, si se quiere cambiar el valor por defecto de alguno de ellos, se puede usar
        configurar_simulacion:

        guardar_resultados : bool | None, optional
            Decidir si guardar los resultados automáticamente al final de la simulación o no.

        path_guardado : str | None, optional
            El path por defecto donde guardar los resultados en el caso de que se decida guardarlos.

        mostrar_progreso : bool | None, optional
            Decidir si mostrar el progreso de la simulación en forma de una barra de progreso, si True,
            reduce el rendimiento de la simulación.

        medir_rendimiento : bool | None, optional
            Decidir si medir el rendimiento durante la simulación, si True, reduce el rendimiento
            según el intervalo de medición del rendimiento.

        intervalo_rendimiento : int | None, optional
            Decidir cada cuántos pasos se mide el rendimiento, a menor intervalo, más se reduce el
            rendimiento de la simulación.

        tamano_batch : int | None, optional
            Tamaño de los batches que se guardan en la gpu antes de transferirse a memoria. Solo
            afecta si la red cargada tiene de backend cupy.

        Returns
        -------
        float
            Tiempo de ejecución de la simulación en segundos, sin incluir la preparación inicial ni
            el posible guardado de los historiales.

        Raises
        ------
        TypeError
            Si alguno de los parámetros no es del tipo de dato correcto.

        ValueError
            Si no hay nada cargado, o si los pasos a simular son negativos.

        Examples
        --------
        Crear y cargar una red de tres neuronas:

        >>> red = RedDeNeuronas({"rs": 3})
        >>> simulador.cargar_red(red)

        Simular 1000 pasos con una corriente constante de 10 para todas las neuronas:

        >>> simulador.simular(1000, I=10)

        Aplicar una corriente fija diferente a cada una de las tres neuronas:

        >>> simulador.simular(1000, I=[5, 10, 15])

        Aplicar una corriente variable temporalmente a una red de tres neuronas:

        >>> corriente = np.array([[5, 10, 15],
                                  [10, 15, 20]])
        >>> simulador.simular(1000, I=corriente)

        En el caso de una única neurona, un vector representa una corriente variable
        temporalmente:

        >>> simulador.limpiar_todo()
        >>> simulador.cargar_neurona(Neurona.predefinida("rs"))
        >>> corriente = [5, 10, 15, 20]
        >>> simulador.simular(4, I=corriente)
        """
        if guardar_resultados is None:
            guardar_resultados = self.__guardar_resultados
        if path_guardado is None:
            path_guardado = self.__path_guardado
        if mostrar_progreso is None:
            mostrar_progreso = self.__mostrar_progreso
        if medir_rendimiento is None:
            medir_rendimiento = self.__medir_rendimiento
        if intervalo_rendimiento is None:
            intervalo_rendimiento = self.__intervalo_rendimiento
        if tamano_batch is None:
            tamano_batch = self.__tamano_batch

        # Comprobaciones iniciales de tipos y valores de los parámetros
        self._validar_parametros_simulacion(guardar_resultados, path_guardado, mostrar_progreso,
                                            medir_rendimiento, intervalo_rendimiento, tamano_batch)

        if self.__red is None:
            raise ValueError("No hay ninguna red o neurona cargada para simular.")

        if not isinstance(pasos, int):
            raise TypeError("Los pasos deben ser un entero.")

        if pasos < 0:
            raise ValueError("Los pasos a simular deben ser un entero mayor o igual a 0.")


        # Reiniciar las métricas de rendimiento
        self.limpiar_rendimiento()

        # 1. Crear referencias locales de los atributos usados, para evitar búsqueda de atributos
        # repetida en el bucle
        historial_spikes = self.__historial_spikes
        historial_v = self.__historial_v
        historial_u = self.__historial_u
        historial_I = self.__historial_I
        red = self.__red
        paso_actual = self.__paso_actual
        num_neuronas = self.__num_neuronas
        dt = self.__dt 
        I = self._convertir_I(I)

        # 2. Reservar o expandir el historial en la RAM de la CPU (NumPy)
        nuevo_tamano = paso_actual + pasos

        # Si el tamaño es 0, cambiarlo a 1 para poder guardar el estado inicial en paso_actual = 0
        if nuevo_tamano == 0:
            nuevo_tamano = 1

        # Si se empieza desde paso actual = 0, y se queire simular algo, sumar 1 al tamaño para que
        # quepa el estado inicial también
        if paso_actual == 0 and pasos > 0:
            nuevo_tamano += 1

        shape_historial = (nuevo_tamano, num_neuronas)

        if historial_v is None:
            historial_spikes = np.empty(shape_historial, dtype=bool)
            historial_v = np.empty(shape_historial, dtype=np.float32)
            historial_u = np.empty(shape_historial, dtype=np.float32)
            historial_I = np.empty(shape_historial, dtype=np.float32)
        else:
            # Si se llama a simular() varias veces seguidas, la matriz de RAM se expande
            nuevo_spikes = np.empty(shape_historial, dtype=bool)
            nuevo_spikes[:paso_actual] = historial_spikes
            historial_spikes = nuevo_spikes

            nuevo_v = np.empty(shape_historial, dtype=np.float32)
            nuevo_v[:paso_actual] = historial_v
            historial_v = nuevo_v

            nuevo_u = np.empty(shape_historial, dtype=np.float32)
            nuevo_u[:paso_actual] = historial_u
            historial_u = nuevo_u

            nuevo_I = np.empty(shape_historial, dtype=np.float32)
            nuevo_I[:paso_actual] = historial_I
            historial_I = nuevo_I

        # 3. Guardar el estado inicial si estamos en el paso cero
        if paso_actual == 0:
            v_actual, u_actual = red._estado()
            if red.uso_gpu:
                v_actual = v_actual.get()
                u_actual = u_actual.get()

            historial_spikes[0] = False
            historial_v[0] = np.asarray(v_actual)
            historial_u[0] = np.asarray(u_actual)
            historial_I[0] = np.zeros(num_neuronas, dtype=np.float32)
            paso_actual += 1

            # Si solo se quería registrar el estado inicial (pasos == 0), se termina la simulación
            if pasos == 0:
                self.__historial_spikes = historial_spikes
                self.__historial_v = historial_v
                self.__historial_u = historial_u
                self.__historial_I = historial_I
                self.__paso_actual = paso_actual
                return 0

        # 4. Determinar si la red interna está usando GPU (CuPy) o CPU (NumPy)
        red_usa_gpu = red.uso_gpu

        # Barra de progreso
        barra = None
        if mostrar_progreso:
            barra = tqdm(total=pasos, desc="Simulando", unit="pasos", leave=True)

        # Acumuladores de rendimiento
        muestras = 0

        cpu_suma = 0
        cpu_max = 0

        ram_suma = 0
        ram_max = 0

        gpu_suma = None
        gpu_max = None

        vram_suma = None
        vram_max = None

        process = None
        gpu_handle = None

        nvml_iniciado = False
        try:
            if medir_rendimiento:
                process = psutil.Process()

                # Primera llamada necesaria para inicializar medición
                process.cpu_percent(None)

                if red_usa_gpu:
                    pynvml.nvmlInit()
                    nvml_iniciado = True

                    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(cp.cuda.Device().id)

                    gpu_suma = 0
                    gpu_max = 0

                    vram_suma = 0
                    vram_max = 0

            # 5. Sincronizar CUDA antes de empezar si se usa GPU para un benchmark preciso
            if red_usa_gpu:
                cp.cuda.Stream.null.synchronize()

            tiempo_inicio = time.perf_counter()

            # 6. Bucle Temporal de Simulación
            # Comprobar si la corriente es temporal
            I_temporal = isinstance(I, np.ndarray) and I.ndim == 2

            if I_temporal:
                long_I = len(I)

            # Duplicación para evitar la comprobación del if en cada paso del bucle
            if red_usa_gpu:
                buffer_spikes_gpu = cp.empty((tamano_batch, num_neuronas), dtype=bool)
                buffer_v_gpu = cp.empty((tamano_batch, num_neuronas), dtype=red.dtype)
                buffer_u_gpu = cp.empty((tamano_batch, num_neuronas), dtype=red.dtype)
                buffer_I_gpu = cp.empty((tamano_batch, num_neuronas), dtype=red.dtype)

                indice_batch = 0
                inicio_batch = paso_actual

                I = cp.asarray(I, dtype=red.dtype)

                for _ in range(pasos):
                    I_actual = I[(inicio_batch + indice_batch - 1) % long_I] if I_temporal else I

                    if num_neuronas == 1 and I_temporal:
                        I_actual = float(I_actual[0])

                    # Avanzar un paso en la simulación.
                    spikes_actual = red._actualizar(I_actual, dt)

                    # Estado actual en GPU
                    v_actual, u_actual = red._estado()

                    # Guardar en buffer de VRAM
                    buffer_spikes_gpu[indice_batch] = spikes_actual
                    buffer_v_gpu[indice_batch] = v_actual
                    buffer_u_gpu[indice_batch] = u_actual
                    buffer_I_gpu[indice_batch] = I_actual

                    indice_batch += 1

                    # Si buffer llenos, transferir a RAM
                    if indice_batch == tamano_batch:
                        historial_spikes[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_spikes_gpu)
                        historial_v[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_v_gpu)
                        historial_u[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_u_gpu)
                        historial_I[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_I_gpu)

                        inicio_batch += tamano_batch
                        indice_batch = 0

                    if barra is not None:
                        barra.update()

                    if medir_rendimiento and ((inicio_batch + indice_batch) % intervalo_rendimiento == 0):
                        (muestras, cpu_suma, cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma,
                         vram_max) = self._actualizar_rendimiento(barra, process, gpu_handle, red_usa_gpu,
                                                                  muestras, cpu_suma, cpu_max, ram_suma,
                                                                  ram_max, gpu_suma, gpu_max, vram_suma,
                                                                  vram_max)

                # Si queda un último batch incompleto, copiarlo.
                if indice_batch > 0:
                    historial_spikes[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_spikes_gpu[:indice_batch])
                    historial_v[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_v_gpu[:indice_batch])
                    historial_u[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_u_gpu[:indice_batch])
                    historial_I[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_I_gpu[:indice_batch])

                paso_actual += pasos

            else:
                for _ in range(pasos):
                    I_actual = I[(paso_actual - 1) % long_I] if I_temporal else I

                    if num_neuronas == 1 and I_temporal:
                        I_actual = float(I_actual[0])

                    # Avanzar un paso en la simulación.
                    spike_actual = red._actualizar(I_actual, dt)

                    # Obtener el nuevo estado
                    v_actual, u_actual = red._estado()

                    historial_spikes[paso_actual] = spike_actual
                    historial_v[paso_actual] = v_actual
                    historial_u[paso_actual] = u_actual
                    historial_I[paso_actual] = I_actual

                    paso_actual += 1

                    if barra is not None:
                        barra.update()

                    if medir_rendimiento and (paso_actual % intervalo_rendimiento == 0):
                        (muestras, cpu_suma, cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma,
                         vram_max) = self._actualizar_rendimiento(barra, process, gpu_handle, red_usa_gpu,
                                                                  muestras, cpu_suma, cpu_max, ram_suma,
                                                                  ram_max, gpu_suma, gpu_max, vram_suma,
                                                                  vram_max)

            if red_usa_gpu:
                cp.cuda.Stream.null.synchronize()

            duracion = time.perf_counter() - tiempo_inicio
            self.__tiempo_ejecucion = duracion

            self.__historial_spikes = historial_spikes
            self.__historial_v = historial_v
            self.__historial_u = historial_u
            self.__historial_I = historial_I
            self.__paso_actual = paso_actual

            # 7. Calcular y guardar información de rendimiento en atributos de la clase.
            if medir_rendimiento and muestras > 0:
                self.__cpu_media = cpu_suma / muestras
                self.__cpu_maxima = cpu_max

                self.__ram_media = ram_suma / muestras
                self.__ram_maxima = ram_max

                if red_usa_gpu:
                    self.__gpu_media = gpu_suma / muestras
                    self.__gpu_maxima = gpu_max

                    self.__vram_media = vram_suma / muestras
                    self.__vram_maxima = vram_max

        finally:
            # Cerrar barra de progreso
            if barra is not None:
                barra.close()
            # Cerrar NVML
            if nvml_iniciado:
                pynvml.nvmlShutdown()

        # 8. Guardar el historial
        if guardar_resultados:
            if path_guardado is None:
                self.guardar_historial()
            else:
                self.guardar_historial(path=path_guardado)

        return duracion

    def limpiar_todo(self) -> None:
        """
        Eliminar la red o neurona cargada del simulador, así como los historiales y los valores de rendimiento
        almacenados.
        """
        if self.__red is not None:
            self.__red = None
            self.__num_neuronas = 0
        self.limpiar_historial()
        self.limpiar_rendimiento()

    def limpiar_historial(self) -> None:
        """
        Eliminar el historial del estado y los disparos de las neuronas a lo largo del tiempo, reiniciando
        el paso actual a 0.
        """
        self.__historial_spikes = None
        self.__historial_v = None
        self.__historial_u = None
        self.__historial_I = None
        self.__paso_actual = 0

    def limpiar_rendimiento(self) -> None:
        """
        Reiniciar todos los valores de rendimiento guardados hasta el momento.
        """
        self.__tiempo_ejecucion = 0

        self.__cpu_media = None
        self.__cpu_maxima = None

        self.__ram_media = None
        self.__ram_maxima = None

        self.__gpu_media = None
        self.__gpu_maxima = None

        self.__vram_media = None
        self.__vram_maxima = None

    def reiniciar(self) -> None:
        """
        Reiniciar el estado del simulador, incluido el estado de la posible neurona o red de neuronas
        cargada, eliminando los historiales en el proceso y reiniciando los valores de rendimiento.
        """
        if self.__red is not None:
            self.__red.reiniciar()
        self.limpiar_historial()
        self.limpiar_rendimiento()

    def guardar_historial(self, path: str | None = None,
                          formato: Literal["npz", "txt", "json", "csv"] | None = None) -> None:
        """
        Guardar el historial en uno o varios archivos, pudiendo elegir el formato entre varios posibles.
        Asimismo, también se guardan metadatos para poder analizar los valores después, incluyendo
        nombres ordenados de las neuronas que haya cargadas, si son excitatorias o inhibitorias,
        y el paso temporal, dt, usado.

        En el caso de que el formato especificado, tanto en el path como en el parámetro formato, no
        esté soportado, se usará el valor por defecto.

        Parameters
        ----------
        path : str | None, optional
            La ruta, absoluta o relativa, al archivo en el que se va a guardar el historial. Debe
            incluir el nombre del archivo y, en el caso de no proporcionar un formato, la extensión
            elegida para guardar los resultados. Por defecto se usa el último valor configurado.

            Si se proporciona un formato distinto de la extensión actual del path, la 
            extensión original se eliminará y se sustituirá por la correspondiente al formato elegido.
            Si el path no tiene extensión y no se especifica formato, se asumirá la extensión "npz".

        formato : Literal["npz", "txt", "json", "csv"] | None, optional
            El formato elegido para guardar los datos. Si es None, se inferirá de la extensión del
            path. Los formatos soportados son:
                - "npz" : (Recomendado) Formato binario comprimido de NumPy/CuPy. Guardando un solo archivo.
                - "json" : Formato de texto estructurado. Los arrays se convertirán automáticamente
                           a listas. Guardando un solo archivo.
                - "csv" : Valores separados por comas. Guardando un archivo para cada historial y metadato.
                - "txt" : Texto en plano. Guardando un archivo para cada historial y metadato.

        Notes
        -----
        Si la red cargada tiene una sola neurona, o se ha cargado una neurona, los historiales se
        aplanarán a vectores antes de guardarse.

        Examples
        --------
        Guardar el historial en el formato configurado por defecto:

        >>> simulador.guardar_historial()

        Guardar el historial en un archivo NPZ concreto:

        >>> simulador.guardar_historial("resultados.npz")

        Guardar el historial en formato JSON independientemente de la extensión
        proporcionada:

        >>> simulador.guardar_historial("resultados.npz", formato="json")
        """
        historial = self._obtener_historial_completo()
        if historial is None:
            print("No hay datos en el historial para guardar.")
            return
        
        # Procesar la extensión del archivo y formato
        formatos_soportados = ["npz", "json", "csv", "txt"]

        if path is None:
            path = self.__path_guardado

        filepath = Path(path)
        if formato is None or formato not in formatos_soportados:
            formato = filepath.suffix.lower().replace(".", "") if filepath.suffix else "npz"
        
        if formato not in formatos_soportados:
            formato = "npz"
            
        filepath = filepath.with_suffix(f".{formato}")

        # Asegurar el directorio de salida
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Cargar los historiales
        spikes_data = historial["spikes"]
        v_data = historial["v"]
        u_data = historial["u"]
        I_data = historial["I"]

        # Cargar los metadatos
        nombres = historial["nombre"]
        excitatorias = historial["es_excitatoria"]
        paso_temporal = historial["dt"]

        # Exportar según el formato
        if formato == "npz":
            np.savez_compressed(filepath, spikes=spikes_data, v=v_data, u=u_data, I=I_data, nombre=nombres,
                                es_excitatoria=excitatorias, dt=paso_temporal)
            
            print(f"Historiales guardados exitosamente en: {filepath}")

        elif formato == "json":
            import json
            estructura_json = {"spikes": spikes_data.tolist(), "v": v_data.tolist(), "u": u_data.tolist(),
                               "I": I_data.tolist(), "nombre": nombres, "es_excitatoria": excitatorias,
                               "dt": paso_temporal}
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(estructura_json, f, indent=2)

            print(f"Historiales guardados exitosamente en: {filepath}")

        elif formato == "csv" or formato == "txt":
            delim = "," if formato == "csv" else " "
            historiales = ("spikes", "v", "u", "I")
            data = (spikes_data, v_data, u_data, I_data)

            # Guardar en cuatro archivos tabulares independientes (Filas: Pasos, Columnas: Neuronas)
            print("Historiales guardados exitosamente en:")
            for historial, datos in zip(historiales, data):
                path_historial = filepath.with_name(f"{filepath.stem}_{historial}.{formato}")
                np.savetxt(path_historial, datos, delimiter=delim)
                print(f"  - {path_historial}")

            # Guardar archivos con los metadatos: nombre, es_excitatoria y dt
            path_nombre = filepath.with_name(f"{filepath.stem}_nombre.{formato}")
            path_excitatorias = filepath.with_name(f"{filepath.stem}_es_excitatoria.{formato}")
            path_dt = filepath.with_name(f"{filepath.stem}_dt.{formato}")

            np.savetxt(path_dt, [paso_temporal], delimiter=delim)

            if formato == "txt":
                delim = "\t"
            np.savetxt(path_nombre, nombres, fmt="%s", delimiter=delim)
            np.savetxt(path_excitatorias, excitatorias, fmt="%u", delimiter=delim)

            print(f"Con metadatos en:\n  - {path_nombre}\n  - {path_excitatorias}\n  - {path_dt}")


    # --------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------

    def _validar_parametros_simulacion(self, guardar_resultados: bool, path_guardado : str,
                                       mostrar_progreso: bool, medir_rendimiento: bool,
                                       intervalo_rendimiento: int, tamano_batch: int) -> None:
        """
        Comprobar que los parámetros de la simulación sean correctos.

        Parameters
        ----------
        guardar_resultados : bool
            Decidir si guardar los resultados automáticamente al final de la simulación o no.

        path_guardado : str
            El path por defecto donde guardar los resultados en el caso de que se decida guardarlos.

        mostrar_progreso : bool
            Decidir si mostrar el progreso de la simulación en forma de una barra de progreso, si True,
            reduce el rendimiento de la simulación.

        medir_rendimiento : bool
            Decidir si medir el rendimiento durante la simulación, si True, reduce el rendimiento
            según el intervalo de medición del rendimiento.

        intervalo_rendimiento : int
            Decidir cada cuántos pasos se mide el rendimiento, a menor intervalo, más se reduce el
            rendimiento de la simulación.

        tamano_batch : int
            Tamaño de los batches que se guardan en la gpu antes de transferirse a memoria. Solo
            afecta si la red cargada tiene de backend cupy.
        
        Raises
        ------
        TypeError
            Si alguno de los parámetros no es del tipo de dato correcto.
        
        ValueError
            Si el intervalo de medir rendimiento o el tamaño de batches en gpu tienen un valor menor o igual que 0.
        """
        if not isinstance(guardar_resultados, bool):
            raise TypeError("guardar_resultados debe ser un booleano.")

        if not isinstance(path_guardado, str):
            raise TypeError("path_guardado debe ser una cadena de texto con el path por defecto donde"
                            " guardar los resultados ")

        if not isinstance(mostrar_progreso, bool):
            raise TypeError("mostrar_progreso debe ser un booleano.")
        
        if not isinstance(medir_rendimiento, bool):
            raise TypeError("medir_rendimiento debe ser un booleano.")

        if not isinstance(intervalo_rendimiento, int):
            raise TypeError("El intervalo de recogida de valores de rendimiento debe ser un entero.")
        
        if intervalo_rendimiento <= 0:
            raise ValueError("El intervalo de recogida de valores de rendimientos debe ser positivo.")

        if not isinstance(tamano_batch, int):
            raise TypeError("El tamaño del batch de gpu debe ser un entero.")
        
        if tamano_batch <= 0:
            raise ValueError("El tamaño del batch debe ser un entero positivo.")

    def _actualizar_rendimiento(self, barra: tqdm | None, process: psutil.Process,
                                 gpu_handle: Any | None, red_usa_gpu: bool,
                                 muestras: int, cpu_suma: float, cpu_max: float, ram_suma: float,
                                 ram_max: float, gpu_suma: float, gpu_max: float, vram_suma: float,
                                 vram_max: float) -> tuple[int, float, float, float, float, float | None,
                                                           float | None, float | None, float | None]:
        """
        Actualizar los acumuladores de rendimiento durante una simulación.

        Recoge una nueva muestra del consumo actual de CPU y memoria RAM.
        En caso de utilizar GPU mediante CuPy, también recoge el uso de GPU
        y memoria VRAM utilizada.

        El método no almacena las muestras individuales, sino que actualiza
        acumuladores de suma, máximo y número de muestras para permitir el
        cálculo posterior de medias sin consumir memoria adicional.

        Parameters
        ----------
        barra : tqdm | None
            Barra de progreso asociada a la simulación.
            Si no es None, se actualiza mostrando los valores actuales.

        process : psutil.Process
            Proceso cuyo consumo de memoria será medido.

        gpu_handle : Any | None
            Identificador del dispositivo GPU proporcionado por NVML.
            Puede ser None si no se utiliza GPU.

        red_usa_gpu : bool
            Indica si la simulación utiliza backend GPU mediante CuPy.

        muestras : int
            Número de muestras recogidas.
        
        cpu_suma : float
            Acumulador del uso total de CPU.

        cpu_max : float
            Máximo uso de CPU registrado.

        ram_suma : float
            Acumulador del uso total de memoria RAM en MB.

        ram_max : float
            Máximo consumo de RAM registrado en MB.

        gpu_suma : float
            Acumulador del uso total de GPU.

        gpu_max : float
            Máximo uso de GPU registrado.

        vram_suma : float
            Acumulador del uso total de VRAM en MB.

        vram_max : float
            Máximo consumo de VRAM registrado en MB.

        Returns
        -------
        tuple[int, float, float, float, float, float | None, float | None, float | None, float | None]
            Nuevos acumuladores actualizados en el mismo orden recibido. Los acumuladores de GPU y
            VRAM serán None si no se utiliza GPU.
        """

        uso_cpu = process.cpu_percent(None)
        uso_ram = process.memory_info().rss / (1024 * 1024)

        cpu_suma += uso_cpu
        cpu_max = max(cpu_max, uso_cpu)
        muestras += 1

        ram_suma += uso_ram
        ram_max = max(ram_max, uso_ram)

        if red_usa_gpu:
            uso_gpu = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            uso_vram = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 * 1024)

            gpu_suma += uso_gpu
            gpu_max = max(gpu_max, uso_gpu)

            vram_suma += uso_vram
            vram_max = max(vram_max, uso_vram)

            if barra is not None:
                barra.set_postfix(CPU=f"{uso_cpu:.0f}%", RAM=f"{uso_ram:.0f}MB",
                                  GPU=f"{uso_gpu:.0f}%", VRAM=f"{uso_vram:.0f}MB")
        elif barra is not None:
            barra.set_postfix(CPU=f"{uso_cpu:.0f}%", RAM=f"{uso_ram:.0f}MB")

        return (muestras, cpu_suma, cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma, vram_max)

    def _convertir_I(self, I: float | list[float] | list[list[float]] | Array) -> float | np.ndarray:
        """
        Convierte la corriente pasada a un vector de numpy de longitud num_neuronas.

        Parameters
        ----------
        I : float | list[float] | list[list[float]] | Array
            Corriente de entrada a convertir. Puede ser:
            - Un escalar.
            - Un vector de tamaño num_neuronas con corriente fija por neurona.
            - Una matriz (pasos, num_neuronas) con corriente variable temporalmente.
            - En caso de una única neurona, un vector (pasos,) se interpreta como corriente temporal.

        Returns
        -------
        float | np.ndarray
            Corriente de entrada convertida a un escalar o un array de numpy.

        Raises
        ------
        TypeError
            Si la corriente pasada no es un número real, un vector o una matriz.

        ValueError
            - Si la corriente pasada está vacía.
            - Si la corriente pasada no tiene tantos elementos como neuronas haya cargadas.
            - Si la corriente es una matriz de más de dos dimensiones.
        """
        if isinstance(I, Real):
            return float(I)
        elif isinstance(I, cp.ndarray):
            I = I.get()
        elif isinstance(I, (list, np.ndarray)):
            I = np.asarray(I)
        else:
            raise TypeError("La corriente I debe ser un número real, un vector o una matriz.")

        if I.size == 0:
            raise ValueError("La corriente I no puede estar vacía.")

        if I.ndim == 1 or (I.ndim == 2 and (I.shape[0] == 1 or I.shape[1] == 1)):
            # Si vector en matriz de nx1 o 1xn
            if I.ndim == 2:
                if I.shape[0] == 1:
                    I = I[0, :]
                elif I.shape[1] == 1:
                    I = I[:, 0]

            # Caso de una única neurona:
            # vector temporal
            if self.__num_neuronas == 1:
                I = I.reshape(-1, 1)
                return I

            # Vector corriente fija por neurona
            if I.shape[0] != self.__num_neuronas:
                raise ValueError("El vector de corriente debe tener una longitud igual al número de neuronas." )

            return I
        elif I.ndim == 2:
            if I.shape[1] != self.__num_neuronas:
                raise ValueError("La matriz de corriente debe tener una columna por neurona.")

            return I
        else:
            raise ValueError("La corriente debe ser un escalar, vector o matriz bidimensional.")

    def _obtener_historial_completo(self) -> dict[str, np.ndarray | list[str] | list[bool] | float] | None:
        """
        Obtener el historial completo, sin separación entre datos y metadatos.

        Returns
        -------
        dict[str, np.ndarray | list[str] | list[bool] | float] | None
            Diccionario con los historiales de spikes, v, u e I, y las listas de nombres de las neuronas
            cargadas y si son excitatorias o inhibitorias, y el paso temporal, dt, de la simulación.
        """
        if self.__historial_spikes is None or self.__historial_v is None or self.__historial_u is None:
            return None

        # Si es una sola neurona (columna única), aplanamos a 1D (Pasos,) para mayor comodidad
        spikes = self.__historial_spikes[:, 0] if self.__num_neuronas == 1 else self.__historial_spikes
        v = self.__historial_v[:, 0] if self.__num_neuronas == 1 else self.__historial_v
        u = self.__historial_u[:, 0] if self.__num_neuronas == 1 else self.__historial_u
        I = self.__historial_I[:, 0] if self.__num_neuronas == 1 else self.__historial_I
        
        return {
            "spikes": spikes.copy(),
            "v": v.copy(),
            "u": u.copy(),
            "I": I.copy(),
            # Si la red es una neurona, convertir nombre y es_excitatoria a listas para mantener el
            # formato de los datos consistente
            "nombre": [self.__red.nombre] if isinstance(self.__red, Neurona) else self.__red.nombre,
            "es_excitatoria": [self.__red.es_excitatoria] if isinstance(self.__red, Neurona) else self.__red.es_excitatoria,
            "dt": self.__dt
        }


    # --------------------------------------------------
    # PROPIEDADES
    # --------------------------------------------------

    @property
    def dt(self) -> float:
        """
        El paso temporal usado en el simulador.
        """
        return self.__dt
    
    @property
    def red(self) -> Neurona | RedDeNeuronas | None:
        """
        Neurona o red de neuronas cargada en el simulador, devuelve None si no hay nada cargado.
        """
        return self.__red

    @property
    def paso_actual(self) -> int:
        """
        Paso actual de la simulación, desde el último reinicio.
        """
        return self.__paso_actual

    @property
    def num_neuronas(self) -> int:
        """
        Número de neuronas cargadas en el simulador.
        Vale 1 en el caso de una sola neurona, y el valor de num_neuronas de la red de neuronas en
        otro caso.
        """
        return self.__num_neuronas

    @property
    def historial(self) -> dict[str, np.ndarray | list[str] | list[bool] | float] | None:
        """
        Devuelve una copia de los historiales almacenados, junto con información adicional para poder
        interpretar los datos, como el nombre de las neuronas cargadas, si son excitatorias o inhibitorias
        y el paso temporal de la simulación.

        Returns
        -------
        dict[str, np.ndarray | list[str] | list[bool] | float] | None
            Diccionario con los historiales almacenados en memoria, o None si no se ha simulado
            nada y no están inicializados.
        """
        historial = self._obtener_historial_completo()
        if historial is None:
            return None
        else:
            return {
                "spikes": historial["spikes"],
                "v": historial["v"],
                "u": historial["u"],
                "I": historial["I"],
                "nombre": historial["nombre"],
                "es_excitatoria": historial["es_excitatoria"],
                "dt": historial["dt"]
            }
    
    @property
    def rendimiento(self) -> dict[str, float | None]:
        """
        Todos los valores de rendimiento actuales, si no se ha recogido alguno, aparecerá como None.

        Estos valores son:
        - tiempo (s)
        - cpu_media (%)
        - cpu_maxima (%)
        - ram_media (MB)
        - ram_maxima (MB)
        - gpu_media (%)
        - gpu_maxima (%)
        - vram_media (MB)
        - vram_maxima (MB)

        Representan el tiempo de ejecución, y los valores de uso de recursos como cpu, memoria ram,
        etc. medios y máximos de la última simulación.

        Returns
        -------
        dict[str, float | None]
            Diccionario con los valores de rendimiento recogidos en la última simulación.
        """
        return {
            "tiempo_ejecucion": self.__tiempo_ejecucion,
            "cpu_media": self.__cpu_media,
            "cpu_maxima": self.__cpu_maxima,
            "ram_media": self.__ram_media,
            "ram_maxima": self.__ram_maxima,
            "gpu_media": self.__gpu_media,
            "gpu_maxima": self.__gpu_maxima,
            "vram_media": self.__vram_media,
            "vram_maxima": self.__vram_maxima
        }

    @property
    def configuracion(self) -> dict[str, bool | str | int]:
        """
        Configuración actual de los valores por defecto de parámetros de la simulación.

        Returns
        -------
        dict[str, bool | str | int]
            Diccionario con los valores de la configuración por defecto.
        """
        return {
            "guardar_resultados": self.__guardar_resultados,
            "path_guardado": self.__path_guardado,
            "mostrar_progreso": self.__mostrar_progreso,
            "medir_rendimiento": self.__medir_rendimiento,
            "intervalo_rendimiento": self.__intervalo_rendimiento,
            "tamano_batch": self.__tamano_batch
        }
