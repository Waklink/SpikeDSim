import time
import numpy as np
import cupy as cp

from typing import Any, Literal
from pathlib import Path
from numbers import Real
from tqdm import tqdm

import psutil
import pynvml

from .Neurona import Neurona
from .RedDeNeuronas import RedDeNeuronas, Array

class Simulador:
    """
    Simulador de una red de neuronas basadas en el modelo de Izhikevich.
    
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

    historico : dict[str, np.ndarray] | None
        Histórico de spikes y variables de estado (v y u).
        None si aún no se ha ejecutado ninguna simulación.
    
    rendimiento : dict[str, float | None]
        Valores de rendimiento de la última simulación ejecutada.
    
    Notes
    -----
    Internamente se almacenan buffers separados para spikes, v y u en memoria CPU para evitar saturación
    de VRAM en ejecuciones con backend GPU, almacenándose buffers temporales de menor tamaño para agrupar
    transferencias. Estos históricos siempre tienen precisión float32, en el caso de v y de u,
    independientemente del dtype interno utilizado por la red.
    """

    def __init__(self, paso_temporal: float = 0.5):
        """
        Inicializa un simulador con el paso temporal indicado.

        Parameters
        ----------
        paso_temporal : float
            Paso temporal de integración expresado en milisegundos.
        
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

        # Históricos en RAM (NumPy) para evitar saturar la VRAM de la GPU
        self.__historico_spikes = None
        self.__historico_v = None
        self.__historico_u = None

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


    def cargar_red(self, red: RedDeNeuronas) -> None:
        """
        Cargar una red de neuronas en el simulador.

        Parameters
        ----------
        red : RedDeNeuronas
            Red a cargar.
        
        Raises
        ------
        ValueError
            Si ya hay otra red o una neurona cargada en el simulador.
        """
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
        ValueError
            Si ya hay otra neurona o una red de neuronas cargada en el simulador.
        """
        if self.__red is not None:
            raise ValueError("Ya hay una neurona o red cargada. Limpia el simulador primero.")
        
        self.__red = neurona
        self.__num_neuronas = 1
    

    def __actualizar_rendimiento(self, barra: tqdm | None, process: psutil.Process,
                                 gpu_handle: Any | None, red_usa_gpu: bool,
                                 muestras: int, cpu_suma: float, cpu_max: float, ram_suma: float,
                                 ram_max: float, gpu_suma: float, gpu_max: float, vram_suma: float,
                                 vram_max: float) -> tuple[int, float, float, float, float, float, float, float, float]:
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
        tuple[int, float, float, float, float, float, float, float, float]
            Nuevos acumuladores actualizados en el mismo orden recibido.
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


    def simular(self, pasos: int = 1000, I: float | Array = 0 , guardar_resultados: bool = False,
                path_guardado: str | None = None, mostrar_progreso: bool = False, medir_rendimiento: bool = False,
                intervalo_rendimiento: int = 100, tamano_batch: int = 100) -> float:
        """
        Realizar un cierto número de pasos de simulación. Pudiendo decidir si guardar los históricos
        al finalizar, así como si mostrar el progreso de forma dinámica o si medir el rendimiento.
        Habrá un mayor rendimiento si no se muestra el progreso ni se mide el rendimiento.

        Parameters
        ----------
        pasos : int, optional
            Número de pasos a simular. En el caso de ser 0, si no hay nada guardado en el histórico,
            se guardará el estado actual como estado inicial, si ya hay estados guardados en el histórico,
            no se guardará nada. Por defecto se simularán 1000 pasos.

        I : float | Array
            Corriente de entrada para las neuronas.

        guardar_resultados : bool
            Decisión de si guardar el histórico al terminar la simulación o no.

        path_guardado : str | None, optional
            Path al archivo donde guardar los resultados en el caso de que guardar_resultados sea True.
            Si no se especifica, entonces se usará "./historico.npz", el valor por defecto de la función
            guardar_historico().

        mostrar_progreso : bool
            Determinar si mostrar el progreso de la simulación de forma dinámica como una barra de progreso.
            Esto reduce el rendimiento de la simulación. Por defecto es False.
        
        medir_rendimiento : bool
            Determinar si medir el rendimiento de la simulación actual, esto reduce el rendimiento de la
            simulación, por lo que no es completamente indicativo del rendimiento máximo real. Por defecto
            es False.
        
        intervalo_rendimiento : int
            Número de pasos intermedios entre recogidas de valores de rendimiento, solo se recogen datos
            cuando medir_rendimiento=True. Por defecto es 100.

        tamano_batch : int, optional
            Número de pasos temporales que se almacenan en GPU antes de transferirlos conjuntamente a la
            memoria RAM. Solo se aplica cuando el backend usado es CuPy.
            Por defecto es 100.
        
        Returns
        -------
        float
            Tiempo de ejecución de la simulación en segundos, sin incluir la preparación inicial ni
            el posible guardado de los históricos.
        
        Raises
        ------
        TypeError
            Si los pasos o el tamaño del batch de gpu no son enteros, o si mostrar_progreso o medir_rendimiento no son booleanos.

        ValueError
            Si no hay nada cargado, o si los pasos a simular son negativos.
        """

        # Comprobaciones iniciales de tipos y valores de los parámetros.
        if self.__red is None:
            raise ValueError("No hay ninguna red o neurona cargada para simular.")
        
        if not isinstance(pasos, int):
            raise TypeError("Los pasos deben ser un entero.")
        
        if pasos < 0:
            raise ValueError("Los pasos a simular deben ser un entero positivo.")
        
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
        
        # Reiniciar las métricas de rendimiento
        self.limpiar_rendimiento()
        
        # 1. Crear referencias locales de los atributos usados, para evitar búsqueda de atributos repetida en el bucle.
        historico_spikes = self.__historico_spikes
        historico_v = self.__historico_v
        historico_u = self.__historico_u
        red = self.__red
        paso_actual = self.__paso_actual
        
        # 2. Reservar o expandir el histórico en la RAM de la CPU (NumPy)
        nuevo_tamano = paso_actual + pasos

        # Si el tamaño es 0, cambiarlo a 1 para poder guardar el estado inicial en paso_actual = 0.
        if nuevo_tamano == 0:
            nuevo_tamano = 1
        
        shape_historico = (nuevo_tamano, self.__num_neuronas)

        if historico_v is None:
            historico_spikes = np.empty(shape_historico, dtype=bool)
            historico_v = np.empty(shape_historico, dtype=np.float32)
            historico_u = np.empty(shape_historico, dtype=np.float32)
        else:
            # Si se llama a simular() varias veces seguidas, la matriz de RAM se expande
            nuevo_spikes = np.empty(shape_historico, dtype=bool)
            nuevo_spikes[:paso_actual] = historico_spikes
            historico_spikes = nuevo_spikes

            nuevo_v = np.empty(shape_historico, dtype=np.float32)
            nuevo_v[:paso_actual] = historico_v
            historico_v = nuevo_v
            
            nuevo_u = np.empty(shape_historico, dtype=np.float32)
            nuevo_u[:paso_actual] = historico_u
            historico_u = nuevo_u

        # 3. Guardar el estado inicial si estamos en el paso cero
        if paso_actual == 0:
            v_actual, u_actual = red._estado()
            historico_spikes[0] = False
            historico_v[0] = np.asarray(v_actual)
            historico_u[0] = np.asarray(u_actual)
            paso_actual += 1
            
            # Si solo se quería registrar el estado inicial (pasos=0), se sale
            if pasos == 0:
                self.__historico_spikes = historico_spikes
                self.__historico_v = historico_v
                self.__historico_u = historico_u
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

                # Primera llamada necesaria para inicializar medición de CPU
                process.cpu_percent(None)

                if red_usa_gpu:
                    pynvml.nvmlInit()
                    nvml_iniciado = True

                    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    gpu_suma = 0
                    gpu_max = 0

                    vram_suma = 0
                    vram_max = 0

            # 5. Sincronizar CUDA antes de empezar si se usa GPU para un benchmark preciso
            if red_usa_gpu:
                cp.cuda.Stream.null.synchronize()
            
            tiempo_inicio = time.perf_counter()

            # 6. Bucle Temporal de Simulación
            # Duplicación para evitar la comprobación del if en cada paso del bucle.
            if red_usa_gpu:
                buffer_spikes_gpu = cp.empty((tamano_batch, self.__num_neuronas), dtype=bool)
                buffer_v_gpu = cp.empty((tamano_batch, self.__num_neuronas), dtype=red.dtype)
                buffer_u_gpu = cp.empty((tamano_batch, self.__num_neuronas), dtype=red.dtype)

                indice_batch = 0

                inicio_batch = paso_actual

                for _ in range(pasos):
                    # Avanzar un paso en la simulación.
                    spikes_actual = red.actualizar(I, self.__dt)
                    
                    # Estado actual en GPU
                    v_actual, u_actual = red._estado()

                    # Guardar en buffer de VRAM
                    buffer_spikes_gpu[indice_batch] = spikes_actual
                    buffer_v_gpu[indice_batch] = v_actual
                    buffer_u_gpu[indice_batch] = u_actual

                    indice_batch += 1

                    # Si buffer llenos, transferir a RAM
                    if indice_batch == tamano_batch:
                        historico_spikes[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_spikes_gpu)
                        historico_v[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_v_gpu)
                        historico_u[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_u_gpu)

                        inicio_batch += tamano_batch
                        indice_batch = 0
                    
                    if barra is not None:
                        barra.update()
                    
                    if medir_rendimiento and ((inicio_batch + indice_batch) % intervalo_rendimiento == 0):
                        (muestras, cpu_suma, cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma, vram_max) = \
                            self.__actualizar_rendimiento(barra, process, gpu_handle, red_usa_gpu, muestras, cpu_suma,
                                                        cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma,
                                                        vram_max)
                
                # Si queda un último batch incompleto, copiarlo.
                if inicio_batch > 0:
                    historico_spikes[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_spikes_gpu[:indice_batch])
                    historico_v[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_v_gpu[:indice_batch])
                    historico_u[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_u_gpu[:indice_batch])

                paso_actual += pasos

            else:
                for _ in range(pasos):
                    # Avanzar un paso en la simulación.
                    spike_actual = red.actualizar(I, self.__dt)
                    
                    # Obtener el nuevo estado
                    v_actual, u_actual = red._estado()

                    historico_spikes[paso_actual] = spike_actual
                    historico_v[paso_actual] = v_actual
                    historico_u[paso_actual] = u_actual

                    paso_actual += 1

                    if barra is not None:
                        barra.update()
                    
                    if medir_rendimiento and (paso_actual % intervalo_rendimiento == 0):
                        (muestras, cpu_suma, cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma, vram_max) = \
                            self.__actualizar_rendimiento(barra, process, gpu_handle, red_usa_gpu, muestras, cpu_suma,
                                                        cpu_max, ram_suma, ram_max, gpu_suma, gpu_max, vram_suma,
                                                        vram_max)

            if red_usa_gpu:
                cp.cuda.Stream.null.synchronize()
            
            duracion = time.perf_counter() - tiempo_inicio
            self.__tiempo_ejecucion = duracion

            self.__historico_spikes = historico_spikes
            self.__historico_v = historico_v
            self.__historico_u = historico_u
            self.__paso_actual = paso_actual

            # 7. Mostrar información de rendimiento por pantalla.
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

        # 8. Guardar el histórico
        if guardar_resultados:
            if path_guardado is None:
                self.guardar_historico()
            else:
                self.guardar_historico(path=path_guardado)
                
        return duracion
    

    def limpiar_todo(self) -> None:
        """
        Eliminar la red o neurona cargada, así como el histórico del estado de las neuronas a lo largo
        del tiempo y los valores de rendimiento almacenados.
        """
        if self.__red is not None:
            self.__red = None
            self.__num_neuronas = 0
        
        self.limpiar_historicos()
        self.limpiar_rendimiento()
    

    def limpiar_historicos(self) -> None:
        """
        Eliminar el histórico del estado de las neuronas a lo largo del tiempo, reinicando el paso
        actual a 0.
        """
        self.__historico_spikes = None
        self.__historico_v = None
        self.__historico_u = None
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
        cargada, eliminando el historico en el proceso y reiniciando los valores de rendimiento.
        """
        if self.__red is not None:
            self.__red.reiniciar()
        
        self.limpiar_historicos()
        self.limpiar_rendimiento()


    def guardar_historico(self, path: str = "./historico.npz",
                          formato: Literal["npz", "txt", "json", "csv"] | None = None) -> None:
        """
        Guardar el histórico en un archivo, pudiendo elegir el formato entre varios posibles.

        En el caso de que el formato especificado, tanto en el path como en el parámetro formato, no
        esté soportado, se usará el valor por defecto.

        Parameters
        ----------
        path : str
            La ruta, absoluta o relativa, al archivo en el que se va a guardar el histórico. Debe incluir
            el nombre del archivo y, en el caso de no proporcionar un formato, la extensión elegida para
            guardar los resultados. Por defecto es "./historico.npz"

            Si se proporciona un formato distinto de la extensión actual del path, la 
            extensión original se eliminará y se sustituirá por la correspondiente al formato elegido.
            Si el path no tiene extensión y no se especifica formato, se asumirá la extensión "npz".

        formato : Literal["npz", "txt", "json", "csv"] | None, optional
            El formato elegido para guardar los datos. Si es None, se inferirá de la extensión del path.
            Los formatos soportados son:
                - "npz" : (Recomendado) Formato binario nativo de NumPy/CuPy. Ideal para arrays de alta densidad.
                - "txt" : Texto en plano.
                - "json" : Formato de texto estructurado. Los arrays se convertirán automáticamente a listas estándar.
                - "csv" : Valores separados por comas. Estructura los datos de forma tabular (aplanada).
        """
        if self.__historico_v is None or self.__historico_u is None:
            print("No hay datos en el histórico para guardar.")
            return
        
        # 1. Procesar la extensión del archivo y formato
        formatos_soportados = ["npz", "txt", "json", "csv"]

        filepath = Path(path)
        if formato is None or formato not in formatos_soportados:
            formato = filepath.suffix.lower().replace(".", "") if filepath.suffix else "npz"
        
        if formato not in formatos_soportados:
            formato = "npz"
            
        filepath = filepath.with_suffix(f".{formato}")

        # Asegurar el directorio de salida
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 2. Si es una sola neurona (columna única), aplanamos a 1D (Pasos,) para mayor comodidad
        spikes_data = self.__historico_spikes[:, 0] if self.__num_neuronas == 1 else self.__historico_spikes
        v_data = self.__historico_v[:, 0] if self.__num_neuronas == 1 else self.__historico_v
        u_data = self.__historico_u[:, 0] if self.__num_neuronas == 1 else self.__historico_u

        # 3. Exportar según el formato elegido
        if formato == "npz":
            np.savez_compressed(filepath, spikes=spikes_data, v=v_data, u=u_data)
            print(f"Histórico guardado exitosamente en: {filepath}")

        elif formato == "csv":
            # Guardar en tres archivos tabulares independientes (Filas: Pasos, Columnas: Neuronas)
            path_spikes = filepath.with_name(f"{filepath.stem}_spikes.csv")
            path_v = filepath.with_name(f"{filepath.stem}_v.csv")
            path_u = filepath.with_name(f"{filepath.stem}_u.csv")
            np.savetxt(path_spikes, spikes_data, delimiter=",")
            np.savetxt(path_v, v_data, delimiter=",")
            np.savetxt(path_u, u_data, delimiter=",")
            print(f"Histórico guardado en archivos CSV:\n - {path_spikes}\n - {path_v}\n - {path_u}")

        elif formato == "txt":
            # Texto plano separado por espacios
            path_spikes = filepath.with_name(f"{filepath.stem}_spikes.txt")
            path_v = filepath.with_name(f"{filepath.stem}_v.txt")
            path_u = filepath.with_name(f"{filepath.stem}_u.txt")
            np.savetxt(path_spikes, spikes_data)
            np.savetxt(path_v, v_data)
            np.savetxt(path_u, u_data)
            print(f"Histórico guardado en archivos de texto:\n - {path_spikes}\n - {path_v}\n - {path_u}")
            
        elif formato == "json":
            import json
            estructura_json = {"spikes": spikes_data.tolist(), "v": v_data.tolist(), "u": u_data.tolist()}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(estructura_json, f)
            print(f"Histórico guardado exitosamente en: {filepath}")
    

    @property
    def dt(self) -> float:
        """
        El paso temporal usado en el simulador.
        """
        return self.__dt
    
    @property
    def red(self) -> Neurona | RedDeNeuronas | None:
        """
        Neurona o red de neuronas cargada en el simulador, devuelve NOne si no hay nada cargado.
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
        Vale 1 en el caso de una sola neuronas, y el valor de num_neuronas de la red de neuronas en
        otro caso.
        """
        return self.__num_neuronas

    @property
    def historico(self) -> dict[str, np.ndarray] | None:
        """
        Devuelve una copia de los históricos almacenados.
        """
        if self.__historico_spikes is None or self.__historico_v is None or self.__historico_u is None:
            return None
        else:
            return {
                "spikes": self.__historico_spikes.copy(),
                "v": self.__historico_v.copy(),
                "u": self.__historico_u.copy()
            }
    
    @property
    def rendimiento(self) -> dict[str, float | None]:
        """
        Todos los valores de rendimiento actuales, si no se ha recogido alguno, aparecerá como None.

        Returns
        -------
        dict[str, float | None]
            Diccionario con los valores de rendimiento recogidos en la última simulación.
        """
        return {
            "tiempo": self.__tiempo_ejecucion,
            "cpu_media": self.__cpu_media,
            "cpu_maxima": self.__cpu_maxima,
            "ram_media": self.__ram_media,
            "ram_maxima": self.__ram_maxima,
            "gpu_media": self.__gpu_media,
            "gpu_maxima": self.__gpu_maxima,
            "vram_media": self.__vram_media,
            "vram_maxima": self.__vram_maxima
        }
