import time
import numpy as np
import cupy as cp
from typing import Literal
from pathlib import Path
from numbers import Real

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
        Índice del último paso simulado.

    historico : dict[str, np.ndarray] | None
        Histórico de variables de estado (v, u).
        None si aún no se ha ejecutado ninguna simulación.
    
    Notes
    -----
    Internamente se almacenan buffers separados para v y u en memoria CPU para evitar saturación
    de VRAM en ejecuciones con backend GPU. Estos históricos siempre tienen precisión float32,
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
        self.__historico_v = None
        self.__historico_u = None

        # Paso actual de la simulación.
        self.__paso_actual = 0
        
        self.__num_neuronas = 0


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
        self.__paso_actual = 0

        self.__historico_v = None
        self.__historico_u = None


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
            Si ya hay otra neurona o una red de neurona cargada en el simulador.
        """
        if self.__red is not None:
            raise ValueError("Ya hay una neurona o red cargada. Limpia el simulador primero.")
        
        self.__red = neurona

        self.__num_neuronas = 1
        self.__paso_actual = 0

        self.__historico_v = None
        self.__historico_u = None


    def simular(self, pasos: int = 1000, I: float | Array = 0 , guardar_resultados: bool = False,
                path_guardado: str | None = None, verbose: bool = False, tamano_batch: int = 100) -> float:
        """
        Realizar un cierto número de pasos de la simulación.

        Parameters
        ----------
        pasos : int, optional
            Número de pasos a simular. Por defecto se simularán 1000 pasos.

        I : float | Array
            Corriente de entrada para las neuronas.

        guardar_resultados : bool
            Decisión de si guardar el histórico al terminar la simulación o no.

        path_guardado : str | None, optional
            Path al archivo donde guardar los resultados ene l caso de que guardar_resultados sea True.
            Si no se especifica, entonces se usará "./historico.npz", el valor por defecto de la función
            guardar_historico().

        verbose : bool
            Determinar si imprimir valores de rendimiento por pantalla.

        tamano_batch : int, optional
            Número de pasos temporales que se almacenan en GPU antes de transferirlos conjuntamente a la
            memoria RAM. Solo se aplica cuando el backend usado es CuPy.
            Por defecto es 100.
        
        Returns
        -------
        float
            Tiempo de ejecución de la simulación en segundos.
        
        Raises
        ------
        TypeError
            Si los pasos o el tamaño del batch de gpu no son enteros.

        ValueError
            Si no hya nada cargado, o si los pasos a simular son negativos.
        """

        if self.__red is None:
            raise ValueError("No hay ninguna red o neurona cargada para simular.")
        
        if not isinstance(pasos, int):
            raise TypeError("Los pasos deben ser un entero.")
        
        if pasos < 0:
            raise ValueError("Los pasos a simular deben ser un entero positivo o 0, para solo guardar \
                             el estado actual en el histórico.")
        
        if not isinstance(tamano_batch, int):
            raise TypeError("El tamaño del batch de gpu debe ser un entero.")
        
        if tamano_batch <= 0:
            raise ValueError("El tamaño del batch debe ser un entero positivo.")
        
        # 1. Crear referencias locales de los atributos usados, para evitar búsqueda de atributos.
        historico_v = self.__historico_v
        historico_u = self.__historico_u
        red = self.__red
        paso_actual = self.__paso_actual
        
        # 2. Reservar o expandir el histórico en la RAM de la CPU (NumPy)
        nuevo_tamano = paso_actual + pasos
        shape_historico = (nuevo_tamano, self.__num_neuronas)

        if historico_v is None:
            historico_v = np.empty(shape_historico, dtype=np.float32)
            historico_u = np.empty(shape_historico, dtype=np.float32)
        else:
            # Si se llama a simular() varias veces seguidas, la matriz de RAM se expande
            nuevo_v = np.empty(shape_historico, dtype=np.float32)
            nuevo_v[:paso_actual] = historico_v
            historico_v = nuevo_v
            nuevo_u = np.empty(shape_historico, dtype=np.float32)
            nuevo_u[:paso_actual] = historico_u
            historico_u = nuevo_u

        # 3. Guardar el estado inicial si estamos en el paso cero
        if paso_actual == 0:
            v_actual, u_actual = red._estado()
            historico_v[0] = np.asarray(v_actual)
            historico_u[0] = np.asarray(u_actual)
            paso_actual += 1
            
            # Si solo queríamos registrar el estado inicial (pasos=0), salimos
            if pasos == 0: return 0

        # 4. Determinar si la red interna está usando GPU (CuPy) o CPU (NumPy)
        red_usa_gpu = red.uso_gpu

        # 5. Sincronizar CUDA antes de empezar si se usa GPU para un benchmark preciso
        if red_usa_gpu:
            cp.cuda.Stream.null.synchronize()
        
        tiempo_inicio = time.perf_counter()

        # 6. Bucle Temporal de Simulación
        # Duplicación para evitar la comprobación del if en cada paso del bucle.
        if red_usa_gpu:
            buffer_v_gpu = cp.empty((tamano_batch, self.__num_neuronas), dtype=red.dtype)
            buffer_u_gpu = cp.empty((tamano_batch, self.__num_neuronas), dtype=red.dtype)

            indice_batch = 0

            inicio_batch = paso_actual

            for _ in range(pasos):
                # Avanzar un paso en la simulación.
                red.actualizar(I, self.__dt)
                
                # Estado actual en GPU
                v_actual, u_actual = red._estado()

                # Guardar en buffer de VRAM
                buffer_v_gpu[indice_batch] = v_actual
                buffer_u_gpu[indice_batch] = u_actual

                indice_batch += 1

                # Si buffer llenos, transferir a RAM
                if indice_batch == tamano_batch:
                    historico_v[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_v_gpu)
                    historico_u[inicio_batch:inicio_batch + tamano_batch] = cp.asnumpy(buffer_u_gpu)

                    inicio_batch += tamano_batch
                    indice_batch = 0
            
            # Si queda un último batch incompleto, copiarlo.
            if inicio_batch > 0:
                historico_v[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_v_gpu[:indice_batch])
                historico_u[inicio_batch:inicio_batch + indice_batch] = cp.asnumpy(buffer_u_gpu[:indice_batch])

            paso_actual += pasos

        else:
            for _ in range(pasos):
                # Avanzar un paso en la simulación.
                red.actualizar(I, self.__dt)
                
                # Obtener el nuevo estado
                v_actual, u_actual = red._estado()

                historico_v[paso_actual] = v_actual
                historico_u[paso_actual] = u_actual

                paso_actual += 1

        if red_usa_gpu:
            cp.cuda.Stream.null.synchronize()
        
        duracion = time.perf_counter() - tiempo_inicio

        self.__historico_v = historico_v
        self.__historico_u = historico_u
        self.__paso_actual = paso_actual

        # 7. Mostrar información de rendimiento por pantalla.
        if verbose:
            print(f"Simulación de {pasos} pasos completada en {duracion:.4f} segundos.")

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
        del tiempo.
        """
        if self.__red is not None:
            self.__red = None
            self.__num_neuronas = 0
            self.__paso_actual = 0
        
        self.__historico_v = None
        self.__historico_u = None
    

    def limpiar_historico(self) -> None:
        """
        Eliminar el histórico del estado de las neuronas a lo largo del tiempo, reinicando el paso
        actual a 0.
        """
        self.__historico_v = None
        self.__historico_u = None
        self.__paso_actual = 0
    

    def reiniciar(self) -> None:
        """
        Reiniciar el estado del simulador, incluido el estado de la posible neurona o red de neuronas
        cargada, eliminando el historico en el proceso.
        """
        if self.__red is not None:
            self.__red.reiniciar()
            self.__paso_actual = 0
        
        self.limpiar_historico()


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
        v_data = self.__historico_v[:, 0] if self.__num_neuronas == 1 else self.__historico_v
        u_data = self.__historico_u[:, 0] if self.__num_neuronas == 1 else self.__historico_u

        # 3. Exportar según el formato elegido
        if formato == "npz":
            np.savez_compressed(filepath, v=v_data, u=u_data)
            print(f"Histórico guardado exitosamente en: {filepath}")

        elif formato == "csv":
            # Guardar en dos archivos tabulares independientes (Filas: Pasos, Columnas: Neuronas)
            path_v = filepath.with_name(f"{filepath.stem}_v.csv")
            path_u = filepath.with_name(f"{filepath.stem}_u.csv")
            np.savetxt(path_v, v_data, delimiter=",")
            np.savetxt(path_u, u_data, delimiter=",")
            print(f"Histórico guardado en archivos CSV:\n - {path_v}\n - {path_u}")

        elif formato == "txt":
            # Texto plano separado por espacios
            path_v = filepath.with_name(f"{filepath.stem}_v.txt")
            path_u = filepath.with_name(f"{filepath.stem}_u.txt")
            np.savetxt(path_v, v_data)
            np.savetxt(path_u, u_data)
            print(f"Histórico guardado en archivos de texto:\n - {path_v}\n - {path_u}")
            
        elif formato == "json":
            import json
            estructura_json = {"v": v_data.tolist(), "u": u_data.tolist()}
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
        Devuelve una copia del histórico almacenado.
        """
        if self.__historico_v is None:
            return None
        else:
            return {
                "v": self.__historico_v.copy(),
                "u": self.__historico_u.copy()
            }
