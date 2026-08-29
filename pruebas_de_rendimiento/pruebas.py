import time
import csv

import numpy as np

from pathlib import Path
from multiprocessing import Process, Queue

from src.Neurona import Neurona
from src.RedDeNeuronas import RedDeNeuronas
from src.Simulador import Simulador

# Comprobar si cupy está disponible, para evitar problemas más adelante
import cupy as cp
cp.zeros(1)
cp = None


# Número de repeticiones por prueba
REP = 10

# Neuronas a usar para las pruebas
N_EXC = Neurona(a=0.02, b=0.2, c=-65, d=8, nombre="Neurona_excitatoria", es_excitatoria=True)
N_INH = Neurona(a=0.02, b=0.25, c=-65, d=2, nombre="Neurona_inhibitoria", es_excitatoria=False)

# Semilla
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
# Aleatorización
ALEAT_PARAM = {"excitatoria": (0, 0, 15, -6), "inhibitoria": (0.08, -0.05, 0, 0)}
ALEAT_CONEX = (0.5, 1)

CONFIGURACIONES_PRUEBAS = [
    {"ids": (0,   1),   "parametros_red": {"neuronas": N_EXC,                      "conexiones": None},     "num_neuronas": 1,     "densidad_conexiones": None},
    {"ids": (2,   17),  "parametros_red": {"neuronas": {N_EXC: 1},                 "conexiones": 0},        "num_neuronas": 1,     "densidad_conexiones": 0},
    {"ids": (18,  33),  "parametros_red": {"neuronas": {N_EXC: 8, N_INH: 2},       "conexiones": 0},        "num_neuronas": 10,    "densidad_conexiones": 0},
    {"ids": (34,  49),  "parametros_red": {"neuronas": {N_EXC: 8, N_INH: 2},       "conexiones": 45},       "num_neuronas": 10,    "densidad_conexiones": 0.5},
    {"ids": (50,  65),  "parametros_red": {"neuronas": {N_EXC: 8, N_INH: 2},       "conexiones": 90},       "num_neuronas": 10,    "densidad_conexiones": 1},
    {"ids": (66,  81),  "parametros_red": {"neuronas": {N_EXC: 80, N_INH: 20},     "conexiones": 0},        "num_neuronas": 100,   "densidad_conexiones": 0},
    {"ids": (82,  97),  "parametros_red": {"neuronas": {N_EXC: 80, N_INH: 20},     "conexiones": 990},      "num_neuronas": 100,   "densidad_conexiones": 0.1},
    {"ids": (98,  113), "parametros_red": {"neuronas": {N_EXC: 80, N_INH: 20},     "conexiones": 4950},     "num_neuronas": 100,   "densidad_conexiones": 0.5},
    {"ids": (114, 129), "parametros_red": {"neuronas": {N_EXC: 80, N_INH: 20},     "conexiones": 9900},     "num_neuronas": 100,   "densidad_conexiones": 1},
    {"ids": (130, 145), "parametros_red": {"neuronas": {N_EXC: 800, N_INH: 200},   "conexiones": 0},        "num_neuronas": 1000,  "densidad_conexiones": 0},
    {"ids": (146, 161), "parametros_red": {"neuronas": {N_EXC: 800, N_INH: 200},   "conexiones": 99900},    "num_neuronas": 1000,  "densidad_conexiones": 0.1},
    {"ids": (162, 177), "parametros_red": {"neuronas": {N_EXC: 800, N_INH: 200},   "conexiones": 249750},   "num_neuronas": 1000,  "densidad_conexiones": 0.25},
    {"ids": (178, 193), "parametros_red": {"neuronas": {N_EXC: 800, N_INH: 200},   "conexiones": 499500},   "num_neuronas": 1000,  "densidad_conexiones": 0.5},
    {"ids": (194, 209), "parametros_red": {"neuronas": {N_EXC: 800, N_INH: 200},   "conexiones": 999000},   "num_neuronas": 1000,  "densidad_conexiones": 1},
    {"ids": (210, 225), "parametros_red": {"neuronas": {N_EXC: 8000, N_INH: 2000}, "conexiones": 0},        "num_neuronas": 10000, "densidad_conexiones": 0},
    {"ids": (226, 241), "parametros_red": {"neuronas": {N_EXC: 8000, N_INH: 2000}, "conexiones": 9999000},  "num_neuronas": 10000, "densidad_conexiones": 0.1},
    {"ids": (242, 257), "parametros_red": {"neuronas": {N_EXC: 8000, N_INH: 2000}, "conexiones": 49995000}, "num_neuronas": 10000, "densidad_conexiones": 0.5},
    {"ids": (258, 273), "parametros_red": {"neuronas": {N_EXC: 8000, N_INH: 2000}, "conexiones": 99990000}, "num_neuronas": 10000, "densidad_conexiones": 1}
]

def ejecutar_neurona(queue, neurona, corriente, param_sim):
    simulador = Simulador(1)
    simulador.cargar_neurona(neurona)

    _simular_y_obtener_resultados(simulador, queue, corriente, param_sim)

def ejecutar_red(queue, parametros_red, corriente, param_sim):
    red = RedDeNeuronas(**parametros_red)
    simulador = Simulador(1)
    simulador.cargar_red(red)

    _simular_y_obtener_resultados(simulador, queue, corriente, param_sim)

def _simular_y_obtener_resultados(simulador, queue, corriente, param_sim):
    tiempo = simulador.simular(pasos=1000, I=corriente, mostrar_progreso=param_sim,
                               medir_rendimiento=param_sim)

    # rendimiento = simulador.rendimiento if param_sim else None

    queue.put((tiempo, simulador.rendimiento))


def lanzar_simulacion(ejecutar, parametros, corriente, param_sim):
    queue = Queue()

    proceso = Process(target=ejecutar, args=(queue, parametros, corriente, param_sim))

    proceso.start()
    proceso.join()

    if proceso.exitcode != 0:
        raise RuntimeError(f"La prueba terminó con un error. Código de salida: {proceso.exitcode}")

    resultado = queue.get()
    queue.close()
    queue.join_thread()

    return resultado


def guardar_resultado(resultado, path):
    path = Path(path)

    existe = path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=resultado.keys(), delimiter=";")

        if not existe:
            escritor.writeheader()

        escritor.writerow(resultado)


def ejecutar_pruebas(start = 0, stop = None, path = "./pruebas_de_rendimiento/resultados.csv"):
    if stop is None:
        stop = len(CONFIGURACIONES_PRUEBAS)
    elif not isinstance(stop, int):
        raise TypeError("stop debe ser un entero o None.")
    elif stop < 1 or stop > len(CONFIGURACIONES_PRUEBAS):
        raise ValueError(f"stop debe estar entre 1 y {len(CONFIGURACIONES_PRUEBAS)}.")

    if not isinstance(start, int):
        raise TypeError("start debe ser un entero.")
    elif start < 0 or start >= len(CONFIGURACIONES_PRUEBAS):
        raise ValueError(f"start debe estar entre 0 y {len(CONFIGURACIONES_PRUEBAS) - 1}.")

    if start >= stop:
        raise ValueError("start debe ser menor que stop.")

    # Tiempo para realizar las pruebas
    tiempo_inicio = time.perf_counter()

    # Número de configuraciones que se van a ejecutar
    num_configuraciones = stop - start

    # Cada configuración de red tiene 16 pruebas.
    # La configuración 0 (neurona) tiene 2 pruebas.
    if start == 0:
        total_pruebas = (2 + (num_configuraciones - 1) * 16) * REP
    else:
        total_pruebas = num_configuraciones * 16 * REP

    inicio_pruebas = CONFIGURACIONES_PRUEBAS[start]["ids"][0] * REP
    fin_pruebas = (CONFIGURACIONES_PRUEBAS[stop - 1]["ids"][1] + 1) * REP - 1

    if num_configuraciones == 1:
        texto_configuraciones = f"Configuración: {start}"
    else:
        texto_configuraciones = f"Configuraciones: {start} a {stop - 1}"

    pruebas_restantes = (2 + (len(CONFIGURACIONES_PRUEBAS) - 1) * 16) * REP - (fin_pruebas + 1)
    pruebas = inicio_pruebas + total_pruebas - 1

    prueba_actual = inicio_pruebas - 1

    print(f"Iniciando pruebas: {inicio_pruebas} a {fin_pruebas} | {texto_configuraciones} | Total: {total_pruebas} pruebas | Restantes: {pruebas_restantes}")
    print(f"Resultados: {path}\n")


    # --------------------------------------------------
    # PRIMERA CONFIGURACIÓN: NEURONA
    # --------------------------------------------------

    if start == 0:
        start += 1
        ids = CONFIGURACIONES_PRUEBAS[0]["ids"]
        neurona = CONFIGURACIONES_PRUEBAS[0]["parametros_red"]["neuronas"]

        # Corriente
        corriente = 5 * rng.standard_normal(1000)

        for id_prueba in ids:
            param_sim = bool(id_prueba)

            # REP repeticiones
            for repeticion in range(1, REP + 1):
                prueba_actual += 1

                print(f"[{prueba_actual}/{pruebas}] ID={id_prueba} | rep={repeticion} | Neurona"
                      f" | progreso={param_sim} | rendimiento={param_sim}", flush=True)

                tiempo, rendimiento = lanzar_simulacion(ejecutar_neurona, neurona, corriente, param_sim)

                resultado = {"id": id_prueba,
                             "repeticion": repeticion,
                             "neuronas": 1,
                             "conexiones": None,
                             "densidad_conexiones": None,
                             "sparse": None,
                             "precision": None,
                             "backend": None,
                             "mostrar_progreso": param_sim,
                             "medir_rendimiento": param_sim,
                             **rendimiento}

                guardar_resultado(resultado, path)

                print(f"\tCompletada: {tiempo:.4f} s", flush=True)


    # --------------------------------------------------
    # CONFIGURACIONES DE RED
    # --------------------------------------------------

    for configuracion in CONFIGURACIONES_PRUEBAS[start:stop]:
        ids = configuracion["ids"]
        parametros_base = configuracion["parametros_red"]
        densidad = configuracion["densidad_conexiones"]
        num_neuronas = configuracion["num_neuronas"]

        # Corriente
        corriente_exc = 5 * rng.standard_normal((1000, parametros_base["neuronas"][N_EXC]))
        if num_neuronas == 1:
            corriente = corriente_exc
        else:
            corriente_inh = 2 * rng.standard_normal((1000, parametros_base["neuronas"][N_INH]))
            corriente = np.concatenate((corriente_exc, corriente_inh), axis=1)

        id_prueba = ids[0]

        for sparse in (False, True):
            for precision in (32, 64):
                for backend in ("numpy", "cupy"):
                    for param_sim in (False, True):
                        # REP repeticiones
                        for repeticion in range(1, REP + 1):
                            prueba_actual += 1

                            print(f"[{prueba_actual}/{pruebas}] ID={id_prueba} | rep={repeticion}"
                                  f" | N={num_neuronas} | conex={parametros_base['conexiones']} | "
                                  f"dens={densidad} | sparse={sparse} | float{precision} | {backend}"
                                  f" | barra={param_sim} | rend={param_sim}", flush=True)

                            parametros_red = {**parametros_base,
                                            "sparse": sparse,
                                            "precision": precision,
                                            "backend": backend,
                                            "semilla": SEMILLA,
                                            "aleat_param": ALEAT_PARAM,
                                            "aleat_conex": ALEAT_CONEX}

                            tiempo, rendimiento = lanzar_simulacion(ejecutar_red, parametros_red, corriente,
                                                            param_sim)

                            resultado = {"id": id_prueba,
                                         "repeticion": repeticion,
                                         "neuronas": num_neuronas,
                                         "conexiones": parametros_base["conexiones"],
                                         "densidad_conexiones": densidad,
                                         "sparse": sparse,
                                         "precision": precision,
                                         "backend": backend,
                                         "mostrar_progreso": param_sim,
                                         "medir_rendimiento": param_sim,
                                         **rendimiento}

                            guardar_resultado(resultado, path)

                            print(f"\tCompletada: {tiempo:.4f} s", flush=True)

                        id_prueba += 1

    duracion = time.perf_counter() - tiempo_inicio

    print(f"\nPruebas completadas: {prueba_actual}/{pruebas}")
    print(f"Resultados guardados en: {path}")
    print(f"Duración de las pruebas: {duracion:.4f} s")


if __name__ == "__main__":
    ejecutar_pruebas()
