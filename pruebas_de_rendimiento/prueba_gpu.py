import numpy as np
import psutil
import os

from neurosim.Izhikevich import Neurona, RedDeNeuronas
from neurosim.backend import cp, CUPY_DISPONIBLE

if not CUPY_DISPONIBLE:
    raise ImportError("cupy no instalado.")


process = psutil.Process(os.getpid())

PASOS = 10

N_EXC = Neurona(a=0.02, b=0.2, c=-65, d=8, nombre="Neurona_excitatoria", es_excitatoria=True)
N_INH = Neurona(a=0.02, b=0.25, c=-65, d=2, nombre="Neurona_inhibitoria", es_excitatoria=False)
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
ALEAT_PARAM = {"excitatoria": (0, 0, 15, -6), "inhibitoria": (0.08, -0.05, 0, 0)}
ALEAT_CONEX = (0.5, 1)

corriente_exc = 5 * rng.standard_normal((PASOS, 8000))
corriente_inh = 2 * rng.standard_normal((PASOS, 2000))
corriente = np.concatenate((corriente_exc, corriente_inh), axis=1)

def ram():
    return process.memory_info().rss / 1024**2

pool = cp.get_default_memory_pool()

print(f"Memoria CPU inicial: {ram():.6f} MB")
print(f"Memoria GPU inicial: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

def medir(texto):
    print()
    print("-" * 70)
    print(texto)
    print("-" * 70)
    print(f"\nRAM inicial: {ram():.6f} MB")
    print(f"GPU inicial: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB\n")

    for i in range(PASOS):
        spikes = red._actualizar(cp.asarray(corriente[i, :], dtype=red.dtype), 1)
        print(f"RSS: {ram():.6f} MB | CuPy used: {pool.used_bytes() / 1024**2:.6f} MB | CuPy total: {pool.total_bytes() / 1024**2:.6f} MB | Disparo={np.count_nonzero(spikes)}")
        
    print(f"\nRAM final: {ram():.6f} MB")
    print(f"GPU final: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

    pool.free_all_blocks()

    print(f"\nRAM + free_all_blocks: {ram():.6f} MB")
    print(f"GPU + free_all_blocks: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, 99990000, "cupy", 64, True, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
medir("Prueba cupy con sparse=True")
del red

pool.free_all_blocks()

red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, 99990000, "cupy", 64, False, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
medir("Prueba cupy con sparse=False")
