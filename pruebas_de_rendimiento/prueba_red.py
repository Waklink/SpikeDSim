import os
import time

import numpy as np
import cupy as cp
import psutil

from neurosim.Izhikevich import Neurona, RedDeNeuronas


process = psutil.Process(os.getpid())

pool = cp.get_default_memory_pool()

def ram():
    return process.memory_info().rss / 1024**2

print(f"RAM inicial: {ram():.6f} MB")
print(f"GPU inicial: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

def medir(texto):
    print()
    print("-" * 70)
    print(texto)
    print("-" * 70)
    print(f"\nRAM: {ram():.6f} MB")
    print(f"GPU: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

    pool.free_all_blocks()

    print(f"\nRAM después de free: {ram():.6f} MB")
    print(f"GPU después de free: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

    print(f"\nDuración total: {duracion:.6f} s")

NUM_CONEXIONES = int(99990000 * 1)
PRECISION = 64

N_EXC = Neurona(a=0.02, b=0.2, c=-65, d=8, nombre="Neurona_excitatoria", es_excitatoria=True)
N_INH = Neurona(a=0.02, b=0.25, c=-65, d=2, nombre="Neurona_inhibitoria", es_excitatoria=False)
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)
ALEAT_PARAM = {"excitatoria": (0, 0, 15, -6), "inhibitoria": (0.08, -0.05, 0, 0)}
ALEAT_CONEX = (0.5, 1)

tiempo_inicio = time.perf_counter()
red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, NUM_CONEXIONES, "cupy", PRECISION, True, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
duracion = time.perf_counter() - tiempo_inicio
medir("Prueba con cupy y sparse=True")
del tiempo_inicio
del red
del duracion
pool.free_all_blocks()

print()
print("-" * 70)
print(f"\nRAM intermedia: {ram():.6f} MB")
print(f"GPU intermedia: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

tiempo_inicio = time.perf_counter()
red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, NUM_CONEXIONES, "cupy", PRECISION, False, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
duracion = time.perf_counter() - tiempo_inicio
medir("Prueba con cupy y sparse=False")
del tiempo_inicio
del red
del duracion
pool.free_all_blocks()

print()
print("-" * 70)
print(f"\nRAM intermedia: {ram():.6f} MB")
print(f"GPU intermedia: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

tiempo_inicio = time.perf_counter()
red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, NUM_CONEXIONES, "numpy", PRECISION, True, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
duracion = time.perf_counter() - tiempo_inicio
medir("Prueba con numpy y sparse=True")
del tiempo_inicio
del red
del duracion

print()
print("-" * 70)
print(f"\nRAM intermedia: {ram():.6f} MB")
print(f"GPU intermedia: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")

tiempo_inicio = time.perf_counter()
red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, NUM_CONEXIONES, "numpy", PRECISION, False, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
duracion = time.perf_counter() - tiempo_inicio
medir("Prueba con numpy y sparse=False")
del tiempo_inicio
del red
del duracion

print()
print("-" * 70)
print(f"\nRAM final: {ram():.6f} MB")
print(f"GPU final: {pool.used_bytes() / 1024**2:.6f} MB, {pool.total_bytes() / 1024**2:.6f} MB")