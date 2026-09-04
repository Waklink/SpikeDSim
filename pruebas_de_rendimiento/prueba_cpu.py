import numpy as np
import psutil
import os

from spikedsim.Izhikevich import Neurona, RedDeNeuronas

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

print(f"Memoria inicial: {ram():.6f} MB")

def medir(texto):
    print()
    print("-" * 70)
    print(texto)
    print("-" * 70)
    print(f"\nInicial: {ram():.6f} MB\n")

    for i in range(PASOS):
        spikes = red._actualizar(corriente[i, :], 1)
        print(f"RSS: {ram():.6f} MB | Disparos={np.count_nonzero(spikes)}")

    print(f"\nDespués: {ram():.6f} MB")

red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, 99990000, "numpy", 64, True, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
medir("Prueba numpy con sparse=True")
del red

red = RedDeNeuronas({N_EXC: 8000, N_INH: 2000}, 99990000, "numpy", 64, False, SEMILLA, ALEAT_PARAM, ALEAT_CONEX)
medir("Prueba numpy con sparse=False")
