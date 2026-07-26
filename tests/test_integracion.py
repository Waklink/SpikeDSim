import pytest
import numpy as np
import cupy as cp
from src.basico.Neurona import Neurona
from src.basico.RedDeNeuronas import RedDeNeuronas
from src.basico.Simulador import Simulador

# Comprobar si hay cupy disponible
try:
    cp.zeros(1)
    CUPY_DISPONIBLE = True
except Exception:
    CUPY_DISPONIBLE = False

# ==================================================
# FIXTURES
# ==================================================

@pytest.fixture
def neurona():
    return Neurona.predefinida("rs")

@pytest.fixture
def red():
    return RedDeNeuronas({"rs": 1})

@pytest.fixture
def simulador_neurona(neurona):
    sim = Simulador()
    sim.cargar_neurona(neurona)
    return sim

@pytest.fixture
def simulador_red(red):
    sim = Simulador()
    sim.cargar_red(red)
    return sim


# ==================================================
# TESTS DE INTEGRACIÓN SIMPLES
# ==================================================

def test_neurona_y_red_de_una_neurona_misma_evolucion():
    n = Neurona.predefinida("rs")
    # Mantener la precisión del float de python
    red = RedDeNeuronas({n.copy(): 1}, precision=64)

    sim_n = Simulador()
    sim_n.cargar_neurona(n)
    sim_red = Simulador()
    sim_red.cargar_red(red)

    pasos = 20

    sim_n.simular(pasos)
    sim_red.simular(pasos)

    hist_n = sim_n.historial
    hist_red = sim_red.historial

    assert np.array_equal(hist_n["spikes"], hist_red["spikes"])
    assert np.allclose(hist_n["v"], hist_red["v"])
    assert np.allclose(hist_n["u"], hist_red["u"])

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
def test_red_neurona_numpy_y_cupy_misma_evolucion():
    red_np = RedDeNeuronas({"rs": 1}, precision=64)
    red_cp = red_np.convertir_backend("cupy")

    sim_red_np = Simulador()
    sim_red_np.cargar_red(red_np)
    sim_red_cp = Simulador()
    sim_red_cp.cargar_red(red_cp)

    pasos = 20

    sim_red_np.simular(pasos)
    sim_red_cp.simular(pasos)

    hist_red_np = sim_red_np.historial
    hist_red_cp = sim_red_cp.historial

    assert np.array_equal(hist_red_np["spikes"], hist_red_cp["spikes"])
    assert np.allclose(hist_red_np["v"], hist_red_cp["v"])
    assert np.allclose(hist_red_np["u"], hist_red_cp["u"])
