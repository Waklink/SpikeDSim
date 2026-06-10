import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Modelo neuronal de Izhikevich
# Adaptación del código original MATLAB -> Python
# ============================================================

# Número de neuronas
Ne = 800   # excitadoras
Ni = 200   # inhibitorias

# ------------------------------------------------------------
# Parámetros aleatorios
# ------------------------------------------------------------

re = np.random.rand(Ne)
ri = np.random.rand(Ni)

# ------------------------------------------------------------
# Parámetros del modelo
# ------------------------------------------------------------

a = np.concatenate([
    0.02 * np.ones(Ne),
    0.02 + 0.08 * ri
])

b = np.concatenate([
    0.2 * np.ones(Ne),
    0.25 - 0.05 * ri
])

c = np.concatenate([
    -65 + 15 * re**2,
    -65 * np.ones(Ni)
])

d = np.concatenate([
    8 - 6 * re**2,
    2 * np.ones(Ni)
])

# ------------------------------------------------------------
# Matriz de conexiones sinápticas
# ------------------------------------------------------------

S = np.hstack([
    0.5 * np.random.rand(Ne + Ni, Ne),   # excitadoras
    -np.random.rand(Ne + Ni, Ni)         # inhibitorias
])

# ------------------------------------------------------------
# Variables de estado
# ------------------------------------------------------------

v = -65 * np.ones(Ne + Ni)   # potencial de membrana
u = b * v                    # variable de recuperación

# Lista de disparos:
# cada elemento será (tiempo, neurona)
firings = []

# ============================================================
# Simulación
# ============================================================

for t in range(1000):   # 1000 ms

    # --------------------------------------------------------
    # Entrada externa (ruido talámico)
    # --------------------------------------------------------

    I = np.concatenate([
        5 * np.random.randn(Ne),
        2 * np.random.randn(Ni)
    ])

    # --------------------------------------------------------
    # Detectar neuronas que disparan
    # --------------------------------------------------------

    fired = np.where(v >= 30)[0]

    # Guardar spikes
    for neuron in fired:
        firings.append((t, neuron))

    # --------------------------------------------------------
    # Reinicio tras spike
    # --------------------------------------------------------

    v[fired] = c[fired]
    u[fired] += d[fired]

    # --------------------------------------------------------
    # Corriente sináptica
    # --------------------------------------------------------

    I += np.sum(S[:, fired], axis=1)

    # --------------------------------------------------------
    # Actualización de v
    # Método de Euler con paso 0.5 ms
    # --------------------------------------------------------

    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)

    # --------------------------------------------------------
    # Actualización de u
    # --------------------------------------------------------

    u += a * (b * v - u)

# ============================================================
# Visualización
# ============================================================

firings = np.array(firings)

plt.figure(figsize=(12, 6))
plt.scatter(firings[:, 0], firings[:, 1], s=2)

plt.title("Raster Plot - Modelo de Izhikevich")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Neurona")

plt.show()
