import numpy as np

from spikedsim.Izhikevich import Neurona, RedDeNeuronas, Simulador
from spikedsim import Visualizar

# Número de pasos a simular
PASOS = 1000

# Paso temporal de la simulación
PASO_TEMPORAL = 1

# Número de neuronas de la red
NUM_NEURONAS = 1000
# Proporción de neuronas excitatorias, el resto serán inhibitorias
PROPORCION_EXC = 0.8

# Densidad de conexiones entre las neuronas de la red
DENSIDAD_CONEXIONES = 1     # Cualquier valor en el intervalo [0, 1]

# Parámetros de la red
BACKEND = "numpy"    # "numpy" o "cupy"
PRECISION = 64      # 32 o 64
SPARSE = True       # True o False

# Neuronas para la red
N_EXC = Neurona(a=0.02, b=0.2, c=-65, d=8, nombre="Neurona_excitatoria", es_excitatoria=True)
N_INH = Neurona(a=0.02, b=0.25, c=-65, d=2, nombre="Neurona_inhibitoria", es_excitatoria=False)

# Valores que determiann la aleatorización de los parámetros de las neuronas de la red
SEMILLA = 42
ALEAT_PARAM = {"excitatoria": (0, 0, 15, -6), "inhibitoria": (0.08, -0.05, 0, 0)}
ALEAT_CONEX = (0.5, 1)

# Valores máximos de corrientes de entrada
MAX_CORRIENTE_EXC = 5
MAX_CORRIENTE_INH = 2

# Crear la red
num_excitatorias = int(NUM_NEURONAS * PROPORCION_EXC)
num_inhibitorias = NUM_NEURONAS - num_excitatorias

max_conexiones = NUM_NEURONAS * (NUM_NEURONAS - 1)
num_conexiones = int(max_conexiones * DENSIDAD_CONEXIONES)

red = RedDeNeuronas(neuronas={N_EXC: num_excitatorias, N_INH: num_inhibitorias},
                    conexiones=num_conexiones, backend=BACKEND, precision=PRECISION,
                    sparse=SPARSE, semilla=SEMILLA, aleat_param=ALEAT_PARAM,
                    aleat_conex=ALEAT_CONEX)

# Crear el simulador y el visualizador
sim = Simulador(PASO_TEMPORAL, mostrar_progreso=True,medir_rendimiento=True, tamano_batch=100)
sim.cargar_red(red)
vis = Visualizar()

# Generar la corriente
rng = np.random.RandomState(SEMILLA)
corriente_exc = MAX_CORRIENTE_EXC * rng.standard_normal((PASOS, num_excitatorias))
corriente_inh = MAX_CORRIENTE_INH * rng.standard_normal((PASOS, num_inhibitorias))
corriente = np.concatenate((corriente_exc, corriente_inh), axis=1)

# Realizar la simulación
tiempo = sim.simular(PASOS, corriente)
print(f"\nLa simulación teminó en {tiempo:.4f} s")

# Cargar el historial en el visualizador
vis.cargar_historial(historial=sim.historial)

# Mostrar el raster plot y la evolución del potencial de emebrana de una neurona
_ = vis.raster_plot()
# n = rng.randint(0, NUM_NEURONAS + 1)
n = 100
_ = vis.potencial_membrana(neuronas=n)
