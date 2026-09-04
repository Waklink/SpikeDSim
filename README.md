# SpikeDSim

SpikeDSim es una librería desarrollada en Python para la simulación de redes
neuronales de disparo (Spiking Neural Networks, SNN). El proyecto proporciona
herramientas para definir neuronas y redes neuronales, ejecutar simulaciones y
visualizar los resultados obtenidos.

La librería está basada en el modelo neuronal de Izhikevich (2003) y permite
simular tanto neuronas individuales como redes de neuronas. Además, incluye
funcionalidades para almacenar los resultados de las simulaciones, visualizar
su actividad y medir el rendimiento de las ejecuciones.

El proyecto ha sido desarrollado como parte de un Trabajo Fin de Grado.

## Características

- Simulación de neuronas mediante el modelo de Izhikevich (2003).
- Simulación de redes neuronales de disparo.
- Configuración de los parámetros de las neuronas y de las conexiones de la red.
- Uso de CPU mediante NumPy y de GPU mediante CuPy.
- Medición de diferentes métricas de rendimiento durante la simulación.
- Almacenamiento del historial de las simulaciones.
- Visualización de la actividad neuronal y de los resultados de las simulaciones.

## Instalación

### Instalación básica

Para instalar este proyecto, se puede ejecutar:

```cmd
pip install .
```

desde la raíz del mismo. También se puede usar -e si se está trabajando directamente
sobre el código de este proyecto:

```cmd
pip install -e .
```

En este último caso se recomienda trabajar en un entorno virtual de Python. Este
se puede crear mediante:

```cmd
python -m venv .venv
```

Una vez creado, se debe activar utilizando el comando correspondiente al sistema
operativo del equipo:

- Windows:

```cmd
.venv\Scripts\activate
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

### Extras

Para poder utilizar los cálculos con la GPU, es necesario tener CuPy correctamente
instalado y funcionando previamente. Las instrucciones de instalación se pueden
consultar en la [documentación oficial de CuPy](https://docs.cupy.dev/en/stable/install.html)

Para realizar los tests se necesita el paquete pytest :

```cmd
pip install pytest
```

Pudiendo después realizarlos con:

```cmd
pytest
```

Para poder ejecutar los cuadernos, se necesita el paquete ipykernel:

```cmd
pip install ipykernel
```

## Ejemplos de uso

### Simulación de una neurona

Se puede crear una neurona y simular directamente su actividaddurante un número
determinado de pasos temporales:

```python
from spikedsim.Izhikevich import Neurona, Simulador

neurona = Neurona(a=0.02, b=0.2, c=-65, d=8, nombre="Neurona")

simulador = Simulador(dt=1)
simulador.cargar_red(neurona)

simulador.simular(pasos=1000, I=10)
```

### Simulación de una red neuronal

También se pueden crear redes neuronales a partir de diferentes tipos de neuronas
y simular su actividad:

```python
from spikedsim import Neurona, RedDeNeuronas, Simulador

neurona_exc = Neurona(a=0.02, b=0.2, c=-65, d=8, nombre="Neurona excitatoria", es_excitatoria=True)
neurona_inh = Neurona(a=0.02, b=0.25, c=-65, d=2, nombre="Neurona inhibitoria", es_excitatoria=False)

red = RedDeNeuronas(neuronas={neurona_exc: 8, neurona_inh:2}, conexiones=90, semilla=42)

simulador = Simulador()
simulador.cargar_red(red)

simulador.simular(pasos=1000, I=10)
```

## Backend

SpikeDSim permite realizar las simulaciones utilizando dos tipos de backend,
que se seleccionan mediante el parámetro `backend` del simulador:

- NumPy ("numpy"): utiliza NumPy para realizar los cálculos en la CPU.
- CuPy ("cupy"): utiliza CuPy para realizar los cálculos en la GPU.

Además, el parámetro `sparse` permite seleccionar la representación utilizada para
la matriz de conexiones. Cuando se activa, esta se almacena como una matriz dispersa
por filas CSR, utilizando `scipy.sparse.csr_matrix` con el backend NumPy o
`cupyx.scipy.sparse.csr_matrix` con el backend CuPy.
