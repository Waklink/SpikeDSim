import pytest
import numpy as np

try:
    import cupy as cp
    try:
        cp.zeros(1)
        CUPY_DISPONIBLE = True
    except Exception:
        CUPY_DISPONIBLE = False
except ImportError:
    cp = None
    CUPY_DISPONIBLE = False

from pathlib import Path
import shutil
import json

from src.Neurona import Neurona
from src.RedDeNeuronas import RedDeNeuronas
from src.Simulador import Simulador

# Comprobar si hay cupy disponible
try:
    cp.zeros(1)
    CUPY_DISPONIBLE = True
except Exception:
    CUPY_DISPONIBLE = False

# Decidir si mantener los archivos temporales
MANTENER_ARCHIVOS_TEMPORALES = False

# Neurona para pruebas básicas
N = Neurona.predefinida("rs")

# Red para pruebas básicas
RED = RedDeNeuronas({N: 1})


# ==================================================
# FIXTURES
# ==================================================

# Simulador básico con una red de una neurona cargada
@pytest.fixture
def simulador():
    sim = Simulador()
    sim.cargar_red(RED.copy())
    return sim

# Simulador básico con una red de dos neuronas ya cargada
@pytest.fixture
def simulador2():
    sim = Simulador()
    sim.cargar_red(RedDeNeuronas({N: 2}))
    return sim

# Eliminar la carpeta temporal si no se quieren mantener los archivos temporales
@pytest.fixture(scope="session", autouse=True)
def limpiar_temporales():
    yield
    if not MANTENER_ARCHIVOS_TEMPORALES:
        carpeta = Path("./tests/tmp_sim")
        if carpeta.exists():
            shutil.rmtree(carpeta)


# ==================================================
# TESTS DEL CONSTRUCTOR
# ==================================================

def test_crear_simulador():
    sim = Simulador(0.1)
    assert sim.dt == 0.1
    assert sim.red is None
    assert sim.num_neuronas == 0
    assert sim.paso_actual == 0
    assert sim.historial is None
    for clave, valor in sim.rendimiento.items():
        if clave != "tiempo_ejecucion":
            assert valor is None
        else:
            assert valor == 0

@pytest.mark.parametrize("dt", ["0.5", [0.5], None])
def test_crear_simulador_dt_no_real(dt):
    with pytest.raises(TypeError):
        Simulador(dt)

@pytest.mark.parametrize("dt", [0, -0.5])
def test_crear_simulador_dt_invalido(dt):
    with pytest.raises(ValueError):
        Simulador(dt)

def test_crear_con_configuracion_por_defecto():
    sim = Simulador()
    config = sim.configuracion
    assert config["guardar_resultados"] is False
    assert config["path_guardado"] == "./historial.npz"
    assert config["mostrar_progreso"] is False
    assert config["medir_rendimiento"] is False
    assert config["intervalo_rendimiento"] == 100
    assert config["tamano_batch"] == 100

def test_crear_con_configuracion_personalizada():
    sim = Simulador(mostrar_progreso=True, medir_rendimiento=True, intervalo_rendimiento=1, tamano_batch=10)
    config = sim.configuracion
    assert config["guardar_resultados"] is False
    assert config["path_guardado"] == "./historial.npz"
    assert config["mostrar_progreso"] is True
    assert config["medir_rendimiento"] is True
    assert config["intervalo_rendimiento"] == 1
    assert config["tamano_batch"] == 10


# ==================================================
# TESTS DE CONFIGURAR SIMULACIÓN
# ==================================================

def test_configurar_simulacion_parcial():
    sim = Simulador()
    sim.configurar_simulacion(mostrar_progreso=True)
    config = sim.configuracion
    assert config["guardar_resultados"] is False
    assert config["path_guardado"] == "./historial.npz"
    assert config["mostrar_progreso"] is True
    assert config["medir_rendimiento"] is False
    assert config["intervalo_rendimiento"] == 100
    assert config["tamano_batch"] == 100

def test_configurar_simulacion_completa():
    sim = Simulador()
    sim.configurar_simulacion(guardar_resultados=True, path_guardado="prueba.npz", mostrar_progreso=True,
                              medir_rendimiento=True, intervalo_rendimiento=10, tamano_batch=50)
    assert sim.configuracion == {"guardar_resultados": True,
                                 "path_guardado": "prueba.npz",
                                 "mostrar_progreso": True,
                                 "medir_rendimiento": True,
                                 "intervalo_rendimiento": 10,
                                 "tamano_batch": 50}

@pytest.mark.parametrize("parametro", ["guardar_resultados", "mostrar_progreso", "medir_rendimiento"])
@pytest.mark.parametrize("valor", ["True", [True]])
def test_configurar_simulacion_booleanos_invalidos(parametro, valor):
    sim = Simulador()
    with pytest.raises(TypeError):
        sim.configurar_simulacion(**{parametro: valor})

@pytest.mark.parametrize("parametro", ["intervalo_rendimiento", "tamano_batch"])
@pytest.mark.parametrize("valor", [0.5, [3], "2"])
def test_configurar_simulacion_enteros_invalidos(parametro, valor):
    sim = Simulador()
    with pytest.raises(TypeError):
        sim.configurar_simulacion(**{parametro: valor})

@pytest.mark.parametrize("parametro", ["intervalo_rendimiento", "tamano_batch"])
@pytest.mark.parametrize("valor", [0, -1])
def test_configurar_simulacion_enteros_no_positivos(parametro, valor):
    sim = Simulador()
    with pytest.raises(ValueError):
        sim.configurar_simulacion(**{parametro: valor})


# ==================================================
# TESTS DE CARGA DE NEURONAS Y REDES
# ==================================================

def test_cargar_red():
    red = RED.copy()
    sim = Simulador()
    sim.cargar_red(red)
    assert sim.red is red
    assert sim.num_neuronas == 1

def test_cargar_red_con_red_cargada(simulador):
    red = RED.copy()
    with pytest.raises(ValueError):
        simulador.cargar_red(red)

def test_cargar_red_con_neurona_cargada():
    red = RED.copy()
    neurona = N.copy()
    sim = Simulador()
    sim.cargar_neurona(neurona)
    with pytest.raises(ValueError):
        sim.cargar_red(red)

def test_cargar_red_despues_de_limpiar_todo(simulador):
    simulador.simular(5)
    simulador.limpiar_todo()
    simulador.cargar_red(RED.copy())
    assert simulador.red is not None

@pytest.mark.parametrize("red", ["red", [RED.copy()], N.copy()])
def test_cargar_red_invalido(red):
    sim = Simulador()
    with pytest.raises(TypeError):
        sim.cargar_red(red)

def test_cargar_neurona():
    neurona = N.copy()
    sim = Simulador()
    sim.cargar_neurona(neurona)
    assert sim.red is neurona
    assert sim.num_neuronas == 1

def test_cargar_neurona_con_red_cargada(simulador):
    neurona = N.copy()
    with pytest.raises(ValueError):
        simulador.cargar_neurona(neurona)

def test_cargar_neurona_con_neurona_cargada():
    neurona = N.copy()
    neurona2 = N.copy()
    sim = Simulador()
    sim.cargar_neurona(neurona2)
    with pytest.raises(ValueError):
        sim.cargar_neurona(neurona)

@pytest.mark.parametrize("neurona", ["neurona", RED.copy(), [N.copy()]])
def test_cargar_neurona_invalido(neurona):
    sim = Simulador()
    with pytest.raises(TypeError):
        sim.cargar_neurona(neurona)


# ==================================================
# TESTS DE SIMULACIÓN
# ==================================================

def test_simular_sin_red_cargada():
    sim = Simulador()
    with pytest.raises(ValueError):
        sim.simular(10)

@pytest.mark.parametrize("pasos", ["2", 2.5, [2]])
def test_simular_pasos_no_enteros(simulador, pasos):
    with pytest.raises(TypeError):
        simulador.simular(pasos)

def test_simular_pasos_negativos(simulador):
    with pytest.raises(ValueError):
        simulador.simular(-1)

@pytest.mark.parametrize("parametro", ["guardar_resultados", "mostrar_progreso", "medir_rendimiento"])
@pytest.mark.parametrize("valor", ["True", [True]])
def test_simular_booleanos_invalidos(simulador, parametro, valor):
    with pytest.raises(TypeError):
        simulador.simular(**{parametro: valor})

@pytest.mark.parametrize("valor", [0.5, [2], "2"])
def test_simular_intervalo_rendimiento_no_entero(simulador, valor):
    with pytest.raises(TypeError):
        simulador.simular(10, medir_rendimiento=True, intervalo_rendimiento=valor)

@pytest.mark.parametrize("valor", [0, -1])
def test_simular_intervalo_rendimiento_invalido(simulador, valor):
    with pytest.raises(ValueError):
        simulador.simular(10, medir_rendimiento=True, intervalo_rendimiento=valor)

def test_simular_cero_pasos_guarda_estado_inicial():
    red = RedDeNeuronas({"RS": 1})
    sim = Simulador()
    sim.cargar_red(red.copy())

    tiempo = sim.simular(pasos=0)

    assert tiempo == pytest.approx(0, abs=1.0e-5)
    assert sim.paso_actual == 1
    assert sim.historial["spikes"].shape == (1,)
    assert sim.historial["v"].shape == (1,)
    assert sim.historial["u"].shape == (1,)
    assert sim.historial["I"].shape == (1,)
    assert not np.any(sim.historial["spikes"])
    assert sim.historial["v"][0] == pytest.approx(red.estado["v"][0])
    assert sim.historial["u"][0] == pytest.approx(red.estado["u"][0])
    assert sim.historial["I"][0] == pytest.approx(0)

def test_simular_cero_pasos_y_continuar(simulador):
    simulador.simular(0)
    assert simulador.paso_actual == 1
    simulador.simular(5)
    assert simulador.paso_actual == 6
    assert simulador.historial["v"].shape == (6,)
    assert simulador.historial["u"].shape == (6,)
    assert simulador.historial["I"].shape == (6,)
    assert simulador.historial["spikes"].shape == (6,)

def test_simular_avanza_pasos_correctamente(simulador):
    simulador.simular(pasos=10)
    assert simulador.paso_actual == 11

    hist = simulador.historial
    assert hist["spikes"].shape == (11,)
    assert hist["v"].shape == (11,)
    assert hist["u"].shape == (11,)
    assert hist["I"].shape == (11,)

def test_simular_varias_llamadas_continua_historial(simulador2):
    simulador2.simular(5, I=[5, 10])
    hist1 = simulador2.historial

    simulador2.simular(5, I=[20, 30])
    hist2 = simulador2.historial

    assert simulador2.paso_actual == 11
    assert hist2["spikes"].shape == (11,2)
    assert hist2["v"].shape == (11,2)
    assert hist2["u"].shape == (11,2)
    assert hist2["I"].shape == (11,2)

    assert np.array_equal(hist1["spikes"], hist2["spikes"][:6])
    assert np.array_equal(hist1["v"], hist2["v"][:6])
    assert np.array_equal(hist1["u"], hist2["u"][:6])
    assert np.array_equal(hist1["I"], hist2["I"][:6])
    assert np.array_equal(hist2["I"][0], [0, 0])
    assert np.all(hist2["I"][1:6] == [5, 10])
    assert np.all(hist2["I"][6:] == [20, 30])

def test_simular_neurona_individual():
    neurona = N.copy()
    sim = Simulador()
    sim.cargar_neurona(neurona)
    sim.simular(10)
    hist = sim.historial

    assert hist["spikes"].shape == (11,)
    assert hist["v"].shape == (11,)
    assert hist["u"].shape == (11,)
    assert len(hist["nombre"]) == 1
    assert len(hist["es_excitatoria"]) == 1
    assert neurona.estado == (hist["v"][10], hist["u"][10])

def test_simular_corriente_escalar(simulador2):
    simulador2.simular(5, I=10)
    hist = simulador2.historial
    assert np.all(hist["I"][0] == 0)
    assert np.all(hist["I"][1:] == 10)

def test_simular_corriente_vector(simulador2):
    simulador2.simular(5, I=[5, 10])
    hist = simulador2.historial
    assert np.array_equal(hist["I"][0], [0, 0])
    assert np.all(hist["I"][1:] == np.array([5, 10]))

def test_simular_corriente_array(simulador2):
    I = np.asarray([5, 10], dtype=np.float64)
    simulador2.simular(5, I=I)
    historial = simulador2.historial
    assert np.array_equal(historial["I"][0], [0, 0])
    assert np.all(historial["I"][1:] == [5, 10])

def test_simular_corriente_tamano_incorrecto(simulador2):
    with pytest.raises(ValueError):
        simulador2.simular(5, I=[1,2,3])

@pytest.mark.parametrize("I", ["10", {"a": 1}])
def test_simular_corriente_tipo_invalido(simulador2, I):
    with pytest.raises(TypeError):
        simulador2.simular(5, I=I)

def test_simular_guarda_resultados(simulador):
    nombre_archivo = "./tests/tmp_sim/historial_desde_simular.npz"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    archivo = Path(nombre_archivo)
    assert archivo.exists()

def test_simular_guardar_resultados_con_configuracion(simulador):
    path = "./tests/tmp_sim/configuracion.npz"
    simulador.configurar_simulacion(guardar_resultados=True, path_guardado=path)
    simulador.simular(5)
    assert Path(path).exists()

def test_simular_no_guarda_resultados_si_se_desactiva(simulador):
    path = "./tests/tmp_sim/no_guardar.npz"
    simulador.simular(5, guardar_resultados=False, path_guardado=path)
    assert not Path(path).exists()

def test_simular_sobrescribe_configuracion(simulador):
    path = "./tests/tmp_sim/sobrescribe_configuracion.npz"
    simulador.configurar_simulacion(guardar_resultados=True, path_guardado=path)
    simulador.simular(5, guardar_resultados=False, path_guardado=path)
    assert not Path(path).exists()


# ==================================================
# TESTS DE PROPIEDADES
# ==================================================

def test_historial_devuelve_copia(simulador):
    simulador.simular(0)
    hist = simulador.historial

    hist["spikes"][0] = True
    hist["v"][0] = 0
    hist["u"][0] = 0
    hist["I"][0] = 10
    hist["nombre"][0] = "Prueba"
    hist["es_excitatoria"][0] = False
    hist["dt"] = 1
    assert not simulador.historial["spikes"][0]
    assert simulador.historial["v"][0] != 0
    assert simulador.historial["u"][0] != 0
    assert simulador.historial["I"][0] != 10
    assert simulador.historial["nombre"][0] != "Prueba"
    assert bool(simulador.historial["es_excitatoria"][0])
    assert simulador.historial["dt"] == 1

def test_historial_dtype_distinto_de_red_dtype():
    red = RedDeNeuronas({"rs": 1}, precision=64)
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(0)
    hist = sim.historial
    assert red.dtype == np.float64
    assert hist["spikes"].dtype == bool
    assert hist["v"].dtype == np.float32
    assert hist["u"].dtype == np.float32
    assert hist["I"].dtype == np.float32


# ==================================================
# TESTS DE GUARDAR HISTORIAL
# ==================================================

def test_guardar_historial_npz_red_con_1_neurona(simulador):
    simulador.simular(5)
    nombre_archivo = "./tests/tmp_sim/historial_1_neurona.npz"
    simulador.guardar_historial(nombre_archivo)

    archivo = Path(nombre_archivo)
    assert archivo.exists()

    datos = np.load(nombre_archivo)

    assert "spikes" in datos
    assert "v" in datos
    assert "u" in datos
    assert "nombre" in datos
    assert "es_excitatoria" in datos
    assert "I" in datos
    assert "dt" in datos

    assert datos["spikes"].shape == (6,)
    assert datos["v"].shape == (6,)
    assert datos["u"].shape == (6,)
    assert len(datos["nombre"]) == 1
    assert len(datos["es_excitatoria"]) == 1
    assert datos["I"].shape == (6,)
    assert datos["dt"] == 1

    datos.close()

def test_guardar_historial_npz_red_con_2_neuronas():
    red = RedDeNeuronas({"rs": 2})
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(5)

    nombre_archivo = "./tests/tmp_sim/historial_2_neuronas.npz"
    sim.guardar_historial(nombre_archivo)

    archivo = Path(nombre_archivo)
    assert archivo.exists()

    datos = np.load(nombre_archivo)

    assert datos["spikes"].shape == (6,2)
    assert datos["v"].shape == (6,2)
    assert datos["u"].shape == (6,2)
    assert len(datos["nombre"]) == 2
    assert len(datos["es_excitatoria"]) == 2
    assert datos["I"].shape == (6,2)
    assert datos["dt"] == 1

    datos.close()

def test_guardar_historial_json():
    red = RedDeNeuronas({"rs": 2})
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(5)

    nombre_archivo = "./tests/tmp_sim/historiales.json"
    sim.guardar_historial(nombre_archivo)

    archivo = Path(nombre_archivo)
    assert archivo.exists()

    with open(nombre_archivo, "r") as f:
        datos = json.loads(f.read())

    assert "spikes" in datos
    assert "v" in datos
    assert "u" in datos
    assert "nombre" in datos
    assert "es_excitatoria" in datos
    assert "I" in datos
    assert "dt" in datos

    assert len(datos["spikes"]) == 6
    assert len(datos["v"]) == 6
    assert len(datos["u"]) == 6
    assert len(datos["I"]) == 6
    assert all(len(datos["spikes"][i]) == 2 for i in range(6))
    assert all(len(datos["v"][i]) == 2 for i in range(6))
    assert all(len(datos["u"][i]) == 2 for i in range(6))
    assert all(len(datos["I"][i]) == 2 for i in range(6))
    assert len(datos["nombre"]) == 2
    assert len(datos["es_excitatoria"]) == 2
    assert datos["dt"] == 1

def test_guardar_historial_csv(simulador):
    simulador.simular(5)
    nombre_archivo = "./tests/tmp_sim/historial.csv"
    simulador.guardar_historial(nombre_archivo)

    archivo_spikes = Path(nombre_archivo)
    archivo_spikes = archivo_spikes.with_name(f"{archivo_spikes.stem}_spikes.csv")
    archivo_v = Path(nombre_archivo)
    archivo_v = archivo_v.with_name(f"{archivo_v.stem}_v.csv")
    archivo_u = Path(nombre_archivo)
    archivo_u = archivo_u.with_name(f"{archivo_u.stem}_u.csv")
    archivo_nombres = Path(nombre_archivo)
    archivo_nombres = archivo_nombres.with_name(f"{archivo_nombres.stem}_nombre.csv")
    archivo_excitatorias = Path(nombre_archivo)
    archivo_excitatorias = archivo_excitatorias.with_name(f"{archivo_excitatorias.stem}_es_excitatoria.csv")
    archivo_I = Path(nombre_archivo)
    archivo_I = archivo_I.with_name(f"{archivo_I.stem}_I.csv")
    archivo_dt = Path(nombre_archivo)
    archivo_dt = archivo_dt.with_name(f"{archivo_dt.stem}_dt.csv")

    assert archivo_spikes.exists()
    assert archivo_v.exists()
    assert archivo_u.exists()
    assert archivo_nombres.exists()
    assert archivo_excitatorias.exists()
    assert archivo_I.exists()
    assert archivo_dt.exists()

def test_guardar_historial_txt(simulador):
    simulador.simular(5)

    nombre_archivo = "./tests/tmp_sim/historial.txt"
    simulador.guardar_historial(nombre_archivo)

    archivo_spikes = Path(nombre_archivo)
    archivo_spikes = archivo_spikes.with_name(f"{archivo_spikes.stem}_spikes.txt")
    archivo_v = Path(nombre_archivo)
    archivo_v = archivo_v.with_name(f"{archivo_v.stem}_v.txt")
    archivo_u = Path(nombre_archivo)
    archivo_u = archivo_u.with_name(f"{archivo_u.stem}_u.txt")
    archivo_nombres = Path(nombre_archivo)
    archivo_nombres = archivo_nombres.with_name(f"{archivo_nombres.stem}_nombre.txt")
    archivo_excitatorias = Path(nombre_archivo)
    archivo_excitatorias = archivo_excitatorias.with_name(f"{archivo_excitatorias.stem}_es_excitatoria.txt")
    archivo_I = Path(nombre_archivo)
    archivo_I = archivo_I.with_name(f"{archivo_I.stem}_I.txt")
    archivo_dt = Path(nombre_archivo)
    archivo_dt = archivo_dt.with_name(f"{archivo_dt.stem}_dt.txt")

    assert archivo_spikes.exists()
    assert archivo_v.exists()
    assert archivo_u.exists()
    assert archivo_nombres.exists()
    assert archivo_excitatorias.exists()
    assert archivo_I.exists()
    assert archivo_dt.exists()

def test_guardar_historial_sin_historial(simulador):
    nombre_archivo = "./tests/tmp_sim/historial_inexistente.npz"
    simulador.guardar_historial(nombre_archivo)
    assert not Path(nombre_archivo).exists()

def test_guardar_historial_formato_pasado(simulador):
    simulador.simular(5)
    nombre_archivo = "./tests/tmp_sim/historial_sin_extension"
    simulador.guardar_historial(nombre_archivo, formato="json")
    archivo = Path(f"{nombre_archivo}.json")
    assert archivo.exists()

def test_guardar_historial_sin_formato(simulador):
    simulador.simular(5)
    nombre_archivo = "./tests/tmp_sim/historial_sin_formato"
    simulador.guardar_historial(nombre_archivo)
    archivo = Path(f"{nombre_archivo}.npz")
    assert archivo.exists()

def test_guardar_historial_formatos_en_conflicto(simulador):
    simulador.simular(5)
    simulador.guardar_historial("./tests/tmp_sim/historial_conflicto.npz", formato="json")
    archivo = Path("./tests/tmp_sim/historial_conflicto.json")
    assert archivo.exists()

def test_guardar_historial_formato_invalido_usa_npz(simulador):
    simulador.simular(5)
    path = "./tests/tmp_sim/historial_sin_formato_existente"
    simulador.guardar_historial(path, formato="inexistente")
    assert Path(path + ".npz").exists()
    assert not Path(path + ".inexistente").exists()

def test_guardar_historial_en_path_por_defecto(simulador):
    simulador.simular(5)
    path = "./tests/tmp_sim/historial_por_defecto.npz"
    simulador.configurar_simulacion(path_guardado=path)
    simulador.guardar_historial()
    assert Path(path).exists()


# ==================================================
# TESTS DE LIMPIAR
# ==================================================

def test_limpiar_historial(simulador):
    simulador.simular(10)
    simulador.limpiar_historial()
    assert simulador.historial is None
    assert simulador.paso_actual == 0
    assert simulador.red is not None

def test_limpiar_rendimiento(simulador):
    simulador.simular(10, medir_rendimiento=True, intervalo_rendimiento=1)
    for clave, valor in simulador.rendimiento.items():
        if clave not in ("tiempo_ejecucion", "gpu_media", "gpu_maxima", "vram_media", "vram_maxima"):
            assert valor is not None
        elif clave == "tiempo_ejecucion":
            assert valor > 0
        else:
            assert valor is None

    simulador.limpiar_rendimiento()
    assert simulador.historial is not None
    assert simulador.red is not None
    for clave, valor in simulador.rendimiento.items():
        if clave != "tiempo_ejecucion":
            assert valor is None
        else:
            assert valor == pytest.approx(0, abs=1.0e-6)

def test_limpiar_todo(simulador):
    simulador.simular(5, medir_rendimiento=True, intervalo_rendimiento=1)
    for clave, valor in simulador.rendimiento.items():
        if clave not in ("tiempo_ejecucion", "gpu_media", "gpu_maxima", "vram_media", "vram_maxima"):
            assert valor is not None
        elif clave == "tiempo_ejecucion":
            assert valor > 0
        else:
            assert valor is None

    simulador.limpiar_todo()
    assert simulador.red is None
    assert simulador.num_neuronas == 0
    assert simulador.historial is None
    assert simulador.paso_actual == 0
    for clave, valor in simulador.rendimiento.items():
        if clave != "tiempo_ejecucion":
            assert valor is None
        else:
            assert valor == 0


# ==================================================
# TEST DE REINICIAR
# ==================================================

def test_reiniciar():
    red = RedDeNeuronas({N: 1})
    estado_inicial = red.estado

    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(5, medir_rendimiento=True, intervalo_rendimiento=1)
    assert red.estado != estado_inicial
    assert sim.historial is not None
    assert sim.paso_actual == 6
    for clave, valor in sim.rendimiento.items():
        if clave not in ("tiempo_ejecucion", "gpu_media", "gpu_maxima", "vram_media", "vram_maxima"):
            assert valor is not None
        elif clave == "tiempo_ejecucion":
            assert valor > 0
        else:
            assert valor is None

    sim.reiniciar()
    assert red.estado == estado_inicial
    assert sim.historial is None
    assert sim.paso_actual == 0
    assert sim.red is not None
    for clave, valor in sim.rendimiento.items():
        if clave != "tiempo_ejecucion":
            assert valor is None
        else:
            assert valor == pytest.approx(0, abs=1.0e-6)


# ==================================================
# TESTS GPU, si cupy disponible
# ==================================================

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
@pytest.mark.parametrize("valor", [1.5, "2", [2]])
def test_tamano_batch_no_entero(valor):
    red = RED.convertir_backend("cupy")
    sim = Simulador()
    sim.cargar_red(red)
    with pytest.raises(TypeError):
        sim.simular(10, tamano_batch=valor)

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
@pytest.mark.parametrize("valor", [0, -1])
def test_tamano_batch_invalido(valor):
    red = RED.convertir_backend("cupy")
    sim = Simulador()
    sim.cargar_red(red)
    with pytest.raises(ValueError):
        sim.simular(10, tamano_batch=valor)

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
def test_simular_en_gpu_historial_en_cpu():
    red = RED.convertir_backend("cupy")
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(5)
    hist = sim.historial
    assert isinstance(hist["spikes"], np.ndarray)
    assert isinstance(hist["v"], np.ndarray)
    assert isinstance(hist["u"], np.ndarray)
    assert isinstance(hist["I"], np.ndarray)
    assert hist["spikes"].dtype == bool
    assert hist["v"].dtype == np.float32
    assert hist["u"].dtype == np.float32
    assert hist["I"].dtype == np.float32

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
def test_simular_en_gpu_rendimiento_gpu_medido():
    red = RED.convertir_backend("cupy")
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(5, medir_rendimiento=True, intervalo_rendimiento=1)
    rend = sim.rendimiento
    assert rend["gpu_media"] is not None
    assert rend["gpu_maxima"] is not None
    assert rend["vram_media"] is not None
    assert rend["vram_maxima"] is not None

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
def test_simular_en_gpu_batch_incompleto():
    red = RED.convertir_backend("cupy")
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(10)
    hist = sim.historial
    assert hist["spikes"].shape == (11,)
    assert hist["v"].shape == (11,)
    assert hist["u"].shape == (11,)
    assert hist["I"].shape == (11,)
    assert len(hist["nombre"]) == 1
    assert len(hist["es_excitatoria"]) == 1
    assert hist["dt"] == 1

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
def test_simular_en_gpu_batch_con_restante():
    red = RED.convertir_backend("cupy")
    sim = Simulador()
    sim.cargar_red(red)
    sim.simular(10, tamano_batch=4)
    hist = sim.historial
    assert hist["spikes"].shape == (11,)
    assert hist["v"].shape == (11,)
    assert hist["u"].shape == (11,)
    assert hist["I"].shape == (11,)
    assert len(hist["nombre"]) == 1
    assert len(hist["es_excitatoria"]) == 1
    assert hist["dt"] == 1
