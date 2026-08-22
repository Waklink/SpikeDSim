import pytest
import numpy as np

from pathlib import Path
import shutil
import json
import matplotlib
matplotlib.use("Agg")

from src.RedDeNeuronas import RedDeNeuronas
from src.Simulador import Simulador
from src.Visualizar import Visualizar

# Decidir si mantener los archivos temporales
MANTENER_ARCHIVOS_TEMPORALES = False

# Claves del historial
CLAVES = ("spikes", "v", "u", "I", "nombre", "es_excitatoria", "dt")


# ==================================================
# FIXTURES
# ==================================================

# Simulador básico con una red de una neurona ya cargada
@pytest.fixture
def simulador():
    sim = Simulador()
    sim.cargar_red(RedDeNeuronas({"rs": 1}))
    return sim

# Simulador básico con una red de dos neuronas ya cargada
@pytest.fixture
def simulador2():
    sim = Simulador()
    sim.cargar_red(RedDeNeuronas({"rs": 2}))
    return sim

# Eliminar la carpeta temporal si no se quieren mantener los archivos temporales
@pytest.fixture(scope="session", autouse=True)
def limpiar_temporales():
    yield
    if not MANTENER_ARCHIVOS_TEMPORALES:
        carpeta = Path("./tests/tmp_vis")
        if carpeta.exists():
            shutil.rmtree(carpeta)


# ==================================================
# FUNCIONES AUXILIARES
# ==================================================

def comprobar_historial_completo(hist, num_neuronas: int, pasos: int = 5):
    assert hist is not None
    assert all((clave in hist) and (hist[clave] is not None) for clave in CLAVES)
    assert all(isinstance(hist[clave], np.ndarray) for clave in CLAVES[:4])
    assert all(isinstance(hist[clave], list) for clave in CLAVES[4:6])
    assert isinstance(hist[CLAVES[6]], float)
    if num_neuronas == 1:
        assert all(hist[clave].shape == (pasos + 1,) for clave in CLAVES[:4])
    else:
        assert all(hist[clave].shape == ((num_neuronas,) if pasos == 0 else (pasos + 1, num_neuronas)) for clave in CLAVES[:4])
    assert all(len(hist[clave]) == num_neuronas for clave in CLAVES[4:6])
    assert hist[CLAVES[6]] == 1

def comprobar_historial_parcial(hist, num_neuronas: int, pasos: int= 5):
    assert hist is not None
    assert all(clave in hist for clave in CLAVES)
    assert all((hist[clave] is None) or isinstance(hist[clave], np.ndarray) for clave in CLAVES[:4])
    assert all((hist[clave] is None) or isinstance(hist[clave], list) for clave in CLAVES[4:6])
    assert (hist[CLAVES[6]] is None) or isinstance(hist[CLAVES[6]], float)
    if num_neuronas == 1:
        assert all((hist[clave] is None) or (hist[clave].shape == (pasos + 1,)) for clave in CLAVES[:4])
    else:
        assert all((hist[clave] is None) or (hist[clave].shape == ((num_neuronas,) if pasos == 0 else (pasos + 1, num_neuronas))) for clave in CLAVES[:4])
    assert all((hist[clave] is None) or (len(hist[clave]) == num_neuronas) for clave in CLAVES[4:6])
    assert (hist[CLAVES[6]] is None) or (hist[CLAVES[6]] == 1)


# ==================================================
# TESTS DEL CONSTRUCTOR Y CARGA DE HISTORIAL
# ==================================================

def test_crear_con_ambos_parametros_pasados(simulador, simulador2):
    path = "./tests/tmp_vis/historial_a_ignorar_en_constructor.npz"
    simulador.simular(10, guardar_resultados=True, path_guardado=path)
    simulador2.simular(5)
    vis = Visualizar(path, simulador2.historial)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_de_una_neurona_con_historial(simulador):
    simulador.simular(5)
    vis = Visualizar(historial=simulador.historial)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_de_dos_neuronas_con_historial(simulador2):
    simulador2.simular(5)
    vis = Visualizar(historial=simulador2.historial)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_de_una_neurona_desde_archivo_npz(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.npz"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_de_dos_neuronas_desde_archivo_npz(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.npz"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_de_una_neurona_desde_archivo_json(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.json"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_de_dos_neuronas_desde_archivo_json(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.json"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_de_una_neurona_desde_archivo_csv(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.csv"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_de_dos_neuronas_desde_archivo_csv(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.csv"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_de_una_neurona_desde_archivo_txt(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.txt"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_de_dos_neuronas_desde_archivo_txt(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.txt"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)


def test_cargar_historial_sin_argumentos():
    vis = Visualizar()
    with pytest.raises(ValueError):
        vis.cargar_historial()

def test_cargar_historial_con_ambos_argumentos(simulador, simulador2):
    simulador.simular(10)
    path = "./tests/tmp_vis/historial_a_ignorar_al_cargar.npz"
    simulador2.simular(5, guardar_resultados=True, path_guardado=path)
    vis = Visualizar()
    vis.cargar_historial(path, simulador.historial)
    hist = vis.historial
    comprobar_historial_completo(hist, 1, 10)

def test_cargar_historial_desde_simulador(simulador):
    simulador.simular(5)
    vis = Visualizar()
    vis.cargar_historial(historial=simulador.historial)
    comprobar_historial_completo(vis.historial, 1)

def test_cargar_historial_desde_archivo(simulador):
    nombre_archivo = "./tests/tmp_vis/historial_cargado.npz"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_cargar_nuevo_historial_sobreescribe_historial_anterior(simulador, simulador2):
    simulador.simular(5)
    nombre_archivo = "./tests/tmp_vis/historial_cargado_para_sobreescribir.npz"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(historial=simulador.historial)
    hist = vis.historial
    comprobar_historial_completo(hist, 1)
    vis.cargar_historial(nombre_archivo)
    hist2 = vis.historial
    comprobar_historial_completo(hist2, 2)

@pytest.mark.parametrize("path", [5, ["./historial.npz"]])
def test_cargar_historial_path_invalido(path):
    vis = Visualizar()
    with pytest.raises(TypeError):
        vis.cargar_historial(path)

def test_cargar_historial_extension_no_soportada():
    vis = Visualizar()
    with pytest.raises(ValueError):
        vis.cargar_historial("archivo.zip")

@pytest.mark.parametrize("formato", ["npz", "json", "csv", "txt"])
def test_cargar_historial_archivo_inexistente(formato):
    vis = Visualizar()
    path = f"./tests/tmp_vis/historial_inexistente.{formato}"
    vis.cargar_historial(path)
    assert vis.historial is None

@pytest.mark.parametrize("contenido, num", [(10, 1), ([1, 2, 3], 2), ("texto", 3)])
def test_cargar_historial_archivo_json_sin_diccionario(contenido, num):
    path = f"./tests/tmp_vis/historial_sin_diccionario{num}.json"
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2)
    vis = Visualizar()
    with pytest.raises(ValueError):
        vis.cargar_historial(path)

@pytest.mark.parametrize("extension", ["json", "npz"])
def test_cargar_historial_archivo_corrupto(extension):
    path = Path(f"./tests/tmp_vis/historial_corrupto.{extension}")
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("contenido corrupto")
    vis = Visualizar()
    with pytest.raises(Exception):
        vis.cargar_historial(path)


@pytest.mark.parametrize("parametro", [*CLAVES])
def test_historial_parcial_desde_npz(simulador, parametro):
    path = "./tests/tmp_vis/historial_incompleto.npz"
    simulador.simular(5)
    hist = simulador.historial
    hist.pop(parametro)
    np.savez_compressed(path, **hist)
    vis = Visualizar(Path(path))
    comprobar_historial_parcial(vis.historial, 1)
    assert vis.historial[parametro] is None

@pytest.mark.parametrize("parametro", [*CLAVES])
def test_historial_parcial_desde_json(simulador, parametro):
    path = "./tests/tmp_vis/historial_incompleto.json"
    simulador.simular(5)
    hist = simulador.historial
    hist.pop(parametro)
    for clave in CLAVES[:4]:
        if clave != parametro:
            hist[clave] = hist[clave].tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    vis = Visualizar(Path(path))
    comprobar_historial_parcial(vis.historial, 1)
    assert vis.historial[parametro] is None

@pytest.mark.parametrize("parametro", [*CLAVES])
def test_historial_parcial_desde_txt(simulador, parametro):
    path = "./tests/tmp_vis/historial_incompleto.txt"
    simulador.simular(5, guardar_resultados=True, path_guardado=path)
    path_eliminado = Path(path)
    path_eliminado = path_eliminado.with_name(f"{path_eliminado.stem}_{parametro}.txt")
    path_eliminado.unlink()
    vis = Visualizar(Path(path))
    comprobar_historial_parcial(vis.historial, 1)
    assert vis.historial[parametro] is None

@pytest.mark.parametrize("parametro", [*CLAVES])
def test_historial_parcial_desde_csv(simulador, parametro):
    path = "./tests/tmp_vis/historial_incompleto.csv"
    simulador.simular(5, guardar_resultados=True, path_guardado=path)
    path_eliminado = Path(path)
    path_eliminado = path_eliminado.with_name(f"{path_eliminado.stem}_{parametro}.csv")
    path_eliminado.unlink()
    vis = Visualizar(Path(path))
    comprobar_historial_parcial(vis.historial, 1)
    assert vis.historial[parametro] is None


# ==================================================
# TESTS DE LA NORMALIZACIÓN DEL HISTORIAL
# ==================================================

def test_historial_vacio_se_convierte_a_None():
    vis = Visualizar(historial={})
    assert vis.historial is None

def test_historial_valores_None_se_convierte_a_None():
    vis = Visualizar(historial={clave: None for clave in CLAVES})
    assert vis.historial is None

@pytest.mark.parametrize("parametro", [*CLAVES[:4]])
@pytest.mark.parametrize("valor, num_neuronas", [([1], 1), ([0, 1], 2)])
def test_normalizacion_arrays(parametro, valor, num_neuronas):
    vis = Visualizar(historial={parametro: valor})
    comprobar_historial_parcial(vis.historial, num_neuronas, 0)

@pytest.mark.parametrize("valor, num_neuronas", [("RS", 1), (np.asarray(["RS", "RS"]), 2)])
def test_normalizacion_nombre(valor, num_neuronas):
    vis = Visualizar(historial={"nombre": valor})
    comprobar_historial_parcial(vis.historial, num_neuronas)

@pytest.mark.parametrize("valor, num_neuronas", [(True, 1), (np.asarray([True, False]), 2), ([1, 0], 2)])
def test_normalizacion_es_excitatoria(valor, num_neuronas):
    vis = Visualizar(historial={"es_excitatoria": valor})
    comprobar_historial_parcial(vis.historial, num_neuronas)

@pytest.mark.parametrize("valor", [1, True, ("RS",), {"RS"}])
def test_normalizacion_nombre_tipo_invalido(valor):
    with pytest.raises(TypeError):
        Visualizar(historial={"nombre": valor})

def test_normalizacion_nombre_elementos_tipo_invalido():
    with pytest.raises(TypeError):
        Visualizar(historial={"nombre": ["RS", 1]})

@pytest.mark.parametrize("valor", [[True, "False"], [True, 2], ["True"]])
def test_normalizacion_es_excitatoria_elementos_invalidos(valor):
    with pytest.raises(TypeError):
        Visualizar(historial={"es_excitatoria": valor})


@pytest.mark.parametrize("valor", [[], "", 4, np.asarray([[1, 2], [3, 4]])])
def test_crear_con_historial_no_diccionario(valor):
    with pytest.raises(TypeError):
        Visualizar(historial=valor)

def test_crear_con_historial_incompleto(simulador):
    simulador.simular(5)
    hist = simulador.historial
    hist.pop("spikes")
    vis = Visualizar(historial=hist)
    hist = vis.historial
    assert hist is not None
    assert "spikes" in hist
    assert hist["spikes"] is None
    assert all((clave in hist) and (hist[clave] is not None) for clave in
                   ("v", "u", "I", "nombre", "es_excitatoria", "dt"))

def test_historiales_inconsistentes():
    hist = {"spikes": np.zeros((5, 2)), "v": np.zeros((6, 2))}
    with pytest.raises(ValueError):
        Visualizar(historial=hist)

@pytest.mark.parametrize("parametro", ["spikes", "v", "u", "I"])
@pytest.mark.parametrize("array", [np.asarray(1), np.zeros((2, 3, 4))])
def test_historiales_con_arrays_dimensiones_incorrectas(parametro, array):
    with pytest.raises(ValueError):
        Visualizar(historial={parametro: array})

@pytest.mark.parametrize("parametro, lista", [("nombre", ["RS", "RS"]), ("es_excitatoria", [True, True])])
def test_historiales_con_listas_longitud_incorrecta(simulador, parametro, lista):
    simulador.simular(5)
    hist = simulador.historial
    hist[parametro] = lista
    with pytest.raises(ValueError):
        Visualizar(historial=hist)

@pytest.mark.parametrize("valor", [1, 0.5, np.float64(2)])
def test_historial_dt_valido(valor):
    vis = Visualizar(historial={"dt": valor})
    assert isinstance(vis.dt, float)
    assert vis.dt == float(valor)

@pytest.mark.parametrize("valor", [0, -1])
def test_historial_dt_invalido(valor):
    with pytest.raises(ValueError):
        Visualizar(historial={"dt": valor})

@pytest.mark.parametrize("valor", ["texto", object()])
def test_historial_dt_no_convertible(valor):
    with pytest.raises((TypeError, ValueError)):
        Visualizar(historial={"dt": valor})


# ==================================================
# TESTS DE PROPIEDADES
# ==================================================

def test_historial_devuelve_copia(simulador):
    simulador.simular(5)
    vis = Visualizar(historial=simulador.historial)
    hist = vis.historial

    hist["spikes"][0] = True
    hist["v"][0] = 0
    hist["u"][0] = 0
    hist["I"][0] = 10
    hist["nombre"][0] = "Prueba"
    hist["es_excitatoria"][0] = False
    hist["dt"] = 0.5
    
    hist2 = vis.historial
    assert not np.array_equal(hist2["spikes"], hist["spikes"])
    assert not np.array_equal(hist2["v"], hist["v"])
    assert not np.array_equal(hist2["u"], hist["u"])
    assert not np.array_equal(hist2["I"], hist["I"])
    assert hist2["nombre"] != hist["nombre"]
    assert hist2["es_excitatoria"] != hist["es_excitatoria"]
    assert hist2["dt"] != hist["dt"]

def test_propiedades_devuelven_None_sin_historial_cargado():
    vis = Visualizar()
    assert vis.historial is None
    assert vis.spikes is None
    assert vis.v is None
    assert vis.u is None
    assert vis.I is None
    assert vis.nombre is None
    assert vis.es_excitatoria is None
    assert vis.dt is None


# ==================================================
# TESTS DE GRÁFICOS
# ==================================================

# Raster plot
def test_raster_plot_varias_neuronas(simulador2):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.raster_plot(mostrar=False)
    assert fig is not None
    assert ax is not None

@pytest.mark.parametrize("neuronas", [0, slice(0, 1), [0, 1], None])
def test_raster_plot_seleccion_neuronas(simulador2, neuronas):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.raster_plot(neuronas=neuronas, mostrar=False)
    assert fig is not None
    assert ax is not None


def test_raster_plot_separar_tipo_reordena_indices():
    hist = {
        "spikes": np.array([[1, 0], [0, 1]]),
        "es_excitatoria": [False, True],
        "nombre": ["I", "E"],
        "dt": 1.0
    }
    vis = Visualizar(historial=hist)
    fig, ax = vis.raster_plot(separar_tipo=True, mostrar=False)
    assert [int(label.get_text()) for label in ax.get_yticklabels()] == [1, 0]


def test_raster_plot_sin_spikes():
    vis = Visualizar(historial={"v": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.raster_plot(mostrar=False)

def test_raster_plot_separar_tipo_sin_datos():
    hist = {"spikes": np.zeros((5,2))}
    vis = Visualizar(historial=hist)
    with pytest.raises(ValueError):
        vis.raster_plot(mostrar=False)

# Potencial de membrana
def test_potencial_membrana_una_neurona(simulador):
    simulador.simular(10)
    vis = Visualizar(historial=simulador.historial)
    fig, ax = vis.potencial_membrana(mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_potencial_membrana_varias_neuronas(simulador2):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.potencial_membrana(neuronas=None, mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_potencial_membrana_sin_v():
    vis = Visualizar(historial={"spikes": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.potencial_membrana(mostrar=False)

# Variable de recuperación
def test_variable_recuperacion_una_neurona(simulador):
    simulador.simular(10)
    vis = Visualizar(historial=simulador.historial)
    fig, ax = vis.variable_recuperacion(mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_variable_recuperacion_varias_neuronas(simulador2):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.variable_recuperacion(neuronas=None, mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_variable_recuperacion_sin_u():
    vis = Visualizar(historial={"v": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.variable_recuperacion(mostrar=False)

# Corriente, I
def test_corriente_basica(simulador):
    simulador.simular(10)
    vis = Visualizar(historial=simulador.historial)
    fig, ax = vis.corriente(mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) == 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_corriente_varias_neuronas(simulador2):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.corriente(neuronas=None, mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_corriente_sin_I():
    vis = Visualizar(historial={"v": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.corriente(mostrar=False)

# Espacio de fase
def test_espacio_fase_varias_neuronas(simulador2):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.espacio_fase(neuronas=None, mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

@pytest.mark.parametrize("neuronas", [0, slice(0, 1), [0, 1], None])
def test_espacio_fase_seleccion_neuronas(simulador2, neuronas):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.espacio_fase(neuronas=neuronas, mostrar=False)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) >= 1
    assert all(len(line.get_xdata()) > 0 for line in ax.lines)
    assert all(len(line.get_ydata()) > 0 for line in ax.lines)

def test_espacio_fase_sin_v():
    vis = Visualizar(historial={"u": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.espacio_fase(mostrar=False)

def test_espacio_fase_sin_u():
    vis = Visualizar(historial={"v": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.espacio_fase(mostrar=False)

# Frecuencia de disparos
def test_frecuencia_disparos_varias_neuronas(simulador2):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.frecuencia_disparos(neuronas=None, mostrar=False)
    assert fig is not None
    assert ax is not None

@pytest.mark.parametrize("neuronas", [0, slice(0, 1), [0, 1], None])
def test_frecuencia_disparos_seleccion_neuronas(simulador2, neuronas):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    fig, ax = vis.frecuencia_disparos(neuronas=neuronas, mostrar=False)
    assert fig is not None
    assert ax is not None

def test_frecuencia_disparos_sin_spikes():
    vis = Visualizar(historial={"v": np.zeros(5)})
    with pytest.raises(ValueError):
        vis.frecuencia_disparos(mostrar=False)

# Tests generales para todos los métodos de gráficos
def test_graficos_con_historial_de_un_instante(simulador):
    simulador.simular(0)
    vis = Visualizar(historial=simulador.historial)
    for grafica in (vis.raster_plot, vis.potencial_membrana, vis.variable_recuperacion, vis.corriente,
                    vis.espacio_fase, vis.frecuencia_disparos,):
        fig, ax = grafica(mostrar=False)
        assert fig is not None
        assert ax is not None


# ==================================================
# TESTS DE VALIDACIONES COMUNES DE GRÁFICOS
# ==================================================

@pytest.mark.parametrize("neuronas", [10, -10])
def test_obtener_indices_neuronas_fuera_de_rango(simulador2, neuronas):
    simulador2.simular(5)
    vis = Visualizar(historial=simulador2.historial)
    with pytest.raises(IndexError):
        vis._obtener_indices_neuronas(neuronas, 2)

def test_obtener_indices_neuronas_lista_vacia(simulador2):
    simulador2.simular(5)
    vis = Visualizar(historial=simulador2.historial)
    with pytest.raises((IndexError, ValueError)):
        vis._obtener_indices_neuronas([], 2)

@pytest.mark.parametrize("neuronas, esperado", [(None, np.asarray([0, 1])), (0, np.asarray([0])),
                                                (-1, np.asarray([-1])), ([0, 1], np.asarray([0, 1])),
                                                (slice(0, 2), np.asarray([0, 1]))])
def test_obtener_indices_neuronas_validos(simulador2, neuronas, esperado):
    simulador2.simular(5)
    vis = Visualizar(historial=simulador2.historial)
    indices = vis._obtener_indices_neuronas(neuronas, 2)
    assert np.array_equal(indices, esperado)

@pytest.mark.parametrize("neuronas", ["0", 1.5, [0, "0"]])
def test_obtener_indices_neuronas_tipo_invalido(simulador2, neuronas):
    simulador2.simular(10)
    vis = Visualizar(historial=simulador2.historial)
    with pytest.raises(TypeError):
        vis._obtener_indices_neuronas(neuronas, 2)
