import pytest
import numpy as np

from pathlib import Path
import shutil
import json

from src.basico.RedDeNeuronas import RedDeNeuronas
from src.basico.Simulador import Simulador
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
    assert hist[CLAVES[6]] == 0.5

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
    assert (hist[CLAVES[6]] is None) or (hist[CLAVES[6]] == 0.5)


# ==================================================
# TESTS DE HISTORIAL
# ==================================================

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

@pytest.mark.parametrize("valor", [0, -1])
def test_historial_dt_invalido(valor):
    with pytest.raises(ValueError):
        Visualizar(historial={"dt": valor})


# ==================================================
# TESTS DE LA NORMALIZACIÓN
# ==================================================

@pytest.mark.parametrize("parametro", [*CLAVES[:4]])
@pytest.mark.parametrize("valor, num_neuronas", [([1], 1), ([0, 1], 2)])
def test_normalizacion_arrays(parametro, valor, num_neuronas):
    vis = Visualizar(historial={parametro: valor})
    comprobar_historial_parcial(vis.historial, num_neuronas, 0)

@pytest.mark.parametrize("valor, num_neuronas", [("RS", 1), (np.asarray(["RS", "RS"]), 2)])
def test_normalizacion_nombre(valor, num_neuronas):
    vis = Visualizar(historial={"nombre": valor})
    comprobar_historial_parcial(vis.historial, num_neuronas)

@pytest.mark.parametrize("valor, num_neuronas", [(True, 1), (np.asarray([True, False]), 2)])
def test_normalizacion_es_excitatoria(valor, num_neuronas):
    vis = Visualizar(historial={"es_excitatoria": valor})
    comprobar_historial_parcial(vis.historial, num_neuronas)

def test_historial_vacio_se_convierte_a_None():
    vis = Visualizar(historial={})
    assert vis.historial is None

def test_historial_valores_None_se_convierte_a_None():
    vis = Visualizar(historial={clave: None for clave in CLAVES})
    assert vis.historial is None


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
    hist["dt"] = 1
    
    hist2 = vis.historial
    assert not hist2["spikes"][0]
    assert hist2["v"][0] != 0
    assert hist2["u"][0] != 0
    assert hist2["I"][0] != 10
    assert hist2["nombre"][0] != "Prueba"
    assert hist2["es_excitatoria"][0] is True
    assert hist2["dt"] == 0.5

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
# TESTS DEL CONSTRUCTOR
# ==================================================

def test_crear_sin_historial():
    vis = Visualizar()
    assert vis.historial is None

def test_crear_con_ambos_parametros_pasados(simulador, simulador2):
    path = "./tests/tmp_vis/historial_a_ignorar_en_constructor.npz"
    simulador.simular(10, guardar_resultados=True, path_guardado=path)
    simulador2.simular(5)
    vis = Visualizar(path, simulador2.historial)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_1_neurona_con_historial(simulador):
    simulador.simular(5)
    vis = Visualizar(historial=simulador.historial)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_1_neurona_desde_archivo_npz(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.npz"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_1_neurona_desde_archivo_json(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.json"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_1_neurona_desde_archivo_csv(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.csv"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_1_neurona_desde_archivo_txt(simulador):
    nombre_archivo = "./tests/tmp_vis/historiales1.txt"
    simulador.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 1)

def test_crear_con_red_2_neuronas_con_historial(simulador2):
    simulador2.simular(5)
    vis = Visualizar(historial=simulador2.historial)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_2_neuronas_desde_archivo_npz(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.npz"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_2_neuronas_desde_archivo_json(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.json"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_2_neuronas_desde_archivo_csv(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.csv"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)

def test_crear_con_red_2_neuronas_desde_archivo_txt(simulador2):
    nombre_archivo = "./tests/tmp_vis/historiales2.txt"
    simulador2.simular(5, guardar_resultados=True, path_guardado=nombre_archivo)
    vis = Visualizar(nombre_archivo)
    comprobar_historial_completo(vis.historial, 2)


# ==================================================
# TESTS DE CARGAR HISTORIAL
# ==================================================

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


# ==================================================
# OTROS TESTS
# ==================================================

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
