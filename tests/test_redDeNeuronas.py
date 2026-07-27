import pytest
import numpy as np
import cupy as cp
from src.basico.Neurona import Neurona
from src.basico.RedDeNeuronas import RedDeNeuronas

# Comprobar si hay cupy disponible
try:
    cp.zeros(1)
    CUPY_DISPONIBLE = True
except Exception:
    CUPY_DISPONIBLE = False

# Diccionario básico usado al probar otros parámetros del constructor
N = {"rs": 1}

# Diccionarios usados para probar cuando se necesiten más de una neurona en la red
N2 = {"rs": 2}
N3 = {"rs": 3}

# ==================================================
# TESTS BÁSICOS DEL CONSTRUCTOR
# ==================================================

def test_crear_red_basica():
    n = Neurona.predefinida("rs")
    red = RedDeNeuronas({n: 1}, precision=64)

    assert red.backend == "numpy"
    assert red.uso_gpu is False
    assert red.sparse is True
    assert red.num_neuronas == 1
    assert red.num_conexiones == 0

    neurona = list(red.neuronas.keys())[0]
    assert neurona.estado == n.estado
    assert neurona.parametros == n.parametros
    assert neurona.nombre == n.nombre
    assert neurona.es_excitatoria == n.es_excitatoria
    assert list(red.neuronas.values())[0] == 1
    assert red.nombre[0] == n.nombre
    assert red.es_excitatoria[0] == n.es_excitatoria
    assert red.conexiones == [[0]]

    v, u = n.estado
    dic_estado = {"v": pytest.approx([v]), "u": pytest.approx([u])}
    assert red.estado == dic_estado
    a, b, c, d = n.parametros
    dic_parametros = {"a": pytest.approx([a]), "b": pytest.approx([b]), "c": pytest.approx([c]), "d": pytest.approx([d])}
    assert red.parametros == dic_parametros

def test_crear_con_varios_tipos():
    tipos = Neurona.tipos_disponibles()
    cantidades = []
    total = 0
    rng = np.random.default_rng(42)
    for _ in range(len(tipos)):
        cantidad = int(rng.integers(1, 5, endpoint=True))
        total += cantidad
        cantidades.append(cantidad)
    neuronas = dict(zip(tipos, cantidades))
    red = RedDeNeuronas(neuronas, precision=64)
    assert red.num_neuronas == total

    a = []
    b = []
    c = []
    d = []
    for tipo, cantidad in zip(tipos, cantidades):
        parametros = Neurona.tipos()[tipo]["parametros"]
        a.extend([parametros[0]] * cantidad)
        b.extend([parametros[1]] * cantidad)
        c.extend([parametros[2]] * cantidad)
        d.extend([parametros[3]] * cantidad)
    dic_parametros = {"a": pytest.approx(a), "b": pytest.approx(b), "c": pytest.approx(c), "d": pytest.approx(d)}
    assert red.parametros == dic_parametros

def test_crear_con_neuronas_mixto():
    neuronas = {"rs": 2, Neurona.predefinida("fs"): 1}
    red = RedDeNeuronas(neuronas)
    assert red.num_neuronas == 3
    assert all(isinstance(n, Neurona) for n in red.neuronas)

@pytest.mark.parametrize("backend_invalido", [None, 1, ["numpy"]])
def test_crear_con_backend_no_texto(backend_invalido):
    with pytest.raises(TypeError):
        RedDeNeuronas(N, backend=backend_invalido)

def test_crear_con_backend_valor_invalido():
    with pytest.raises(ValueError):
        RedDeNeuronas(N, backend="cpu")

def test_crear_con_backend_correcto():
    red = RedDeNeuronas(N, backend=("  NumPY "))
    assert red.backend == "numpy"

@pytest.mark.parametrize("precision_invalida", ["32", [32], 32.0])
def test_crear_con_precision_no_entera(precision_invalida):
    with pytest.raises(TypeError):
        RedDeNeuronas(N, precision=precision_invalida)

def test_crear_con_precision_valor_invalido():
    with pytest.raises(ValueError):
        RedDeNeuronas(N, precision=16)

@pytest.mark.parametrize("sparse_invalido", [[True], None, "False"])
def test_crear_con_sparse_invalido(sparse_invalido):
    with pytest.raises(TypeError):
        RedDeNeuronas(N, sparse=sparse_invalido)

@pytest.mark.parametrize("valor_invalido", [[], (), None])
def test_crear_con_neuronas_invalido(valor_invalido):
    with pytest.raises(TypeError):
        RedDeNeuronas(valor_invalido)

@pytest.mark.parametrize("clave", [["rs"], 4])
def test_crear_con_claves_de_neuronas_no_texto(clave):
    with pytest.raises(TypeError):
        RedDeNeuronas({clave: 1})

def test_crear_con_clave_de_neurona_no_existente():
    with pytest.raises(ValueError):
        RedDeNeuronas({"neurona": 1})

@pytest.mark.parametrize("cantidad", [[2], "3", 0.5])
def test_crear_con_cantidades_de_neuronas_no_enteras(cantidad):
    with pytest.raises(TypeError):
        RedDeNeuronas({"rs": cantidad})

def test_crear_con_cantidades_de_neuronas_invalidas():
    with pytest.raises(ValueError):
        RedDeNeuronas({"rs": -1})


# ==================================================
# TESTS DEL CONSTRUCTOR CON CONEXIONES
# ==================================================

def test_crear_con_conexiones_aleatorias():
    red = RedDeNeuronas(N3, conexiones=5)
    assert red.num_conexiones == 5

def test_crear_con_conexiones_aleatorias_usando_semilla():
    red1 = RedDeNeuronas(N3, conexiones=5, semilla=10)
    red2 = RedDeNeuronas(N3, conexiones=5, semilla=10)
    red3 = RedDeNeuronas(N3, conexiones=5, semilla=11)
    assert red1.conexiones == red2.conexiones
    assert red1.conexiones != red3.conexiones

def test_crear_con_demasiadas_conexiones():
    with pytest.raises(ValueError):
        RedDeNeuronas(N3, conexiones=100)

def test_crear_con_conexiones_negativas():
    with pytest.raises(ValueError):
        RedDeNeuronas(N3, conexiones=-1)

def test_crear_con_matriz_de_conexiones():
    matriz_conexiones = [[0, 0.5],
                         [-0.2, 0]]
    red = RedDeNeuronas(N2, conexiones=matriz_conexiones)
    assert red.conexiones == [pytest.approx(conexion) for conexion in matriz_conexiones]

def test_crear_con_matriz_de_conexiones_de_dimensiones_incorrectas():
    matriz_conexiones = [[0, 0.5],
                         [-0.2, 0]]
    with pytest.raises(ValueError):
        RedDeNeuronas(N3, conexiones=matriz_conexiones)

def test_crear_con_matriz_de_conexiones_de_diagonal_no_cero():
    matriz_conexiones = [[0.3, 0.5],
                         [-0.2, 0]]
    with pytest.raises(ValueError):
        RedDeNeuronas(N2, conexiones=matriz_conexiones)

@pytest.mark.parametrize("peso", [1, -1, 3, -2])
def test_crear_con_matriz_de_conexiones_pesos_invalidos(peso):
    matriz_conexiones = [[0, peso],
                         [-peso, 0]]
    with pytest.raises(ValueError):
        RedDeNeuronas(N2, conexiones=matriz_conexiones)

def test_comprobar_num_conexiones_con_matriz_de_conexiones_densa():
    matriz_conexiones = [[0, 0.5],
                         [-0.5, 0]]
    red = RedDeNeuronas(N2, conexiones=matriz_conexiones, sparse=False)
    assert red.num_conexiones == 2

def test_crear_con_array_numpy_de_conexiones():
    matriz_conexiones = np.asarray([[0, 0.5], [-0.5, 0]])
    red = RedDeNeuronas(N2, conexiones=matriz_conexiones)
    assert red.conexiones == [pytest.approx(conexion) for conexion in matriz_conexiones.tolist()]

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
def test_crear_con_array_cupy_de_conexiones():
    matriz_conexiones = cp.asarray([[0, 0.5], [-0.5, 0]])
    red = RedDeNeuronas(N2, conexiones=matriz_conexiones, backend="cupy")
    assert red.conexiones == [pytest.approx(conexion) for conexion in matriz_conexiones.tolist()]


# ==================================================
# TESTS DE LAS PROPIEDADES
# ==================================================

def test_comprobar_estado_interno_devuelve_referencia_para_simulacion():
    red = RedDeNeuronas(N)
    v, u = red._estado()
    v[0] = 100
    u[0] = 20
    assert red.estado == {"v": [100], "u": [20]}

def test_comprobar_estado_devuelve_copia():
    red = RedDeNeuronas(N)
    estado = red.estado
    estado["v"][0] = 100
    estado["u"][0] = 20
    assert red.estado != estado

def test_comprobar_parametros_devuelve_copia():
    red = RedDeNeuronas(N)
    parametros = red.parametros
    parametros["a"][0] = 0
    parametros["b"][0] = 0
    parametros["c"][0] = 0
    parametros["d"][0] = 0
    assert red.parametros != parametros

def test_comprobar_neuronas_devuelve_copia():
    red = RedDeNeuronas(N)
    neuronas = red.neuronas
    neuronas.clear()
    assert len(red.neuronas) == 1

def test_comprobar_conexiones_devuelve_copia():
    matriz_conexiones = [[0, 0.5],
                         [-0.5, 0]]
    red = RedDeNeuronas(N2, conexiones=matriz_conexiones)
    matriz_conexiones[0][0] = -1
    assert red.conexiones != matriz_conexiones
    matriz_conexiones = red.conexiones
    matriz_conexiones[0][0] = 1
    assert red.conexiones != matriz_conexiones

def test_comprobar_nombre_es_correcto():
    n = Neurona.predefinida("rs")
    n2 = Neurona.predefinida("fs")
    red = RedDeNeuronas({n: 2, n2: 3})

    assert red.nombre == [n.nombre] * 2 + [n2.nombre] * 3

def test_comprobar_es_excitatoria_es_correcto():
    red = RedDeNeuronas({"rs": 2, "fs": 3})
    assert red.es_excitatoria == [True] * 2 + [False] * 3


# ==================================================
# TESTS DE ACTUALIZAR
# ==================================================

def test_actualizar_sin_spike():
    red = RedDeNeuronas(N)
    estado_inicial = red.estado
    spikes = red.actualizar(0)
    assert red.estado != estado_inicial
    assert spikes.shape == (1,)

def test_actualizar_genera_spike():
    red = RedDeNeuronas(N)
    spikes = []
    for _ in range(100):
        spikes.append(red.actualizar(20).tolist())
    assert any(spikes)

def test_actualizar_reset():
    n = Neurona.predefinida("rs", v_inicial=31, u_inicial=-13)
    red = RedDeNeuronas({n: 1})
    red.actualizar(0, 1)
    estado = red.estado
    v = estado["v"]
    u = estado["u"]
    assert v == pytest.approx([-76])
    assert u == pytest.approx([-5.204])

def test_actualizar_corriente_como_escalar():
    red = RedDeNeuronas(N2)
    estado_inicial = red.estado
    spikes = red.actualizar(10)
    assert red.estado != estado_inicial
    assert spikes.shape == (2,)

def test_actualizar_corriente_como_vector():
    red = RedDeNeuronas(N2)
    estado_inicial = red.estado
    corriente = np.asarray([10, 5])
    spikes = red.actualizar(corriente)
    assert red.estado != estado_inicial
    assert spikes.shape == (2,)

def test_actualizar_corriente_longitud_incorrecta():
    red = RedDeNeuronas(N3)
    corriente = np.asarray([10, 5])
    with pytest.raises(ValueError):
        red.actualizar(corriente)

@pytest.mark.parametrize("valor", [0, -1])
def test_actualizar_dt_invalido(valor):
    red = RedDeNeuronas(N)
    with pytest.raises(ValueError):
        red.actualizar(0, valor)

@pytest.mark.parametrize("I", ["10", [10]])
@pytest.mark.parametrize("dt", ["0.5", [0.5]])
def test_actualizar_entradas_no_reales(I, dt):
    red = RedDeNeuronas(N)
    with pytest.raises(TypeError):
        red.actualizar(I, dt)


# ==================================================
# TESTS DEL ESTADO
# ==================================================

def test_reiniciar():
    red = RedDeNeuronas(N)
    estado_inicial = red.estado
    red.actualizar(0)
    assert red.estado != estado_inicial
    red.reiniciar()
    assert red.estado == estado_inicial

@pytest.mark.parametrize("valor", [np.asarray([10, 2]), [10, 2]])
def test_establecer_estado_v(valor):
    red = RedDeNeuronas(N2)
    estado = red.estado
    estado["v"] = [10, 2]
    assert red.estado != estado
    red.establecer_estado(v=valor)
    assert red.estado == estado

@pytest.mark.parametrize("valor", [np.asarray([10, 2]), [10, 2]])
def test_establecer_estado_u(valor):
    red = RedDeNeuronas(N2)
    estado = red.estado
    estado["u"] = [10, 2]
    assert red.estado != estado
    red.establecer_estado(u=valor)
    assert red.estado == estado

@pytest.mark.parametrize("valor", [np.asarray([10, 2]), [10, 2]])
def test_establecer_estado_completo(valor):
    red = RedDeNeuronas(N2)
    estado = red.estado
    estado["v"] = [10, 2]
    estado["u"] = [10, 2]
    assert red.estado != estado
    red.establecer_estado(v=valor, u=valor)
    assert red.estado == estado

def test_establecer_estado_vacio():
    red = RedDeNeuronas(N2)
    estado = red.estado
    red.establecer_estado()
    assert red.estado == estado

@pytest.mark.parametrize("parametro", ["v", "u"])
def test_establecer_estado_longitud_incorrecta(parametro):
    parametros = {"v": None, "u": None}
    parametros[parametro] = np.asarray([10, 10, 10])
    red = RedDeNeuronas(N2)
    with pytest.raises(ValueError):
        red.establecer_estado(**parametros)


# ==================================================
# TESTS DE CONVERSIONES
# ==================================================

@pytest.mark.skipif(not CUPY_DISPONIBLE, reason="CuPy/GPU no disponible.")
@pytest.mark.parametrize(("backend", "nuevo_backend"), [("numpy", "cupy"), ("cupy", "numpy"),
                                                        ("cupy", "cupy")])
def test_convertir_backend_con_cupy(backend, nuevo_backend):
    red = RedDeNeuronas(N, backend=backend)
    nueva_red = red.convertir_backend(nuevo_backend)
    assert red is not nueva_red
    assert red.estado == nueva_red.estado
    assert red.parametros == nueva_red.parametros
    assert red.conexiones == nueva_red.conexiones
    assert red.num_neuronas == nueva_red.num_neuronas
    assert red.sparse == nueva_red.sparse
    assert red.backend == backend
    assert nueva_red.backend == nuevo_backend
    assert red.dtype == nueva_red.dtype

def test_convertir_backend_sin_cupy():
    red = RedDeNeuronas(N, backend="numpy")
    nueva_red = red.convertir_backend("numpy")
    assert red is not nueva_red
    assert red.estado == nueva_red.estado
    assert red.parametros == nueva_red.parametros
    assert red.conexiones == nueva_red.conexiones
    assert red.num_neuronas == nueva_red.num_neuronas
    assert red.sparse == nueva_red.sparse
    assert red.backend == "numpy"
    assert nueva_red.backend == "numpy"
    assert red.dtype == nueva_red.dtype

@pytest.mark.parametrize("formato", [True, False])
@pytest.mark.parametrize("nuevo_formato", [True, False])
def test_convertir_formato_conexiones(formato, nuevo_formato):
    red = RedDeNeuronas(N, sparse=formato)
    nueva_red = red.convertir_formato(nuevo_formato)
    assert red is not nueva_red
    assert red.estado == nueva_red.estado
    assert red.parametros == nueva_red.parametros
    assert red.conexiones == nueva_red.conexiones
    assert red.num_neuronas == nueva_red.num_neuronas
    assert red.sparse == formato
    assert nueva_red.sparse == nuevo_formato
    assert red.backend == nueva_red.backend
    assert red.dtype == nueva_red.dtype

@pytest.mark.parametrize("precision", [32, 64])
@pytest.mark.parametrize("nueva_precision", [32, 64])
def test_cambiar_precision(precision, nueva_precision):
    red = RedDeNeuronas(N, precision=precision)
    nueva_red = red.cambiar_precision(nueva_precision)
    assert red is not nueva_red

    # Comparar estado y parametros uno a uno, en vez de todo el diccionario, por el posible cambio
    # de valor al haber cambiado la precision
    assert all(red.estado[clave] == pytest.approx(nueva_red.estado[clave]) for clave in ("v", "u"))
    assert all(red.parametros[clave] == pytest.approx(nueva_red.parametros[clave]) for clave in 
               ("a", "b", "c", "d"))

    assert red.conexiones == nueva_red.conexiones
    assert red.num_neuronas == nueva_red.num_neuronas
    assert red.sparse == nueva_red.sparse
    assert red.backend == nueva_red.backend
    dtype = np.float32 if precision == 32 else np.float64
    nuevo_dtype = np.float32 if nueva_precision == 32 else np.float64
    assert red.dtype == dtype
    assert nueva_red.dtype == nuevo_dtype
