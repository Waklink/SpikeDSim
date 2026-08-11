import pytest
from src.Neurona import Neurona

# Parámetros básicos a usar en las pruebas al crear neuronas personalizadas.
# Siguen el formato (a, b, c, d)
PARAMETROS_BASICOS = (0.02, 0.2, -65, 8)

# Tipos predefinidos
TIPOS = ("rs", "ib", "ch", "fs", "lts", "tc", "rz")

# Parámetros de los tipos predefinidos
PARAMETROS_TIPOS = {
        "rs": (0.02, 0.2, -65, 8),
        "ib": (0.02, 0.2, -55, 4),
        "ch": (0.02, 0.2, -50, 2),
        "fs": (0.1, 0.2, -65, 2),
        "lts": (0.02, 0.25, -65, 2),
        "tc": (0.02, 0.25, -65, 0.05),
        "rz": (0.1, 0.26, -65, 2)
    }


# ==================================================
# TESTS DEL CONSTRUCTOR
# ==================================================

def test_crear_neurona_personalizada():
    n = Neurona(*PARAMETROS_BASICOS, v_inicial=-50, u_inicial=-20, nombre="Prueba", es_excitatoria=False)
    assert isinstance(n, Neurona)
    assert n.parametros == (0.02, 0.2, -65, 8)
    assert n.estado == (-50, -20)
    assert n.nombre == "Prueba"
    assert n.es_excitatoria is False
    assert n.uso_gpu is False

def test_u_inicial_calculado():
    n = Neurona(*PARAMETROS_BASICOS, v_inicial=-50)
    assert n.estado == (-50, -10)

@pytest.mark.parametrize("parametro", ["a", "b", "c", "d", "v_inicial"])
@pytest.mark.parametrize("valor_invalido", [None, "0.1", [2]])
def test_parametros_deben_ser_reales(parametro, valor_invalido):
    parametros = {"a": 0.02, "b": 0.2, "c": -65, "d": 8, "v_inicial": -65}
    parametros[parametro] = valor_invalido
    with pytest.raises(TypeError):
        Neurona(**parametros)

@pytest.mark.parametrize("valor_invalido", ["10", [-13]])
def test_u_inicial_debe_ser_real(valor_invalido):
    with pytest.raises(TypeError):
        Neurona(*PARAMETROS_BASICOS, u_inicial=valor_invalido)

@pytest.mark.parametrize("nombre_invalido", [123, ["Texto"], None, b"texto"])
def test_nombre_debe_ser_string(nombre_invalido):
    with pytest.raises(TypeError):
        Neurona(*PARAMETROS_BASICOS, nombre=nombre_invalido)

@pytest.mark.parametrize("valor_invalido", [None, 0, 1, "True", [True]])
def test_es_excitatoria_debe_ser_booleano(valor_invalido):
    with pytest.raises(TypeError):
        Neurona(*PARAMETROS_BASICOS, es_excitatoria=valor_invalido)


# ==================================================
# TESTS DE CREAR NEURONAS PREDEFINIDAS
# ==================================================

@pytest.mark.parametrize("tipo", TIPOS)
def test_crear_predefinido(tipo):
    n = Neurona.predefinida(tipo)
    assert isinstance(n, Neurona)
    assert n.parametros == PARAMETROS_TIPOS[tipo]

def test_predefinida_u_inicial_calculado():
    n = Neurona.predefinida("rs", v_inicial=-50)
    assert n.estado == (-50, -10)

def test_predefinida_estado_inicial_personalizado():
    n = Neurona.predefinida("rs", v_inicial=-50, u_inicial=-20)
    assert n.estado == (-50, -20)


# ==================================================
# TESTS DE ACTUALIZAR
# ==================================================

def test_actualizar_sin_spike():
    n = Neurona.predefinida("rs")

    estado_inicial = n.estado
    spike = n.actualizar(0)

    assert spike is False
    assert n.estado != estado_inicial

def test_actualizar_genera_spike():
    n = Neurona.predefinida("rs", v_inicial=29)
    assert n.actualizar(10, 1) is True
    assert n.estado[0] == 30

def test_actualizar_reset():
    n = Neurona.predefinida("rs", v_inicial=30, u_inicial=-13)
    n.actualizar(0, 1)
    v, u = n.estado

    # Reinicio
    # v = c = -65
    # u = u + d = -13 + 8 = -5
    # Integración
    # v = v + 0.5 * (0.04*v**2 + 5*v + 140 - u + I) = v + 0.5 * (0.04 * (-65)**2 + 5 * (-65) + 140 - (-5) + 0) = v + 0.5 * (-11) = -65 -5.5 = -70.5
    # v = v + 0.5 * (0.04*v**2 + 5*v + 140 - u + I) = v + 0.5 * (0.04 * (-70.5)**2 + 5 * (-70.5) + 140 - (-5) + 0) = v + 0.5 * (-8.69) = -70.5 - 4.345 = -74.845
    # u = u + a*(b*v - u) = u + 0.02 * (0.2 * (-74.845) - (-5)) = u - 0.19938 = -5 - 0.19938 = -5.19938

    assert v == pytest.approx(-74.845)
    assert u == pytest.approx(-5.19938)

@pytest.mark.parametrize("I", [-10, -5, 0, 5, 10])
@pytest.mark.parametrize("dt", [0.01, 0.1, 0.5, 1, 2, 10])
def test_actualizar_entradas_validas(I, dt):
    n = Neurona.predefinida("rs")
    a, b, _, _ = n.parametros
    v, u = n.estado

    v_calculado = v + 0.5 * dt * (0.04*v*v + 5*v + 140 - u + I)
    v_calculado = v_calculado + 0.5 * dt * (0.04*v_calculado*v_calculado + 5*v_calculado + 140 - u + I)
    u_calculado = u + dt * a * (b*v_calculado - u)
    if v_calculado >= 30:
        v_calculado = 30

    n.actualizar(I, dt)
    v_actual, u_actual = n.estado

    assert v_actual == pytest.approx(v_calculado)
    assert u_actual == pytest.approx(u_calculado)

@pytest.mark.parametrize("dt", [0, -1])
def test_actualizar_dt_invalido(dt):
    n = Neurona.predefinida("rs")
    with pytest.raises(ValueError):
        n.actualizar(0, dt)

@pytest.mark.parametrize("I, dt", [("5", 0.5), ([-4], 0.5), (0, "0.5"), (0, [0.5])])
def test_actualizar_entradas_no_reales(I, dt):
    n = Neurona.predefinida("rs")
    with pytest.raises(TypeError):
        n.actualizar(I, dt)


# ==================================================
# TESTS DE MÉTODOS PÚBLICOS
# ==================================================

def test_reiniciar_restaura_estado():
    n = Neurona.predefinida("rs")
    estado_inicial = n.estado

    n.actualizar(20)
    n.reiniciar()
    assert n.estado == estado_inicial

def test_establecer_estado_v():
    n = Neurona.predefinida("rs")
    _, u = n.estado
    n.establecer_estado(v=10)
    assert n.estado == (10, u)

def test_establecer_estado_u():
    n = Neurona.predefinida("rs")
    v, _ = n.estado
    n.establecer_estado(u=5)
    assert n.estado == (v, 5)

def test_establecer_estado_completo():
    n = Neurona.predefinida("rs")
    n.establecer_estado(v=10, u=5)
    assert n.estado == (10, 5)

def test_establecer_estado_vacio():
    n = Neurona.predefinida("rs")
    estado = n.estado
    n.establecer_estado()
    assert n.estado == estado

@pytest.mark.parametrize("parametro", ["v", "u"])
@pytest.mark.parametrize("valor_invalido", ["10", [-5]])
def test_establecer_estado_invalido(parametro, valor_invalido):
    parametros = {"v": None, "u": None}
    parametros[parametro] = valor_invalido
    n = Neurona.predefinida("rs")
    with pytest.raises(TypeError):
        n.establecer_estado(**parametros)

def test_copy_devuelve_copia():
    n1 = Neurona.predefinida("rs")
    n2 = n1.copy()
    assert n1 is not n2
    assert n1.estado == n2.estado
    assert n1.parametros == n2.parametros
    assert n1.nombre == n2.nombre
    assert n1.es_excitatoria is n2.es_excitatoria
    n2.establecer_estado(v=100)
    assert n1.estado != n2.estado


# ==================================================
# TESTS DE MÉTODOS DE LA CLASE
# ==================================================

def test_tipos_disponibles():
    assert Neurona.tipos_disponibles() == TIPOS

def test_alias_devuelve_copia():
    alias = Neurona.alias()
    alias["nuevo"] = "rs"
    assert "nuevo" not in Neurona.alias()

def test_tipos_devuelve_copia():
    tipos = Neurona.tipos()
    tipos["rs"]["nombre"] = "Cambio"
    assert Neurona.tipos()["rs"]["nombre"] != "Cambio"

@pytest.mark.parametrize("alias,tipo", [("regular spiking", "rs"), ("regular-spiking", "rs"),
                                        ("intrinsically bursting", "ib"), ("intrinsically-bursting", "ib"),
                                        ("chattering", "ch"), ("fast spiking", "fs"), ("fast-spiking", "fs"),
                                        ("low threshold spiking", "lts"), ("low-threshold-spiking", "lts"),
                                        ("thalamocortical", "tc"), ("thalamo cortical", "tc"),
                                        ("thalamo-cortical", "tc"), ("resonator", "rz")])
def test_alias_tipos_predefinidos(alias, tipo):
    assert Neurona.predefinida(alias).parametros == Neurona.predefinida(tipo).parametros

def test_tipo_inexistente():
    with pytest.raises(ValueError):
        Neurona.predefinida("Neurona")

@pytest.mark.parametrize("tipo", [None, 123, ["rs"]])
def test_tipo_no_texto(tipo):
    with pytest.raises(TypeError):
        Neurona.predefinida(tipo)


# ==================================================
# TESTS DE REPRESENTACIÓN
# ==================================================

def test_repr_contiene_informacion_basica():
    n = Neurona.predefinida("rs")
    texto = repr(n)
    assert "Regular Spiking" in texto
    assert "a=" in texto
    assert "b=" in texto
    assert "c=" in texto
    assert "d=" in texto
    assert "v=" in texto
    assert "u=" in texto
