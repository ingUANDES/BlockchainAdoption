"""Pruebas de la capa de acoplamiento propia (``caballero_dual.py``).

Una primera versión de esta batería, bajo el nombre ``test_venegas_dual.py``,
quedó fuera del repositorio porque su grupo D1 codificaba la objeción
refutada el 20-oct-2026. Cuatro grupos, con propósitos distintos.

``test_validacion_*``   fijan las cifras publicadas que la capa SÍ reproduce.
``test_discrepancia_*`` fijan las que NO reproduce, para que la brecha quede
                        dentro de la suite en vez de fuera de ella.
``test_C*``             documentan los hallazgos sobre la especificación de
                        la fuente, con su verificación ejecutable.
``test_guarda_*``       cubren las validaciones de entrada del módulo.

Ejecutar desde la raíz del repositorio, sin variables de entorno

    python -m pytest code/caballero/test_caballero_dual.py -v
"""

import numpy as np
import pytest
import sympy as sp

from caballero_dual import (
    calcular_w_estrella, tolerancia_simetria, resolver_acoplamiento,
    resolver_acoplamiento_precio_unico, bienestar_riesgo_combinado,
    bienestar_riesgo_separado, brecha_fragmentacion, cuota_red_alternativa,
)
from venegas_model_corregido import EJEMPLO5_COV, EJEMPLO5_MEAN, EJEMPLO5_GAMMA
from metricas_tesis import bilateral_volume, resumen, COV4, MEAN4, GAMMA4
import venegas_model as orig

SIGMA, GAMMA, M = EJEMPLO5_COV, EJEMPLO5_GAMMA, EJEMPLO5_MEAN
RESTRICCION = [(0, 2)]          # arista Banco Nacional - Firma Local


@pytest.fixture(scope="module")
def w_star():
    return calcular_w_estrella(SIGMA, M, GAMMA)


@pytest.fixture(scope="module")
def tol(w_star):
    """Tolerancia de convergencia del motor, umbral para juzgar factibilidad."""
    return tolerancia_simetria(w_star[0])


@pytest.fixture(scope="module")
def simetrico(w_star):
    return resolver_acoplamiento(w_star[0], SIGMA, GAMMA, 0.0, 0.0, [])


@pytest.fixture(scope="module")
def restringido(w_star):
    return resolver_acoplamiento(w_star[0], SIGMA, GAMMA, 0.0, 0.0, RESTRICCION)


# =========================================================== validación
# Cifras publicadas que la capa reproduce. Fuente
# docs/MemoriasTesis/Venegas/chapters/chapter04.tex, tablas
# tab:3f_dual_baseline y tab:3f_dual_restricted.

def test_validacion_w_estrella_reproduce_benchmark_red_unica(w_star):
    """$w_i^*$ debe dar el bienestar 1,4338 ya fijado en test_venegas_model.py."""
    W, P = w_star
    assert resumen(W, P, SIGMA, M, GAMMA)["total_welfare"] == pytest.approx(1.4338, abs=5e-4)


def test_validacion_escenario_simetrico_reparto_mitad(w_star, simetrico):
    """Con tau=0, lambda_A=0 y ambas redes sin restricciones el reparto es 50/50."""
    W_T, W_A, _ = simetrico
    np.testing.assert_allclose(W_T, 0.5 * w_star[0], atol=1e-6)
    np.testing.assert_allclose(W_A, 0.5 * w_star[0], atol=1e-6)


def test_validacion_escenario_simetrico_diferencial_nulo(simetrico):
    """tab:3f_dual_baseline reporta un diferencial de precio promedio de 0,000."""
    assert np.abs(simetrico[2]).max() < 1e-6


def test_validacion_restriccion_bloquea_la_arista_en_T(restringido):
    assert restringido[0][0, 2] == pytest.approx(0.0, abs=1e-9)


def test_validacion_contrato_bloqueado_migra_completo_a_A(restringido):
    """tab:3f_dual_restricted reporta W_13 = 1,030 en la red Alternativa.

    Tolerancia 5e-3 porque la tabla redondea a tres decimales.
    """
    assert restringido[1][0, 2] == pytest.approx(1.030, abs=5e-3)


@pytest.mark.parametrize("escenario", ["simetrico", "restringido"])
def test_validacion_utilidad_total_publicada(request, w_star, escenario):
    """Ambos escenarios duales reportan una utilidad total de 1,434.

    Se reproduce sólo bajo la contabilidad de riesgo combinado, y por el
    motivo que documenta C1, a saber que bajo esa contabilidad el total
    depende únicamente de $w_i^*$.
    """
    W_T, W_A, _ = request.getfixturevalue(escenario)
    w = bienestar_riesgo_combinado(W_A, w_star[0], M, SIGMA, GAMMA, 0.0, 0.0)
    assert w == pytest.approx(1.434, abs=2e-3)


def test_validacion_utilidad_total_cuatro_agentes():
    """El escenario dual de cuatro agentes reporta una utilidad total de 1,741."""
    W_star4 = orig.run_price_dynamics(COV4, MEAN4, [], num_iterations=500,
                                      eta=0.5, Gamma=GAMMA4)[0][-1]
    W_T4, W_A4, _ = resolver_acoplamiento(W_star4, COV4, GAMMA4, 0.0, 0.0, RESTRICCION)
    w = bienestar_riesgo_combinado(W_A4, W_star4, MEAN4, COV4, GAMMA4, 0.0, 0.0)
    assert w == pytest.approx(1.741, abs=2e-3)


def test_validacion_mejora_sobre_la_red_unica_restringida(w_star, restringido):
    """La fuente reporta +103,0 por ciento sobre la utilidad 0,7061."""
    W_T, W_A, _ = restringido
    w = bienestar_riesgo_combinado(W_A, w_star[0], M, SIGMA, GAMMA, 0.0, 0.0)
    assert (w - 0.7061) / 0.7061 * 100 == pytest.approx(103.0, abs=0.5)


# =========================================================== discrepancia
# Cifras publicadas que la capa NO reproduce con los parámetros declarados
# por la fuente, es decir tau=0 y lambda_B=0. Quedan fijadas aquí para que la
# brecha sea visible dentro de la suite. Si alguna empieza a coincidir, hay
# que averiguar por qué antes de celebrarlo.

def test_discrepancia_adopcion_tres_agentes(restringido):
    """Publicado 72,4 por ciento, reproducido 81,6 por ciento."""
    cuota = cuota_red_alternativa(restringido[0], restringido[1])
    assert cuota == pytest.approx(0.816, abs=2e-3)
    assert abs(cuota - 0.724) > 0.05


def test_discrepancia_adopcion_cuatro_agentes():
    """Publicado 47,9 por ciento, reproducido 75,0 por ciento."""
    W_star4 = orig.run_price_dynamics(COV4, MEAN4, [], num_iterations=500,
                                      eta=0.5, Gamma=GAMMA4)[0][-1]
    W_T4, W_A4, _ = resolver_acoplamiento(W_star4, COV4, GAMMA4, 0.0, 0.0, RESTRICCION)
    cuota = cuota_red_alternativa(W_T4, W_A4)
    assert cuota == pytest.approx(0.750, abs=2e-3)
    assert abs(cuota - 0.479) > 0.2


def test_discrepancia_volumenes_por_red(restringido):
    """Publicado 0,568 y 1,486; reproducido 0,411 y 1,826."""
    W_T, W_A, _ = restringido
    assert bilateral_volume(W_T) == pytest.approx(0.411, abs=2e-3)
    assert bilateral_volume(W_A) == pytest.approx(1.826, abs=2e-3)


def test_discrepancia_diferencial_de_precio(restringido):
    """Publicado 0,059; reproducido 0,082."""
    P_eff = restringido[2]
    promedio = float(np.abs(P_eff[np.triu_indices(3, 1)]).mean())
    assert promedio == pytest.approx(0.082, abs=2e-3)


# =========================================================== C1
# Las dos contabilidades del riesgo son incompatibles.

def test_C1_riesgo_conjunto_degenera_la_optimizacion():
    """Con el riesgo sobre la exposición combinada el Hessiano pierde Sigma.

    Queda $H_i = \\lambda_A I$, de modo que con $\\lambda_A = 0$ el problema es
    lineal en $w_i^T$, no hay óptimo interior y el 50/50 no se puede derivar.
    """
    n = 3
    Sig = sp.Matrix(n, n, lambda i, j: sp.Symbol(f'S{min(i,j)}{max(i,j)}', real=True))
    mu = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f'mu{i}', real=True))
    ws = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f'ws{i}', real=True))
    w = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f'w{i}', real=True))
    gam, lamA, tau = sp.symbols('gamma lambda_A tau', real=True, positive=True)
    PTe = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f'PT{i}', real=True))
    PAe = sp.Matrix(n, 1, lambda i, j: sp.Symbol(f'PA{i}', real=True))
    wA = ws - w

    g = sp.expand((w.T * (mu - PTe))[0] + (wA.T * (mu + tau * sp.ones(n, 1) - PAe))[0]
                  - gam * (ws.T * Sig * ws)[0] - lamA * (wA.T * wA)[0])
    H = sp.simplify(-sp.Matrix(n, n, lambda i, j: sp.diff(g, w[i], w[j])) / 2)
    assert sp.simplify(H - lamA * sp.eye(n)) == sp.zeros(n, n)


def test_C1_las_dos_contabilidades_difieren_en_el_escenario_simetrico(w_star, simetrico):
    """La contabilidad literal del capítulo 3 no entrega el 1,434 publicado."""
    W_T, W_A, _ = simetrico
    sep = bienestar_riesgo_separado(W_T, W_A, M, SIGMA, GAMMA, 0.0, 0.0)
    comb = bienestar_riesgo_combinado(W_A, w_star[0], M, SIGMA, GAMMA, 0.0, 0.0)
    assert sep == pytest.approx(2.1507, abs=1e-3)
    assert comb == pytest.approx(1.4339, abs=1e-3)
    assert sep > comb


def test_C1_riesgo_separado_no_es_invariante_a_la_restriccion(w_star, simetrico, restringido):
    """Bajo la especificación literal el bienestar SÍ cae con la restricción.

    Reformula la objeción D2.4 original. La invariancia es una propiedad de la
    contabilidad de riesgo combinado, no del modelo, dado que bajo el riesgo
    separado el bienestar baja un 16,9 por ciento.
    """
    sep_sin = bienestar_riesgo_separado(*simetrico[:2], M, SIGMA, GAMMA, 0.0, 0.0)
    sep_con = bienestar_riesgo_separado(*restringido[:2], M, SIGMA, GAMMA, 0.0, 0.0)
    assert sep_con == pytest.approx(1.7869, abs=1e-3)
    assert (sep_con - sep_sin) / sep_sin * 100 == pytest.approx(-16.9, abs=0.2)


def test_C1_riesgo_combinado_si_es_invariante(w_star, simetrico, restringido):
    """La otra cara de lo mismo, y el motivo por el que el 1,434 no informa."""
    a = bienestar_riesgo_combinado(simetrico[1], w_star[0], M, SIGMA, GAMMA, 0.0, 0.0)
    b = bienestar_riesgo_combinado(restringido[1], w_star[0], M, SIGMA, GAMMA, 0.0, 0.0)
    assert a == pytest.approx(b, abs=1e-6)


# =========================================================== C2
# La sobreestimación por fragmentación, en forma cerrada.

def test_C2_forma_cerrada_predice_la_brecha(w_star, simetrico):
    """La brecha medida coincide con $2f(1-f)\\sum_i \\gamma_i w^{*'}\\Sigma w^*$."""
    W_T, W_A, _ = simetrico
    medida = (bienestar_riesgo_separado(W_T, W_A, M, SIGMA, GAMMA, 0.0, 0.0)
              - bienestar_riesgo_combinado(W_A, w_star[0], M, SIGMA, GAMMA, 0.0, 0.0))
    assert medida == pytest.approx(brecha_fragmentacion(w_star[0], SIGMA, GAMMA, 0.5), abs=1e-4)


def test_C2_la_brecha_es_maxima_en_el_reparto_que_elige_la_FOC(w_star):
    """Crece de forma monótona en [0, 1/2] y su máximo cae en f = 1/2.

    Que el máximo coincida con el reparto que la condición de primer orden
    elige cuando tau=lambda_A=0 es lo que hace que la sobreestimación no sea
    un caso de borde sino el caso que el modelo produce.
    """
    fs = [0.0, 0.1, 0.25, 0.4, 0.5]
    brechas = [brecha_fragmentacion(w_star[0], SIGMA, GAMMA, f) for f in fs]
    assert brechas == sorted(brechas)
    assert brechas[-1] == max(brecha_fragmentacion(w_star[0], SIGMA, GAMMA, f)
                              for f in np.linspace(0, 1, 101))


# =========================================================== C3
# El precio único no admite punto factible bajo restricción.

def test_C3_precio_unico_es_factible_sin_restriccion(w_star, tol):
    """Sin restricción el reparto hereda la simetría de $W^*$, de modo que
    el supuesto de precio único no genera problema alguno."""
    W_T, _ = resolver_acoplamiento_precio_unico(w_star[0], SIGMA, GAMMA, 0.0, 0.0, [])
    assert np.abs(W_T - W_T.T).max() <= tol


def test_C3_precio_unico_da_el_reparto_mitad_no_cien_cero(w_star):
    """Refuta la objeción D1 original.

    Bajo precio único la condición de primer orden entrega $w_i^T = w_i^*/2$,
    es decir el 50/50 que la fuente publica, y no el 100/0 que D1 alegaba.
    """
    W_T, _ = resolver_acoplamiento_precio_unico(w_star[0], SIGMA, GAMMA, 0.0, 0.0, [])
    np.testing.assert_allclose(W_T, 0.5 * w_star[0], atol=1e-6)


def test_C3_precio_unico_viola_la_simetria_bajo_restriccion(w_star, tol):
    """La objeción que sí se sostiene contra el supuesto de precio único.

    La asimetría supera la tolerancia del motor en dos órdenes de magnitud,
    de manera que $W^T$ no es un punto factible según la propia definición
    de la fuente.
    """
    W_T, _ = resolver_acoplamiento_precio_unico(w_star[0], SIGMA, GAMMA, 0.0, 0.0, RESTRICCION)
    asimetria = np.abs(W_T - W_T.T).max()
    assert asimetria == pytest.approx(0.227, abs=2e-3)
    assert asimetria > 100 * tol


def test_C3_el_diferencial_de_precio_restaura_la_simetria(restringido, tol):
    """Con $P^T \\neq P^A$ el mismo escenario sí admite punto factible."""
    assert np.abs(restringido[0] - restringido[0].T).max() <= tol


# =========================================================== guardas

def test_guarda_rechaza_gamma_heterogenea(w_star):
    """``venegas_dual.py`` leía Gamma[0,0] e ignoraba el resto en silencio."""
    Gamma_het = np.diag([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="homogenea"):
        resolver_acoplamiento(w_star[0], SIGMA, Gamma_het, 0.0, 0.0, [])


def test_guarda_rechaza_parametros_negativos(w_star):
    with pytest.raises(ValueError, match="no negativos"):
        resolver_acoplamiento(w_star[0], SIGMA, GAMMA, -0.1, 0.0, [])
    with pytest.raises(ValueError, match="no negativos"):
        resolver_acoplamiento(w_star[0], SIGMA, GAMMA, 0.0, -0.1, [])
