"""Pruebas del módulo corregido (H1, H3, H7, H4b + validación de Sigma).

No reemplazan test_venegas_model.py: ese archivo sigue documentando el
comportamiento (con defectos) del original y debe seguir pasando sin
cambios. Esta suite prueba que:
1. Las correcciones se comportan como se espera de forma aislada.
2. El Ejemplo 5 (con y sin restricción) da EXACTAMENTE los mismos números
   publicados con el módulo corregido -- las correcciones no alteran ningún
   resultado de la tesis.
3. eta_estable_teorema4 reproduce los valores ya verificados a mano.

Ejecutar desde la raíz del repositorio, sin variables de entorno

    python -m pytest code/caballero/test_venegas_model_corregido.py -v
"""

import numpy as np
import pytest

import venegas_model as original
import venegas_model_corregido as corregido
from metricas_tesis import resumen


# --------------------------------------------------------- H1: get_percent_change

def test_h1_acuerdo_perfecto_ya_no_reporta_discrepancia_maxima():
    assert corregido.get_percent_change(3.0, 3.0) == 0.0
    assert corregido.get_percent_change(0.0, 0.0) == 0.0


def test_h1_original_sigue_con_el_defecto():
    """Documenta que el original NO cambió (no se tocó venegas_model.py)."""
    assert original.get_percent_change(3.0, 3.0) == 100.0


def test_h1_caso_normal_sin_cambios():
    # fuera de la guardia H1, el cálculo es idéntico al original
    assert corregido.get_percent_change(1.5, 1.0) == original.get_percent_change(1.5, 1.0)


# --------------------------------------------------------- validación de Sigma

def test_sigma_no_simetrica_falla():
    Sigma = np.array([[1.0, 0.5], [0.6, 1.0]])
    with pytest.raises(ValueError, match="simétrica"):
        corregido.validar_covarianza(Sigma)


def test_sigma_no_definida_positiva_falla():
    Sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # autovalores 3, -1
    with pytest.raises(ValueError, match="definida positiva"):
        corregido.validar_covarianza(Sigma)


def test_sigma_valida_no_lanza():
    corregido.validar_covarianza(corregido.EJEMPLO5_COV)  # no debe lanzar


def test_construct_q_matrices_valida_por_defecto():
    n = 3
    Rm = corregido.construct_prohibition_matrices(n, [])
    Sigma_mala = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        corregido.construct_q_matrices([Sigma_mala.copy() for _ in range(n)], Rm)


def test_construct_q_matrices_validar_false_permite_saltarse_el_chequeo():
    n = 3
    Rm = corregido.construct_prohibition_matrices(n, [])
    Sigma_mala = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # no debe lanzar, aunque Sigma sea invalida, porque validar=False
    corregido.construct_q_matrices([Sigma_mala.copy() for _ in range(n)], Rm, validar=False)


# --------------------------------------------------------- H7: generadores sin semilla

def test_h7_rng_obligatorio():
    with pytest.raises(ValueError, match="rng es obligatorio"):
        corregido.generate_firm_covariances(None, num_firms=3)
    with pytest.raises(ValueError, match="rng es obligatorio"):
        corregido.generate_means(None, num_firms=3)


def test_h7_misma_semilla_reproduce():
    a = corregido.generate_firm_covariances(42, num_firms=3)
    b = corregido.generate_firm_covariances(42, num_firms=3)
    np.testing.assert_array_equal(a, b)


def test_h7_semillas_distintas_dan_resultados_distintos():
    a = corregido.generate_firm_covariances(1, num_firms=3)
    b = corregido.generate_firm_covariances(2, num_firms=3)
    assert not np.allclose(a, b)


def test_h7_acepta_generator_explicito():
    rng = np.random.default_rng(7)
    a = corregido.generate_means(rng, num_firms=3)
    assert a.shape == (3, 3)
    assert np.all(np.diag(a) == 0.0)


# --------------------------------------------------------- H3: eta sin default

def test_h3_eta_obligatorio():
    with pytest.raises(TypeError):
        corregido.run_price_dynamics(corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN,
                                     corregido.EJEMPLO5_PROHIBIDAS_CON, num_iterations=5)


def test_h3_eta_cero_explicito_sigue_permitido():
    # eta=0 explícito es una elección válida (aunque degenerada); lo que se
    # corrige es el DEFAULT silencioso, no la posibilidad de pedirlo.
    W_list, P_list = corregido.run_price_dynamics(
        corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN, corregido.EJEMPLO5_PROHIBIDAS_CON,
        num_iterations=5, eta=0.0)
    assert np.allclose(P_list[-1], 0.0)


# --------------------------------------------------------- H4b: aristas prohibidas

def test_h4b_arista_prohibida_hereda_p_init_no_nulo():
    n = 3
    prohib = [(0, 2)]
    Rm = corregido.construct_prohibition_matrices(n, prohib)
    Q = corregido.construct_q_matrices([corregido.EJEMPLO5_COV.copy() for _ in range(n)], Rm)

    P_init = np.zeros((n, n))
    P_init[0, 2], P_init[2, 0] = 0.77, -0.77  # valor en la arista PROHIBIDA

    _, P_new = corregido.run_price_dynamics_one_step(Q, corregido.EJEMPLO5_MEAN, P_init,
                                                      prohibited_edges_tuples=prohib, eta=0.5)
    assert P_new[0, 2] == 0.77
    assert P_new[2, 0] == -0.77


def test_h4b_original_pierde_el_valor_no_nulo():
    """Documenta que el original (no tocado) sigue con el defecto."""
    n = 3
    prohib = [(0, 2)]
    Rm = original.construct_prohibition_matrices(n, prohib)
    Q = original.construct_q_matrices([original.EJEMPLO5_COV.copy() for _ in range(n)], Rm)

    P_init = np.zeros((n, n))
    P_init[0, 2], P_init[2, 0] = 0.77, -0.77

    _, P_new = original.run_price_dyanmics_one_step(Q, original.EJEMPLO5_MEAN, P_init,
                                                     prohibited_edges_tuples=prohib, eta=0.5)
    assert P_new[0, 2] == 0.0  # se perdió


# --------------------------------------------------------- Teorema 4 (helper nuevo)

def test_teorema4_reproduce_valores_verificados_a_mano():
    n = 3
    prohib = corregido.EJEMPLO5_PROHIBIDAS_CON
    Rm = corregido.construct_prohibition_matrices(n, prohib)
    Q = corregido.construct_q_matrices([corregido.EJEMPLO5_COV.copy() for _ in range(n)], Rm,
                                       Gamma=corregido.EJEMPLO5_GAMMA)
    r = corregido.eta_estable_teorema4(Q, prohib)
    assert r["lambda_min"] == pytest.approx(0.4467, abs=1e-3)
    assert r["lambda_max"] == pytest.approx(1.5533, abs=1e-3)
    assert r["eta_star"] == pytest.approx(1.2876, abs=1e-3)
    assert r["eta_opt"] == pytest.approx(1.0, abs=1e-3)


def test_teorema4_sin_restriccion_da_distinta_region():
    n = 3
    Rm = corregido.construct_prohibition_matrices(n, [])
    Q = corregido.construct_q_matrices([corregido.EJEMPLO5_COV.copy() for _ in range(n)], Rm,
                                       Gamma=corregido.EJEMPLO5_GAMMA)
    r = corregido.eta_estable_teorema4(Q, [])
    assert r["eta_star"] > 0
    assert len(r["R"]) == 6  # 3 pares x 2 ordenamientos, todos permitidos


# --------------------------------------------------------- Regresión: Ejemplo 5 intacto

def test_regresion_ejemplo5_con_restriccion_bienestar_identico():
    W_list, P_list = corregido.run_price_dynamics(
        corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN, corregido.EJEMPLO5_PROHIBIDAS_CON,
        num_iterations=corregido.EJEMPLO5_ITERS, eta=corregido.EJEMPLO5_ETA,
        Gamma=corregido.EJEMPLO5_GAMMA)
    r = resumen(W_list[-1], P_list[-1], corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN,
                corregido.EJEMPLO5_GAMMA)
    assert r["total_welfare"] == pytest.approx(0.7061, abs=5e-4)


def test_regresion_ejemplo5_sin_restriccion_bienestar_identico():
    W_list, P_list = corregido.run_price_dynamics(
        corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN, corregido.EJEMPLO5_PROHIBIDAS_SIN,
        num_iterations=corregido.EJEMPLO5_ITERS, eta=corregido.EJEMPLO5_ETA,
        Gamma=corregido.EJEMPLO5_GAMMA)
    r = resumen(W_list[-1], P_list[-1], corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN,
                corregido.EJEMPLO5_GAMMA)
    assert r["total_welfare"] == pytest.approx(1.4338, abs=5e-4)


def test_regresion_identico_al_original_iterado_a_iterado():
    """El módulo corregido y el original deben dar los MISMOS W,P en cada
    iteración del Ejemplo 5 -- ninguna corrección activa en este escenario."""
    n = 3
    Rm_o = original.construct_prohibition_matrices(n, original.EJEMPLO5_PROHIBIDAS_CON)
    Q_o = original.construct_q_matrices(
        [original.EJEMPLO5_COV.copy() for _ in range(n)], Rm_o, Gamma=original.EJEMPLO5_GAMMA)
    W_o, P_o = original.run_price_dynamics(
        original.EJEMPLO5_COV, original.EJEMPLO5_MEAN, original.EJEMPLO5_PROHIBIDAS_CON,
        num_iterations=50, eta=original.EJEMPLO5_ETA, Gamma=original.EJEMPLO5_GAMMA)

    W_c, P_c = corregido.run_price_dynamics(
        corregido.EJEMPLO5_COV, corregido.EJEMPLO5_MEAN, corregido.EJEMPLO5_PROHIBIDAS_CON,
        num_iterations=50, eta=corregido.EJEMPLO5_ETA, Gamma=corregido.EJEMPLO5_GAMMA)

    for k in range(len(W_o)):
        np.testing.assert_allclose(W_o[k], W_c[k], atol=1e-12)
        np.testing.assert_allclose(P_o[k], P_c[k], atol=1e-12)
