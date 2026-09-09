"""Suite de auditoría del código de Venegas.

Dos clases de prueba, con propósitos opuestos:

1. REGRESIÓN (``test_regresion_*``): fija los números publicados. Si alguna
   falla, algo alteró el resultado del Ejemplo 5 y hay que investigar.

2. CAPTURA DE DEFECTO (``test_defecto_*``): documenta el comportamiento
   defectuoso ACTUAL. Estas pruebas pasan mientras el defecto exista. Si una
   falla, el defecto fue corregido -- y entonces hay que actualizar la prueba y
   el informe, no "arreglar" la prueba.

Ejecutar:  cd code/audit && python -m pytest test_venegas_model.py -v
"""

import warnings

import numpy as np
import pytest

import audit_variantes as av
import venegas_model as vm

# ---------------------------------------------------------------- utilidades

ESCENARIOS = {
    "con_restriccion": [(0, 2)],
    "sin_restriccion": [],
}


def _corre(prohib):
    return vm.run_price_dynamics(
        vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, prohib,
        num_iterations=vm.EJEMPLO5_ITERS, eta=vm.EJEMPLO5_ETA,
        Gamma=vm.EJEMPLO5_GAMMA,
    )


# ============================================================== REGRESIÓN

#: P_infinito y W_infinito re-ejecutados y validados contra las cuatro figuras
#: commiteadas en Financial_Networks_Stability_files/figure-html/ (128 de 128
#: celdas dentro del redondeo a 2 decimales de las figuras).
ESPERADO = {
    "con_restriccion": dict(
        P=np.array([[0.0, -0.261663, 0.0],
                    [0.261663, 0.0, 0.230888],
                    [0.0, -0.230888, 0.0]]),
        W=np.array([[-0.065112, 0.260273, 0.0],
                    [0.260447, -0.401894, 0.561806],
                    [0.0, 0.561376, -0.337084]]),
    ),
    "sin_restriccion": dict(
        P=np.array([[0.0, -0.025634, 0.347007],
                    [0.025634, 0.0, 0.241436],
                    [-0.347007, -0.241436, 0.0]]),
        W=np.array([[-0.756940, -0.067616, 1.032559],
                    [-0.067667, -0.509303, 0.876240],
                    [1.031809, 0.877012, -1.300163]]),
    ),
}


@pytest.mark.parametrize("esc", list(ESCENARIOS))
def test_regresion_ejemplo5(esc):
    """El Ejemplo 5 reproduce los precios y exposiciones publicados."""
    W_list, P_list = _corre(ESCENARIOS[esc])
    np.testing.assert_allclose(P_list[-1], ESPERADO[esc]["P"], atol=1e-6)
    np.testing.assert_allclose(W_list[-1], ESPERADO[esc]["W"], atol=1e-6)


def test_regresion_restriccion_reduce_intermediacion():
    """Prohibir la arista (0,2) reduce la exposición total del sistema.

    Es el resultado económico que la tesis reporta; se fija como invariante.
    """
    W_con, _ = _corre(ESCENARIOS["con_restriccion"])
    W_sin, _ = _corre(ESCENARIOS["sin_restriccion"])
    assert np.abs(W_con[-1]).sum() < np.abs(W_sin[-1]).sum()
    assert W_con[-1][0, 2] == 0.0 and W_con[-1][2, 0] == 0.0


def test_regresion_precios_antisimetricos():
    """P debe ser antisimétrico: el precio que i paga a j es -el que j paga a i."""
    for prohib in ESCENARIOS.values():
        _, P_list = _corre(prohib)
        np.testing.assert_allclose(P_list[-1], -P_list[-1].T, atol=1e-12)


def test_regresion_variante_parametrica_equivale_al_original():
    """`audit_variantes.dinamica(tol_pct=1e-1)` == `run_price_dynamics`."""
    for esc, prohib in ESCENARIOS.items():
        W_list, P_list = _corre(prohib)
        W, P, _ = av.dinamica(vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, tuple(prohib),
                              num_iterations=vm.EJEMPLO5_ITERS, eta=vm.EJEMPLO5_ETA,
                              Gamma=vm.EJEMPLO5_GAMMA, tol_pct=1e-1)
        assert np.abs(P - P_list[-1]).max() == 0.0
        assert np.abs(W - W_list[-1]).max() == 0.0


# ======================================================= CAPTURA DE DEFECTOS


def test_defecto_H1_guarda_invertida_en_get_percent_change():
    """H1: devuelve 100 % justo cuando los valores coinciden.

    La guarda `if current == previous: return 100.0` reporta discrepancia
    máxima en el caso de acuerdo perfecto, que es la condición que la dinámica
    busca. Lo correcto sería 0.
    """
    assert vm.get_percent_change(0.5, 0.5) == 100.0
    assert vm.get_percent_change(0.0, 0.0) == 100.0
    # y sí reporta 0 cuando la discrepancia es diminuta pero no nula
    assert vm.get_percent_change(0.5 + 1e-15, 0.5) < 1e-10


def test_defecto_H1_latente_con_los_datos_del_ejemplo5():
    """H1 no se activa en el Ejemplo 5: los flotantes no coinciden bit a bit.

    Sobre las aristas que el bucle evalúa (las permitidas) la igualdad exacta
    nunca ocurre, así que el defecto no afecta ningún resultado publicado.
    En la arista PROHIBIDA sí ocurre -- ambas entradas de W son 0.0 -- pero el
    bucle la salta antes de llamar a `get_percent_change`. Quien reutilice la
    función como diagnóstico global sí toparía con el 100 % espurio ahí.
    """
    prohib = ESCENARIOS["con_restriccion"]
    W_list, _ = _corre(prohib)
    permitidas = [(i, j) for i in range(3) for j in range(i)
                  if (i, j) not in prohib and (j, i) not in prohib]
    activaciones = sum(1 for W in W_list for i, j in permitidas if W[i, j] == W[j, i])
    assert activaciones == 0

    # en la arista prohibida la igualdad se cumple siempre (0.0 == 0.0)
    assert all(W[2, 0] == W[0, 2] == 0.0 for W in W_list)


def test_defecto_H1_se_activa_con_calibracion_simetrica():
    """H1 sí se activa con Sigma = I y medias simétricas: 100 % reportado, 0 % real.

    Es el escenario relevante para calibrar con datos agregados o simétricos.
    El precio de equilibrio (P = 0) sigue siendo correcto porque el
    desplazamiento también es nulo, pero `get_percent_change` queda inservible
    como diagnóstico de convergencia.
    """
    cov_i = np.eye(3)
    mean_sim = np.array([[0.0, 0.8, 0.6],
                         [0.8, 0.0, 0.7],
                         [0.6, 0.7, 0.0]])
    W, P, info = av.dinamica(cov_i, mean_sim, (), num_iterations=50, eta=0.5,
                             Gamma=np.eye(3), tol_pct=1e-1)
    np.testing.assert_allclose(W, W.T, atol=1e-14)        # acuerdo real perfecto
    assert info["desacuerdo_abs"] == 0.0                   # discrepancia real = 0
    assert info["desacuerdo_pct"] == 100.0                 # discrepancia reportada = 100 %
    np.testing.assert_allclose(P, 0.0, atol=1e-14)


def test_defecto_H2_zerodivisionerror_es_inerte_con_escalares_numpy():
    """H2: la rama `except ZeroDivisionError` nunca se ejecuta con datos reales.

    Las entradas de W son np.float64: dividir por cero emite RuntimeWarning y
    devuelve inf, no lanza la excepción. El rescate solo funcionaría si las
    entradas fueran floats de Python.
    """
    # np.float64: emite RuntimeWarning y devuelve inf. NO entra al except.
    # (el aviso de NumPy se emite una sola vez por ubicación en el proceso, así
    # que se afirma sobre el valor de retorno, que es la evidencia del defecto)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = vm.get_percent_change(np.float64(1.0), np.float64(0.0))
        out_neg = vm.get_percent_change(np.float64(-2.0), np.float64(0.0))
    assert np.isinf(out) and np.isinf(out_neg)   # NO devolvió el 0 del except
    # el caso 0/0 (nan) es inalcanzable: 0.0 == -0.0, así que lo intercepta H1
    assert vm.get_percent_change(np.float64(0.0), np.float64(-0.0)) == 100.0
    assert isinstance(vm.run_price_dynamics(
        vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, [(0, 2)], num_iterations=1,
        eta=0.5, Gamma=vm.EJEMPLO5_GAMMA)[0][-1][0, 1], np.float64)  # W es np.float64

    # solo con floats de Python la excepción ocurre y el except devuelve 0
    assert vm.get_percent_change(1.0, 0.0) == 0


def test_defecto_H3_eta_por_defecto_anula_los_precios():
    """H3: `eta=0.0` por defecto congela P en cero y la dinámica es un no-op.

    Con eta=0, `price_update_naive` devuelve `0*p' + 1*old = old`. Quien llame
    a `run_price_dynamics` sin pasar eta obtiene precios idénticamente nulos,
    un resultado que parece plausible (sin intermediación) pero es un artefacto.
    """
    W_list, P_list = vm.run_price_dynamics(
        vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, [(0, 2)],
        num_iterations=100, Gamma=vm.EJEMPLO5_GAMMA)   # eta omitido -> 0.0
    assert np.abs(P_list[-1]).max() == 0.0
    assert np.abs(P_list[-1] - P_list[0]).max() == 0.0
    # y difiere del resultado publicado en ~50 veces la precisión impresa
    assert np.abs(P_list[-1] - ESPERADO["con_restriccion"]["P"]).max() > 0.25


def test_defecto_H4_sin_criterio_de_convergencia():
    """H4: se ejecutan 500 iteraciones fijas; la trayectoria se congela mucho antes.

    Con eta=0.5 la trayectoria deja de cambiar en k=29 (con restricción) y
    k=17 (sin restricción): 94 % y 97 % de las iteraciones no hacen nada, y el
    código no reporta ningún residual que permita saberlo.
    """
    congelacion = {"con_restriccion": 29, "sin_restriccion": 17}
    for esc, prohib in ESCENARIOS.items():
        _, P_list = _corre(prohib)
        k = next(i for i in range(1, len(P_list))
                 if np.array_equal(P_list[i], P_list[i - 1]))
        assert k == congelacion[esc]
        assert np.array_equal(P_list[-1], P_list[k])


def test_defecto_H4_umbral_deja_desacuerdo_residual():
    """H4: el punto final no es pairwise-estable exacto, sino a 0.1 % de tolerancia.

    El umbral está fijo en el código como 1e-1 (por ciento). La trayectoria se
    detiene con un desacuerdo residual justo por debajo de él.
    """
    for esc, prohib in ESCENARIOS.items():
        W_list, _ = _corre(prohib)
        W = W_list[-1]
        pares = [(i, j) for i in range(3) for j in range(i)
                 if (i, j) not in prohib and (j, i) not in prohib]
        peor = max(abs(vm.get_percent_change(W[i, j], W[j, i])) for i, j in pares)
        assert 0.05 < peor < 0.1        # residual, pero bajo el umbral del código


def test_defecto_H4b_aristas_prohibidas_no_heredan_P_init():
    """H4b: `P_new = np.zeros(...)` y las aristas prohibidas nunca se escriben.

    Un paso de la dinámica pone en cero el precio de una arista prohibida en
    lugar de conservar el de entrada. Coincide con el original solo porque P
    arranca en ceros.
    """
    n = 3
    Rm = vm.construct_prohibition_matrices(n, [(0, 2)])
    Q = vm.construct_q_matrices([vm.EJEMPLO5_COV.copy() for _ in range(n)], Rm,
                                Gamma=vm.EJEMPLO5_GAMMA)
    P_init = np.zeros((n, n))
    P_init[0, 2], P_init[2, 0] = 0.7, -0.7      # precio arbitrario en la arista prohibida
    _, P_new = vm.run_price_dyanmics_one_step(Q, vm.EJEMPLO5_MEAN, P_init,
                                              [(0, 2)], eta=0.5)
    assert P_new[0, 2] == 0.0 and P_new[2, 0] == 0.0   # se perdió, no se heredó
    assert P_init[0, 2] == 0.7                          # el input no se mutó


def test_defecto_H5_price_soln_sylvester_lanza_nameerror():
    """H5: el nombre `la` no está definido en ninguna parte del .qmd original.

    Los imports traen `solve_sylvester` directo al espacio de nombres, sin
    alias de módulo. La solución cerrada -- la verificación independiente
    natural de la dinámica iterativa -- nunca fue ejecutable.
    """
    S_inv = np.linalg.inv(vm.EJEMPLO5_COV)
    with pytest.raises(NameError, match=r"\bla\b"):
        vm.price_soln_sylvester(S_inv, vm.EJEMPLO5_MEAN)


def test_defecto_H7_generadores_sin_semilla():
    """H7: `generate_firm_covariances` y `generate_means` sortean sin semilla.

    Dos llamadas consecutivas devuelven matrices distintas: ningún análisis de
    sensibilidad construido sobre ellas es reproducible.
    """
    a = vm.generate_firm_covariances(num_firms=4)
    b = vm.generate_firm_covariances(num_firms=4)
    assert not np.allclose(a, b)
    assert not np.allclose(vm.generate_means(4), vm.generate_means(4))
    # y no aceptan argumento de semilla
    with pytest.raises(TypeError):
        vm.generate_firm_covariances(num_firms=4, seed=0)


def test_defecto_H8_codigo_muerto():
    """H8: tres funciones definidas y nunca invocadas en el .qmd.

    `opt_portfolio_mat` aparece además en dos llamadas COMENTADAS dentro de la
    dinámica (líneas 193 y 214): es la versión sin restricciones de aristas que
    `opt_w_with_edge_constraints` reemplazó, y el autor dejó el reemplazo a la
    vista sin borrar el código anterior.
    """
    lineas = open("../Financial_Networks_Stability.qmd", encoding="utf-8").read().splitlines()
    activas = [ln for ln in lineas if not ln.lstrip().startswith("#")]
    for nombre in ("get_iden_missing_row", "opt_portfolio_mat", "price_soln_sylvester"):
        usos = [ln for ln in activas if nombre in ln]
        assert len(usos) == 1 and usos[0].lstrip().startswith("def "), (nombre, usos)

    comentadas = [ln.strip() for ln in lineas
                  if ln.lstrip().startswith("#") and "opt_portfolio_mat" in ln]
    assert len(comentadas) == 2


def test_defecto_H9_divergencia_silenciosa_para_eta_alto():
    """H9: eta >= 1.5 diverge a ~1e12 sin excepción, aviso ni valor centinela."""
    W, P, info = av.dinamica(vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, ((0, 2),),
                             num_iterations=500, eta=1.5,
                             Gamma=vm.EJEMPLO5_GAMMA, tol_pct=1e-1)
    assert info["divergio"]
    assert np.abs(P).max() > 1e6
    assert np.all(np.isfinite(P))     # ni siquiera desborda a inf: parece un número


def test_defecto_H9_region_estable():
    """H9: la región estable es 0 < eta <~ 1.4; el código no la documenta."""
    for eta in (0.25, 0.5, 1.0, 1.25):
        _, _, info = av.dinamica(vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, ((0, 2),),
                                 num_iterations=500, eta=eta,
                                 Gamma=vm.EJEMPLO5_GAMMA, tol_pct=1e-1)
        assert not info["divergio"] and info["P_max_abs"] < 1.0
    for eta in (1.5, 2.0):
        _, _, info = av.dinamica(vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, ((0, 2),),
                                 num_iterations=500, eta=eta,
                                 Gamma=vm.EJEMPLO5_GAMMA, tol_pct=1e-1)
        assert info["divergio"]


# ============================================================== ÁLGEBRA
# Verifican que el código implementa la CPO del modelo declarado en la tesis
# (docs/MemoriasTesis/Venegas/chapters/chapter03.tex). Derivaciones en
# verificacion_algebraica.md. Si alguna falla, el código dejó de implementar
# el modelo -- no es un defecto documentado, es una regresión algebraica.

from scipy.linalg import solve_sylvester

COV, MEAN, GAM = vm.EJEMPLO5_COV, vm.EJEMPLO5_MEAN, vm.EJEMPLO5_GAMMA


def _P_aleatoria(rng, n=3):
    A = rng.normal(size=(n, n))
    return A - A.T


def test_algebra_foc_no_restringida():
    """opt_portfolio_mat resuelve (mu_i - P e_i) - 2 gamma_i Sigma w_i = 0."""
    rng = np.random.default_rng(0)
    Gamma = np.diag([0.7, 1.3, 2.0])
    P = _P_aleatoria(rng)
    W = vm.opt_portfolio_mat(np.linalg.inv(COV), MEAN, P, Gamma=Gamma)
    for i in range(3):
        resid = (MEAN[:, i] - P[:, i]) - 2 * Gamma[i, i] * COV @ W[:, i]
        assert np.abs(resid).max() < 1e-12, i


def test_algebra_optimo_restringido_resuelve_el_sistema_reducido():
    """Q_i es la inversa del sistema REDUCIDO, no filas anuladas de Sigma^-1."""
    rng = np.random.default_rng(1)
    P = _P_aleatoria(rng)
    R = vm.construct_prohibition_matrices(3, [(0, 2)])
    Q = vm.construct_q_matrices([COV.copy()] * 3, R, Gamma=GAM)
    W = vm.opt_w_with_edge_constraints(Q, MEAN, P)

    # (a) cero exacto en la contraparte prohibida
    assert W[2, 0] == 0.0 and W[0, 2] == 0.0

    # (b) CPO proyectada sobre las contrapartes permitidas
    for i in (0, 2):
        resid = R[i] @ ((MEAN[:, i] - P[:, i]) - 2 * GAM[i, i] * COV @ W[:, i])
        assert np.abs(resid).max() < 1e-12, i

    # (c) NO coincide con anular filas de la inversa (la forma ingenua e incorrecta)
    ingenuo = 0.5 * np.linalg.inv(COV) @ (MEAN[:, 0] - P[:, 0])
    ingenuo[2] = 0.0
    assert np.abs(ingenuo - W[:, 0]).max() > 1e-3


def test_algebra_paso_de_precio_es_exacto():
    """El shift de price_update_naive iguala W_ij = W_ji en un paso (eta=1)."""
    rng = np.random.default_rng(2)
    P0 = _P_aleatoria(rng)
    Q = vm.construct_q_matrices([COV.copy()] * 3,
                                vm.construct_prohibition_matrices(3, []), Gamma=GAM)
    i, j = 0, 1
    W0 = vm.opt_w_with_edge_constraints(Q, MEAN, P0)
    shift = (W0[i, j] - W0[j, i]) / (Q[i][j, j] + Q[j][i, i])
    P1 = P0.copy()
    P1[i, j] = P0[i, j] + shift
    P1[j, i] = -P1[i, j]
    W1 = vm.opt_w_with_edge_constraints(Q, MEAN, P1)
    assert abs(W1[i, j] - W1[j, i]) < 1e-12


def test_algebra_sylvester_coincide_con_el_punto_fijo():
    """Validación independiente: solución cerrada vs. iteración amortiguada."""
    Q = 0.5 * np.linalg.inv(COV)
    P_cerrada = solve_sylvester(Q, Q, Q @ MEAN - MEAN.T @ Q)
    _, P_iter, _ = av.dinamica(COV, MEAN, (), num_iterations=20000,
                               eta=vm.EJEMPLO5_ETA, Gamma=GAM, tol_pct=1e-10)
    assert np.abs(P_cerrada - P_iter).max() < 1e-9
    assert np.allclose(P_cerrada, -P_cerrada.T, atol=1e-13)


def test_algebra_sylvester_requiere_gamma_identidad():
    """Resuelve el TODO de la linea 135: la forma cerrada asume Gamma = I."""
    Gamma = np.diag([0.7, 1.3, 2.0])
    Q = 0.5 * np.linalg.inv(COV)          # lo que el codigo usa, ignora Gamma
    P_cerrada = solve_sylvester(Q, Q, Q @ MEAN - MEAN.T @ Q)
    _, P_iter, _ = av.dinamica(COV, MEAN, (), num_iterations=20000,
                               eta=vm.EJEMPLO5_ETA, Gamma=Gamma, tol_pct=1e-10)
    assert np.abs(P_cerrada - P_iter).max() > 1e-2


def test_algebra_iterados_publicados_en_la_tesis():
    """W y P en t = 0, 5, 10, inf del cap. 3 de la tesis, a 2 decimales."""
    X = np.nan
    esperado = {
        0: (np.array([[-0.10, 0.20, X], [0.40, -0.31, 0.74], [X, 0.44, -0.45]]),
            np.array([[0, 0, X], [0, 0, 0], [X, 0, 0]], dtype=float)),
        5: (np.array([[-0.08, 0.24, X], [0.30, -0.38, 0.61], [X, 0.53, -0.37]]),
            np.array([[0, -0.19, X], [0.19, 0, 0.17], [X, -0.17, 0]])),
        10: (np.array([[-0.07, 0.25, X], [0.27, -0.40, 0.58], [X, 0.55, -0.35]]),
             np.array([[0, -0.24, X], [0.24, 0, 0.21], [X, -0.21, 0]])),
        -1: (np.array([[-0.07, 0.26, X], [0.26, -0.40, 0.56], [X, 0.56, -0.34]]),
             np.array([[0, -0.26, X], [0.26, 0, 0.23], [X, -0.23, 0]])),
    }
    W_list, P_list = _corre(ESCENARIOS["con_restriccion"])
    for t, (We, Pe) in esperado.items():
        for nom, calc, esp in (("W", W_list[t], We), ("P", P_list[t], Pe)):
            d = np.where(np.isnan(esp), 0.0, np.abs(calc - esp))
            assert np.nanmax(d) <= 0.005, (t, nom, float(np.nanmax(d)))


def test_algebra_q_dual_no_es_generalizacion_directa():
    """Trampa del factor 1/2: Q_dual(lambda_B=0) = Q_Jalan / 2, no Q_Jalan."""
    gam, lam = 1.3, 0.0
    Q_jalan = np.linalg.inv(2 * gam * COV)
    Q_dual = 0.5 * np.linalg.inv(2 * gam * COV + lam * np.eye(3))
    assert np.abs(Q_dual - Q_jalan / 2).max() < 1e-12
    assert np.abs(Q_dual - Q_jalan).max() > 1e-2


# ============================================================== COBERTURA
# Fijan los resultados de la tesis que ESTA auditoría demostró reproducibles.
# Las definiciones de métrica viven en metricas_tesis.py (recuperadas por
# ingeniería inversa y confirmadas contra las tablas publicadas).

import metricas_tesis as mt


def _resumen(prohib, Sigma, M, Gamma):
    Wl, Pl = vm.run_price_dynamics(Sigma, M, prohib, num_iterations=500,
                                   eta=0.5, Gamma=Gamma)
    return mt.resumen(Wl[-1], Pl[-1], Sigma, M, Gamma)


def test_cobertura_tabla_baseline_cap4():
    """Tabla 'Baseline Comparison (3-Firm)' del cap. 4, a la precisión impresa."""
    con = _resumen(ESCENARIOS["con_restriccion"], COV, MEAN, GAM)
    sin = _resumen(ESCENARIOS["sin_restriccion"], COV, MEAN, GAM)
    assert abs(con["bilateral_volume"] - 0.822) < 5e-4
    assert abs(sin["bilateral_volume"] - 1.976) < 5e-4
    assert abs(con["total_positions"] - 1.626) < 5e-4
    assert abs(sin["total_positions"] - 4.543) < 5e-4
    for calc, esp in zip(con["utilities"], (0.0636, 0.4405, 0.2020)):
        assert abs(calc - esp) < 5e-5, (calc, esp)
    for calc, esp in zip(sin["utilities"], (0.4125, 0.4254, 0.5959)):
        assert abs(calc - esp) < 5e-5, (calc, esp)


def test_cobertura_numero_titular_perdida_de_bienestar():
    """El 50,8% de pérdida de bienestar del cap. 1 es reproducible."""
    con = _resumen(ESCENARIOS["con_restriccion"], COV, MEAN, GAM)
    sin = _resumen(ESCENARIOS["sin_restriccion"], COV, MEAN, GAM)
    assert abs(con["total_welfare"] - 0.7061) < 5e-5
    assert abs(sin["total_welfare"] - 1.4338) < 5e-5
    perdida = 100 * (sin["total_welfare"] - con["total_welfare"]) / sin["total_welfare"]
    assert abs(perdida - 50.8) < 0.05


def test_cobertura_escenario_fintech_4_firmas():
    """El escenario de 4 firmas del cap. 4 corre con el núcleo versionado."""
    con = _resumen(mt.PROHIBIDAS_4F_WITH, mt.COV4, mt.MEAN4, mt.GAMMA4)
    sin = _resumen(mt.PROHIBIDAS_4F_WITHOUT, mt.COV4, mt.MEAN4, mt.GAMMA4)
    assert abs(con["total_welfare"] - 1.0341) < 5e-5
    assert abs(sin["total_welfare"] - 1.7413) < 5e-5
    assert abs(con["utilities"][3] - 0.2390) < 5e-5      # utilidad de la Fintech
    assert abs(con["bilateral_volume"] - 1.351) < 5e-4
    # El escenario es EXACTAMENTE simetrico bajo permutar los agentes 2 y 4
    # (filas/columnas 2 y 4 de Sigma y M coinciden), asi que sus utilidades
    # deberian ser identicas. Con el umbral fijo del codigo queda un residuo:
    residuo = abs(con["utilities"][1] - con["utilities"][3])
    assert 1e-7 < residuo < 1e-5

    # y el residuo ES el umbral (H4), no la simetria del escenario: al apretarlo
    # cae varios ordenes de magnitud
    W, P, _ = av.dinamica(mt.COV4, mt.MEAN4, tuple(mt.PROHIBIDAS_4F_WITH),
                          num_iterations=20000, eta=0.5, Gamma=mt.GAMMA4,
                          tol_pct=1e-10)
    u = mt.utilities(W, P, mt.COV4, mt.MEAN4, mt.GAMMA4)
    assert abs(u[1] - u[3]) < residuo / 100


def test_cobertura_brecha_residual_27_9_pct():
    """La competencia mejora el bienestar pero no cierra la brecha (27,9%)."""
    sin3 = _resumen(ESCENARIOS["sin_restriccion"], COV, MEAN, GAM)["total_welfare"]
    con4 = _resumen(mt.PROHIBIDAS_4F_WITH, mt.COV4, mt.MEAN4, mt.GAMMA4)["total_welfare"]
    assert abs(100 * (sin3 - con4) / sin3 - 27.9) < 0.05


def test_cobertura_volumen_requiere_valor_absoluto():
    """Sin |.| el volumen del escenario 3F sin restriccion no cuadra con la tesis."""
    Wl, _ = vm.run_price_dynamics(COV, MEAN, [], num_iterations=500,
                                  eta=0.5, Gamma=GAM)
    W = Wl[-1]
    iu = np.triu_indices(3, 1)
    assert abs(np.abs(W[iu]).sum() - 1.976) < 5e-4       # definicion de la tesis
    assert abs(W[iu].sum() - 1.976) > 0.1                # suma con signo: no cuadra
    assert (W[iu] < 0).sum() == 1                        # hay una exposicion negativa
