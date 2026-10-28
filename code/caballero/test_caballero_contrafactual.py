"""Pruebas de la rama contrafactual del mandato de segundo piso.

Cuatro grupos, con propósitos distintos.

``test_primitivos_*``  fijan la restricción de cuatro a tres agentes y el cierre
                       del par CORFO--MiPyME, que es el supuesto agregado.
``test_identidad_*``   fijan que la fracción recuperada vale exactamente 100 por
                       ciento con tau = lambda_A = 0, que NO es un resultado sino
                       la identidad algebraica del capítulo 4, y que por eso la
                       métrica se reporta desagregada.
``test_topologia_*``   fijan que esta topología sí admite la capa de
                       acoplamiento, a diferencia del caso principal de cuatro.
``test_cifras_*``      fijan las cifras que el informe reporta.

Ejecutar desde code/caballero

    python -m pytest test_caballero_contrafactual.py -v
"""

import numpy as np
import pytest

import caballero_caso_chileno as cc
import caballero_contrafactual_mandato as cm


RW, RM, RS = np.mean(cc.R_W), np.mean(cc.R_M), np.mean(cc.R_S_DECLARADO)
DELTA = cc.subsidio(RS, RM) / 2
V = 0.02


@pytest.fixture
def M():
    return cm.construir_M3(RW, RS, RM, DELTA, V)


@pytest.fixture
def sigmas():
    return cm.sigmas3()


# ======================================================= primitivos y cierre

def test_primitivos_restriccion_elimina_al_banco(M):
    """La construcción de tres sale de la de cuatro sin el banco."""
    M4 = cc.construir_M(RW, RS, RM, DELTA, m_banco=0.0, v_corfo=V)
    for a in range(3):
        for b in range(3):
            if (a, b) in ((cm.CORFO, cm.MIPYME), (cm.MIPYME, cm.CORFO)):
                continue           # esas dos las completa el cierre
            assert M[a, b] == pytest.approx(M4[cm.IDX[a], cm.IDX[b]])


def test_primitivos_el_par_del_mandato_queda_vacio_al_restringir():
    """Sin cierre el par que el contrafactual habilita no tiene creencias.

    Es el supuesto que hubo que agregar, y la razón por la que la corrida se
    detuvo a consultarlo antes de elegir.
    """
    M4 = cc.construir_M(RW, RS, RM, DELTA, m_banco=0.0, v_corfo=V)
    M3_crudo = M4[np.ix_(cm.IDX, cm.IDX)]
    assert M3_crudo[cm.CORFO, cm.MIPYME] == 0.0
    assert M3_crudo[cm.MIPYME, cm.CORFO] == 0.0
    assert M3_crudo[cm.CORFO, cm.MIPYME] + M3_crudo[cm.MIPYME, cm.CORFO] == 0.0


def test_primitivos_cierre_mayorista_da_excedente_positivo(M):
    """Con el cierre adoptado el par sí tiene ganancia de intercambio."""
    assert M[cm.CORFO, cm.MIPYME] == pytest.approx(RW + V)
    assert M[cm.MIPYME, cm.CORFO] == pytest.approx((RM - RW) + DELTA)
    assert M[cm.CORFO, cm.MIPYME] + M[cm.MIPYME, cm.CORFO] > 0


def test_primitivos_el_cierre_casi_no_mueve_la_fraccion(sigmas):
    """Los dos cierres difieren en el nivel pero no en el cociente.

    Es lo que permitió acotar el riesgo del supuesto agregado: la fracción
    recuperada es un cociente donde el nivel de las tasas se cancela.
    """
    Ma = cm.construir_M3(RW, RS, RM, DELTA, V, cierre="mayorista")
    Mb = cm.construir_M3(RW, RS, RM, DELTA, V, cierre="subsidiada")
    for S in sigmas.values():
        fa = cm.fraccion_recuperada(S, Ma, cc.LAMBDA_A_PROPUESTO)["fraccion"]
        fb = cm.fraccion_recuperada(S, Mb, cc.LAMBDA_A_PROPUESTO)["fraccion"]
        assert abs(fa - fb) < 0.01


def test_primitivos_cierre_desconocido_se_rechaza():
    with pytest.raises(ValueError, match="cierre desconocido"):
        cm.construir_M3(RW, RS, RM, DELTA, V, cierre="inventado")


# ================================================ la identidad del capítulo 4

def test_identidad_cien_por_ciento_sin_friccion_ni_transparencia(M, sigmas):
    """Con tau = lambda_A = 0 la fracción es exactamente 100 por ciento.

    NO es un resultado. Bajo riesgo combinado con portafolio total fijo el
    bienestar depende sólo de w*, que no cambia entre escenarios, de modo que
    la recuperación completa es la identidad que el capítulo 4 documenta. Esta
    prueba existe para que el capítulo no la cite como hallazgo.
    """
    for S in sigmas.values():
        r = cm.fraccion_recuperada(S, M, lambda_A=0.0, tau=0.0)
        assert r["fraccion"] == pytest.approx(1.0, abs=1e-6)


def test_identidad_es_independiente_de_los_pares_habilitados(M, sigmas):
    """La identidad vale cualquiera sea el conjunto de pares de la red A.

    Es la razón por la que el bienestar recuperado NO es atribuible por par, y
    por la que la descomposición reparte el déficit y no la recuperación.
    """
    S = sigmas["equicorrelada"]
    base = cm.fraccion_recuperada(S, M, 0.0, 0.0)["fraccion"]
    Sigma_otra = cm.sigmas3(sigma_corfo_mipyme=0.6)["equicorrelada"]
    otra = cm.fraccion_recuperada(Sigma_otra, M, 0.0, 0.0)["fraccion"]
    assert base == pytest.approx(otra, abs=1e-6) == pytest.approx(1.0, abs=1e-6)


def test_identidad_tau_positivo_rompe_la_interpretacion(M, sigmas):
    """Con tau > 0 la fracción supera 100 por ciento y deja de ser una fracción.

    La referencia sin restricción no contiene tau, de modo que el cociente
    mezcla deshacer la restricción con agregar un beneficio que la referencia
    nunca tuvo. Por eso el informe reporta el efecto de tau sobre el volumen.
    """
    for S in sigmas.values():
        r = cm.fraccion_recuperada(S, M, cc.LAMBDA_A_PROPUESTO, tau=0.20)
        assert r["fraccion"] > 1.0


def test_identidad_el_deficit_se_descompone_de_forma_exacta(M, sigmas):
    """La suma del déficit por par reproduce el déficit total.

    Es lo que hace legítima la descomposición de la entrega 3: la desviación
    respecto de la referencia es una suma sobre coordenadas, es decir pares.
    """
    S = sigmas["equicorrelada"]
    lam = cc.LAMBDA_A_PROPUESTO
    r = cm.fraccion_recuperada(S, M, lam, 0.0)
    d = cm.descomponer_por_par(r["W_A"], r["brecha"], lam, 0.0)
    assert sum(d["share_deficit"].values()) == pytest.approx(1.0)
    assert d["deficit_total"] == pytest.approx(r["fraccion"] - 1.0, rel=1e-3)


# ========================================================= topología

def test_topologia_la_red_alternativa_si_reconecta_el_par_del_mandato():
    """A diferencia del caso principal, aquí la capa de acoplamiento aplica.

    Con Psi^A = I el par que la restricción separa vuelve a estar disponible,
    que es el mecanismo de recuperación de la fuente. En el caso chileno de
    cuatro agentes ese par NO se restaura, según fija
    ``caballero_caso_chileno.diagnostico_topologia``.
    """
    assert cm.PAR_MANDATO in cm.PARES
    d4 = cc.diagnostico_topologia()
    assert d4["arista_bloqueada_se_restaura"] is False


def test_topologia_el_par_del_mandato_carga_el_volumen(M, sigmas):
    """El par que el mandato prohíbe concentra el movimiento hacia la red A."""
    for S in sigmas.values():
        r = cm.fraccion_recuperada(S, M, cc.LAMBDA_A_PROPUESTO, 0.0)
        d = cm.descomponer_por_par(r["W_A"], r["brecha"], cc.LAMBDA_A_PROPUESTO, 0.0)
        clave = f"({cm.CORFO},{cm.MIPYME}) CORFO-MiPyME"
        assert d["share_volumen"][clave] > 0.60
        assert d["share_volumen"][clave] == max(d["share_volumen"].values())


# ============================================================ cifras del informe

def test_cifras_costo_del_mandato(M, sigmas):
    """El costo del mandato en los valores centrales de cada intervalo."""
    esperado = {"equicorrelada": 0.499, "por ciclo": 0.547}
    for nom, S in sigmas.items():
        c = cm.costo_del_mandato(S, M)
        assert c["brecha_relativa"] == pytest.approx(esperado[nom], abs=2e-3)


def test_cifras_fraccion_recuperada_al_lambda_propuesto(M, sigmas):
    """Punto central de los intervalos, que es lo que alimenta la figura."""
    esperado = {"equicorrelada": 0.667, "por ciclo": 0.675}
    for nom, S in sigmas.items():
        r = cm.fraccion_recuperada(S, M, cc.LAMBDA_A_PROPUESTO, 0.0)
        assert r["fraccion"] == pytest.approx(esperado[nom], abs=2e-3)


def test_cifras_cuartiles_de_la_fraccion_sobre_la_rejilla(sigmas):
    """Fija los CUARTILES, no sólo la mediana, de la tabla de la Sección 4.1.

    La versión anterior fijaba sólo la mediana, de modo que pasaba con los
    cuartiles mal transcritos en el informe: el límite superior publicado para
    la matriz equicorrelada era el verdadero de la de ciclo, un corrimiento de
    fila al copiar. Fijarlos aquí impide que vuelva a ocurrir.

    Se recorre la misma rejilla que la corrida pero sólo en las dimensiones que
    la tabla agrega, es decir con lambda_A y tau fijos, de manera que son 16
    combinaciones por matriz y no hace falta regenerar el barrido completo.
    """
    esperado = {"equicorrelada": (0.654, 0.665, 0.679),
                "por ciclo": (0.665, 0.674, 0.685)}
    for nom, S in sigmas.items():
        fracs = []
        for r_w in cm.REJILLA["r_w"]:
            for r_m in cm.REJILLA["r_m"]:
                for r_s in cm.REJILLA["r_s"]:
                    if cc.margen_ifnb(r_s, r_w) <= 0:
                        continue
                    for delta in [0.0, cc.subsidio(r_s, r_m)]:
                        for v in cm.REJILLA["v_corfo"]:
                            Mx = cm.construir_M3(r_w, r_s, r_m, delta, v)
                            fracs.append(cm.fraccion_recuperada(
                                S, Mx, cc.LAMBDA_A_PROPUESTO, 0.0)["fraccion"])
        assert len(fracs) == 16, f"{nom}: se esperaban 16 combinaciones admisibles"
        q1, med, q3 = np.percentile(fracs, [25, 50, 75])
        e1, emed, e3 = esperado[nom]
        assert q1 == pytest.approx(e1, abs=1e-3)
        assert med == pytest.approx(emed, abs=1e-3)
        assert q3 == pytest.approx(e3, abs=1e-3)
        assert (q3 - q1) < 0.03, f"{nom}: el IQR dejó de ser de ~3 puntos"


def test_cifras_friccion_de_equilibrio(M, sigmas):
    """Fija las DOS fricciones de equilibrio, que no son una sola.

    El informe las reportaba como un único valor aproximado de 0,79, que es sólo
    el de la matriz equicorrelada.
    """
    esperado = {"equicorrelada": 0.787, "por ciclo": 0.810}
    for nom, S in sigmas.items():
        lam = cm.friccion_de_equilibrio(S, M)
        assert lam == pytest.approx(esperado[nom], abs=1e-3)
        assert cm.fraccion_recuperada(S, M, lam - 0.05, 0.0)["fraccion"] > 0
        assert cm.fraccion_recuperada(S, M, lam + 0.05, 0.0)["fraccion"] < 0
        # El barrido de la corrida sólo la encierra entre estos dos nodos.
        assert 0.5 < lam < 1.0


def test_cifras_mayor_covarianza_cuesta_mas_y_recupera_menos(M):
    """Las dos monotonías que sostienen los paneles (b) y (c) del informe."""
    valores = [-0.30, 0.0, 0.30, 0.60]
    for nom in ("equicorrelada", "por ciclo"):
        costos, fracs = [], []
        for v in valores:
            S = cm.sigmas3(sigma_corfo_mipyme=v)[nom]
            assert cc.es_definida_positiva(S)
            r = cm.fraccion_recuperada(S, M, cc.LAMBDA_A_PROPUESTO, 0.0)
            costos.append(r["brecha_relativa"]); fracs.append(r["fraccion"])
        assert costos == sorted(costos), f"{nom}: el costo no es monótono creciente"
        assert fracs[-1] < fracs[0], f"{nom}: la fracción no decae"
