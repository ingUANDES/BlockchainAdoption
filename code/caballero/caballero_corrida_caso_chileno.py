"""Corrida del caso chileno bajo supuestos declarados, con sus barridos.

CORRIDA BAJO SUPUESTOS DECLARADOS. No es la calibración del objetivo
específico 3, que sigue bloqueada por el resultado de identificabilidad del
capítulo 3. Toda cifra que este script imprime es condicional a los supuestos
que ``caballero_caso_chileno`` documenta, y ninguna proviene de contrastar el
modelo con una red observada.

Ejecutar desde la raíz del repositorio, sin variables de entorno

    python code/caballero/caballero_corrida_caso_chileno.py

Produce ``caballero_caso_chileno_resultados.csv`` con el detalle de los barridos,
en el directorio desde el que se ejecuta.

CUÁL COLUMNA DEFINE LA ADMISIBILIDAD. El CSV trae dos columnas que parecen
significar lo mismo y no lo significan, de modo que conviene decir cuál manda
antes de que alguien recuente sobre la equivocada.

``arista_A_negativa``  es la que manda. Marca SÓLO la arista de la red
                       Alternativa, MiPyME--Banco. Las 312 combinaciones
                       admisibles del bloque ``rango`` son las que cumplen
                       ``margen_ifnb > 0`` y ``arista_A_negativa == False``:
                       384 tienen margen positivo y de ésas 312 tienen además la
                       arista A no negativa.
``contrato_negativo``  es diagnóstica y NO define admisibilidad. Lista cualquier
                       arista con contrato negativo, incluida CORFO--IFNB, que es
                       levemente negativa en casi toda la rejilla por el
                       excedente conjunto nulo de esa arista. Filtrar por ella
                       deja 456 filas, cifra que no corresponde a ninguna
                       magnitud reportada. Se conserva porque distingue qué
                       arista es la negativa, que es información que
                       ``arista_A_negativa`` no da.

Que el complemento de ``contrato_negativo`` dé también 312 es coincidencia
numérica y no la procedencia de la cifra. **No produce la figura**: los cuatro
paneles de ``caballero_caso_chileno_sensibilidad.png`` se construyeron fuera de
este script y por lo tanto no son reproducibles desde el repositorio, igual que
las demás figuras del documento. Las cifras que la figura muestra sí salen del
CSV.

Versionar este archivo no compromete al autor con sus cifras. Sigue abierta la
decisión de imponer una restricción de balance que ate lo que el intermediario
coloca a lo que se fondea, que es la vía estructural discutida en el docstring de
``caballero_caso_chileno.construir_M``; si se adopta, hay que rehacer esta
corrida y con ella las cifras de la Sección 4.5. Lo que el versionado hace es
volver auditable lo que el documento ya afirma, no declararlo definitivo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import caballero_caso_chileno as cc  # noqa: E402

GAMMA = np.eye(cc.N)


def bloque(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


# ------------------------------------------------------------------ escenario
def primitivos(r_w, r_s, r_m, delta, m_banco, v_corfo):
    return cc.construir_M(r_w, r_s, r_m, delta, m_banco, v_corfo)


def corrida(Sigma, M, lambda_A, tau):
    """Un punto estable con todo lo que interesa reportar de él."""
    r = cc.resolver_caso(Sigma, M, GAMMA, lambda_A, tau)
    d = cc.descomponer(r["W"])
    u = cc.utilidades(r["W"], r["P"], Sigma, M, GAMMA, lambda_A, tau)
    return dict(
        W_corfo_ifnb=float(r["W"][cc.CORFO, cc.IFNB]),
        W_ifnb_mipyme=float(r["W"][cc.IFNB, cc.MIPYME]),
        W_mipyme_banco=float(r["W"][cc.MIPYME, cc.BANCO]),
        volumen_T=d["volumen_T"], volumen_A=d["volumen_A"],
        adopcion_A=d["adopcion_A"],
        contrato_negativo=";".join(d["contrato_negativo"]),
        # La adopción sólo es interpretable como uso del canal si la arista de
        # la red Alternativa es no negativa. Un W negativo en CORFO--IFNB es
        # otro problema, y no invalida la lectura de la adopción.
        arista_A_negativa=bool(r["W"][cc.MIPYME, cc.BANCO] < -1e-9),
        u_corfo=float(u[cc.CORFO]), u_ifnb=float(u[cc.IFNB]),
        u_mipyme=float(u[cc.MIPYME]), u_banco=float(u[cc.BANCO]),
        utilidad_total=float(u.sum()), eta=r["eta"], asimetria=r["asimetria"],
    )


# --------------------------------------------------------- valores de partida
R_W0, R_M0 = float(np.mean(cc.R_W)), float(np.mean(cc.R_M))
R_S0 = float(np.mean(cc.R_S_DECLARADO))
S0 = cc.subsidio(R_S0, R_M0)
DELTA0, M_BANCO0, V_CORFO0 = S0 / 2, 0.02, 0.02
LAMBDA0, TAU0 = cc.LAMBDA_A_PROPUESTO, 0.0

SIGMAS = {"neutra": cc.sigma_neutra(), "economica": cc.sigma_economica()}


def main():
    filas = []

    # ============================================================= 1. topología
    bloque("1. ¿Corre la capa de acoplamiento sobre esta topología?")
    d = cc.diagnostico_topologia()
    print("Contrapartes admisibles de cada agente en cada red")
    for ag, v in d["soportes"].items():
        print(f"  {ag:7s} T={str(v['en_T']):22s} A={str(v['en_A']):12s} "
              f"comparten={v['comparten'] or 'ninguna'}")
    print(f"\n  soportes disjuntos ............... {d['soportes_disjuntos']}")
    print(f"  el reparto es una decisión ....... {d['reparto_es_decision']}")
    print(f"  la red A restaura {d['arista_bloqueada']} .... "
          f"{d['arista_bloqueada_se_restaura']}")
    print(f"\n  resolver_acoplamiento fija Psi^A = I, de modo que admitiría en la")
    print(f"  red Alternativa {len(d['psi_A_implicito']['admite_resolver_acoplamiento'])} "
          f"pares cuando la red real tiene "
          f"{len(d['psi_A_implicito']['red_A_real'])}.")
    print(f"  Pares espurios: {', '.join(d['psi_A_implicito']['pares_espurios'])}")

    # ============================================= 2. primitivos y admisibilidad
    bloque("2. Primitivos y región admisible")
    print(f"  r_w = {cc.R_W}   mayorista CORFO->IFNB        [verificado, DIPRES]")
    print(f"  r_m = {cc.R_M}   mercado Banco->MiPyME        [verificado, ELE-7]")
    print(f"  r_s = {cc.R_S_DECLARADO}   declarado por el cap. 3")
    print(f"        {tuple(round(x, 4) for x in cc.r_s_derivado())}   "
          f"derivado de su propia aritmética (r_m menos 7-10 pp)")
    print("  Los dos intervalos NO coinciden. Se usa el declarado.")

    ra = cc.region_admisible()
    print(f"\n  Margen del IFNB = r_s - r_w, entre {ra['margen_min']:+.4f} y "
          f"{ra['margen_max']:+.4f}.")
    print(f"  Sólo el {100*ra['fraccion_admisible']:.1f} % del rectángulo (r_s, r_w) "
          f"deja margen positivo.")
    print("  En el resto el modelo predice que el IFNB no coloca y el programa no opera.")

    M0 = primitivos(R_W0, R_S0, R_M0, DELTA0, M_BANCO0, 0.0)
    print(f"\n  Excedente conjunto por arista con v_corfo = 0")
    for k, v in cc.excedente_conjunto(M0).items():
        print(f"    {k:16s} {v:+.4f}")
    print("  CORFO--IFNB da exactamente cero, porque el ingreso de CORFO y el costo")
    print("  del IFNB se cancelan, de modo que esa arista no tiene ganancia de")
    print("  intercambio que la sostenga. El contrato NO se anula por eso, ver el")
    print("  bloque 5: queda un residuo de cobertura cuyo signo lo fija Sigma.")

    # ================================================= 3. punto estable predicho
    bloque("3. Punto estable predicho (valores centrales de cada intervalo)")
    M = primitivos(R_W0, R_S0, R_M0, DELTA0, M_BANCO0, V_CORFO0)
    print(f"  lambda_A = {LAMBDA0:.4f} (propuesto), tau = {TAU0}, "
          f"delta = {DELTA0:.4f} = s/2, v_corfo = {V_CORFO0}, m_banco = {M_BANCO0}")
    print(f"\n  {'Sigma':10s} {'W(C-I)':>9s} {'W(I-M)':>9s} {'W(M-B)':>9s} "
          f"{'adopA':>7s} {'U_total':>9s}")
    base = {}
    for nom, S in SIGMAS.items():
        o = corrida(S, M, LAMBDA0, TAU0)
        base[nom] = o
        filas.append(dict(bloque="punto_estable", sigma=nom, **o))
        print(f"  {nom:10s} {o['W_corfo_ifnb']:9.4f} {o['W_ifnb_mipyme']:9.4f} "
              f"{o['W_mipyme_banco']:9.4f} {100*o['adopcion_A']:6.1f}% "
              f"{o['utilidad_total']:9.5f}")
    print("\n  Utilidad por agente")
    for nom in SIGMAS:
        o = base[nom]
        print(f"    {nom:10s} CORFO {o['u_corfo']:+.5f}  IFNB {o['u_ifnb']:+.5f}  "
              f"MiPyME {o['u_mipyme']:+.5f}  Banco {o['u_banco']:+.5f}")
    print("\n  Los niveles de contrato no tienen unidad interpretable, puesto que el")
    print("  modelo no fija escala de balance. Lo comparable son las razones entre")
    print("  aristas y la adopción, no los montos.")

    # ====================================================== 4. sensibilidad
    bloque("4. Sensibilidad")

    print("4a. Fricción de acceso lambda_A, adopción de la red Alternativa")
    print(f"  {'lambda_A':>9s} " + " ".join(f"{n:>12s}" for n in SIGMAS))
    for lam in [0.0, 0.1, LAMBDA0, 0.5, 1.0, 2.0, 5.0]:
        celdas = []
        for nom, S in SIGMAS.items():
            o = corrida(S, M, lam, TAU0)
            filas.append(dict(bloque="lambda_A", sigma=nom, lambda_A=lam, **o))
            celdas.append(f"{100*o['adopcion_A']:11.1f}%")
        marca = "  <- propuesto" if abs(lam - LAMBDA0) < 1e-9 else ""
        print(f"  {lam:9.3f} " + " ".join(celdas) + marca)

    print("\n4b. Beneficio de transparencia tau, adopción de la red Alternativa")
    print(f"  {'tau':>9s} " + " ".join(f"{n:>12s}" for n in SIGMAS))
    for tau in [0.0, 0.01, 0.025, 0.05, 0.10, 0.20]:
        celdas = []
        for nom, S in SIGMAS.items():
            o = corrida(S, M, LAMBDA0, tau)
            filas.append(dict(bloque="tau", sigma=nom, tau=tau, **o))
            celdas.append(f"{100*o['adopcion_A']:11.1f}%")
        print(f"  {tau:9.3f} " + " ".join(celdas))

    print("\n4c. Barrido de las dos covarianzas candidatas a dominar")
    print("    Base equicorrelada en 0,30; se mueve una entrada a la vez.")
    print(f"  {'valor':>7s} {'Sig[CORFO,MiPyME]':>19s} {'Sig[IFNB,Banco]':>17s}")
    for v in [-0.30, 0.0, 0.30, 0.45, 0.60, 0.75, 0.90]:
        cel = []
        for etiqueta, kw in [("corfo_mipyme", dict(corfo_mipyme=v)),
                             ("ifnb_banco", dict(ifnb_banco=v))]:
            S = _sigma_una_entrada(**kw)
            if not cc.es_definida_positiva(S):
                cel.append(f"{'no def.pos.':>17s}")
                continue
            o = corrida(S, M, LAMBDA0, TAU0)
            filas.append(dict(bloque=f"sigma_{etiqueta}", sigma=etiqueta,
                              sigma_valor=v, **o))
            neg = "*" if o["arista_A_negativa"] else " "
            cel.append(f"{100*o['adopcion_A']:16.1f}%{neg}")
        print(f"  {v:+7.2f} {cel[0]} {cel[1]}")
    print("  * el contrato MiPyME--Banco es negativo, de modo que la adopción no")
    print("    es interpretable como uso del canal.")

    _resumen_dominancia(filas)

    # ========================================================= 5. rango global
    bloque("5. Rango de resultados sobre todos los intervalos declarados")
    rango = _rango_global(filas)
    return pd.DataFrame(filas), rango


def _sigma_una_entrada(ifnb_banco=0.30, corfo_mipyme=0.30, base=0.30):
    S = np.full((cc.N, cc.N), base)
    np.fill_diagonal(S, 1.0)
    S[cc.IFNB, cc.BANCO] = S[cc.BANCO, cc.IFNB] = ifnb_banco
    S[cc.CORFO, cc.MIPYME] = S[cc.MIPYME, cc.CORFO] = corfo_mipyme
    return S


def _resumen_dominancia(filas):
    """Amplitud por entrada de Sigma, con y sin el filtro de arista negativa.

    El filtro descarta los puntos con contrato MiPyME--Banco negativo, que no es
    interpretable como uso del canal. Se reportan las DOS cifras y el conteo de
    puntos descartados, y ambas quedan escritas como filas de resumen en el CSV
    con ``bloque = "resumen_dominancia"``, para que el capítulo pueda declarar
    el filtro citando un archivo en vez de un cálculo al vuelo.
    """
    df = pd.DataFrame([f for f in filas if f["bloque"].startswith("sigma_")])
    out = {}
    for etiqueta in ["sigma_corfo_mipyme", "sigma_ifnb_banco"]:
        sub = df[df["bloque"] == etiqueta]
        keep = sub[~sub["arista_A_negativa"]]
        amp_f = 100 * (keep["adopcion_A"].max() - keep["adopcion_A"].min())
        amp_s = 100 * (sub["adopcion_A"].max() - sub["adopcion_A"].min())
        out[etiqueta] = amp_f
        filas.append(dict(
            bloque="resumen_dominancia", sigma=etiqueta,
            puntos_totales=len(sub), puntos_retenidos=len(keep),
            puntos_descartados=len(sub) - len(keep),
            amplitud_pp_filtrada=float(amp_f),
            amplitud_pp_sin_filtrar=float(amp_s),
        ))
    print(f"\n  Amplitud de la adopción al recorrer cada entrada, en puntos porcentuales")
    print(f"  {'entrada':22s} {'filtrada':>9s} {'sin filtrar':>12s} {'retenidos':>10s} "
          f"{'descartados':>12s}")
    for etiqueta, nombre in [("sigma_corfo_mipyme", "Sigma[CORFO, MiPyME]"),
                             ("sigma_ifnb_banco", "Sigma[IFNB,  Banco]")]:
        f = [r for r in filas if r["bloque"] == "resumen_dominancia"
             and r["sigma"] == etiqueta][0]
        print(f"  {nombre:22s} {f['amplitud_pp_filtrada']:8.1f} pp "
              f"{f['amplitud_pp_sin_filtrar']:9.1f} pp "
              f"{f['puntos_retenidos']:>4d}/{f['puntos_totales']:<4d} "
              f"{f['puntos_descartados']:>11d}")
    print("  La cifra publicada es la filtrada, que es la conservadora.")
    print("  En esta topología manda Sigma[IFNB, Banco], que mide cuán sustituibles")
    print("  son las dos contrapartes de la MiPyME, y no Sigma[CORFO, MiPyME], que es")
    print("  la que domina en la configuración de la fuente. La razón es estructural,")
    print("  dado que allá la red Alternativa reconecta el par que la restricción")
    print("  separa, y aquí agrega una contraparte distinta.")


def _rango_global(filas):
    """Recorre el producto de los intervalos declarados y resume el resultado."""
    rejilla = dict(
        r_w=list(cc.R_W), r_m=list(cc.R_M), r_s=list(cc.R_S_DECLARADO),
        delta=[0.0, None], m_banco=[0.005, 0.04], v_corfo=[0.0, 0.05],
        lambda_A=[0.0, LAMBDA0, 2.0], tau=[0.0, 0.05],
    )
    reg = []
    for r_w in rejilla["r_w"]:
        for r_m in rejilla["r_m"]:
            for r_s in rejilla["r_s"]:
                s = cc.subsidio(r_s, r_m)
                for delta in [0.0, s]:
                    for m_banco in rejilla["m_banco"]:
                        for v_corfo in rejilla["v_corfo"]:
                            for lam in rejilla["lambda_A"]:
                                for tau in rejilla["tau"]:
                                    for nom, S in SIGMAS.items():
                                        M = primitivos(r_w, r_s, r_m, delta,
                                                       m_banco, v_corfo)
                                        try:
                                            o = corrida(S, M, lam, tau)
                                        except Exception:
                                            continue
                                        o.update(bloque="rango", sigma=nom,
                                                 r_w=r_w, r_m=r_m, r_s=r_s,
                                                 delta=delta, m_banco=m_banco,
                                                 v_corfo=v_corfo,
                                                 lambda_A=lam, tau=tau,
                                                 margen_ifnb=cc.margen_ifnb(r_s, r_w))
                                        reg.append(o)
    df = pd.DataFrame(reg)
    filas.extend(reg)

    adm = df[(df["margen_ifnb"] > 0) & (~df["arista_A_negativa"])]
    print(f"  Combinaciones evaluadas .......................... {len(df)}")
    print(f"  Con margen del IFNB positivo y arista A no negativa  {len(adm)}")
    print(f"\n  {'magnitud':22s} {'min':>9s} {'p25':>9s} {'mediana':>9s} "
          f"{'p75':>9s} {'max':>9s}")
    for col, etq, esc in [("adopcion_A", "adopción red A (%)", 100),
                          ("W_ifnb_mipyme", "W IFNB--MiPyME", 1),
                          ("W_mipyme_banco", "W MiPyME--Banco", 1),
                          ("utilidad_total", "utilidad total", 1)]:
        q = adm[col].quantile([0, .25, .5, .75, 1]).values * esc
        print(f"  {etq:22s} " + " ".join(f"{x:9.4f}" for x in q))

    sinv = df[(df["v_corfo"] == 0.0)]
    print(f"\n  Arista mayorista con v_corfo = 0 ({len(sinv)} combinaciones)")
    print(f"    exactamente cero .......... {int((sinv['W_corfo_ifnb'].abs() < 1e-9).sum())}")
    print(f"    |W| mediana ............... {sinv['W_corfo_ifnb'].abs().median():.6f}")
    for nom in SIGMAS:
        v = sinv[sinv["sigma"] == nom]["W_corfo_ifnb"]
        print(f"    mediana con {nom:10s} .. {v.median():+.6f}")
    print("    El contrato no se anula; persiste un residuo de cobertura cuyo signo")
    print("    depende de Sigma y no de la economía del programa.")

    print(f"\n  Por supuesto de covarianza, adopción de la red Alternativa")
    for nom in SIGMAS:
        sub = adm[adm["sigma"] == nom]["adopcion_A"] * 100
        print(f"    {nom:10s} mediana {sub.median():5.1f} %   "
              f"rango intercuartil {sub.quantile(.25):5.1f} a {sub.quantile(.75):5.1f} %   "
              f"min-max {sub.min():5.1f} a {sub.max():5.1f} %")
    return adm


if __name__ == "__main__":
    df, adm = main()
    df.to_csv("caballero_caso_chileno_resultados.csv", index=False)
    print(f"\nEscrito caballero_caso_chileno_resultados.csv con {len(df)} filas.")
