"""Corrida de la rama contrafactual del mandato de segundo piso.

CORRIDA BAJO SUPUESTOS DECLARADOS. No es la calibración del objetivo
específico 3. Los supuestos agregados y quién los decidió están en el
encabezado de ``caballero_contrafactual_mandato``.

Ejecutar desde la raíz del repositorio, sin variables de entorno

    python code/caballero/caballero_corrida_contrafactual.py

Produce ``caballero_contrafactual_resultados.csv`` en el directorio desde el
que se ejecuta. **No produce la figura**: los cuatro paneles de
``caballero_contrafactual_sensibilidad.png`` se construyeron fuera de este
script y por lo tanto no son reproducibles desde el repositorio, igual que las
demás figuras del documento. Las cifras que la figura muestra sí salen del CSV,
salvo las dos fricciones de equilibrio, que vienen de
``caballero_contrafactual_mandato.friccion_de_equilibrio``.

Versionar este archivo no compromete al autor con sus cifras. Siguen abiertas la
decisión sobre la restricción de balance y el costo de originación minorista de
CORFO que el cierre adoptado fija en cero; si cualquiera de las dos cambia, hay
que rehacer esta corrida. Lo que el versionado hace es volver auditable lo que el
documento ya afirma, no declararlo definitivo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import caballero_caso_chileno as cc  # noqa: E402
import caballero_contrafactual_mandato as cm  # noqa: E402


def bloque(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)




def main():
    filas = []
    SIG = cm.sigmas3()

    # ====================================================== 0. la rejilla
    bloque("0. Rejilla tras eliminar al banco")
    n_comb = (len(cm.REJILLA["r_w"]) * len(cm.REJILLA["r_m"]) *
              len(cm.REJILLA["r_s"]) * 2 *                      # delta: 0 y s
              len(cm.REJILLA["v_corfo"]) * len(cm.REJILLA["lambda_A"]) *
              len(cm.REJILLA["tau"]) * len(SIG))
    print(f"  El caso principal recorre 768 combinaciones. Al eliminar al banco")
    print(f"  desaparece la dimensión de su creencia de retorno (m_banco), de modo")
    print(f"  que aquí quedan {n_comb}. No se reutilizan las 312 admisibles del caso")
    print(f"  principal, porque no son las mismas combinaciones.")

    # ============================================ 1a. costo del mandato
    bloque("1a. Costo del mandato de segundo piso (entrega 1, primera parte)")
    print("  Brecha entre el bienestar sin restricción y el de la red Tradicional")
    print("  sola con el par CORFO--MiPyME prohibido. No interviene la capa dual ni")
    print("  ningún parámetro no observable, de modo que es la cifra más robusta.")
    print(f"\n  {'Sigma':16s} {'U_ref':>9s} {'U_T':>9s} {'brecha':>9s} {'% de U_ref':>11s}")
    for nom, S in SIG.items():
        c = cm.costo_del_mandato(S, _M_central(0.02))
        print(f"  {nom:16s} {c['U_ref']:9.5f} {c['U_T']:9.5f} {c['brecha']:9.5f} "
              f"{100*c['brecha_relativa']:10.1f}%")

    # ======================================= 1b/4. fracción recuperada y tau
    bloque("1b. Fracción del costo que recupera la red Alternativa, con tau = 0")
    print(f"  Barrido de lambda_A sobre el intervalo declarado {cm.INTERVALO_LAMBDA},")
    print(f"  el mismo del caso principal.")
    print(f"\n  {'lambda_A':>9s} " + " ".join(f"{n:>16s}" for n in SIG))
    for lam in [0.0, 0.1, cc.LAMBDA_A_PROPUESTO, 0.5, 1.0, 2.0, 5.0]:
        cel = []
        for nom, S in SIG.items():
            r = cm.fraccion_recuperada(S, _M_central(0.02), lam, 0.0)
            filas.append(dict(bloque="lambda_A", sigma=nom, **_plano(r)))
            cel.append(f"{100*r['fraccion']:15.1f}%")
        marca = "   <- propuesto" if abs(lam - cc.LAMBDA_A_PROPUESTO) < 1e-9 else ""
        print(f"  {lam:9.3f} " + " ".join(cel) + marca)
    print("\n  En lambda_A = 0 da exactamente 100 %, y eso NO es un resultado: es la")
    print("  identidad algebraica del capítulo 4, dado que con portafolio total fijo")
    print("  el bienestar bajo riesgo combinado depende sólo de w*, que no cambia.")

    bloque("4. Efecto de tau, reportado sobre el volumen de la red Alternativa")
    print(f"  Intervalo declarado {cm.INTERVALO_TAU}, el mismo del caso principal.")
    print("  Se reporta el volumen y no la fracción recuperada, porque la referencia")
    print("  sin restricción no contiene tau y el cociente dejaría de ser")
    print("  interpretable, según se documenta en el encabezado del módulo.")
    print(f"\n  {'tau':>7s} " + " ".join(f"{n:>26s}" for n in SIG))
    print(f"  {'':7s} " + " ".join(f"{'vol A':>12s} {'frac (no usar)':>13s}" for _ in SIG))
    for tau in [0.0, 0.01, 0.025, 0.05, 0.10, 0.20]:
        cel = []
        for nom, S in SIG.items():
            r = cm.fraccion_recuperada(S, _M_central(0.02), cc.LAMBDA_A_PROPUESTO, tau)
            vol = sum(abs(r["W_A"][a, b]) for a, b in cm.PARES)
            filas.append(dict(bloque="tau", sigma=nom, volumen_A=float(vol), **_plano(r)))
            cel.append(f"{vol:12.5f} {100*r['fraccion']:12.0f}%")
        print(f"  {tau:7.3f} " + " ".join(cel))

    # ======================================= 2. barrido de Sigma[CORFO,MiPyME]
    bloque("2. Variación con Sigma[CORFO,MiPyME] (entrega 2)")
    print("  Es la única entrada que la estructura bipartita identifica, de modo que")
    print("  es el único resultado del proyecto que no descansa sobre un supuesto.")
    print(f"  Rango recorrido {cm.RANGO_SIGMA_CM}, verificando definición positiva.")
    print(f"\n  {'Sigma_CM':>9s} {'def.pos.':>9s} " +
          " ".join(f"{n+' brecha%':>18s} {n+' frac%':>16s}" for n in SIG))
    for v in np.linspace(*cm.RANGO_SIGMA_CM, 10):
        S3 = cm.sigmas3(sigma_corfo_mipyme=v)
        ok = all(cc.es_definida_positiva(S) for S in S3.values())
        if not ok:
            print(f"  {v:9.3f} {'no':>9s}")
            continue
        cel = []
        for nom, S in S3.items():
            r = cm.fraccion_recuperada(S, _M_central(0.02), cc.LAMBDA_A_PROPUESTO, 0.0)
            filas.append(dict(bloque="sigma_corfo_mipyme", sigma=nom,
                              sigma_cm=float(v), **_plano(r)))
            cel.append(f"{100*r['brecha_relativa']:17.1f}% {100*r['fraccion']:15.1f}%")
        print(f"  {v:9.3f} {'si':>9s} " + " ".join(cel))

    # =========================================== 3. descomposición por par
    bloque("3. Descomposición por par (entrega 3)")
    print("  Con Psi^A = I la red Alternativa habilita los tres pares, no sólo el del")
    print("  mandato, de modo que el cambio no es atribuible sólo a que CORFO preste")
    print("  directo. Hay que separar dos cosas que no se descomponen igual.")
    print("\n  El VOLUMEN de la red Alternativa sí se reparte por par.")
    print("  El BIENESTAR RECUPERADO no es atribuible por par en esta contabilidad,")
    print("  porque con tau = lambda_A = 0 vale 100 % cualquiera sea el conjunto de")
    print("  pares habilitados: no proviene de ningún par sino del portafolio fijo.")
    print("  Lo que sí se descompone de forma exacta es el DÉFICIT respecto de ese")
    print("  100 %, que es lo que se reporta.")
    for nom, S in SIG.items():
        r = cm.fraccion_recuperada(S, _M_central(0.02), cc.LAMBDA_A_PROPUESTO, 0.0)
        d = cm.descomponer_por_par(r["W_A"], r["brecha"], cc.LAMBDA_A_PROPUESTO, 0.0)
        print(f"\n  {nom} (fracción recuperada {100*r['fraccion']:.1f} %)")
        print(f"    {'coordenada':28s} {'W^A':>10s} {'% volumen':>10s} {'% déficit':>10s}")
        for k in d["deficit_fraccion"]:
            wa = d["volumen"].get(k)
            sv = d["share_volumen"].get(k)
            print(f"    {k:28s} "
                  f"{(f'{wa:+10.5f}' if wa is not None else f'{chr(8212):>10s}')} "
                  f"{(f'{100*sv:9.1f}%' if sv is not None else f'{chr(8212):>10s}')} "
                  f"{100*d['share_deficit'][k]:9.1f}%")
            filas.append(dict(bloque="descomposicion", sigma=nom, par=k,
                              W_A_par=wa, share_volumen=sv,
                              share_deficit=d["share_deficit"][k],
                              fraccion=r["fraccion"]))
        print(f"    {'suma':28s} {chr(8212):>10s} {100*sum(d['share_volumen'].values()):9.1f}% "
              f"{100*sum(d['share_deficit'].values()):9.1f}%")
        print("    La posición propia es la diagonal de W^A, que no es un contrato con")
        print("    nadie; se deja como fila aparte en vez de repartirla entre los pares.")

    # ============================================= 5. rango sobre la rejilla
    bloque("5. Rango sobre la rejilla, combinaciones admisibles (entrega 1 y 5)")
    adm = _rejilla(filas, SIG)
    return pd.DataFrame(filas), adm


def _M_central(v_corfo):
    rw, rm_, rs = np.mean(cc.R_W), np.mean(cc.R_M), np.mean(cc.R_S_DECLARADO)
    return cm.construir_M3(rw, rs, rm_, cc.subsidio(rs, rm_) / 2, v_corfo)


def _plano(r):
    """Aplana el resultado dejando fuera las matrices, para el CSV."""
    return {k: (float(v) if np.isscalar(v) else v)
            for k, v in r.items() if k not in ("W_A", "W_T_dual", "P_eff")}


def _rejilla(filas, SIG):
    reg = []
    for r_w in cm.REJILLA["r_w"]:
        for r_m in cm.REJILLA["r_m"]:
            for r_s in cm.REJILLA["r_s"]:
                s = cc.subsidio(r_s, r_m)
                for delta in [0.0, s]:
                    for v_corfo in cm.REJILLA["v_corfo"]:
                        for lam in cm.REJILLA["lambda_A"]:
                            for tau in cm.REJILLA["tau"]:
                                M = cm.construir_M3(r_w, r_s, r_m, delta, v_corfo)
                                for nom, S in SIG.items():
                                    try:
                                        r = cm.fraccion_recuperada(S, M, lam, tau)
                                    except Exception:
                                        continue
                                    reg.append(dict(
                                        bloque="rejilla", sigma=nom, r_w=r_w,
                                        r_m=r_m, r_s=r_s, delta=delta,
                                        v_corfo=v_corfo,
                                        margen_ifnb=cc.margen_ifnb(r_s, r_w),
                                        **_plano(r)))
    df = pd.DataFrame(reg)
    filas.extend(reg)

    adm = df[df["margen_ifnb"] > 0]
    print(f"  Combinaciones evaluadas .......................... {len(df)}")
    print(f"  Admisibles, margen del IFNB positivo ............. {len(adm)}")
    print(f"  (el caso principal tiene 312 admisibles de 768; no son las mismas)")

    sub = adm[adm["tau"] == 0.0]
    print(f"\n  Costo del mandato, como % del bienestar de referencia")
    print(f"  {'Sigma':16s} {'mediana':>9s} {'IQR':>17s} {'min-max':>17s}")
    for nom in SIG:
        v = 100 * adm[adm["sigma"] == nom]["brecha_relativa"]
        print(f"  {nom:16s} {v.median():8.1f}% {v.quantile(.25):7.1f}-{v.quantile(.75):<8.1f}% "
              f"{v.min():7.1f}-{v.max():<8.1f}%")

    print(f"\n  Fracción recuperada con tau = 0, DESAGREGADA por lambda_A")
    print("  No se agrupa sobre lambda_A porque la rejilla sólo tiene tres valores")
    print("  discretos y es el factor dominante, de modo que los cuartiles de la")
    print("  mezcla no serían interpretables.")
    print(f"\n  {'lambda_A':>9s} {'Sigma':16s} {'n':>4s} {'mediana':>9s} {'IQR':>17s} "
          f"{'min-max':>17s}")
    for lam in sorted(sub["lambda_A"].unique()):
        for nom in SIG:
            v = 100 * sub[(sub["sigma"] == nom) & (sub["lambda_A"] == lam)]["fraccion"]
            print(f"  {lam:9.3f} {nom:16s} {len(v):4d} {v.median():8.1f}% "
                  f"{v.quantile(.25):7.1f}-{v.quantile(.75):<8.1f}% "
                  f"{v.min():7.1f}-{v.max():<8.1f}%")
    return adm


if __name__ == "__main__":
    df, adm = main()
    df.to_csv("caballero_contrafactual_resultados.csv", index=False)
    print(f"\nEscrito caballero_contrafactual_resultados.csv con {len(df)} filas.")
