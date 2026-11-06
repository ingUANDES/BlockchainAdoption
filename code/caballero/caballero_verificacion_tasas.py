"""Recálculo independiente de las tasas que declara el Capítulo 3.

Verificación numérica, no corrida del modelo. Recalcula desde los datos y sólo
al final compara contra lo que el documento afirma, para que el cálculo no se
ajuste a la cifra buscada.

===========================================================================
ALCANCE: QUÉ SE VERIFICA CONTRA LA FUENTE Y QUÉ NO
===========================================================================

Esta distinción es la más importante del archivo y conviene no perderla.

El Capítulo 3 declara que las tasas salen de los microdatos completos de la
ELE-7 (``ele7-full.csv``, 6.592 empresas), tomando la variable ``B066``
segmentada por ``TAMANO`` y anualizando según ``B067``/``B068``. **Ese archivo
no está disponible aquí.** Lo que hay es un extracto ya procesado,
``ele7_financiamiento_mipyme.csv``, cuya columna ``tasa_anual_pct`` viene
anualizada de antemano y que no conserva ``B066``.

En consecuencia:

  SÍ se verifica   la agregación, es decir que las medianas y los tamaños de
                   muestra publicados son los que salen de esos 1.252 casos.
  SÍ se infiere    la regla de anualización, por vía forense, invirtiendo los
                   705 casos declarados mensuales y viendo cuál de las dos
                   reglas candidatas produce tasas mensuales redondas.
  NO se verifica   que esos 1.252 casos sean los correctos, es decir que la
                   lectura y el filtrado de ``B066`` desde el microdato
                   original hayan sido los adecuados. Eso exige el archivo
                   original.

===========================================================================
DECISIONES PROPIAS, QUE EL DOCUMENTO NO FIJA
===========================================================================

1. Periodicidad ambigua. Hay 44 casos con ``tasa_declarada_mensual`` y
   ``tasa_declarada_anual`` ambas en cero, de modo que la encuesta no dice en
   qué base vienen. El extracto los trata como anuales. Se reporta cada
   mediana con y sin ellos.
2. Mediana ponderada. El capítulo no declara si pondera por el factor de
   expansión. Las cifras publicadas corresponden a la mediana SIN ponderar; se
   reportan ambas, porque el otro análisis de la misma encuesta en esta memoria
   (``caballero_ele7_canales.py``) sí pondera.
3. Atípicos. No se recorta ninguno. Las medianas son robustas y recortar sería
   introducir un criterio que el documento no tiene.

Ejecutar desde la raíz del repositorio

    python code/caballero/caballero_verificacion_tasas.py <ruta>/ele7_financiamiento_mipyme.csv
"""

import sys

import numpy as np
import pandas as pd

#: Orden de tramos y cifras que el Capítulo 3 declara en tab:ele7-tasas.
DECLARADO = {
    "Grande":    (10.43, 574),
    "Mediana":   (12.01, 225),
    "Pequeña 2": (12.68, 150),
    "Pequeña 1": (15.39, 151),
    "Micro":     (14.00, 152),
}
TOTAL_DECLARADO = 1252

#: Intervalos que el capítulo adopta, en decimal anual.
#: R_M se actualizó a la mediana PONDERADA de Micro y Pequeña 1 el 06-nov-2026;
#: el valor sin ponderar era (0.1400, 0.1540).
R_M_PONDERADO = (0.126825, 0.140286)
R_W_DECLARADO = (0.0423, 0.0473)   # mayorista, DIPRES, no recalculable
R_S_DECLARADO = (0.03, 0.06)       # subsidiada, supuesto de trabajo
R_M_DECLARADO = (0.14, 0.154)      # mercado, de la encuesta
DESCUENTO_PP = (7.0, 10.0)         # puntos porcentuales, DIPRES

#: Fricción de acceso que propone el módulo del caso.
RECHAZO_MIPYME = 0.15
AUTOCENSURA_NUM, AUTOCENSURA_DEN = 111, 1018
LAMBDA_A_DECLARADO = 0.2427


def mediana_ponderada(valores, pesos, conv="corte"):
    """Mediana ponderada, en las dos convenciones que dan resultados distintos.

    ``corte``  el primer valor cuyo peso acumulado alcanza el 50 por ciento.
    ``interp`` interpolación lineal sobre el peso acumulado centrado.

    La convención importa más de lo normal con estos datos: la distribución
    tiene puntos de masa, de modo que las dos convenciones pueden caer en
    valores distintos. Ver ``puntos_de_masa``. La adoptada por defecto es
    ``corte``, por ser la que no inventa valores que nadie declaró.
    """
    orden = np.argsort(valores)
    v, w = np.asarray(valores)[orden], np.asarray(pesos)[orden]
    if conv == "corte":
        return float(v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())])
    if conv == "interp":
        return float(np.interp(0.5, (np.cumsum(w) - 0.5 * w) / w.sum(), v))
    raise ValueError(f"convención desconocida: {conv!r}")


def puntos_de_masa(t, k=6):
    """Valores más frecuentes de la tasa anual, con su mensual implícita.

    Las empresas declaran tasas mensuales redondas que al capitalizarse
    producen valores anuales repetidos. Por eso las medianas no se apoyan en
    una distribución suave sino en puntos de masa, y un intervalo que parece
    medido a dos decimales en realidad está eligiendo entre puntos contiguos.
    """
    vc = t["tasa_anual_pct"].round(4).value_counts().head(k)
    return [dict(anual=float(a), n=int(n), cuota_pct=100 * n / len(t),
                 mensual_implicita=float(((1 + a / 100) ** (1 / 12) - 1) * 100))
            for a, n in vc.items()]


def cargar(ruta):
    d = pd.read_csv(ruta)
    return d[d["tasa_anual_pct"].notna()].copy(), len(d)


def medianas(t):
    """Mediana por tramo, con y sin ponderar, y con y sin los casos ambiguos."""
    ambiguos = t[(t["tasa_declarada_mensual"] == 0) & (t["tasa_declarada_anual"] == 0)]
    filas = []
    for tramo in DECLARADO:
        s = t[t["tamano"] == tramo]
        sin_amb = s.drop(ambiguos.index, errors="ignore")
        med_dec, n_dec = DECLARADO[tramo]
        filas.append(dict(
            tramo=tramo, n=len(s), n_sin_ambiguos=len(sin_amb),
            ponderada=mediana_ponderada(s["tasa_anual_pct"].values,
                                        s["fe_transversal"].values),
            ponderada_interp=mediana_ponderada(s["tasa_anual_pct"].values,
                                               s["fe_transversal"].values, "interp"),
            ponderada_sin_amb=mediana_ponderada(sin_amb["tasa_anual_pct"].values,
                                                sin_amb["fe_transversal"].values),
            sin_ponderar=float(s["tasa_anual_pct"].median()),
            mediana_declarada=med_dec, n_declarado=n_dec,
        ))
    return pd.DataFrame(filas), ambiguos


def regla_de_anualizacion(t):
    """Infiere si la anualización fue compuesta o simple, sin el microdato.

    Invierte los casos declarados mensuales con ambas reglas candidatas y
    cuenta cuántos devuelven una tasa mensual exacta a dos decimales. La regla
    verdadera es la que produce valores redondos, dado que son los que una
    empresa declara.
    """
    a = t[t["tasa_declarada_mensual"] == 1]["tasa_anual_pct"].values
    compuesta = ((1 + a / 100) ** (1 / 12) - 1) * 100
    simple = a / 12
    exactos = {nombre: int((np.abs(r - np.round(r, 2)) < 1e-6).sum())
               for nombre, r in [("compuesta", compuesta), ("simple", simple)]}
    return dict(n=len(a), exactos_2_decimales=exactos,
                regla=max(exactos, key=exactos.get))


def benchmark_implicito():
    """Qué benchmark reproduce exactamente el intervalo subsidiado declarado.

    El capítulo relaciona la tasa subsidiada con un descuento de 7 a 10 puntos
    respecto de un benchmark de mercado, pero no reporta el nivel de ese
    benchmark. Aplicar el descuento al intervalo 14 a 15,4 da 4 a 8,4, que es
    lo que el propio capítulo deriva y después no adopta. Resolver en cambio
    cuál benchmark PUNTUAL produce el 3 a 6 adoptado da un valor único.
    """
    b_desde_inferior = R_S_DECLARADO[0] * 100 + DESCUENTO_PP[1]
    b_desde_superior = R_S_DECLARADO[1] * 100 + DESCUENTO_PP[0]
    return dict(
        desde_limite_inferior=b_desde_inferior,
        desde_limite_superior=b_desde_superior,
        unico=abs(b_desde_inferior - b_desde_superior) < 1e-9,
        benchmark=b_desde_inferior,
        aplicado_al_intervalo=(R_M_DECLARADO[0] * 100 - DESCUENTO_PP[1],
                               R_M_DECLARADO[1] * 100 - DESCUENTO_PP[0]),
        desde_mediana_agrupada=(12.6825 - DESCUENTO_PP[1], 12.6825 - DESCUENTO_PP[0]),
    )


def friccion():
    """Recalcula la fricción de acceso como probabilidad de no acceder."""
    a = AUTOCENSURA_NUM / AUTOCENSURA_DEN
    return dict(rechazo=RECHAZO_MIPYME, autocensura=a,
                disjuntas=RECHAZO_MIPYME + a * (1 - RECHAZO_MIPYME),
                suma_simple=RECHAZO_MIPYME + a)


def margen_y_admisibilidad(n_muestra=20_000_000, semilla=0):
    """Margen del intermediario y fracción del rectángulo con margen positivo.

    La fracción continua se obtiene por simulación sobre el rectángulo con
    distribución uniforme, que es el supuesto implícito de tratar cada
    intervalo como un rango sin información adicional. Se reporta aparte la
    fracción de la REJILLA, que es otra cosa: la rejilla usa sólo los dos
    extremos de cada intervalo y por eso da exactamente la mitad por
    construcción, con independencia de la geometría real.
    """
    rng = np.random.default_rng(semilla)
    r_s = rng.uniform(*R_S_DECLARADO, n_muestra)
    r_w = rng.uniform(*R_W_DECLARADO, n_muestra)
    return dict(
        margen_min_pp=100 * (R_S_DECLARADO[0] - R_W_DECLARADO[1]),
        margen_max_pp=100 * (R_S_DECLARADO[1] - R_W_DECLARADO[0]),
        fraccion_continua_pct=100 * float((r_s - r_w > 0).mean()),
        fraccion_rejilla_pct=50.0,
    )


def main(ruta):
    t, n_total = cargar(ruta)
    tabla, ambiguos = medianas(t)
    print("=" * 78)
    print("1. MEDIANAS POR TRAMO, PONDERADAS  (contra el extracto, no el microdato)")
    print("=" * 78)
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  casos con tasa: {len(t)} de {n_total} (declarado {TOTAL_DECLARADO}); "
          f"sin dato: {n_total - len(t)}")
    print(f"  periodicidad ambigua: {len(ambiguos)} casos, tratados como anuales")

    print("\n  Puntos de masa de la distribución, que es sobre lo que caen las medianas:")
    for pm in puntos_de_masa(t):
        print(f"    {pm['anual']:9.4f} % anual  n={pm['n']:4d} ({pm['cuota_pct']:4.1f} %)"
              f"   <- {pm['mensual_implicita']:.4f} % mensual capitalizado")
    print("  Por eso la precisión aparente de dos decimales es engañosa: las medianas")
    print("  saltan entre puntos de masa contiguos, no se mueven de forma continua.")

    r = regla_de_anualizacion(t)
    print("\n" + "=" * 78)
    print("2. REGLA DE ANUALIZACIÓN, inferida por inversión")
    print("=" * 78)
    print(f"  sobre {r['n']} casos declarados mensuales, tasas mensuales exactas a 2 decimales:")
    for nombre, cuenta in r["exactos_2_decimales"].items():
        print(f"    {nombre:10s} {cuenta:4d}")
    print(f"  regla inferida: {r['regla'].upper()}")

    b = benchmark_implicito()
    print("\n" + "=" * 78)
    print("3. EL INTERVALO SUBSIDIADO 3 A 6")
    print("=" * 78)
    print(f"  descuento aplicado al intervalo de mercado -> "
          f"{b['aplicado_al_intervalo'][0]:.1f} a {b['aplicado_al_intervalo'][1]:.1f} %")
    print(f"  descuento sobre la mediana MiPyME AGRUPADA -> "
          f"{b['desde_mediana_agrupada'][0]:.2f} a {b['desde_mediana_agrupada'][1]:.2f} % "
          "  <- la derivación correcta, deja 0,32 pp de residuo")
    print(f"  benchmark puntual que reproduce 3 a 6      -> {b['benchmark']:.1f} % "
          f"(único: {b['unico']})")

    f = friccion()
    print("\n" + "=" * 78)
    print("4. FRICCIÓN DE ACCESO")
    print("=" * 78)
    print(f"  rechazo {f['rechazo']:.4f} + autocensura {f['autocensura']:.6f} sobre el resto")
    print(f"  disjuntas   {f['disjuntas']:.6f}   (declarado {LAMBDA_A_DECLARADO})")
    print(f"  suma simple {f['suma_simple']:.6f}")

    m = margen_y_admisibilidad()
    print("\n" + "=" * 78)
    print("5. CONSECUENCIAS ARITMÉTICAS")
    print("=" * 78)
    print(f"  margen del intermediario: {m['margen_min_pp']:+.2f} a {m['margen_max_pp']:+.2f} pp")
    print(f"  fracción continua con margen positivo: {m['fraccion_continua_pct']:.3f} %")
    print(f"  fracción de la rejilla: {m['fraccion_rejilla_pct']:.1f} %, "
          "exacta por construcción al usar sólo los extremos")
    return tabla


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("uso: caballero_verificacion_tasas.py <ruta>/ele7_financiamiento_mipyme.csv")
    main(sys.argv[1])
