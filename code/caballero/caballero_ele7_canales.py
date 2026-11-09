"""Reparto de canales de financiamiento de las MiPyMEs en la ELE-7.

Produce las cifras que cita la Seccion 4.5 del Capitulo 4 de la memoria, a saber
la tabla de uso de canales por tramo de tamano y las tres cifras derivadas que el
texto comenta.

Entrada
-------
``ele7_financiamiento_mipyme.csv``, extracto de los microdatos de la Septima
Encuesta Longitudinal de Empresas con una fila por empresa, las columnas
``fuente_*`` como indicadores de uso de cada fuente de financiamiento, la columna
``tamano`` con el tramo y la columna ``fe_transversal`` con el factor de
expansion. Los microdatos originales no se versionan en el repositorio por su
tamano y se obtienen del sitio de la encuesta.

Convenciones
------------
Todas las cifras van ponderadas por ``fe_transversal`` y restringidas a
``tamano != 'Grande'``. El canal del programa agrupa la fuente de instituciones
financieras no bancarias y la estatal, puesto que son las dos vias por las que
opera un programa del tipo Credito Corfo Mipyme. Queda pendiente verificar
contra el cuestionario si la fuente estatal corresponde a credito o incluye
subsidios.
"""

import pandas as pd

TRAMOS = ["Micro", "Pequeña 1", "Pequeña 2", "Mediana"]


def cargar(ruta):
    """Lee el extracto y lo restringe a MiPyMEs."""
    d = pd.read_csv(ruta)
    return d[d["tamano"] != "Grande"].copy()


def _pct(mask, peso):
    """Porcentaje ponderado de la mascara sobre el total de peso."""
    return float(100 * (peso * mask).sum() / peso.sum())


def reparto_por_tramo(d):
    """Tabla de uso de canales por tramo, en porcentaje ponderado de empresas."""
    filas = []
    for tramo in TRAMOS + ["MiPyME total"]:
        sub = d if tramo == "MiPyME total" else d[d["tamano"] == tramo]
        w = sub["fe_transversal"]
        banco = sub["fuente_bancos"].astype(bool)
        ifnb = sub["fuente_ifnb"].astype(bool)
        estatal = sub["fuente_estatal"].astype(bool)
        programa = ifnb | estatal
        filas.append({
            "tramo": tramo,
            "n": int(len(sub)),
            "representadas": float(w.sum()),
            "usa_banco": _pct(banco, w),
            "usa_programa": _pct(programa, w),
            "usa_ifnb": _pct(ifnb, w),
            "usa_estatal": _pct(estatal, w),
            "usa_ambos": _pct(banco & programa, w),
            "solo_banco": _pct(banco & ~programa, w),
            "solo_programa": _pct(programa & ~banco, w),
            "ninguno": _pct(~banco & ~programa, w),
        })
    return pd.DataFrame(filas).round(4)


def cifras_derivadas(d):
    """Las tres cifras que el capitulo comenta fuera de la tabla.

    ``coexistencia`` es la fraccion de quienes usan el canal del programa que
    mantienen ademas un banco. ``banco_entre_formales`` es la fraccion de
    quienes tienen alguna deuda formal que usa banco. ``sin_fuente_alguna`` es la
    declaracion de no usar fuente de financiamiento alguna, que NO coincide con
    la columna ``ninguno`` de la tabla, dado que esta ultima solo mira los tres
    canales de deuda formal considerados.
    """
    w = d["fe_transversal"]
    banco = d["fuente_bancos"].astype(bool)
    programa = d["fuente_ifnb"].astype(bool) | d["fuente_estatal"].astype(bool)
    formal = banco | programa
    return {
        "coexistencia": float(100 * (w * (banco & programa)).sum()
                              / (w * programa).sum()),
        "banco_entre_formales": float(100 * (w * banco).sum()
                                      / (w * formal).sum()),
        "sin_fuente_alguna": _pct(d["fuente_ninguna"].astype(bool), w),
    }


if __name__ == "__main__":
    import sys
    ruta = sys.argv[1] if len(sys.argv) > 1 else "ele7_financiamiento_mipyme.csv"
    datos = cargar(ruta)
    tabla = reparto_por_tramo(datos)
    tabla.to_csv("ele7_reparto_canales_mipyme.csv", index=False)
    print(tabla.to_string(index=False))
    for clave, valor in cifras_derivadas(datos).items():
        print(f"{clave}: {valor:.2f} %")
