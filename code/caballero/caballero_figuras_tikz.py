"""Productor de las dos figuras TikZ del Capítulo 4.

Las figuras se agregaron al documento con las coordenadas escritas a mano, de
modo que si las corridas se rehacían quedaban viejas sin que nada lo avisara.
Este script las regenera desde los dos CSV versionados, que son la misma fuente
que sustenta las tablas, y así la figura y la tabla no pueden separarse.

Escribe

    docs/MemoriasTesis/Caballero/fig_escenarios_admisibles.tex
    docs/MemoriasTesis/Caballero/fig_contrafactual_curvas.tex

El pie y la etiqueta de cada figura se conservan literales, porque el Capítulo 4
las referencia por ``\\label``. Lo único que el script calcula son las
coordenadas y las dos medianas.

Ejecutar desde la raíz del repositorio, sin variables de entorno

    python code/caballero/caballero_figuras_tikz.py

Sobre el desplazamiento horizontal de la nube. Los escenarios comparten muchos
valores de cuota, de modo que sin desplazamiento los puntos se superponen y la
densidad no se ve. Se usa un desplazamiento uniforme en más menos 0,17 con
semilla fija, sorteado primero para la matriz equicorrelada y después para la
de ciclo, para que la figura sea idéntica en cada corrida. La semilla no tiene
ningún contenido, sólo fija el sorteo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CSV_CASO = RAIZ / "code/caballero/caballero_caso_chileno_resultados.csv"
CSV_CONTRA = RAIZ / "code/caballero/caballero_contrafactual_resultados.csv"
DESTINO = RAIZ / "docs/MemoriasTesis/Caballero"

#: Parámetros del desplazamiento de la nube. No son resultados.
SEMILLA = 7
DESPLAZAMIENTO = 0.17

#: Orden de las dos matrices, que fija también el orden del sorteo.
GRUPOS = [("neutra", 1, "equicorrelada"), ("economica", 2, "por ciclo")]


def _coma(x, dec=1):
    """Número con coma decimal, que es la convención del documento."""
    return f"{x:.{dec}f}".replace(".", ",")


def admisibles(ruta=CSV_CASO):
    """Los escenarios admisibles del caso principal.

    El criterio es el mismo que usa la corrida y que su encabezado declara:
    margen del intermediario positivo y arista de la red Alternativa no
    negativa. La columna booleana vuelve del CSV como texto, de modo que hay
    que convertirla antes de negarla; hacerlo con ``~`` directamente sobre la
    columna produce un entero sin sentido en vez de una máscara.
    """
    d = pd.read_csv(ruta)
    rango = d[d["bloque"] == "rango"]
    negativa = (rango["arista_A_negativa"].astype(str).str.strip().str.lower()
                .isin(["true", "1", "1.0"]))
    return rango[(rango["margen_ifnb"] > 0) & (~negativa)]


def nube(adm):
    """Coordenadas de la nube y mediana de cada grupo."""
    rng = np.random.default_rng(SEMILLA)
    salida = []
    for sigma, centro, _ in GRUPOS:
        y = 100 * adm[adm["sigma"] == sigma]["adopcion_A"].values
        x = centro + rng.uniform(-DESPLAZAMIENTO, DESPLAZAMIENTO, len(y))
        salida.append(dict(sigma=sigma, centro=centro, x=x, y=y,
                           mediana=float(np.median(y))))
    return salida


def tex_escenarios(grupos):
    marcas = []
    for g in grupos:
        puntos = " ".join(f"({a:.3f},{b:.2f})" for a, b in zip(g["x"], g["y"]))
        marcas.append(
            "\\addplot[only marks, mark=*, mark size=1.4pt, draw=none,\n"
            "         fill=blue!45!black, fill opacity=0.45] coordinates {"
            + puntos + " };")
    lineas, nodos = [], []
    for g in grupos:
        c, m = g["centro"], g["mediana"]
        lineas.append("\\addplot[red!75!black, line width=1.8pt, mark=none]\n"
                      f"  coordinates {{({c - 0.26:.2f},{m:.2f}) ({c + 0.26:.2f},{m:.2f})}};")
        nodos.append("\\node[red!75!black, font=\\small, anchor=west] at "
                     f"(axis cs:{c + 0.30:.2f},{m:.2f}) {{{_coma(m)}\\%}};")
    n_eq, n_ci = (len(g["y"]) for g in grupos)
    return f"""%% Nube de los {n_eq + n_ci} escenarios admisibles. Generada desde
%% code/caballero/caballero_caso_chileno_resultados.csv.
\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.80\\textwidth,
    height=0.46\\textwidth,
    xmin=0.5, xmax=2.6,
    ymin=-3, ymax=58,
    xtick={{1,2}},
    xticklabels={{equicorrelada, por ciclo}},
    ylabel={{Cuota de la red Alternativa (\\%)}},
    ytick={{0,10,20,30,40,50}},
    tick label style={{font=\\small}},
    label style={{font=\\small}},
]
{marcas[0]}
{marcas[1]}
{lineas[0]}
{lineas[1]}
{nodos[0]}
{nodos[1]}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption[Cuota de la red Alternativa sobre los escenarios admisibles]{{Cuota de
la actividad que el modelo asigna a la red Alternativa, con un punto por cada
uno de los {n_eq + n_ci} escenarios admisibles y la mediana de cada grupo en rojo. El
grupo de la matriz equicorrelada tiene {n_eq} escenarios y el de la matriz por
ciclo tiene {n_ci}. La dispersión horizontal dentro de cada grupo es un
desplazamiento aleatorio que separa puntos de igual valor. Elaboración propia.}}
\\label{{fig:escenarios-admisibles}}
\\end{{figure}}
"""


def curvas(ruta=CSV_CONTRA):
    """Las dos series de cada panel, ordenadas por la covarianza supuesta."""
    d = pd.read_csv(ruta)
    sg = d[d["bloque"] == "sigma_corfo_mipyme"].sort_values("sigma_cm")
    salida = {}
    for etiqueta in ("equicorrelada", "por ciclo"):
        s = sg[sg["sigma"] == etiqueta]
        salida[etiqueta] = dict(
            x=s["sigma_cm"].values,
            costo=100 * s["brecha_relativa"].values,
            fraccion=100 * s["fraccion"].values)
    return salida


def tex_contrafactual(series):
    def serie(etiqueta, campo, estilo):
        s = series[etiqueta]
        pts = " ".join(f"({a:.2f},{b:.1f})" for a, b in zip(s["x"], s[campo]))
        return (f"\\addplot[{estilo}]\n  coordinates {{" + pts + " };\n"
                f"\\addlegendentry{{{etiqueta}}}")

    sup = "\n".join([
        serie("equicorrelada", "costo", "red!75!black, line width=1pt, mark=*, mark size=1.7pt"),
        serie("por ciclo", "costo", "orange!85!black, line width=1pt, mark=triangle*, mark size=2pt, dashed"),
    ])
    inf = "\n".join([
        serie("equicorrelada", "fraccion", "green!35!black, line width=1pt, mark=square*, mark size=1.7pt"),
        serie("por ciclo", "fraccion", "teal!70!black, line width=1pt, mark=diamond*, mark size=2pt, dashed"),
    ])
    return f"""%% Costo del mandato y fracción recuperada, ambas matrices. Generada desde
%% code/caballero/caballero_contrafactual_resultados.csv.
\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{groupplot}}[
    group style={{group size=1 by 2, vertical sep=1.4cm}},
    width=0.78\\textwidth,
    height=0.30\\textwidth,
    xmin=-0.55, xmax=0.95,
    tick label style={{font=\\small}},
    label style={{font=\\small}},
    legend style={{font=\\small, draw=none, fill=none}},
]
\\nextgroupplot[
    ymin=30, ymax=95, ytick={{40,60,80}},
    xlabel={{}}, ylabel={{\\% del bienestar sin restricción}},
    legend pos=north west]
{sup}
\\nextgroupplot[
    ymin=-135, ymax=90, ytick={{-120,-60,0,60}},
    xlabel={{Covarianza supuesta entre CORFO y la MiPyME}},
    ylabel={{\\% de ese costo}},
    legend pos=south west]
{inf}
\\end{{groupplot}}
\\end{{tikzpicture}}
\\caption[Costo del mandato y fracción recuperada]{{Costo del mandato de segundo
piso, en el panel superior, y fracción de ese costo que recupera la red
Alternativa, en el inferior, según la covarianza entre CORFO y la MiPyME, en el
punto central de los intervalos de tasas y con beneficio de transparencia nulo.
Los dos paneles no comparten denominador, puesto que el superior es porcentaje
del bienestar sin restricción y el inferior es porcentaje de ese costo. Los
valores exactos están en la Tabla~\\ref{{tab:contrafactual-sigma}}. Elaboración
propia.}}
\\label{{fig:contrafactual-curvas}}
\\end{{figure}}
"""


def main(destino=DESTINO):
    grupos = nube(admisibles())
    series = curvas()
    escritos = []
    for nombre, texto in [("fig_escenarios_admisibles.tex", tex_escenarios(grupos)),
                          ("fig_contrafactual_curvas.tex", tex_contrafactual(series))]:
        (Path(destino) / nombre).write_text(texto, encoding="utf-8")
        escritos.append(nombre)
    print(f"  escenarios admisibles: {sum(len(g['y']) for g in grupos)} puntos "
          f"({', '.join(str(len(g['y'])) + ' ' + g['sigma'] for g in grupos)})")
    for g in grupos:
        print(f"    mediana {g['sigma']:10s} {g['mediana']:.4f} %")
    for etiqueta, s in series.items():
        print(f"  contrafactual {etiqueta:14s} {len(s['x'])} puntos por panel")
    print("  escritos: " + ", ".join(escritos))
    return grupos, series


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
