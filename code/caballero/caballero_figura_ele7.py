"""Genera la figura de distribución de tasas de la ELE-7 del Capítulo 3.

Todas las síntesis se ponderan por ``fe_transversal``, que es la convención
única que el Capítulo 3 declara para esta encuesta. La mediana ponderada usa la
convención del primer valor cuyo peso acumulado alcanza el cincuenta por ciento.

Los microdatos no se versionan, por tamaño y por condiciones de uso, de modo que
la ruta del extracto se pasa como argumento.

    python caballero_figura_ele7.py <ruta>/ele7_financiamiento_mipyme.csv \
        <ruta>/fig_ele7_distribucion_tasas.png
"""
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

AZUL, ROJO, GRIS = "#1f4e79", "#c0504d", "#8c8c8c"
TRAMOS = ["Grande", "Mediana", "Pequeña 2", "Pequeña 1", "Micro"]


def wquant(x, w, q):
    """Cuantil ponderado, primer valor cuyo peso acumulado alcanza q."""
    orden = np.argsort(x)
    x, w = np.asarray(x)[orden], np.asarray(w)[orden]
    return float(x[np.searchsorted(np.cumsum(w), q * w.sum())])


def estilo():
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.titlelocation": "left", "axes.titlepad": 8,
    })


def figura(d, salida):
    coma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                                     gridspec_kw={"width_ratios": [1.6, 1]})
    rng = np.random.default_rng(0)

    for i, t in enumerate(TRAMOS):
        g = d[d.tamano == t]
        x, w = g.tasa_anual_pct.values, g.fe_transversal.values
        q1, q2, q3 = (wquant(x, w, q) for q in (0.25, 0.50, 0.75))
        bajo, alto = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        dentro = x[(x >= bajo) & (x <= alto)]
        ax_a.scatter(x, i + rng.uniform(-0.17, 0.17, len(x)), s=4, color=GRIS,
                     alpha=0.28, linewidths=0, zorder=1)
        ax_a.add_patch(plt.Rectangle((q1, i - 0.26), q3 - q1, 0.52, fill=False,
                                     edgecolor=AZUL, linewidth=1.0, zorder=3))
        ax_a.plot([q2, q2], [i - 0.26, i + 0.26], color=AZUL, lw=2.0, zorder=4)
        ax_a.plot([dentro.min(), q1], [i, i], color=AZUL, lw=1.0, zorder=2)
        ax_a.plot([q3, dentro.max()], [i, i], color=AZUL, lw=1.0, zorder=2)
        for extremo in (dentro.min(), dentro.max()):
            ax_a.plot([extremo, extremo], [i - 0.14, i + 0.14], color=AZUL, lw=1.0, zorder=2)

    maximo = d.tasa_anual_pct.max()
    fila = TRAMOS.index(d.loc[d.tasa_anual_pct.idxmax(), "tamano"])
    etiqueta = "máx. " + f"{maximo:.1f}".replace(".", ",") + " %"
    ax_a.annotate(etiqueta, xy=(maximo, fila),
                  xytext=(90, 1.75), color=ROJO, fontsize=7, ha="center",
                  arrowprops=dict(arrowstyle="-", color=ROJO, lw=0.7))
    ax_a.set_xscale("log")
    ax_a.set_xticks([0.1, 1, 10, 100])
    ax_a.xaxis.set_major_formatter(coma)
    ax_a.set_yticks(range(len(TRAMOS)))
    ax_a.set_yticklabels([f"{t} (n={int((d.tamano == t).sum())})" for t in TRAMOS])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Tasa de interés anualizada 2022 (%, escala logarítmica)")
    ax_a.set_title("Distribución asimétrica a la derecha en los cinco tramos")
    ax_a.margins(y=0.08)

    for i, t in enumerate(TRAMOS):
        g = d[d.tamano == t]
        med = wquant(g.tasa_anual_pct.values, g.fe_transversal.values, 0.50)
        media = float(np.average(g.tasa_anual_pct, weights=g.fe_transversal))
        ax_b.plot([med, media], [i, i], color=GRIS, lw=1.0, zorder=1)
        ax_b.scatter([med], [i], s=38, color=AZUL, zorder=3,
                     label="Mediana" if i == 0 else None)
        ax_b.scatter([media], [i], s=38, color=ROJO, zorder=3,
                     label="Media" if i == 0 else None)
    ax_b.set_yticks(range(len(TRAMOS)))
    ax_b.set_yticklabels(TRAMOS)
    ax_b.invert_yaxis()
    ax_b.xaxis.set_major_formatter(coma)
    ax_b.set_xlabel("Tasa anual (%)")
    ax_b.set_title("La media invierte el orden entre\nGrande y Mediana")
    ax_b.margins(x=0.12, y=0.08)
    ax_b.legend(frameon=False, loc="upper right", handletextpad=0.3)

    for ax, letra in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.02, 1.06, letra, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom", ha="right")

    fig.subplots_adjust(wspace=0.35)
    fig.savefig(salida)
    return fig


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("uso: caballero_figura_ele7.py <ruta>/ele7_financiamiento_mipyme.csv <ruta>/fig.png")
    estilo()
    datos = pd.read_csv(sys.argv[1]).dropna(subset=["tasa_anual_pct"])
    figura(datos, sys.argv[2])
    print(f"figura escrita en {sys.argv[2]} sobre {len(datos)} empresas con tasa")
