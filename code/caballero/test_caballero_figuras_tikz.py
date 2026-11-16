"""Pruebas del productor de las dos figuras TikZ.

Dos grupos, con propósitos distintos.

``test_genera_*``  comprueban que el script produce lo que debe a partir de los
                   CSV versionados, es decir los conteos y las medianas.
``test_deriva_*``  comparan el ``.tex`` QUE ESTÁ EN EL REPOSITORIO contra los
                   CSV. Son las que importan: si alguien rehace una corrida y no
                   vuelve a correr el productor, éstas fallan y la figura vieja
                   deja de pasar inadvertida, que es el problema que el script
                   existe para resolver.

Ninguna prueba escribe en ``docs/``.

Ejecutar desde la raíz del repositorio

    python -m pytest code/caballero/test_caballero_figuras_tikz.py -v
"""

import re

import numpy as np
import pytest

import caballero_figuras_tikz as fig


@pytest.fixture(scope="module")
def grupos():
    return fig.nube(fig.admisibles())


@pytest.fixture(scope="module")
def series():
    return fig.curvas()


def _coordenadas(texto):
    """Todos los bloques ``coordinates {...}`` de un .tex, ya parseados."""
    return [[(float(a), float(b))
             for a, b in re.findall(r"\(([-\d.]+),([-\d.]+)\)", bloque)]
            for bloque in re.findall(r"coordinates \{(.*?)\}", texto, re.S)]


def _tex_en_el_repo(nombre):
    return (fig.DESTINO / nombre).read_text(encoding="utf-8")


# ===================================================== generación

def test_genera_los_312_escenarios_repartidos_168_y_144(grupos):
    assert [len(g["y"]) for g in grupos] == [168, 144]
    assert sum(len(g["y"]) for g in grupos) == 312


def test_genera_nueve_puntos_por_serie_del_contrafactual(series):
    for etiqueta, s in series.items():
        assert len(s["x"]) == 9, etiqueta
        assert len(s["costo"]) == 9 and len(s["fraccion"]) == 9
        assert list(s["x"]) == sorted(s["x"]), f"{etiqueta}: no viene ordenada"


def test_genera_la_nube_dentro_del_desplazamiento_declarado(grupos):
    """Cada grupo queda centrado en su abscisa, sin invadir al otro."""
    for g in grupos:
        assert np.abs(g["x"] - g["centro"]).max() <= fig.DESPLAZAMIENTO + 1e-12


def test_genera_un_sorteo_reproducible(grupos):
    """La semilla fija hace que dos corridas den la misma figura."""
    otra = fig.nube(fig.admisibles())
    for a, b in zip(grupos, otra):
        assert np.array_equal(a["x"], b["x"])


# ============================== el .tex del repositorio contra los CSV

def test_deriva_las_medianas_del_tex_coinciden_con_el_csv(grupos):
    """La guardia contra figuras viejas.

    Falla si se rehace una corrida sin volver a correr el productor. Tolerancia
    de media centésima, que es la precisión con la que el .tex escribe la
    coordenada de la mediana.
    """
    bloques = _coordenadas(_tex_en_el_repo("fig_escenarios_admisibles.tex"))
    for g, bloque in zip(grupos, bloques[2:4]):
        escrita = {y for _, y in bloque}
        assert len(escrita) == 1, "el segmento de la mediana no es horizontal"
        assert escrita.pop() == pytest.approx(g["mediana"], abs=5e-3)


def test_deriva_la_nube_del_tex_coincide_con_el_csv(grupos):
    bloques = _coordenadas(_tex_en_el_repo("fig_escenarios_admisibles.tex"))
    for g, bloque in zip(grupos, bloques[:2]):
        assert len(bloque) == len(g["y"])
        assert [y for _, y in bloque] == [round(float(v), 2) for v in g["y"]]


def test_deriva_las_curvas_del_tex_coinciden_con_el_csv(series):
    bloques = _coordenadas(_tex_en_el_repo("fig_contrafactual_curvas.tex"))
    esperado = [("equicorrelada", "costo"), ("por ciclo", "costo"),
                ("equicorrelada", "fraccion"), ("por ciclo", "fraccion")]
    assert len(bloques) == 4
    for (etiqueta, campo), bloque in zip(esperado, bloques):
        s = series[etiqueta]
        assert len(bloque) == 9, f"{etiqueta}/{campo}"
        assert [x for x, _ in bloque] == [round(float(v), 2) for v in s["x"]]
        assert [y for _, y in bloque] == [round(float(v), 1) for v in s[campo]]


def test_deriva_las_etiquetas_que_cita_el_capitulo_siguen_ahi():
    """El Capítulo 4 referencia las dos figuras por ``\\label``."""
    for nombre, etiqueta in [("fig_escenarios_admisibles.tex", "fig:escenarios-admisibles"),
                             ("fig_contrafactual_curvas.tex", "fig:contrafactual-curvas")]:
        assert f"\\label{{{etiqueta}}}" in _tex_en_el_repo(nombre)


def test_deriva_el_pie_declara_los_conteos_correctos(grupos):
    """Los tres números del pie se calculan, de modo que no pueden desfasarse."""
    texto = _tex_en_el_repo("fig_escenarios_admisibles.tex")
    n_eq, n_ci = (len(g["y"]) for g in grupos)
    assert f"uno de los {n_eq + n_ci} escenarios admisibles" in texto
    assert f"equicorrelada tiene {n_eq} escenarios" in texto
    assert f"ciclo tiene {n_ci}" in texto
