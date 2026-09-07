"""Capa de métricas de la tesis de Venegas, recuperada por ingeniería inversa.

El capítulo 4 de la tesis reporta volumen bilateral, posiciones totales, utilidad
por agente y bienestar total. NINGÚN archivo de código versionado calcula esas
magnitudes: el `.qmd` no contiene las palabras `utility`, `welfare`, `volume` ni
`networkx`. Este módulo reconstruye esas definiciones a partir de las matrices W
y P que sí produce `venegas_model.py`, y las definiciones están CONFIRMADAS:
reproducen la tabla baseline del cap. 4 en 14/14 métricas y el escenario de la
fintech en sus valores titulares (ver `cobertura_resultados.csv`).

Los parámetros del escenario de 4 firmas están declarados en la tesis
(cap. 4, "Experimental Design", ecuaciones eq:cov_mat_4 / eq:mean_mat_4 /
eq:gamma_4 / eq:psi_4) pero no existen en ningún archivo de código: se
transcriben aquí para que el escenario sea ejecutable.
"""

import numpy as np

# --------------------------------------------------------------- métricas

def bilateral_volume(W):
    """Suma de |W_ij| sobre el triángulo superior estricto.

    El valor absoluto es necesario: en el escenario 3F sin restricción hay una
    exposición bilateral negativa, y sin |.| el volumen da 1.841 en vez del
    1.976 que reporta la tesis.
    """
    return float(np.abs(W[np.triu_indices(W.shape[0], 1)]).sum())


def total_positions(W):
    """Suma de |W_ij| sobre el triángulo superior INCLUYENDO la diagonal."""
    return float(np.abs(W[np.triu_indices(W.shape[0], 0)]).sum())


def utilities(W, P, Sigma, M, Gamma):
    """Utilidad media-varianza por agente: g_i = w_i'(mu_i - P e_i) - gamma_i w_i' Sigma w_i.

    Es la utilidad declarada en el cap. 3 de la tesis, evaluada en el óptimo.
    """
    n = W.shape[0]
    return np.array([
        W[:, i] @ (M[:, i] - P[:, i]) - Gamma[i, i] * W[:, i] @ Sigma @ W[:, i]
        for i in range(n)
    ])


def total_welfare(W, P, Sigma, M, Gamma):
    """Bienestar total del sistema = suma de utilidades individuales."""
    return float(utilities(W, P, Sigma, M, Gamma).sum())


def resumen(W, P, Sigma, M, Gamma):
    u = utilities(W, P, Sigma, M, Gamma)
    return dict(bilateral_volume=bilateral_volume(W),
                total_positions=total_positions(W),
                utilities=u, total_welfare=float(u.sum()))


# ------------------------------------ escenario de 4 firmas (cap. 4, tesis)

#: Correlación entre Local Bank y Fintech: los hace casi indistinguibles.
RHO_4 = 0.99

COV4 = np.array([[1.00, 0.25, 0.75, 0.25],
                 [0.25, 1.00, 0.60, RHO_4],
                 [0.75, 0.60, 1.00, 0.60],
                 [0.25, RHO_4, 0.60, 1.00]])

MEAN4 = np.array([[0.00, 0.9, 0.90, 0.9],
                  [0.75, 0.0, 0.95, 0.0],
                  [0.50, 0.8, 0.00, 0.8],
                  [0.75, 0.0, 0.95, 0.0]])

GAMMA4 = np.eye(4)

#: Restricción regulatoria: prohíbe W_13 y W_31 (índices base 0).
PROHIBIDAS_4F_WITH = [(0, 2)]
PROHIBIDAS_4F_WITHOUT = []
