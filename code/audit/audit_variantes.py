"""Variantes paramétricas del código de Venegas, SOLO para la auditoría.

`venegas_model.py` es la transcripción fiel y no se modifica. Este archivo
contiene copias con parámetros expuestos, para medir cuánto dependen los
resultados publicados de constantes que el original tiene fijas en el código.

Cambio único respecto del original: el umbral `1e-1` de
`run_price_dyanmics_one_step` pasa a ser el argumento `tol_pct`. Con
`tol_pct=1e-1` el resultado es idéntico al original (se verifica por assert en
`auditoria_convergencia.py`).
"""

import numpy as np

from venegas_model import (
    construct_prohibition_matrices,
    construct_q_matrices,
    get_percent_change,
    opt_w_with_edge_constraints,
    price_update_naive,
)


def paso_con_tolerancia(Q_matrices_list, mean_matrix, P_init,
                        prohibited_edges_tuples=(), eta=1.0, tol_pct=1e-1):
    """`run_price_dyanmics_one_step` con el umbral de acuerdo expuesto."""
    W_current = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_init)
    P_new = np.zeros(shape=P_init.shape)
    n = P_init.shape[0]
    for i in range(n):
        for j in range(i):
            if (j, i) not in prohibited_edges_tuples and (i, j) not in prohibited_edges_tuples:
                pct_w_diff = get_percent_change(W_current[i, j], W_current[j, i])
                if np.abs(pct_w_diff) > tol_pct:
                    p_ij = price_update_naive(i, j, P_init, Q_matrices_list,
                                              mean_matrix, eta=eta)
                    P_new[i, j] = p_ij
                    P_new[j, i] = -1 * p_ij
                else:
                    P_new[i, j] = P_init[i, j]
                    P_new[j, i] = P_init[j, i]
    W_new = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_new)
    return W_new, P_new


def dinamica(sigma_shared, mean_matrix, prohibited_edges_tuples=(),
             num_iterations=50, eta=0.0, Gamma=None, tol_pct=1e-1):
    """`run_price_dynamics` con el umbral expuesto y diagnósticos de parada.

    Devuelve ``(W_final, P_final, info)`` donde ``info`` trae la iteración en
    que la trayectoria se congela (ninguna arista se actualiza), el desacuerdo
    residual y si la iteración divergió.
    """
    n = mean_matrix.shape[0]
    Rm = construct_prohibition_matrices(n, list(prohibited_edges_tuples))
    Q = construct_q_matrices([sigma_shared.copy() for _ in range(n)], Rm, Gamma=Gamma)
    pares = [(i, j) for i in range(n) for j in range(i)
             if (i, j) not in prohibited_edges_tuples
             and (j, i) not in prohibited_edges_tuples]

    P = np.zeros_like(sigma_shared)
    iter_congelada = None
    divergio = False
    for k in range(1, num_iterations + 1):
        W_new, P_new = paso_con_tolerancia(Q, mean_matrix, P, prohibited_edges_tuples,
                                            eta=eta, tol_pct=tol_pct)
        if not np.all(np.isfinite(P_new)) or np.abs(P_new).max() > 1e12:
            divergio = True
            P = P_new
            break
        if iter_congelada is None and np.array_equal(P_new, P):
            iter_congelada = k
        P = P_new
    W = opt_w_with_edge_constraints(Q, mean_matrix, P)
    desac = max(abs(W[i, j] - W[j, i]) for i, j in pares) if pares else 0.0
    desac_pct = (max(abs(get_percent_change(W[i, j], W[j, i])) for i, j in pares)
                 if pares else 0.0)
    info = dict(iter_congelada=iter_congelada, divergio=divergio,
                desacuerdo_abs=float(desac), desacuerdo_pct=float(desac_pct),
                P_max_abs=float(np.abs(P).max()) if np.all(np.isfinite(P)) else np.inf)
    return W, P, info
