"""Núcleo numérico de la tesis de Hernán Venegas, extraído para auditoría.

Transcripción FIEL de ``code/Financial_Networks_Stability.qmd`` (commit e2f9e39,
hrvenegas, 2024-11-04). La lógica no se altera: se conservan los nombres
originales -- incluido el typo ``run_price_dyanmics_one_step`` -- y los defectos
detectados se preservan tal cual para poder documentarlos con pruebas.

Único cambio respecto del original: los tres ``import *`` del .qmd se
reemplazaron por imports explícitos. Ver ``equivalencia_imports.md`` para la
resolución de cada nombre ambiguo en el orden original.

Referencia: Jalan & Chakrabarti (2024), "Incentive-Aware Models of Financial
Networks", Operations Research. DOI 10.1287/opre.2022.0678.
"""

import numpy as np

DEFAULT_NUM_FIRMS = 3

####################
## Gen fns
####################


def generate_firm_covariances(num_firms=DEFAULT_NUM_FIRMS,
                              num_samples=DEFAULT_NUM_FIRMS,
                              risk_aversion=1.0, entry_distribution='gaussian',
                              noise_param=1.0, subtract_mean=False,
                              return_inv=False):
    """Matriz de covarianza sintética como suma de matrices aleatorias rank-1.

    Origen: .qmd líneas 54-85.

    ATENCIÓN (hallazgo H7): usa ``np.random`` sin semilla. No interviene en el
    Ejemplo 5, que usa matrices fijas, pero cualquier análisis de sensibilidad
    o calibración construido sobre esta función es irreproducible tal cual.
    """
    cov_matrix = np.zeros(shape=(num_firms, num_firms))
    # need n iterations. at each iteration add a random rank-1 matrix.

    for i in range(num_samples):
        if entry_distribution == 'gaussian':
            A = np.random.normal(loc=0.0, scale=noise_param, size=(num_firms, 1))
        elif entry_distribution == 'exponential':
            A = np.random.exponential(noise_param, size=(num_firms, 1))
        elif entry_distribution == 'eye':
            A = np.eye(1, num_firms, i).T
        else:
            raise Exception(f'Distribution {entry_distribution} is unknown')
        symmA = np.matmul(A, A.T)
        cov_matrix += symmA
    cov_matrix_normalized = np.copy(cov_matrix)
    if subtract_mean:
        cov_matrix_normalized = cov_matrix - cov_matrix.mean(axis=1, keepdims=True)

    # normalize by num_samples so it has a reasonable scale.
    if entry_distribution != 'eye':
        cov_matrix_normalized *= (1.0 / num_samples)

    # scale by risk aversion parameter.
    cov_matrix_normalized *= risk_aversion

    if return_inv:
        return cov_matrix_normalized, np.linalg.inv(cov_matrix_normalized)
    return cov_matrix_normalized


def generate_means(num_firms=DEFAULT_NUM_FIRMS):
    """Matriz de retornos esperados con diagonal nula. Origen: .qmd 87-91.

    ATENCIÓN (hallazgo H7): ``np.random`` sin semilla.
    """
    init = np.random.normal(loc=5, scale=1.0, size=(num_firms, num_firms))
    for i in range(num_firms):
        init[i][i] = 0.0
    return init


####################
## Solver fns
####################


def construct_prohibition_matrices(n, prohibited_tuples):
    """Matrices de selección R_i que codifican la restricción regulatoria.

    Origen: .qmd 97-113. Para cada agente i, R_i es la identidad n x n a la que
    se le eliminan las filas correspondientes a las contrapartes prohibidas,
    de modo que R_i tiene forma (n - |prohibidas(i)|, n).
    """
    # tuples belong to {0, 1, ..., n-1}

    # Initialize an empty list to store the matrices
    matrices = [np.eye(n) for _ in range(n)]
    rows_to_delete = {i: [] for i in range(n)}

    # Create the identity matrix of size n x n
    # identity_matrix = np.identity(n)

    # Loop through each prohibited tuple
    for prohibited_edge in prohibited_tuples:
        i, j = prohibited_edge
        rows_to_delete[i].append(j)
        rows_to_delete[j].append(i)
    matrices = [np.delete(np.eye(n), rows_to_delete[i], axis=0) for i in range(n)]
    return matrices


def construct_q_matrices(Sigma_matrices, prohibition_matrices, Gamma=None):
    """Q_i = R_i^T (2 Gamma_ii R_i Sigma_i R_i^T)^{-1} R_i. Origen: .qmd 115-124.

    Es la inversa restringida al subespacio de contrapartes permitidas; el
    factor 2 y la posición de Gamma se verifican en verificacion_algebraica.md.
    """
    out = []
    n = len(Sigma_matrices)
    if Gamma is None:
        Gamma = np.eye(n)
    for i in range(n):
        inner = 2 * Gamma[i, i] * (prohibition_matrices[i] @ Sigma_matrices[i] @ prohibition_matrices[i].T)
        outer = prohibition_matrices[i].T @ np.linalg.inv(inner) @ prohibition_matrices[i]
        out.append(outer.copy())
    return out


def opt_portfolio_mat(S_inv, M, P, Gamma=None):
    """Portafolio óptimo sin restricciones de arista. Origen: .qmd 129-133.

    No se usa en el Ejemplo 5 (está reemplazada por
    ``opt_w_with_edge_constraints``); se conserva por completitud.
    """
    sol_mat = np.matmul(0.5 * S_inv, M - P)
    if Gamma is None:
        return sol_mat
    return np.matmul(sol_mat, np.linalg.inv(Gamma))


# todo update price sylvester for Gamma
def price_soln_sylvester(S_inv, mean_matrix):
    """Solución cerrada de precios vía ecuación de Sylvester. Origen: .qmd 135-139.

    DEFECTO PRESERVADO (hallazgo H5): el cuerpo referencia ``la.solve_sylvester``
    y el nombre ``la`` no está definido en ningún punto del .qmd original -- los
    imports traen ``solve_sylvester`` directamente desde ``scipy.linalg``, sin
    alias ``la``. Llamar a esta función lanza NameError. Se transcribe tal cual
    para que la prueba lo documente. El TODO de la línea 135 es del autor.
    """
    Q = 0.5 * S_inv
    return la.solve_sylvester(Q, Q,  # noqa: F821  (defecto preservado)
                              np.matmul(Q, mean_matrix) - np.matmul(mean_matrix.T, Q))


################################
################################
# Dynamics
################################
################################


def get_percent_change(current, previous):
    """Cambio porcentual usado como test de acuerdo entre exposiciones.

    Origen: .qmd 146-152.

    DEFECTOS PRESERVADOS:
    - H1: devuelve 100.0 cuando ``current == previous``, es decir reporta
      discrepancia máxima justamente en el caso de acuerdo perfecto, que es la
      condición que la dinámica busca. La guarda está invertida respecto de su
      propósito.
    - H2: ``except ZeroDivisionError`` es inerte con escalares de NumPy: la
      división por cero emite RuntimeWarning y devuelve inf o nan en lugar de
      lanzar la excepción, así que la rama de rescate nunca se ejecuta.
    """
    if current == previous:
        return 100.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0


def get_iden_missing_row(n, i):
    """Identidad con el elemento (i,i) anulado. Origen: .qmd 154-157.

    No es invocada en ninguna parte del .qmd (código muerto).
    """
    iden = np.eye(n)
    iden[i, i] = 0.0
    return iden


def price_update_naive(i, j, P_init, Q_matrices_list, mean_matrix, eta=1.0):
    """Precio negociado de la arista (i,j) con amortiguamiento eta.

    Origen: .qmd 159-174. ``diff_scalar`` es W[i,j] - W[j,i], la discrepancia
    de exposiciones deseadas; ``denom`` es Q_i[j,j] + Q_j[i,i].
    """
    if i == j:
        return 0.0

    diff_mat = mean_matrix - P_init
    prod_i = Q_matrices_list[i] @ diff_mat
    prod_j = Q_matrices_list[j] @ diff_mat

    diff_scalar = prod_j[i, j] - prod_i[j, i]
    denom = Q_matrices_list[i][j, j] + Q_matrices_list[j][i, i]

    shift = diff_scalar / denom
    old_price = P_init[i, j]
    p_prime = shift + old_price
    new_price = eta * p_prime + (1.0 - eta) * old_price
    return new_price


# assuming shared Sigma and Gamma = I
def opt_w_with_edge_constraints(Q_matrices_list, M, P):
    """Exposiciones óptimas por columna: W[:, i] = Q_i (M[:, i] - P[:, i]).

    Origen: .qmd 177-187. El comentario ``assuming shared Sigma and Gamma = I``
    es del autor.
    """
    n = M.shape[0]
    # prohibition_matrices = construct_prohibition_matrices(n, prohibited_edges_tuples)
    # Q_matrices_list = construct_q_matrices([Sigma_shared.copy() for _ in range(n)], prohibition_matrices)
    w_list = []
    for i in range(n):
        w_i = Q_matrices_list[i] @ (M[:, i] - P[:, i])
        w_list.append(w_i)
    W_mat = np.vstack(w_list).T
    # print('W opt is ', W_mat)
    return W_mat


def run_price_dyanmics_one_step(Q_matrices_list, mean_matrix, P_init,
                                prohibited_edges_tuples=[], eta=1.0):
    """Un paso de la dinámica de negociación. Origen: .qmd 189-215.

    El typo ``dyanmics`` es del original y se conserva.

    DEFECTO PRESERVADO (H4b): ``P_new`` se inicializa en ceros y las entradas
    de aristas prohibidas nunca se escriben, por lo que quedan en 0 en vez de
    heredar ``P_init``. Coincide con el valor esperado mientras P arranque en
    ceros, pero no es equivalente en general.
    """
    # prohibited_edges_tuples is a list of pairs, e.g. [(1, 3), (2, 3), (2, 5)]

    W_current = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_init)
    P_new = np.zeros(shape=P_init.shape)
    n = P_init.shape[0]

    for i in range(n):
        for j in range(i):
            # check that contract prices disagree
            if (j, i) not in prohibited_edges_tuples and (i, j) not in prohibited_edges_tuples:
                pct_w_diff = get_percent_change(W_current[i, j], W_current[j, i])
                if np.abs(pct_w_diff) > 1e-1:
                    updated_price_ij = price_update_naive(i, j, P_init, Q_matrices_list,
                                                          mean_matrix, eta=eta)
                    P_new[i, j] = updated_price_ij
                    P_new[j, i] = -1 * updated_price_ij
                else:
                    P_new[i, j] = P_init[i, j]
                    P_new[j, i] = P_init[j, i]
    W_new = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_new)
    return W_new, P_new


def run_price_dynamics(sigma_shared, mean_matrix, prohibited_edges_tuples=[],
                       num_iterations=50, eta=0.0, Gamma=None):
    """Itera la dinámica ``num_iterations`` veces y devuelve las trayectorias.

    Origen: .qmd 217-245.

    DEFECTOS PRESERVADOS:
    - H3: ``eta=0.0`` por defecto. Con eta=0 la actualización es
      ``new = 0*p' + 1*old``: los precios no se mueven nunca y la dinámica
      "converge" trivialmente. Los ejemplos del .qmd pasan eta=0.5 explícito.
    - H4: no hay criterio de convergencia ni residual reportado; se ejecuta un
      número fijo de iteraciones sin verificar que se alcanzó el punto
      pairwise-estable.
    """
    n = mean_matrix.shape[0]
    prohibition_matrices = construct_prohibition_matrices(n, prohibited_edges_tuples)
    Q_matrices_list = construct_q_matrices([sigma_shared.copy() for _ in range(n)],
                                           prohibition_matrices, Gamma=Gamma)
    P_current = np.zeros_like(sigma_shared)
    W_current = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_current)

    W_list = []
    P_list = []
    W_list.append(W_current)
    P_list.append(P_current)

    for idx in range(num_iterations):
        W_current, P_current = run_price_dyanmics_one_step(
            Q_matrices_list, mean_matrix, P_current, prohibited_edges_tuples, eta=eta)
        W_list.append(W_current)
        P_list.append(P_current)
    return W_list, P_list


####################
## Parámetros del Ejemplo 5 (Jalan & Chakrabarti 2024), .qmd 249-273 y 397-420
####################

EJEMPLO5_COV = np.array([
    [1.0, 0.25, 0.75],
    [0.25, 1.0, 0.6],
    [0.75, 0.6, 1.0],
])

EJEMPLO5_MEAN = np.array([
    [0.0, 0.9, 0.9],
    [0.75, 0.0, 0.95],
    [0.5, 0.8, 0.0],
])

EJEMPLO5_GAMMA = np.eye(3)

#: Arista prohibida en el escenario con restricción regulatoria.
EJEMPLO5_PROHIBIDAS_CON = [(0, 2)]
#: Escenario benchmark sin restricciones.
EJEMPLO5_PROHIBIDAS_SIN = []
#: Parámetros de corrida usados en el .qmd original.
EJEMPLO5_ETA = 0.5
EJEMPLO5_ITERS = 500
