"""Núcleo numérico de Venegas, con el subconjunto de hallazgos H1/H3/H7
corregido más validación de Sigma y el hallazgo H4b, previo a construir la
capa de acoplamiento T/A (modelo dual).

Alcance de esta corrección (decidido junto con el usuario, 2026-09-15):
corrige SOLO lo que la propia auditoría marca como bloqueante para calibrar
con datos externos (AUDITORIA_VENEGAS.md, sección 5, puntos 1-4) más H4b,
que se vuelve directamente relevante para la dinámica de precios dual (ver
docstring de ``run_price_dynamics`` más abajo). NO toca H2, H5, H8, H9 --
quedan documentados y diferidos, tal como están en ``venegas_model.py``.

``venegas_model.py`` (la transcripción fiel del .qmd, con los defectos
preservados como evidencia de auditoría) NO se modifica. Este módulo es un
reemplazo API-compatible para el código nuevo (capa dual y en adelante);
las funciones sin defectos (``construct_prohibition_matrices``,
``opt_w_with_edge_constraints``, ``price_update_naive``, las constantes
``EJEMPLO5_*``) se re-exportan sin cambios desde ``venegas_model``.

Referencia: Jalan & Chakrabarti (2024), "Incentive-Aware Models of Financial
Networks", Operations Research. DOI 10.1287/opre.2022.0678. El helper
``eta_estable_teorema4`` implementa el Teorema 4 de ese paper (convergencia
de Algorithm 1): eta* = 2/lambda_max, eta_opt = 2/(lambda_min+lambda_max),
con lambda los autovalores de (L^+ K)|_R restringida a los pares permitidos.
"""

import warnings

import numpy as np

import _ruta_audit  # noqa: F401  deja code/audit importable, ver ese modulo

# Piezas sin defectos: se reexportan tal cual desde el original,
# que vive en code/audit porque preserva la evidencia de la auditoria.
from venegas_model import (  # noqa: F401
    DEFAULT_NUM_FIRMS,
    construct_prohibition_matrices,
    opt_portfolio_mat,
    opt_w_with_edge_constraints,
    price_update_naive,
    EJEMPLO5_COV,
    EJEMPLO5_MEAN,
    EJEMPLO5_GAMMA,
    EJEMPLO5_PROHIBIDAS_CON,
    EJEMPLO5_PROHIBIDAS_SIN,
    EJEMPLO5_ETA,
    EJEMPLO5_ITERS,
)

####################
## H1 -- guardia de convergencia
####################


def get_percent_change(current, previous):
    """Cambio porcentual entre dos exposiciones, para el test de acuerdo.

    CORRIGE H1: la versión original de ``venegas_model.get_percent_change``
    devuelve 100.0 (discrepancia MÁXIMA) cuando ``current == previous``, es
    decir en el caso de acuerdo perfecto -- exactamente lo opuesto de lo que
    la guardia debía señalar. Aquí se corrige a 0.0 (sin discrepancia).

    Con los parámetros del Ejemplo 5 esta rama nunca se activa (el propio
    informe de auditoría lo documenta), así que este cambio no altera ningún
    resultado publicado -- ver ``test_venegas_model_corregido.py::
    test_regresion_ejemplo5_sin_cambios``.

    H2 (``except ZeroDivisionError`` inerte con escalares de NumPy) NO se
    corrige aquí -- queda fuera del alcance acordado. Si ``previous == 0`` y
    ``current != 0`` el resultado sigue siendo ``inf``/``nan`` sin aviso.
    """
    if current == previous:
        return 0.0
    try:
        return (abs(current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return 0


####################
## Validación de Sigma (recomendación 3 de AUDITORIA_VENEGAS.md, sección 5)
####################


def validar_covarianza(Sigma, nombre="Sigma", tol=1e-10):
    """Chequea simetría y definición positiva de una matriz de covarianza.

    La versión original no valida nada antes de invertir Sigma (tres
    llamadas a ``np.linalg.inv`` sin chequeo previo, ver AUDITORIA_VENEGAS.md
    sección 5, punto 3). Una Sigma estimada de datos del BID/CORFO con
    valores faltantes puede no ser definida positiva, y el resultado sería
    silenciosamente inválido. Usa autovalores (``eigvalsh``, para matrices
    simétricas) en vez de Cholesky para poder reportar el autovalor mínimo
    exacto en el mensaje de error.

    Lanza ``ValueError`` con un mensaje explícito si falla; no devuelve nada
    si Sigma es válida.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"{nombre} debe ser cuadrada; forma recibida: {Sigma.shape}")

    asimetria = np.abs(Sigma - Sigma.T).max()
    if asimetria > tol:
        raise ValueError(
            f"{nombre} no es simétrica: max|Sigma - Sigma.T| = {asimetria:.3e} "
            f"(tolerancia {tol:.1e}). Una covarianza estimada de datos reales "
            "con missing values distintos por par puede romper la simetría; "
            "simetrizar explícitamente con 0.5*(Sigma+Sigma.T) antes de llamar "
            "a esta función si esa es la causa."
        )

    autovalores = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
    lambda_min = autovalores.min()
    if lambda_min <= tol:
        raise ValueError(
            f"{nombre} no es definida positiva: autovalor mínimo = "
            f"{lambda_min:.3e} (tolerancia {tol:.1e}). construct_q_matrices "
            "invierte esta matriz; con datos reales revisa missing values, "
            "ventana muestral insuficiente (menos observaciones que activos) "
            "o agrega regularización (shrinkage) antes de calibrar."
        )


def construct_q_matrices(Sigma_matrices, prohibition_matrices, Gamma=None, validar=True):
    """Igual que ``venegas_model.construct_q_matrices``, con validación de
    Sigma_i opcional (activada por defecto) antes de invertir.

    La construcción algebraica (factor 2, posición de Gamma) no cambia --
    está verificada y correcta en el original (ver verificacion_algebraica.md).
    Solo se agrega el chequeo de la recomendación 3 de la auditoría.
    """
    n = len(Sigma_matrices)
    if Gamma is None:
        Gamma = np.eye(n)
    if validar:
        for i, Sigma_i in enumerate(Sigma_matrices):
            validar_covarianza(Sigma_i, nombre=f"Sigma_{i}")

    out = []
    for i in range(n):
        inner = 2 * Gamma[i, i] * (prohibition_matrices[i] @ Sigma_matrices[i] @ prohibition_matrices[i].T)
        outer = prohibition_matrices[i].T @ np.linalg.inv(inner) @ prohibition_matrices[i]
        out.append(outer.copy())
    return out


####################
## H7 -- generadores sin semilla
####################


def _as_rng(rng):
    """Normaliza ``rng`` (int, None explícitamente rechazado, o Generator)."""
    if rng is None:
        raise ValueError(
            "rng es obligatorio (hallazgo H7): pasa un entero de semilla o un "
            "np.random.Generator. La versión original usaba np.random global "
            "sin semilla, así que cualquier resultado producido con ella era "
            "irreproducible -- ver AUDITORIA_VENEGAS.md, hallazgo H7."
        )
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def generate_firm_covariances(rng, num_firms=DEFAULT_NUM_FIRMS,
                              num_samples=DEFAULT_NUM_FIRMS,
                              risk_aversion=1.0, entry_distribution='gaussian',
                              noise_param=1.0, subtract_mean=False,
                              return_inv=False):
    """Igual que ``venegas_model.generate_firm_covariances``, pero exige un
    generador reproducible en vez de usar ``np.random`` global (H7).

    ``rng`` es ahora el primer argumento posicional (antes no existía) para
    que sea imposible de omitir por accidente.
    """
    rng = _as_rng(rng)
    cov_matrix = np.zeros(shape=(num_firms, num_firms))

    for i in range(num_samples):
        if entry_distribution == 'gaussian':
            A = rng.normal(loc=0.0, scale=noise_param, size=(num_firms, 1))
        elif entry_distribution == 'exponential':
            A = rng.exponential(noise_param, size=(num_firms, 1))
        elif entry_distribution == 'eye':
            A = np.eye(1, num_firms, i).T
        else:
            raise Exception(f'Distribution {entry_distribution} is unknown')
        symmA = np.matmul(A, A.T)
        cov_matrix += symmA
    cov_matrix_normalized = np.copy(cov_matrix)
    if subtract_mean:
        cov_matrix_normalized = cov_matrix - cov_matrix.mean(axis=1, keepdims=True)

    if entry_distribution != 'eye':
        cov_matrix_normalized *= (1.0 / num_samples)
    cov_matrix_normalized *= risk_aversion

    if return_inv:
        return cov_matrix_normalized, np.linalg.inv(cov_matrix_normalized)
    return cov_matrix_normalized


def generate_means(rng, num_firms=DEFAULT_NUM_FIRMS):
    """Igual que ``venegas_model.generate_means``, con ``rng`` obligatorio (H7)."""
    rng = _as_rng(rng)
    init = rng.normal(loc=5, scale=1.0, size=(num_firms, num_firms))
    for i in range(num_firms):
        init[i][i] = 0.0
    return init


####################
## H4b -- aristas prohibidas no heredaban P_init
####################


def run_price_dynamics_one_step(Q_matrices_list, mean_matrix, P_init,
                                prohibited_edges_tuples=(), eta=1.0):
    """Un paso de la dinámica de negociación (Algorithm 1 de Jalan et al.).

    Nombre sin el typo del original (``run_price_dyanmics_one_step``) --
    esta función ya no es una transcripción fiel, así que no hay razón para
    preservarlo.

    CORRIGE H4b: la versión original inicializa ``P_new`` en ceros y nunca
    escribe las entradas de aristas prohibidas, así que un ``P_init`` no nulo
    en una arista prohibida se pierde silenciosamente. Aquí ``P_new`` arranca
    como copia de ``P_init``, de modo que toda entrada que el loop no
    modifique explícitamente -- incluidas las prohibidas -- conserva su
    valor previo. Con ``P_init`` arrancando en ceros (el único caso que usa
    ``run_price_dynamics`` más abajo) el comportamiento es idéntico al
    original; difiere solo si se arranca con un ``P_init`` no nulo, que es
    exactamente lo que hará el acoplamiento T/A al recalentar la dinámica de
    cada red entre rondas externas.

    La actualización sigue siendo síncrona (tipo Jacobi): cada par lee
    siempre ``P_init``, nunca ``P_new`` -- ver la verificación numérica hecha
    en el chat sobre el Ejemplo 5 (diferencia de ~0.035 si se leyera el valor
    recién actualizado de otro par).
    """
    prohibited_edges_tuples = list(prohibited_edges_tuples)
    W_current = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_init)
    P_new = np.array(P_init, copy=True)
    n = P_init.shape[0]

    for i in range(n):
        for j in range(i):
            if (j, i) not in prohibited_edges_tuples and (i, j) not in prohibited_edges_tuples:
                pct_w_diff = get_percent_change(W_current[i, j], W_current[j, i])
                if np.abs(pct_w_diff) > 1e-1:
                    updated_price_ij = price_update_naive(i, j, P_init, Q_matrices_list,
                                                          mean_matrix, eta=eta)
                    P_new[i, j] = updated_price_ij
                    P_new[j, i] = -1 * updated_price_ij
                # si no hay discrepancia, P_new[i,j] ya heredó P_init[i,j]
                # (queda explícito para que quede claro que no es un no-op accidental)
                else:
                    P_new[i, j] = P_init[i, j]
                    P_new[j, i] = P_init[j, i]
    W_new = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_new)
    return W_new, P_new


####################
## H3 -- eta sin default silencioso
####################


def run_price_dynamics(sigma_shared, mean_matrix, prohibited_edges_tuples=(),
                       num_iterations=50, *, eta, Gamma=None, validar_sigma=True):
    """Itera la dinámica de precios ``num_iterations`` veces.

    CORRIGE H3: ``eta`` es ahora un argumento de solo palabra clave SIN
    default -- omitirlo lanza ``TypeError`` en vez de caer silenciosamente en
    ``eta=0.0`` (que en el original deja los precios congelados en cero para
    siempre, con apariencia de resultado válido). Ver ``eta_estable_teorema4``
    para elegir un valor con respaldo teórico (Teorema 4 de Jalan et al.) en
    vez de adivinar.

    H4b corregido vía ``run_price_dynamics_one_step`` (ver arriba). H9 (sin
    aviso de divergencia) sigue sin corregirse -- fuera del alcance acordado.
    """
    n = mean_matrix.shape[0]
    prohibited_edges_tuples = list(prohibited_edges_tuples)
    prohibition_matrices = construct_prohibition_matrices(n, prohibited_edges_tuples)
    Q_matrices_list = construct_q_matrices(
        [sigma_shared.copy() for _ in range(n)], prohibition_matrices,
        Gamma=Gamma, validar=validar_sigma)
    P_current = np.zeros_like(sigma_shared, dtype=float)
    W_current = opt_w_with_edge_constraints(Q_matrices_list, mean_matrix, P_current)

    W_list = [W_current]
    P_list = [P_current]

    for _ in range(num_iterations):
        W_current, P_current = run_price_dynamics_one_step(
            Q_matrices_list, mean_matrix, P_current, prohibited_edges_tuples, eta=eta)
        W_list.append(W_current)
        P_list.append(P_current)
    return W_list, P_list


####################
## Helper nuevo: región estable de eta (Teorema 4, Jalan & Chakrabarti 2024)
####################


def eta_estable_teorema4(Q_matrices_list, prohibited_edges_tuples=()):
    """Calcula eta* y eta_opt del Teorema 4 para una calibración dada.

    Implementa: K = sum_r [e_r e_r^T (kron) Q_r + Q_r (kron) e_r e_r^T],
    L diagonal con L[(i,j)] = Q_i[j,j] + Q_j[i,i], restringidos ambos a los
    pares permitidos R (excluye pares prohibidos y los self-pares i==j).
    eta* = 2/lambda_max((L^+ K)|_R); eta_opt = 2/(lambda_min+lambda_max).

    Para eta < eta* la convergencia es exponencial (Teorema 4); en
    eta = eta* se rompe la garantía; para eta > eta* el error se amplifica.
    eta_opt minimiza la tasa de convergencia asintótica.

    Validado contra el Ejemplo 5 con restricción (arista (0,2) prohibida):
    lambda_min=0.4467, lambda_max=1.5533, eta*=1.2876, eta_opt=1.0 -- los
    mismos valores obtenidos a mano y por simulación en la sesión previa
    sobre el paper (ver test_venegas_model_corregido.py::test_teorema4_*).

    AUDITORIA_VENEGAS.md, sección 5, punto 2: "Fijar eta explícitamente y
    re-verificar la región estable [...] Hay que medirla, no asumirla." Esta
    función es esa medición, en vez de reusar el eta=0.5 de la tesis con
    datos para los que nunca se verificó.

    Devuelve un dict con ``lambda_min``, ``lambda_max``, ``eta_star``,
    ``eta_opt`` y ``R`` (los índices de par usados, para depuración).
    """
    prohibited_edges_tuples = list(prohibited_edges_tuples)
    n = len(Q_matrices_list)

    def idx(i, j):
        return i * n + j

    K = np.zeros((n * n, n * n))
    for r in range(n):
        e_r = np.zeros((n, 1))
        e_r[r, 0] = 1.0
        err = e_r @ e_r.T
        K += np.kron(err, Q_matrices_list[r]) + np.kron(Q_matrices_list[r], err)

    Ldiag = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            Ldiag[idx(i, j)] = Q_matrices_list[i][j, j] + Q_matrices_list[j][i, i]

    with np.errstate(divide='ignore'):
        Lpinv_diag = np.where(Ldiag != 0, 1.0 / Ldiag, 0.0)
    M_full = np.diag(Lpinv_diag) @ K

    def permitido(i, j):
        return (i, j) not in prohibited_edges_tuples and (j, i) not in prohibited_edges_tuples

    R = [idx(i, j) for i in range(n) for j in range(n) if i != j and permitido(i, j)]
    if not R:
        raise ValueError("No hay pares permitidos: la red no tiene aristas activas.")

    M_R = M_full[np.ix_(R, R)]
    eigs = np.linalg.eigvals(M_R)
    if np.abs(eigs.imag).max() > 1e-8:
        warnings.warn(
            "Autovalores de (L^+K)|_R con parte imaginaria no despreciable "
            f"(max|Im|={np.abs(eigs.imag).max():.2e}); se usa la parte real. "
            "Puede indicar una Q_i mal condicionada -- revisar validar_covarianza.",
            RuntimeWarning)
    lam_min, lam_max = eigs.real.min(), eigs.real.max()

    if lam_max <= 0:
        raise ValueError(f"lambda_max no positivo ({lam_max:.3e}); revisar Q_matrices_list.")

    return dict(lambda_min=float(lam_min), lambda_max=float(lam_max),
                eta_star=float(2.0 / lam_max),
                eta_opt=float(2.0 / (lam_min + lam_max)), R=R)
