"""Capa de acoplamiento Tradicional/Alternativa del modelo dual.

Implementación propia de esta memoria, y el único módulo de la capa dual que
se versiona. Una primera versión de esta capa, escrita bajo el nombre
``venegas_dual.py``, quedó deliberadamente fuera del repositorio porque su
encabezado documentaba la versión equivocada de la objeción D1, refutada el
20-oct-2026 por la re-derivación simbólica del bloque C de
``caballero_verificacion_objeciones.py``. De aquí en adelante los cambios de
la capa dual se hacen sobre este archivo.

El motor de negociación por red se reusa sin modificar desde
``venegas_model_corregido.py``, que a su vez preserva intacto
``venegas_model.py``, en ``code/audit``. Esos dos conservan su nombre porque
documentan el código de la tesis que se continúa, mientras que esta capa es
aporte propio, y de ahí la separación entre ``code/audit`` y
``code/caballero``.

===========================================================================
LA REDUCCIÓN
===========================================================================

El agente tiene un portafolio total fijo $w_i^*$, que es su óptimo en la red
única sin restricciones, y decide qué fracción ejecutar en la red Tradicional.
Lo que no ejecuta ahí va a la Alternativa por construcción, ya que
$w_i^A = w_i^* - w_i^T$.

Sustituyendo esa restricción dentro de la utilidad dual del capítulo 3 de
[Venegas, cap. 3](../../docs/MemoriasTesis/Venegas/chapters/chapter03.tex),
sección *The Dual-Network Framework*, y agrupando términos, la condición de
primer orden queda

    w_i^{T*} = (1/2) H_i^{-1} a_i
    H_i = 2*gamma_i*Sigma_i + lambda_A*I
    a_i = (P^A - P^T)e_i - tau*1 + 2*gamma_i*Sigma_i*w_i^* + 2*lambda_A*w_i^*

que es la misma forma algebraica que la condición estándar de Jalan,
$w_i = Q_i(\\mu_i - P e_i)$, bajo el cambio de variable

    Sigma_eff = H_i        Gamma_eff = 1
    M_eff[:, i] = 2*gamma_i*Sigma_i*w_i^* + 2*lambda_A*w_i^* - tau*1
    P_eff = P^T - P^A      (el DIFERENCIAL de precio, no P^T por sí solo)

Por lo tanto resolver el acoplamiento es correr el mismo Algorithm 1 ya
auditado, con los primitivos transformados, sobre las restricciones
$\\Psi^T$ de la red Tradicional. No hace falta un algoritmo nuevo, y
$w_i^A$ sale como residuo, simétrico porque es diferencia de dos matrices
simétricas.

El término lineal y el Hessiano fueron re-derivados desde cero y coinciden
con el Apéndice A.1 de la tesis
([anexo_a.tex](../../docs/MemoriasTesis/Venegas/attachments/anexo_a.tex)),
verificado en el bloque A de ``caballero_verificacion_objeciones.py``.

===========================================================================
HALLAZGOS SOBRE LA ESPECIFICACIÓN DE LA FUENTE
===========================================================================

Cada uno indica entre corchetes el bloque de
``caballero_verificacion_objeciones.py`` que lo comprueba, de modo que la
afirmación y su verificación quedan ligadas. El argumento completo, con su
estatus dentro del trabajo, se desarrolla en el capítulo 4 y en el Anexo de
la memoria, que son su lugar único; aquí sólo se enuncia lo necesario para
entender qué hace el código y por qué.

C1 — Las dos contabilidades del riesgo son incompatibles [bloque F].
Es el hallazgo central. Si el riesgo se evalúa sobre la exposición combinada
$w_i^*$, el término cuadrático queda constante, dado que $w_i^*$ está fijo
por supuesto, y el Hessiano colapsa a $H_i = \\lambda_A I$, sin $\\Sigma$. Con
$\\lambda_A = 0$, que es el valor declarado en los escenarios de validación,
el problema del agente es lineal en $w_i^T$, de modo que no hay óptimo
interior y el agente queda indiferente entre todos los repartos. El 50/50 no
se puede derivar bajo esa contabilidad. La tesis necesita el riesgo separado
por red para que exista el óptimo del que sale el reparto, y el riesgo
combinado para que el bienestar total coincida con el benchmark de red
única, pero ninguna de las dos por sí sola entrega ambos resultados.
Este módulo expone las dos contabilidades por separado, en
``bienestar_riesgo_separado`` y ``bienestar_riesgo_combinado``, justamente
para que la diferencia sea explícita y medible en vez de quedar implícita.

C2 — La sobreestimación por fragmentación, en forma cerrada [bloque E].
Escribiendo $w_i^T = f w_i^*$, la diferencia entre ambas contabilidades es
exactamente $2f(1-f)\\sum_i \\gamma_i w_i^{*\\top}\\Sigma_i w_i^*$, cuya
derivada se anula sólo en $f = 1/2$. Es decir que la sobreestimación crece
de forma monótona y alcanza su máximo justo en el reparto que la condición
de primer orden elige cuando $\\tau = \\lambda_A = 0$. Fragmentar una
posición en dos mitades reduce el riesgo contabilizado sin que cambie nada
económico, lo que explica C1.

C3 — El precio único no admite punto factible bajo restricción [bloque D].
La definición de punto factible dual del capítulo 3 exige un único precio
compartido entre ambas redes. Bajo ese supuesto $a_i$ no depende del precio,
por lo que cada agente resuelve su reparto por separado y no queda
instrumento alguno con el cual negociar. Sin restricción eso es inocuo, ya
que $w_i^T = w_i^*/2$ hereda la simetría de $W^*$. Con la restricción activa,
en cambio, la proyección $\\Psi_i^T$ difiere por agente y el resultado deja
de ser simétrico, con una asimetría de $2{,}27\\times 10^{-1}$ frente a una
tolerancia de convergencia del motor de $7{,}7\\times 10^{-4}$, lo que viola
la condición de simetría que la propia definición exige. El diferencial
$P^T - P^A$ es el único instrumento que la restaura. Ver
``resolver_acoplamiento_precio_unico``.

C4 — La fórmula de Q^dual está mal transcrita [bloque B].
La forma impresa en el capítulo 3 y repetida en el Apéndice A.1,
$\\Psi^\\top(\\Psi H^{-1}\\Psi^\\top)^{-1}\\Psi H^{-1}$ con $H^{-1}$ dentro de
la inversa, no satisface la condición de estacionariedad proyectada, y
evaluada en $\\Psi = I$ da la identidad, lo que contradice la simplificación
$Q^{dual} = H^{-1}$ que el mismo enunciado declara dos líneas después. La
forma que sí resuelve el problema restringido es
$(1/2)\\Psi^\\top(\\Psi H \\Psi^\\top)^{-1}\\Psi$, con $H$ dentro de la
inversa, análoga a la de Jalan. Es un error de transcripción sin
consecuencia numérica, dado que este módulo implementa la forma correcta.

ADVERTENCIA sobre la objeción D1 original, hoy refutada [bloque C].
``venegas_dual.py`` afirmaba que bajo precio único el reparto simétrico
daría 100 por ciento a la red Tradicional, porque el término de precio
desaparece de $a_i$. El término en efecto desaparece, pero la conclusión no
se sigue, ya que la condición de primer orden da $w_i^T = w_i^*/2$ exacto,
es decir el 50/50 que la tesis publica. El reparto 100/0 aparece sólo al
usar la Proposición del capítulo 3 sin el factor $1/2$ que sí tiene el
apéndice. La objeción correcta al precio único es C3, no aquélla.

===========================================================================
ALCANCE DE REPRODUCCIÓN
===========================================================================

Se reproducen exactamente las cifras publicadas que no dependen del reparto
T/A, y ninguna de las que sí dependen de él [bloque G]. Las utilidades
totales coinciden al cuarto decimal, con 1,4339 contra 1,434 publicado en
tres agentes y 1,7417 contra 1,741 en cuatro, pero lo hacen porque bajo
riesgo combinado sólo dependen de $w_i^*$, que no cambia entre escenarios,
de modo que no contienen información sobre la capa dual. Las tasas de
adopción, los volúmenes por red y el diferencial de precio, que son las
únicas cifras que sí miden el reparto, no se reproducen con los parámetros
que la tesis declara. Ver ``test_caballero_dual.py``, grupo
``test_discrepancia_*``, donde las cifras no reproducidas quedan fijadas
como regresión en vez de quedar fuera de la suite.
"""

import numpy as np

from venegas_model_corregido import construct_prohibition_matrices, run_price_dynamics


def _verificar_gamma_homogenea(Gamma, tol=1e-12):
    """Rechaza una Gamma con aversión al riesgo heterogénea.

    La reducción pasa un único ``H`` como covarianza efectiva a
    ``run_price_dynamics``, lo que presupone $H_i = H$ para todo agente y por
    lo tanto $\\gamma_i = \\gamma$. Con $\\gamma_i$ distintos entre agentes
    cada $H_i$ difiere y un único arreglo no puede representarlos, de modo
    que habría que construir las $Q_i$ directamente en vez de delegar en el
    motor. ``venegas_dual.py`` leía ``Gamma[0, 0]`` e ignoraba el resto en
    silencio, lo que daba resultados incorrectos sin aviso; aquí se exige
    homogeneidad de forma explícita.

    La extensión a $\\gamma_i \\neq \\gamma_j$ corresponde al Hito 2 y no está
    implementada.
    """
    d = np.diag(Gamma)
    if np.abs(d - d[0]).max() > tol:
        raise ValueError(
            "Esta capa supone aversion al riesgo homogenea (gamma_i iguales); "
            f"se recibio diag(Gamma) = {d}. La extension a agentes heterogeneos "
            "requiere construir las Q_i por agente en vez de delegar en "
            "run_price_dynamics, y corresponde al Hito 2."
        )
    off = Gamma - np.diag(d)
    if np.abs(off).max() > tol:
        raise ValueError("Gamma debe ser diagonal; se recibieron entradas fuera de la diagonal.")


def calcular_w_estrella(Sigma, M, Gamma, eta=0.5, num_iterations=500):
    """$w_i^*$, el punto estable de la red única sin restricciones.

    Es el insumo que la definición de *dual network setting* del capítulo 3
    toma como dado, descrito allí como el portafolio óptimo del agente en el
    caso de red única sin restricciones.
    """
    W_list, P_list = run_price_dynamics(Sigma, M, [], num_iterations=num_iterations,
                                        eta=eta, Gamma=Gamma)
    return W_list[-1], P_list[-1]


def tolerancia_simetria(W_star):
    """Asimetría residual de $W^*$, que fija el umbral de comparación.

    El propio motor converge a una $W^*$ simétrica sólo hasta su tolerancia
    de iteración, de manera que cualquier asimetría por debajo de este valor
    es ruido de convergencia y no una violación de factibilidad. Se usa para
    juzgar C3 en vez de comparar contra cero.
    """
    return float(np.abs(W_star - W_star.T).max())


def resolver_acoplamiento(W_star, Sigma, Gamma, tau, lambda_A, prohibidas_T,
                          eta=0.5, num_iterations=500):
    """Resuelve el reparto T/A reduciéndolo a una corrida del motor auditado.

    Parameters
    ----------
    W_star : (n, n) -- columna i es $w_i^*$, de ``calcular_w_estrella``.
    Sigma, Gamma : los primitivos originales con los que se obtuvo ``W_star``.
        ``Gamma`` debe ser diagonal y homogénea, ver
        ``_verificar_gamma_homogenea``.
    tau : beneficio de transparencia de la red Alternativa, no negativo.
    lambda_A : fricción operacional de la red Alternativa, no negativa.
    prohibidas_T : aristas prohibidas en la red Tradicional. La red
        Alternativa se supone sin restricciones, es decir $\\Psi^A = I$;
        generalizar a una $\\Psi^A$ propia queda pendiente de que se defina
        con el profesor guía qué constituye la red Alternativa concreta.

    Returns
    -------
    W_T, W_A : la asignación en cada red, con $W^A = W^* - W^T$ por
        construcción.
    P_eff : el DIFERENCIAL $P^T - P^A$, antisimétrico. No es $P^T$ por sí
        solo, y en los pares que la red Tradicional nunca negocia queda en su
        valor inicial sin reflejar el diferencial verdadero. Recuperar el
        nivel de $P^T$ requiere información que las ecuaciones publicadas no
        determinan, lo que impide reproducir las utilidades por agente.
    """
    _verificar_gamma_homogenea(Gamma)
    if tau < 0 or lambda_A < 0:
        raise ValueError(f"tau y lambda_A deben ser no negativos; se recibio tau={tau}, lambda_A={lambda_A}")

    n = W_star.shape[0]
    H = 2 * Gamma[0, 0] * Sigma + lambda_A * np.eye(n)
    unos = np.ones(n)

    M_eff = np.column_stack([
        2 * Gamma[i, i] * Sigma @ W_star[:, i] + 2 * lambda_A * W_star[:, i] - tau * unos
        for i in range(n)
    ])

    W_list, P_list = run_price_dynamics(H, M_eff, prohibidas_T,
                                        num_iterations=num_iterations, eta=eta,
                                        Gamma=np.eye(n))
    W_T = W_list[-1]
    return W_T, W_star - W_T, P_list[-1]


def resolver_acoplamiento_precio_unico(W_star, Sigma, Gamma, tau, lambda_A, prohibidas_T):
    """El reparto bajo el supuesto de precio único del capítulo 3, para C3.

    Con $P^T = P^A$ el término $(P^A - P^T)e_i$ se anula dentro de $a_i$, de
    modo que el reparto no depende del precio y cada agente resuelve por
    separado, sin negociación. No hay dinámica que iterar.

    Existe para hacer ejecutable la objeción C3, ya que el $W^T$ que devuelve
    no es simétrico bajo restricción y por lo tanto no es un punto factible
    según la propia definición de la fuente. No se usa para reportar
    resultados.
    """
    _verificar_gamma_homogenea(Gamma)
    n = W_star.shape[0]
    H = 2 * Gamma[0, 0] * Sigma + lambda_A * np.eye(n)
    Psis = construct_prohibition_matrices(n, list(prohibidas_T))
    cols = []
    for i in range(n):
        a_i = (-tau * np.ones(n) + 2 * Gamma[i, i] * Sigma @ W_star[:, i]
               + 2 * lambda_A * W_star[:, i])
        Ps = Psis[i]
        # Proyeccion correcta (1/2)Psi'(Psi H Psi')^-1 Psi, ver C4.
        cols.append(0.5 * Ps.T @ np.linalg.inv(Ps @ H @ Ps.T) @ Ps @ a_i)
    W_T = np.column_stack(cols)
    return W_T, W_star - W_T


def bienestar_riesgo_combinado(W_A, W_star, M, Sigma, Gamma, tau, lambda_A):
    """Bienestar total con el riesgo evaluado sobre la exposición combinada.

    Es la contabilidad que reproduce las utilidades totales publicadas, con
    1,4339 contra 1,434 en tres agentes y 1,7417 contra 1,741 en cuatro. Los
    términos de precio se cancelan de forma exacta en la suma sobre agentes,
    dado que $\\operatorname{tr}(WP) = 0$ para toda $W$ simétrica y $P$
    antisimétrica, por lo que no hacen falta $P^T$ ni $P^A$.

    Advertencia de interpretación, ver C1. Bajo esta contabilidad el
    bienestar total depende sólo de $w_i^*$, que está fijo por supuesto y no
    cambia entre escenarios, de manera que su invariancia frente a la
    restricción, a $\\tau$ y a $\\lambda_A$ es una identidad algebraica y no
    un resultado emergente del modelo. No es la contabilidad consistente con
    la condición de primer orden que produce el reparto.
    """
    n = W_star.shape[0]
    return float(sum(
        W_star[:, i] @ M[:, i] + tau * W_A[:, i].sum()
        - Gamma[i, i] * (W_star[:, i] @ Sigma @ W_star[:, i])
        - lambda_A * (W_A[:, i] @ W_A[:, i])
        for i in range(n)
    ))


def bienestar_riesgo_separado(W_T, W_A, M, Sigma, Gamma, tau, lambda_A):
    """Bienestar total con el riesgo evaluado por separado en cada red.

    Es la forma literal de la utilidad del capítulo 3, y la única consistente
    con la condición de primer orden de la que sale el reparto. Da 2,1507 en
    el escenario simétrico y 1,7869 en el restringido, de modo que no
    reproduce el 1,434 publicado y, a diferencia de la otra contabilidad, no
    es invariante frente a la restricción, ya que cae un 16,9 por ciento.

    La diferencia respecto de ``bienestar_riesgo_combinado`` es la brecha de
    fragmentación de C2, con valor $2f(1-f)\\sum_i \\gamma_i
    w_i^{*\\top}\\Sigma_i w_i^*$ cuando $w_i^T = f w_i^*$.
    """
    n = W_T.shape[0]
    total = 0.0
    for i in range(n):
        wT, wA = W_T[:, i], W_A[:, i]
        total += (wT @ M[:, i] - Gamma[i, i] * (wT @ Sigma @ wT)
                  + wA @ (M[:, i] + tau) - Gamma[i, i] * (wA @ Sigma @ wA)
                  - lambda_A * (wA @ wA))
    return float(total)


def brecha_fragmentacion(W_star, Sigma, Gamma, f):
    """La brecha entre ambas contabilidades en forma cerrada, para C2.

    Vale $2f(1-f)\\sum_i \\gamma_i w_i^{*\\top}\\Sigma_i w_i^*$ cuando el
    reparto es $w_i^T = f w_i^*$, y alcanza su máximo en $f = 1/2$.
    """
    return float(2 * f * (1 - f) * sum(
        Gamma[i, i] * W_star[:, i] @ Sigma @ W_star[:, i] for i in range(W_star.shape[0])
    ))


def cuota_red_alternativa(W_T, W_A):
    """Fracción del volumen bilateral que capta la red Alternativa.

    Es la métrica que la fuente reporta como *market share* o tasa de
    adopción, calculada sobre el triángulo superior estricto en valor
    absoluto. Con los parámetros declarados por la tesis entrega 81,6 por
    ciento en tres agentes y 75,0 por ciento en cuatro, frente a 72,4 y 47,9
    publicados, discrepancia que queda fijada en la suite de pruebas.
    """
    from metricas_tesis import bilateral_volume
    vT, vA = bilateral_volume(W_T), bilateral_volume(W_A)
    return float(vA / (vT + vA))
