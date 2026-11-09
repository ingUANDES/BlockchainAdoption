"""Rama contrafactual: cuánto bienestar deja sobre la mesa el mandato de
segundo piso.

CORRIDA BAJO SUPUESTOS DECLARADOS. No es la calibración del objetivo
específico 3. El caso principal de cuatro agentes no se toca; esto es una rama
aparte.

Tres agentes, CORFO (0), intermediario financiero no bancario (1) y MiPyME (2).
No hay banco de mercado. La red Tradicional prohíbe el par (0,2), igual que el
caso principal, y la red Alternativa va sin restricciones, es decir
$\\Psi^A = I$, que es lo que ``caballero_dual.resolver_acoplamiento`` hace de
fábrica y que aquí se reusa sin modificar. En esa red CORFO sí puede prestar
directo a la MiPyME.

A diferencia del caso principal de cuatro agentes, esta topología SÍ es la de la
tesis que se continúa, puesto que la red Alternativa reconecta el mismo par que
la restricción separa. Por eso la capa de acoplamiento corre aquí tal cual,
mientras que allá no, según documenta ``caballero_caso_chileno``.

===========================================================================
LOS DOS SUPUESTOS QUE HUBO QUE AGREGAR, Y QUIÉN LOS DECIDIÓ
===========================================================================

Ambos se consultaron antes de elegir, porque ninguno se sostiene con lo
declarado en el capítulo 3, y los dos los resolvió el autor el 26-oct-2026.

PRIMERO, las creencias del par CORFO--MiPyME. Al restringir la construcción de
cuatro agentes eliminando la fila y columna del banco, ese par queda en cero,
porque en el caso principal está prohibido en ambas redes y nunca se le
asignaron creencias. Con excedente conjunto cero el par no contrata, que es
justo el par que este contrafactual existe para habilitar. El capítulo 3 no
dice a qué tasa prestaría CORFO directo, porque es precisamente lo que el
mandato impide y por eso nunca se observa.

Decisión adoptada, ``CIERRE = "mayorista"``. CORFO presta directo a su propia
tasa de fondeo, de modo que $M_{\\text{CORFO,MiPyME}} = r_w + v$ y
$M_{\\text{MiPyME,CORFO}} = (r_m - r_w) + \\delta$. Es la lectura literal de
saltarse al intermediario, dado que CORFO cobra lo mismo que cobra hoy al IFNB
y la MiPyME captura además el margen de intermediación. **Supone que CORFO
puede originar crédito minorista al mismo costo que mayorista, es decir sin
costo de evaluación ni de cobranza, que es parte de lo que justifica el
mandato**, y esa es la principal limitación del ejercicio. El cierre
alternativo, a tasa subsidiada, queda implementado en ``construir_M3`` para que
la comparación sea reproducible; mueve la fracción recuperada entre 67,0 y
67,1 por ciento y deja la descomposición por par idéntica, porque es un
cociente donde el nivel se cancela.

SEGUNDO, cómo medir el bienestar recuperado. La fracción de la brecha no se
comporta como métrica única. Con la contabilidad de riesgo combinado, que es la
que reproduce las cifras publicadas, da exactamente 100 por ciento cuando
$\\tau$ y $\\lambda_A$ son cero, y no porque la red Alternativa repare algo sino
porque es la identidad algebraica que el capítulo 4 documenta: con el
portafolio total fijo, el bienestar bajo riesgo combinado depende sólo de
$w^*$, que no cambia entre escenarios. Al mover esos parámetros se dispara
fuera de todo rango interpretable, porque la referencia sin restricción no
contiene ni $\\tau$ ni $\\lambda_A$ y el cociente termina mezclando deshacer la
restricción con agregar un beneficio que la referencia nunca tuvo.

Decisión adoptada, separar las dos preguntas.

  (i)  El COSTO DEL MANDATO es la brecha $U_{\\text{ref}} - U_T$, que no
       necesita la capa dual ni ningún parámetro no observable, y es por lo
       tanto lo más fuerte que la sección puede afirmar.
  (ii) La FRACCIÓN RECUPERADA se reporta con $\\tau = 0$, barriendo sólo
       $\\lambda_A$, donde sí es interpretable como cuánto del costo del mandato
       sobrevive a la fricción de acceso.

El efecto de $\\tau$ se reporta aparte, sobre el volumen de la red Alternativa,
que es la magnitud que $\\tau$ sí mueve sin contaminar la referencia.
"""

import numpy as np

import _ruta_audit  # noqa: F401  deja code/audit importable

import caballero_caso_chileno as cc
from caballero_dual import (resolver_acoplamiento, bienestar_riesgo_combinado,
                            calcular_w_estrella)
from venegas_model_corregido import run_price_dynamics

# ------------------------------------------------------------------- agentes
CORFO, IFNB, MIPYME = 0, 1, 2
NOMBRES = ["CORFO", "IFNB", "MiPyME"]
N = 3
IDX = [cc.CORFO, cc.IFNB, cc.MIPYME]      # para restringir la construcción de 4
GAMMA = np.eye(N)

#: Par que el mandato de segundo piso prohíbe en la red Tradicional.
PAR_MANDATO = (CORFO, MIPYME)
PROHIBIDAS_T = [PAR_MANDATO]

#: Cierre decidido por el autor el 26-oct-2026, ver encabezado.
CIERRE = "mayorista"

#: Pares de la red Alternativa, que va sin restricciones.
PARES = [(CORFO, IFNB), (CORFO, MIPYME), (IFNB, MIPYME)]


def construir_M3(r_w, r_s, r_m, delta, v_corfo, cierre=CIERRE):
    """Creencias de tres agentes, restringiendo la construcción de cuatro.

    Se elimina la fila y la columna del banco y se completan las dos entradas
    del par CORFO--MiPyME, que quedan en cero al restringir. Ver el encabezado
    para por qué hay que completarlas y con qué criterio.
    """
    M = cc.construir_M(r_w, r_s, r_m, delta, m_banco=0.0,
                       v_corfo=v_corfo)[np.ix_(IDX, IDX)].copy()
    if cierre == "mayorista":
        M[CORFO, MIPYME] = r_w + v_corfo
        M[MIPYME, CORFO] = (r_m - r_w) + delta
    elif cierre == "subsidiada":
        M[CORFO, MIPYME] = r_s + v_corfo
        M[MIPYME, CORFO] = cc.subsidio(r_s, r_m) + delta
    else:
        raise ValueError(f"cierre desconocido: {cierre!r}")
    return M


def sigmas3(sigma_corfo_mipyme=None):
    """Las dos variantes de covarianza del caso principal, restringidas a tres.

    ``sigma_corfo_mipyme`` sobreescribe esa entrada en ambas, que es la del
    barrido pedido por ser la única que la estructura bipartita identifica.
    """
    neutra = cc.sigma_neutra()[np.ix_(IDX, IDX)].copy()
    ciclo = cc.sigma_economica()[np.ix_(IDX, IDX)].copy()
    if sigma_corfo_mipyme is not None:
        for S in (neutra, ciclo):
            S[CORFO, MIPYME] = S[MIPYME, CORFO] = float(sigma_corfo_mipyme)
    return {"equicorrelada": neutra, "por ciclo": ciclo}


def _bienestar_red_unica(Sigma, M, prohibidas):
    """Punto estable y bienestar de UNA red, sin capa dual.

    Usa la misma contabilidad de riesgo combinado que la capa dual con
    $W^A = 0$, de modo que las tres magnitudes del cociente son comparables.
    """
    W = run_price_dynamics(Sigma, M, prohibidas, num_iterations=500, eta=0.5,
                           Gamma=GAMMA)[0][-1]
    U = bienestar_riesgo_combinado(np.zeros((N, N)), W, M, Sigma, GAMMA, 0.0, 0.0)
    return W, U


def costo_del_mandato(Sigma, M):
    """Entrega (i): la brecha que abre la prohibición del par CORFO--MiPyME.

    No interviene la capa dual ni ningún parámetro no observable, de modo que
    es la cifra más robusta del ejercicio. ``brecha_relativa`` la expresa como
    fracción del bienestar de referencia, que es lo comparable entre
    escenarios dado que los niveles no tienen unidad interpretable.
    """
    W_ref, U_ref = _bienestar_red_unica(Sigma, M, [])
    W_T, U_T = _bienestar_red_unica(Sigma, M, PROHIBIDAS_T)
    return dict(U_ref=U_ref, U_T=U_T, brecha=U_ref - U_T,
                brecha_relativa=(U_ref - U_T) / U_ref if U_ref != 0 else np.nan,
                W_ref=W_ref, W_T=W_T)


def fraccion_recuperada(Sigma, M, lambda_A, tau=0.0):
    """Entrega (i) segunda parte: cuánto del costo del mandato sobrevive.

    Con ``tau = 0`` es interpretable como la fracción del costo del mandato que
    la red Alternativa recupera pese a la fricción de acceso. Con ``tau > 0``
    deja de serlo, porque la referencia no contiene $\\tau$; se permite pasarlo
    sólo para poder exhibir ese problema, y el informe no lo usa como titular.
    """
    c = costo_del_mandato(Sigma, M)
    W_T_dual, W_A, P_eff = resolver_acoplamiento(c["W_ref"], Sigma, GAMMA, tau,
                                                 lambda_A, PROHIBIDAS_T)
    U_dual = bienestar_riesgo_combinado(W_A, c["W_ref"], M, Sigma, GAMMA, tau,
                                        lambda_A)
    frac = ((U_dual - c["U_T"]) / c["brecha"]) if abs(c["brecha"]) > 1e-12 else np.nan
    return dict(**{k: v for k, v in c.items() if k not in ("W_ref", "W_T")},
                U_dual=U_dual, fraccion=frac, W_A=W_A, W_T_dual=W_T_dual,
                P_eff=P_eff, lambda_A=lambda_A, tau=tau)


def descomponer_por_par(W_A, brecha, lambda_A, tau):
    """Entrega (3): reparto por par del volumen de A y del déficit de bienestar.

    Hay que distinguir dos cosas que no se descomponen igual.

    El VOLUMEN de la red Alternativa sí se reparte por par de forma directa, y
    es la atribución que el capítulo puede citar para decir cuánto del
    movimiento pasa por el par del mandato.

    El BIENESTAR RECUPERADO, en cambio, **no es atribuible por par** en esta
    contabilidad, y conviene que el capítulo lo diga. Bajo riesgo combinado con
    portafolio total fijo se tiene $U_{\\text{dual}} = U_{\\text{ref}}$ cuando
    $\\tau = \\lambda_A = 0$, cualquiera sea el conjunto de pares que la red
    Alternativa habilite. Es decir que la recuperación no proviene de ningún
    par, sino del supuesto de portafolio fijo. Lo que sí se descompone de forma
    exacta es la DESVIACIÓN respecto de la referencia, puesto que

        U_dual - U_ref = -lambda_A * sum_i ||Pi_A w_i||^2 + tau * sum_i (w_i^A . 1)

    y ambos términos son sumas sobre coordenadas. Por eso se reporta el déficit
    respecto del 100 por ciento repartido por coordenada, que es exacto, en vez
    de un reparto del bienestar recuperado, que no lo sería.

    Una advertencia sobre la contabilidad del reparto. La suma recorre TODAS las
    coordenadas de $W^A$, y eso incluye la diagonal, que es la posición propia de
    cada agente y no un contrato con nadie. Omitirla deja el reparto corto en un
    8 por ciento aproximadamente, de modo que se reporta como una fila aparte
    llamada ``posicion propia`` en vez de repartirla entre los pares, que sería
    atribuirle a un contrato algo que no lo es. Cada par $(a,b)$ aparece en la
    columna de $a$ y en la de $b$, de manera que entra con
    $W^A_{ab}{}^2 + W^A_{ba}{}^2$; no se usa $2 W^A_{ab}{}^2$ porque $W^A$ sólo
    es simétrica hasta la tolerancia de convergencia.
    """
    vol = {f"({a},{b}) {NOMBRES[a]}-{NOMBRES[b]}": float(W_A[a, b]) for a, b in PARES}
    total = sum(abs(v) for v in vol.values())
    share_vol = {k: (abs(v) / total if total > 0 else np.nan) for k, v in vol.items()}

    deficit = {}
    for a, b in PARES:
        cuad = W_A[a, b] ** 2 + W_A[b, a] ** 2
        lineal = W_A[a, b] + W_A[b, a]
        deficit[f"({a},{b}) {NOMBRES[a]}-{NOMBRES[b]}"] = float(
            (-lambda_A * cuad + tau * lineal) / brecha)
    cuad_d = sum(W_A[i, i] ** 2 for i in range(N))
    lineal_d = sum(W_A[i, i] for i in range(N))
    deficit["posicion propia (diagonal)"] = float(
        (-lambda_A * cuad_d + tau * lineal_d) / brecha)

    tot_def = sum(deficit.values())
    share_def = {k: (v / tot_def if tot_def != 0 else np.nan) for k, v in deficit.items()}
    return dict(volumen=vol, share_volumen=share_vol,
                deficit_fraccion=deficit, share_deficit=share_def,
                deficit_total=tot_def)


# ---------------------------------------------------------------- la rejilla
#: Al eliminar al banco desaparece la dimensión de su creencia de retorno
#: (``m_banco``), de modo que la rejilla de 768 del caso principal se reduce a
#: la mitad. Se reporta el conteo efectivo en la corrida, no se asume.
REJILLA = dict(
    r_w=list(cc.R_W), r_m=list(cc.R_M), r_s=list(cc.R_S_DECLARADO),
    v_corfo=[0.0, 0.05], lambda_A=[0.0, cc.LAMBDA_A_PROPUESTO, 2.0],
    tau=[0.0, 0.05],
)

#: Intervalos de los dos barridos, declarados para que la comparación de
#: amplitudes no quede implícita. Son los mismos del caso principal.
INTERVALO_LAMBDA = (0.0, 5.0)
INTERVALO_TAU = (0.0, 0.20)

#: Rango admisible de Sigma[CORFO,MiPyME] bajo la construcción chilena. Se
#: verifica en la corrida que cada valor deje la matriz definida positiva.
RANGO_SIGMA_CM = (-0.45, 0.90)


def friccion_de_equilibrio(Sigma, M, lo=0.01, hi=5.0):
    """Fricción por sobre la cual la red Alternativa cuesta más que el mandato.

    Es la raíz de ``fraccion_recuperada`` como función de $\\lambda_A$ con
    $\\tau = 0$. Existe en ``[lo, hi]`` porque la fracción vale exactamente 1 en
    $\\lambda_A = 0$ y decrece de forma monótona, de modo que cruza cero una sola
    vez; ``brentq`` exige que el intervalo cambie de signo y lanza si no lo hace.

    Vive en el módulo, y no suelta en un notebook, porque el informe cita su
    valor al tercer decimal y esa precisión tiene que ser reproducible desde un
    archivo. El barrido de la corrida sólo la encierra entre sus nodos contiguos
    0,5 y 1,0; interpolar entre ellos coincide con la raíz al tercer decimal,
    pero no quedaría respaldado por el módulo.
    """
    from scipy.optimize import brentq
    return float(brentq(
        lambda L: fraccion_recuperada(Sigma, M, L, 0.0)["fraccion"],
        lo, hi, xtol=1e-9))
