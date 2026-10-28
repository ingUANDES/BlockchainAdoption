"""Corrida del modelo dual sobre el caso chileno BAJO SUPUESTOS DECLARADOS.

NO es la calibración del objetivo específico 3 y no debe presentarse como tal.
El objetivo 3 pide calibrar con las series levantadas del programa y contrastar
con la evidencia observada, y eso sigue bloqueado por el resultado de
identificabilidad del capítulo 3, a saber que tomando las dos redes en conjunto
no hay ninguna diferencia $\\Delta W(t)$ utilizable. Lo que este módulo produce
es una predicción del modelo condicional a supuestos, todos declarados y
trazados a su fuente o marcados como no fijados por el capítulo 3.

===========================================================================
TOPOLOGÍA DEL CASO, Y POR QUÉ NO ES LA DE VENEGAS
===========================================================================

Cuatro agentes, CORFO (0), Intermediario financiero no bancario (1), MiPyME (2)
y Banco tradicional (3), repartidos en dos redes.

  Red Tradicional   CORFO--IFNB y IFNB--MiPyME. La arista CORFO--MiPyME está
                    prohibida de forma permanente, porque CORFO opera como
                    banca de segundo piso. El Banco no participa.
  Red Alternativa   Banco--MiPyME, un único par sin restricciones. CORFO y el
                    IFNB no participan.

En la configuración de cuatro agentes de la tesis que se continúa, en cambio,
el cuarto agente es un intermediario competidor DENTRO de la misma red, todos
los agentes participan en ambas redes y la red Alternativa reconecta al mismo
par que la restricción separó. Aquí ninguna de esas tres cosas ocurre, y las
consecuencias son las tres que siguen.

PRIMERA. ``caballero_dual.resolver_acoplamiento`` no corre sobre esta topología
sin adaptarla, porque fija $\\Psi^A = I$, es decir supone la red Alternativa sin
restricciones sobre TODOS los pares. Aquí la red Alternativa tiene un solo par
admitido, de modo que esa función asignaría contrato a aristas que no existen,
por ejemplo CORFO--Banco. Ver ``diagnostico_topologia``.

SEGUNDA, y es la que obliga a cambiar de formulación. Bajo el supuesto de
portafolio fijo $w_i^T + w_i^A = w_i^*$ de la fuente, el reparto de la MiPyME
deja de ser una decisión. Sus contrapartes admisibles son el IFNB en la red
Tradicional y el Banco en la Alternativa, conjuntos DISJUNTOS, de modo que
$w_i^T$ y $w_i^A$ viven en subespacios de coordenadas ortogonales y la
restricción de portafolio determina cada parte sin que haya nada que optimizar.
En consecuencia $\\lambda_A$ y $\\tau$ no podrían mover el reparto, que es
precisamente el objeto de interés. Verificado en ``diagnostico_topologia``.

TERCERA. La red Alternativa NO reconecta el par que la restricción separa. En
la fuente, el contrato bloqueado CORFO--MiPyME migra a la red Alternativa y de
ahí viene la recuperación de bienestar. Aquí el Banco es una contraparte
distinta, no una vía alternativa hacia CORFO, y la arista CORFO--MiPyME sigue
ausente en ambas redes. El mecanismo de recuperación de la fuente, por lo
tanto, no se traslada a este caso, y esperar aquí la recuperación completa que
aquélla reporta sería un error de lectura.

ADAPTACIÓN ADOPTADA. Se abandona el supuesto de portafolio fijo, que en esta
topología es vacío, y se resuelve el sistema como UNA negociación sobre la
unión de aristas de ambas redes, donde la pertenencia a la red Alternativa no
cambia quién puede contratar con quién sino qué términos enfrenta esa arista.
Formalmente, la utilidad del agente $i$ es

    g_i = w_i'(mu_i + tau*1_A - P e_i) - gamma_i w_i' Sigma w_i
          - lambda_A * ||Pi_A w_i||^2

con $\\Pi_A$ la proyección sobre las coordenadas de contrapartes de la red
Alternativa. La condición de primer orden conserva la forma estándar con

    Q_i = Psi_i' (Psi_i (2 gamma_i Sigma + 2 lambda_A Pi_A) Psi_i')^{-1} Psi_i

de modo que la mecánica de negociación es la misma ya auditada y sólo cambian
los primitivos. Se reusa ``run_price_dynamics_one_step`` del motor corregido
pasándole estas $Q_i$, sin reimplementar la dinámica.

Nótese que esta formulación SÍ acopla ambas redes, pero por una vía distinta de
la de la fuente. El acoplamiento es el término cruzado
$2 w_{i,\\mathrm{IFNB}} \\Sigma_{\\mathrm{IFNB},\\mathrm{Banco}} w_{i,\\mathrm{Banco}}$
en el riesgo de la MiPyME, es decir la covarianza entre sus dos contrapartes, y
no una restricción de portafolio.

===========================================================================
ESTADO DE CADA PARÁMETRO
===========================================================================

VERIFICADO, con fuente en el capítulo 3
  r_w   tasa mayorista CORFO hacia IFNB, 4,23 a 4,73 por ciento anual.
  r_m   tasa de mercado que enfrenta la MiPyME, 14 a 15,4 por ciento anual.
  r_s   tasa subsidiada IFNB hacia MiPyME. El capítulo declara 3 a 6 por
        ciento, pero su propia derivación da otro intervalo, ver
        ``INCONSISTENCIA_R_S`` más abajo.
  gamma aversión al riesgo igual a uno para todos, por alcance del capítulo 1.

SUPUESTO EN ESTE MÓDULO, porque el capítulo 3 no lo fija
  rho       retorno esperado del proyecto de la MiPyME, que fija el NIVEL de
            su columna de creencias. Ver ``construir_M``.
  m_banco   margen del Banco sobre su costo de fondo.
  lambda_A  el capítulo ancla la fricción de forma cualitativa y dice que es
            sustancialmente mayor que la de la red Tradicional, sin asignarle
            valor. Ver ``LAMBDA_A_PROPUESTO``.
  tau       no observable, va por sensibilidad.
  Sigma     no estimable, ver ``sigma_neutra`` y ``sigma_economica``.
"""

import numpy as np

import _ruta_audit  # noqa: F401  deja code/audit importable

from venegas_model_corregido import (construct_prohibition_matrices,
                                     run_price_dynamics_one_step,
                                     opt_w_with_edge_constraints,
                                     eta_estable_teorema4)

# --------------------------------------------------------------------- agentes
CORFO, IFNB, MIPYME, BANCO = 0, 1, 2, 3
NOMBRES = ["CORFO", "IFNB", "MiPyME", "Banco"]
N = 4

#: Aristas de cada red. La unión es el conjunto de pares que pueden contratar.
ARISTAS_T = [(CORFO, IFNB), (IFNB, MIPYME)]
ARISTAS_A = [(MIPYME, BANCO)]

#: Pares prohibidos en el sistema completo: todo lo que no está en la unión.
#: Incluye CORFO--MiPyME, que es la restricción permanente de banca de segundo
#: piso, y CORFO--Banco e IFNB--Banco, que el capítulo 3 reporta como aristas
#: concebibles pero sin series disponibles, de modo que aquí se excluyen.
PROHIBIDAS = [(CORFO, MIPYME), (CORFO, BANCO), (IFNB, BANCO)]

# ------------------------------------------------------------------ parámetros
R_W = (0.0423, 0.0473)      # mayorista CORFO -> IFNB        [DIPRES, verificado]
R_M = (0.1400, 0.1540)      # mercado Banco -> MiPyME        [ELE-7,  verificado]
R_S_DECLARADO = (0.03, 0.06)  # subsidiada IFNB -> MiPyME, cifra que declara el cap. 3
DESCUENTO_PP = (0.07, 0.10)   # puntos bajo el benchmark, cifra que declara el cap. 3

#: El capítulo 3 declara r_s entre 3 y 6 por ciento, pero lo deriva restando un
#: descuento de 7 a 10 puntos al benchmark de mercado de 14 a 15,4 por ciento,
#: lo que da 4 a 8,4 por ciento. Los dos intervalos no coinciden. Se reportan
#: ambos y se usa el declarado como base, por ser el que el documento afirma.
INCONSISTENCIA_R_S = dict(
    declarado=R_S_DECLARADO,
    derivado=(R_M[0] - DESCUENTO_PP[1], R_M[1] - DESCUENTO_PP[0]),
)


def r_s_derivado():
    """Intervalo de r_s que se obtiene de la propia aritmética del capítulo 3."""
    return INCONSISTENCIA_R_S["derivado"]


def margen_ifnb(r_s, r_w):
    """Margen del IFNB, que es lo que cobra menos lo que paga por el fondeo.

    Si sale negativo el IFNB pierde dinero por cada unidad colocada y el modelo
    predice que no participa, de modo que el programa no opera. Es una
    restricción de admisibilidad sobre la combinación (r_s, r_w), no un
    resultado, y conviene chequearla antes de interpretar cualquier corrida.
    """
    return r_s - r_w


def region_admisible(r_s_rango=R_S_DECLARADO, r_w_rango=R_W):
    """Fracción del rectángulo (r_s, r_w) donde el IFNB tiene margen positivo."""
    rs = np.linspace(*r_s_rango, 201)
    rw = np.linspace(*r_w_rango, 201)
    RS, RW = np.meshgrid(rs, rw, indexing="ij")
    ok = margen_ifnb(RS, RW) > 0
    return dict(fraccion_admisible=float(ok.mean()),
                margen_min=float(margen_ifnb(RS, RW).min()),
                margen_max=float(margen_ifnb(RS, RW).max()))


# ------------------------------------------------------- matrices del problema
def psi_sistema():
    """Proyecciones $\\Psi_i$ sobre la unión de aristas de ambas redes."""
    return construct_prohibition_matrices(N, PROHIBIDAS)


def proyectores_alternativa():
    """$\\Pi_A^i$ por agente, diagonal con uno en las contrapartes de la red A.

    Sólo la MiPyME y el Banco tienen coordenadas en la red Alternativa, de modo
    que para CORFO y el IFNB el proyector es nulo y la fricción no los toca.
    """
    Pis = []
    for i in range(N):
        d = np.zeros(N)
        for (a, b) in ARISTAS_A:
            if i == a:
                d[b] = 1.0
            elif i == b:
                d[a] = 1.0
        Pis.append(np.diag(d))
    return Pis


def construir_M(r_w, r_s, r_m, delta, m_banco, v_corfo=0.0):
    """Matriz de creencias de retorno esperado, en tasa anual decimal.

    Este es el paso que el capítulo 3 no deja explícito, de modo que la
    traducción se documenta entrada por entrada. La convención del modelo es
    que $M_{ij}$ es el retorno esperado que el agente $i$ anticipa por unidad
    de contrato con $j$, y que ambas partes deben anticipar retorno positivo
    para que el contrato se forme, ya que el precio $P$ sólo reparte el
    excedente y no lo crea.

    Cada entrada registra la ganancia NETA del propio agente en esa arista, lo
    que evita el error de contar dos veces. Conviene ser explícito sobre ese
    error, porque aquí es fácil de cometer. El IFNB gana un margen $r_s - r_w$
    por intermediar, y si ese margen se anota en sus dos aristas, la que lo une
    a CORFO y la que lo une a la MiPyME, su utilidad queda proporcional a
    $W_{01} + W_{12}$, es decir cuenta dos veces el mismo dinero, que entra por
    una arista y sale por la otra. Separar costo de ingreso lo corrige, puesto
    que deja la utilidad del IFNB en $r_s W_{12} - r_w W_{01}$, que recupera el
    margen una sola vez cuando ambos contratos coinciden en tamaño.

      M[CORFO, IFNB]   = +r_w        ingreso de CORFO por colocar al mayorista.
                                     Su costo de fondo es fiscal y el cap. 3 no
                                     lo cuantifica, de modo que se toma nulo.
      M[IFNB, CORFO]   = -r_w        costo del fondeo para el IFNB. Es negativo
                                     y debe serlo, ya que tomar fondeo sólo
                                     tiene valor por lo que permite colocar.
      M[IFNB, MiPyME]  = +r_s        ingreso del IFNB al colocar al subsidio.
      M[MiPyME, IFNB]  = s + delta   con s = r_m - r_s el subsidio observado.
      M[Banco, MiPyME] = m_banco     margen neto del Banco sobre su costo de
                                     fondo, que el cap. 3 no observa.
      M[MiPyME, Banco] = delta       excedente de la MiPyME en el canal de
                                     mercado.

    Sobre ``delta``. La columna de la MiPyME es $\\rho - r$, con $\\rho$ el
    retorno esperado de lo que financia, cantidad que el capítulo 3 no
    determina. Escribiendo $\\delta = \\rho - r_m$ queda que sólo $\\delta$ es
    desconocido, mientras que la DIFERENCIA entre ambos canales es
    $s = r_m - r_s$, el subsidio, que sí está determinado por cantidades
    observadas. Dicho de otro modo, los datos fijan cuánto más conviene el
    canal subsidiado, pero no cuánto conviene cada uno en términos absolutos.

    $\\delta = 0$ significa que la MiPyME marginal es indiferente ante el
    crédito bancario, lo que anula el valor del canal de mercado y, por esa
    vía, de la red Alternativa completa. Es el borde inferior del rango
    admisible y no un valor central, puesto que las MiPyMEs efectivamente se
    endeudan con bancos a tasa de mercado. Como ninguna fuente lo fija, se
    recorre $\\delta \\in [0, s]$, cota superior que equivale a suponer que el
    canal de mercado, por sí solo, vale tanto como el subsidio entero.

    Sobre ``v_corfo``. Con sólo los supuestos del capítulo 3, el ingreso de
    CORFO y el costo del IFNB en la arista mayorista son exactamente opuestos,
    de modo que su excedente conjunto es cero y esa arista no tiene ganancia de
    intercambio que la sostenga.

    Conviene no exagerar la consecuencia, porque el contrato NO se anula. Sobre
    las 384 combinaciones del barrido con $v = 0$ queda en una mediana de
    0,0045 en valor absoluto, nunca exactamente cero, y lo que persiste es una
    posición de cobertura cuyo signo lo fija $\\Sigma$ y no la economía del
    programa, con mediana -0,0057 bajo la matriz equicorrelada y +0,0034 bajo
    la de ciclo, es decir el 9 por ciento de la arista IFNB--MiPyME. Sin
    valoración de política el modelo no representa el programa en esa arista,
    puesto que lo que predice ahí es un residuo de cobertura que puede ir en
    cualquier dirección.

    La causa es que el modelo no tiene restricción de balance que ate lo que el
    IFNB coloca a lo que se fondea, de manera que nada lo obliga a tomar el
    fondeo mayorista. Se cierra agregando a CORFO una valoración de política
    $v$ por unidad colocada, que es lo que distingue a una agencia de fomento
    de un prestamista comercial. El capítulo 3 no la cuantifica, de modo que
    entra al barrido. La alternativa sería imponer la restricción de balance,
    lo que cambia el modelo y excede este encargo; queda anotada como la vía
    estructural.
    """
    s = subsidio(r_s, r_m)
    M = np.zeros((N, N))
    M[CORFO, IFNB] = r_w + v_corfo
    M[IFNB, CORFO] = -r_w
    M[IFNB, MIPYME] = r_s
    M[MIPYME, IFNB] = s + delta
    M[BANCO, MIPYME] = m_banco
    M[MIPYME, BANCO] = delta
    return M


def subsidio(r_s, r_m):
    """$s = r_m - r_s$, la única diferencia de la columna de la MiPyME que los
    datos del capítulo 3 determinan."""
    return r_m - r_s


def excedente_conjunto(M):
    """$M_{ij} + M_{ji}$ por arista admitida.

    Si es menor o igual a cero el par no gana nada por contratar y el modelo
    lleva ese contrato a cero, dado que el precio sólo reparte excedente y no
    lo crea. Conviene revisarlo antes de interpretar cualquier corrida, porque
    un contrato nulo puede venir de aquí y no de la estructura de riesgo.
    """
    return {f"{NOMBRES[a]}--{NOMBRES[b]}": float(M[a, b] + M[b, a])
            for a, b in ARISTAS_T + ARISTAS_A}


# ------------------------------------------------------------------ covarianzas
def sigma_neutra(rho_c=0.3):
    """Equicorrelación. Todos los pares comparten la misma correlación.

    Es la alternativa sin contenido económico, útil justamente porque no
    privilegia ninguna estructura y sirve de referencia contra la cual medir
    cuánto aporta imponer una.
    """
    S = np.full((N, N), float(rho_c))
    np.fill_diagonal(S, 1.0)
    return S


def sigma_economica(rho_fin=0.75, rho_real=0.35, rho_publico=-0.20,
                    sigma_corfo_mipyme=None):
    """Correlaciones por exposición al ciclo, con tres bloques.

    El criterio es el siguiente. El IFNB y el Banco son ambos oferta de crédito
    privada y comparten exposición al ciclo financiero, de modo que su
    correlación es alta. La MiPyME es sector real y se correlaciona de forma
    moderada con los oferentes de crédito. CORFO es contracíclico, puesto que
    un programa público de fomento se expande cuando el crédito privado se
    contrae, de modo que su correlación con los agentes privados es negativa.

    ``sigma_corfo_mipyme`` permite fijar directamente la entrada
    CORFO--MiPyME, que es la del barrido principal. Si queda en None toma
    ``rho_publico``.
    """
    s_cm = rho_publico if sigma_corfo_mipyme is None else float(sigma_corfo_mipyme)
    S = np.eye(N)
    S[IFNB, BANCO] = S[BANCO, IFNB] = rho_fin
    S[MIPYME, IFNB] = S[IFNB, MIPYME] = rho_real
    S[MIPYME, BANCO] = S[BANCO, MIPYME] = rho_real
    S[CORFO, IFNB] = S[IFNB, CORFO] = rho_publico
    S[CORFO, BANCO] = S[BANCO, CORFO] = rho_publico
    S[CORFO, MIPYME] = S[MIPYME, CORFO] = s_cm
    return S


def es_definida_positiva(S, tol=1e-10):
    return bool(np.linalg.eigvalsh(0.5 * (S + S.T)).min() > tol)


# ------------------------------------------------------------------- solución
def construir_Q(Sigma, Gamma, lambda_A):
    """$Q_i$ con la fricción de la red Alternativa dentro de la inversa.

    Es la forma $(1/2)\\Psi_i'(\\Psi_i H_i \\Psi_i')^{-1}\\Psi_i$ verificada en el
    bloque B de ``caballero_verificacion_objeciones.py``, con
    $H_i = 2\\gamma_i\\Sigma + 2\\lambda_A\\Pi_A^i$. Se escribe sin el factor
    $1/2$ para que coincida con la convención del motor, que absorbe el 2 en
    $H$, a saber $Q_i = \\Psi_i'(\\Psi_i(2\\gamma_i\\Sigma+2\\lambda_A\\Pi_A^i)
    \\Psi_i')^{-1}\\Psi_i$.
    """
    Psis, Pis = psi_sistema(), proyectores_alternativa()
    Qs = []
    for i in range(N):
        H_i = 2 * Gamma[i, i] * Sigma + 2 * lambda_A * Pis[i]
        Ps = Psis[i]
        Qs.append(Ps.T @ np.linalg.inv(Ps @ H_i @ Ps.T) @ Ps)
    return Qs


def construir_M_efectiva(M, tau):
    """Agrega el beneficio de transparencia sólo a las coordenadas de la red A."""
    Pis = proyectores_alternativa()
    M_eff = M.copy().astype(float)
    for i in range(N):
        M_eff[:, i] = M[:, i] + tau * np.diag(Pis[i])
    return M_eff


def resolver_caso(Sigma, M, Gamma, lambda_A, tau, eta=None, num_iterations=500):
    """Punto estable del sistema de cuatro agentes sobre la unión de aristas.

    Si ``eta`` es None lo elige con el Teorema 4, tomando
    $\\eta = \\eta_{opt}$ si cae bajo $\\eta^*$, que es la medición que la
    auditoría recomienda en vez de heredar un valor sin verificar.
    """
    if not es_definida_positiva(Sigma):
        raise ValueError("Sigma no es definida positiva.")
    Q = construir_Q(Sigma, Gamma, lambda_A)
    M_eff = construir_M_efectiva(M, tau)

    diag_eta = eta_estable_teorema4(Q, PROHIBIDAS)
    if eta is None:
        eta = min(diag_eta["eta_opt"], 0.9 * diag_eta["eta_star"])

    P = np.zeros((N, N))
    W = opt_w_with_edge_constraints(Q, M_eff, P)
    for _ in range(num_iterations):
        W, P = run_price_dynamics_one_step(Q, M_eff, P, PROHIBIDAS, eta=eta)

    if not np.all(np.isfinite(W)) or np.abs(P).max() > 1e10:
        raise RuntimeError(f"la dinamica divergio con eta={eta}")

    return dict(W=W, P=P, eta=eta, diag_eta=diag_eta,
                asimetria=float(np.abs(W - W.T).max()))


def descomponer(W):
    """Separa el contrato de cada red y devuelve volúmenes y adopción.

    La adopción se calcula sobre valores absolutos, siguiendo la convención de
    volumen bilateral de la capa de métricas, pero un contrato NEGATIVO no es
    interpretable como crédito, puesto que significaría que la MiPyME presta al
    Banco. Por eso se devuelve además ``contrato_negativo``, y una adopción
    alta con esa marca activa no debe leerse como adopción de la red
    Alternativa sino como que el modelo se salió del dominio que representa.
    """
    wT = sum(abs(W[a, b]) for a, b in ARISTAS_T)
    wA = sum(abs(W[a, b]) for a, b in ARISTAS_A)
    negativos = [f"{NOMBRES[a]}--{NOMBRES[b]}" for a, b in ARISTAS_T + ARISTAS_A
                 if W[a, b] < -1e-9]
    return dict(volumen_T=float(wT), volumen_A=float(wA),
                adopcion_A=float(wA / (wT + wA)) if (wT + wA) > 0 else float("nan"),
                contrato_negativo=negativos)


def utilidades(W, P, Sigma, M, Gamma, lambda_A, tau):
    """Utilidad por agente, con el riesgo sobre la exposición total del agente.

    Es la contabilidad de riesgo combinado, que es la única consistente en este
    caso porque aquí no hay partición de portafolio y cada agente tiene una
    sola exposición. La tensión entre contabilidades que el capítulo 4
    documenta para la configuración de la fuente no se presenta aquí.
    """
    Pis = proyectores_alternativa()
    M_eff = construir_M_efectiva(M, tau)
    u = np.zeros(N)
    for i in range(N):
        w = W[:, i]
        u[i] = (w @ (M_eff[:, i] - P[:, i])
                - Gamma[i, i] * (w @ Sigma @ w)
                - lambda_A * (w @ Pis[i] @ w))
    return u


# -------------------------------------------------- fricción de la red A
#: El capítulo 3 ancla lambda_A de forma cualitativa, con una tasa de rechazo
#: de 15 por ciento para MiPyMEs frente a 3 por ciento para empresas grandes y
#: 111 de 1.018 empresas en autocensura, pero no le asigna valor. La propuesta
#: de este módulo es anclarlo en la probabilidad de NO obtener el crédito en el
#: canal bancario, que combina rechazo y autocensura, y escalarlo por el
#: término de riesgo para que sea adimensional respecto de las tasas. Es una
#: propuesta de cierre, no una medición, y por eso el barrido la acompaña.
TASA_RECHAZO_MIPYME = 0.15
TASA_RECHAZO_GRANDE = 0.03
AUTOCENSURA = 111 / 1018


def lambda_A_propuesto(escala=1.0):
    """Fricción de acceso propuesta, como probabilidad de no acceder al canal.

    Combina rechazo condicional y autocensura como eventos sobre poblaciones
    disjuntas, según precisa el capítulo 3, de modo que la probabilidad de no
    acceder es la suma ponderada y no el producto. El factor ``escala`` queda
    expuesto porque la traducción de una probabilidad a una penalización
    cuadrática no es única y es justamente lo que el barrido debe cubrir.
    """
    p_no_accede = TASA_RECHAZO_MIPYME + AUTOCENSURA * (1 - TASA_RECHAZO_MIPYME)
    return float(escala * p_no_accede)


LAMBDA_A_PROPUESTO = lambda_A_propuesto()


# ------------------------------------------------- diagnóstico de la topología
def diagnostico_topologia():
    """Comprueba las dos razones por las que la capa dual no corre tal cual.

    Devuelve un diccionario con la evidencia, para que la afirmación del
    encabezado sea verificable y no una aserción del autor.

    ``soportes``        contrapartes admisibles de cada agente en cada red, y
                        si las dos listas comparten alguna. Si ninguna las
                        comparte, la restricción de portafolio
                        $w_i^T + w_i^A = w_i^*$ determina ambas partes sin que
                        quede nada que optimizar, de modo que $\\lambda_A$ y
                        $\\tau$ no pueden mover el reparto.
    ``psi_A_implicito`` pares que ``caballero_dual.resolver_acoplamiento``
                        admitiría en la red Alternativa, dado que fija
                        $\\Psi^A = I$, contra los que esta red realmente tiene.
    ``arista_bloqueada_se_restaura`` si la red Alternativa reconecta el par que
                        la restricción separa, que es el mecanismo de
                        recuperación de la fuente.
    """
    PsiT = construct_prohibition_matrices(N, _prohibidas_de(ARISTAS_T))
    PsiA = construct_prohibition_matrices(N, _prohibidas_de(ARISTAS_A))

    soportes, disjuntos = {}, True
    for i in range(N):
        cT, cA = _contrapartes(PsiT[i], i), _contrapartes(PsiA[i], i)
        comun = sorted(set(cT) & set(cA))
        disjuntos &= (len(comun) == 0)
        soportes[NOMBRES[i]] = dict(en_T=[NOMBRES[j] for j in cT],
                                    en_A=[NOMBRES[j] for j in cA],
                                    comparten=[NOMBRES[j] for j in comun])

    todos = [(a, b) for a in range(N) for b in range(a + 1, N)]
    return dict(
        soportes=soportes,
        soportes_disjuntos=bool(disjuntos),
        reparto_es_decision=not bool(disjuntos),
        psi_A_implicito=dict(
            admite_resolver_acoplamiento=[f"{NOMBRES[a]}--{NOMBRES[b]}" for a, b in todos],
            red_A_real=[f"{NOMBRES[a]}--{NOMBRES[b]}" for a, b in ARISTAS_A],
            pares_espurios=[f"{NOMBRES[a]}--{NOMBRES[b]}" for a, b in todos
                            if (a, b) not in ARISTAS_A and (b, a) not in ARISTAS_A],
        ),
        arista_bloqueada=f"{NOMBRES[CORFO]}--{NOMBRES[MIPYME]}",
        arista_bloqueada_se_restaura=bool(
            (CORFO, MIPYME) in ARISTAS_A or (MIPYME, CORFO) in ARISTAS_A),
    )


def _prohibidas_de(aristas):
    """Pares prohibidos si sólo se admiten las aristas dadas."""
    permitidas = {tuple(sorted(e)) for e in aristas}
    return [(a, b) for a in range(N) for b in range(a + 1, N)
            if (a, b) not in permitidas]


def _contrapartes(Psi_i, i):
    s = {int(j) for j in np.flatnonzero(np.abs(Psi_i).sum(axis=0) > 0)}
    return sorted(s - {i})
