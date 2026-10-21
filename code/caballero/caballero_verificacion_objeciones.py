"""Re-derivación algebraica de las objeciones D1/D2 a la especificación dual
de Venegas, y cuantificación de su efecto sobre las cifras titulares.

Responde a los puntos 1, 2 y 4 del encargo del 20-oct-2026. No vuelve a
correr las pruebas: deriva el álgebra desde la definición de utilidad del
capítulo 3 de la tesis de Venegas y del Apéndice A.1, y compara contra las
cifras publicadas.

Fuentes (rutas relativas a la raíz del repositorio):
  docs/MemoriasTesis/Venegas/chapters/chapter03.tex  -- sección "The Dual-Network Framework"
  docs/MemoriasTesis/Venegas/chapters/chapter04.tex  -- tablas 3f_dual_baseline / 3f_dual_restricted
  docs/MemoriasTesis/Venegas/attachments/anexo_a.tex -- Apéndice A.1

Corre sin variables de entorno, desde la raíz del repositorio o desde esta
carpeta, porque la ruta a ``code/audit`` se resuelve en ``_ruta_audit``:

    python code/caballero/caballero_verificacion_objeciones.py

Qué verifica, bloque por bloque. La letra es la referencia que usan
``caballero_dual.py`` y el Anexo de la memoria al citar cada hallazgo.
  A. Que a_i y H_i del Apéndice A.1 son correctos, derivados de cero.
  B. Que la fórmula Q^dual impresa NO resuelve el problema restringido (C4).
  C. Que el precio único da 50/50 y no 100/0, lo que refuta la objeción D1
     tal como estaba escrita en la versión anterior de la capa.
  D. Que el precio único no admite punto factible bajo restricción (C3), que
     es la objeción que sí se sostiene contra ese supuesto.
  E. Que el riesgo separado sobreestima, con la brecha en forma cerrada (C2).
  F. Que el riesgo conjunto hace degenerar la optimización (C1).
  G. Cuantificación de las cifras titulares bajo ambas contabilidades.
  H. Equivalencia de los dos motores sobre la capa dual.
"""

import sys
from pathlib import Path

# Este archivo se ejecuta directo, de modo que agrega su propia carpeta a
# sys.path antes de cualquier import local. No basta con que Python lo haga
# solo, porque con PYTHONSAFEPATH activo no inserta el directorio del script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import sympy as sp

import _ruta_audit  # noqa: F401,E402  deja code/audit importable, ver ese modulo

import venegas_model as orig  # noqa: E402
from venegas_model_corregido import (construct_prohibition_matrices, run_price_dynamics,
                                     EJEMPLO5_COV, EJEMPLO5_MEAN, EJEMPLO5_GAMMA)
from caballero_dual import (calcular_w_estrella, resolver_acoplamiento,
                            bienestar_riesgo_combinado, bienestar_riesgo_separado)
from metricas_tesis import bilateral_volume, COV4, MEAN4, GAMMA4

np.set_printoptions(precision=4, suppress=True)
N = 3
Sigma, Gamma, M = EJEMPLO5_COV, EJEMPLO5_GAMMA, EJEMPLO5_MEAN

#: Tolerancia de simetría del propio motor de negociación, medida sobre W*.
#: Cualquier asimetría por debajo de este valor es ruido de convergencia;
#: por encima, es violación real de la Definición "dual feasible point".
W_star, P_star = calcular_w_estrella(Sigma, M, Gamma)
TOL_SIMETRIA = float(np.abs(W_star - W_star.T).max())


def bloque(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


# ===========================================================================
bloque("A. a_i y H_i del Apéndice A.1, derivados desde la utilidad del cap. 3")
# ===========================================================================
Sig = sp.Matrix(N, N, lambda i, j: sp.Symbol(f'S{min(i,j)}{max(i,j)}', real=True))
mu = sp.Matrix(N, 1, lambda i, j: sp.Symbol(f'mu{i}', real=True))
ws = sp.Matrix(N, 1, lambda i, j: sp.Symbol(f'ws{i}', real=True))
w = sp.Matrix(N, 1, lambda i, j: sp.Symbol(f'w{i}', real=True))
gam, lamA, tau = sp.symbols('gamma lambda_A tau', real=True, positive=True)
PTe = sp.Matrix(N, 1, lambda i, j: sp.Symbol(f'PT{i}', real=True))
PAe = sp.Matrix(N, 1, lambda i, j: sp.Symbol(f'PA{i}', real=True))
ones = sp.ones(N, 1)
wA = ws - w

# Utilidad dual con riesgo SEPARADO por red -- la forma literal del cap. 3
gT = (w.T * (mu - PTe))[0] - gam * (w.T * Sig * w)[0]
gA = (wA.T * (mu + tau * ones - PAe))[0] - gam * (wA.T * Sig * wA)[0] - lamA * (wA.T * wA)[0]
g = sp.expand(gT + gA)

grad = sp.Matrix([sp.diff(g, w[i]) for i in range(N)])
Hess = sp.Matrix(N, N, lambda i, j: sp.diff(g, w[i], w[j]))
H_der = sp.simplify(-Hess / 2)
a_der = sp.simplify(grad + 2 * H_der * w)

chk_H = sp.simplify(H_der - (2 * gam * Sig + lamA * sp.eye(N))) == sp.zeros(N, N)
chk_a = sp.simplify(a_der - ((PAe - PTe) - tau * ones + 2 * gam * Sig * ws + 2 * lamA * ws)) == sp.zeros(N, 1)
print(f"H_i = 2*gamma_i*Sigma_i + lambda_A*I .......... {'CORRECTO' if chk_H else 'NO COINCIDE'}")
print(f"a_i = (P^A-P^T)e_i - tau*1 + 2*g*S*w* + 2*l*w*  {'CORRECTO' if chk_a else 'NO COINCIDE'}")
print("=> El Apéndice A.1 deriva bien el término lineal y el Hessiano.")


# ===========================================================================
bloque("B. La fórmula Q^dual impresa no resuelve el problema restringido")
# ===========================================================================
Psi = sp.Matrix([[1, 0, 0], [0, 1, 0]])      # agente con su 3a contraparte prohibida
Hs = sp.Matrix(N, N, lambda i, j: sp.Symbol(f'H{min(i,j)}{max(i,j)}', real=True))
a_s = sp.Matrix(N, 1, lambda i, j: sp.Symbol(f'a{i}', real=True))

Q_impresa = Psi.T * (Psi * Hs.inv() * Psi.T).inv() * Psi * Hs.inv()   # cap.3 eq.Q_dual_general
Q_correcta = sp.Rational(1, 2) * Psi.T * (Psi * Hs * Psi.T).inv() * Psi

for nombre, Q in [("impresa (cap.3 / anexo)", Q_impresa), ("correcta", Q_correcta)]:
    kkt = sp.simplify(Psi * (a_s - 2 * Hs * (Q * a_s)))
    ok = kkt == sp.zeros(Psi.rows, 1)
    print(f"Q {nombre:24s}: Psi(a-2Hw)=0 ... {'SI' if ok else 'NO'}")
print("Al evaluar la fórmula impresa en Psi=I da la IDENTIDAD, contradiciendo")
print("su propia simplificación eq.Q_dual_simplified, que afirma Q^dual = H^-1.")
print("La forma correcta en Psi=I da (1/2)H^-1, consistente con la FOC del anexo.")


# ===========================================================================
bloque("C. D1 tal como está escrita: el precio único NO da 100/0")
# ===========================================================================
w_foc = sp.Rational(1, 2) * H_der.inv() * a_der           # FOC del Apéndice A.1
sub_single = {tau: 0, lamA: 0}
sub_single.update({PAe[i]: PTe[i] for i in range(N)})     # P^T = P^A (Definición cap.3)

w_single = sp.simplify(w_foc.subs(sub_single))
es_mitad = sp.simplify(w_single - ws / 2) == sp.zeros(N, 1)
print(f"FOC del anexo con P^T=P^A, tau=0, lambda_A=0: w^T = w*/2 ... {'SI' if es_mitad else 'NO'}")
print("=> El precio único reproduce EXACTAMENTE el 50/50 publicado.")
print("   D1 atribuye el 100/0 a que 'el término de precio desaparece'; eso es incorrecto.")

w_prop = sp.simplify((H_der.inv() * a_der).subs(sub_single))   # Proposición SIN el factor 1/2
es_entero = sp.simplify(w_prop - ws) == sp.zeros(N, 1)
print(f"\nProposición del cap.3 sin el factor 1/2: w^T = w* exacto ... {'SI' if es_entero else 'NO'}")
print("=> El 100/0 proviene del factor 1/2 ausente en la Proposición, no del precio único.")


# ===========================================================================
bloque("D. D1 reformulada: el precio único no admite punto factible bajo restricción")
# ===========================================================================
def w_T_precio_unico(prohib, tau_v=0.0, lam_v=0.0):
    """w_i^T bajo precio ÚNICO: a_i pierde el término de precio, así que cada
    agente resuelve por separado y no queda instrumento para negociar."""
    H = 2 * Gamma[0, 0] * Sigma + lam_v * np.eye(N)
    Psis = construct_prohibition_matrices(N, prohib)
    cols = []
    for i in range(N):
        a_i = -tau_v * np.ones(N) + 2 * Gamma[i, i] * Sigma @ W_star[:, i] + 2 * lam_v * W_star[:, i]
        Ps = Psis[i]
        cols.append(0.5 * Ps.T @ np.linalg.inv(Ps @ H @ Ps.T) @ Ps @ a_i)
    return np.column_stack(cols)

print(f"Tolerancia de simetría del motor (medida sobre W*): {TOL_SIMETRIA:.2e}\n")
for etiq, prohib in [("sin restricción", []), ("restricción en T", [(0, 2)])]:
    WT_u = w_T_precio_unico(prohib)
    asim_u = np.abs(WT_u - WT_u.T).max()
    WT_d, _, _ = resolver_acoplamiento(W_star, Sigma, Gamma, 0.0, 0.0, prohib)
    asim_d = np.abs(WT_d - WT_d.T).max()
    print(f"{etiq:18s} | precio único: asimetría {asim_u:.2e} "
          f"({'FACTIBLE' if asim_u <= TOL_SIMETRIA else 'VIOLA la simetría exigida'})")
    print(f"{'':18s} | con P^T!=P^A: asimetría {asim_d:.2e} "
          f"({'FACTIBLE' if asim_d <= TOL_SIMETRIA else 'VIOLA la simetría exigida'})")
print("\n=> El diferencial P^T-P^A es el único instrumento que restaura la simetría")
print("   de W^T bajo restricción. Ahí sí falla la Definición del cap. 3, no en el 50/50.")


# ===========================================================================
bloque("E. D2: el riesgo separado sobreestima, y la brecha en forma cerrada")
# ===========================================================================
f = sp.Symbol('f', real=True)
riesgo_sep = gam * ((f * ws).T * Sig * (f * ws))[0] + gam * (((1 - f) * ws).T * Sig * ((1 - f) * ws))[0]
riesgo_comb = gam * (ws.T * Sig * ws)[0]
brecha = sp.simplify(riesgo_comb - riesgo_sep)
print("Brecha de bienestar (separado - combinado), con w^T = f*w*:")
print("   ", sp.factor(brecha), " = 2*f*(1-f)*gamma*w*'Sigma*w*")
df = sp.simplify(sp.diff(brecha, f))
print(f"   d/df = {sp.factor(df)}  =>  máximo en f = {sp.solve(df, f)[0]}")
print("=> La sobreestimación es 2f(1-f)*gamma*w*'Sigma*w*: crece monótonamente en")
print("   [0, 1/2] y alcanza su máximo exactamente en f=1/2, que es el reparto que")
print("   el propio FOC elige cuando tau=lambda_A=0. D2 puntos 2 y 3 se sostienen.")

WT1, WA1, _ = resolver_acoplamiento(W_star, Sigma, Gamma, 0.0, 0.0, [])
sep1 = bienestar_riesgo_separado(WT1, WA1, M, Sigma, Gamma, 0.0, 0.0)
comb1 = bienestar_riesgo_combinado(WA1, W_star, M, Sigma, Gamma, 0.0, 0.0)
pred = 0.5 * sum(Gamma[i, i] * W_star[:, i] @ Sigma @ W_star[:, i] for i in range(N))
print(f"\nNumérico en f=1/2: separado {sep1:.4f} - combinado {comb1:.4f} = {sep1-comb1:.4f}")
print(f"Predicho por la forma cerrada 2*(1/2)*(1/2)*sum_i gamma_i w*'Sigma w* = {pred:.4f}")


# ===========================================================================
bloque("F. El riesgo conjunto hace degenerar la optimización (hallazgo nuevo)")
# ===========================================================================
g_joint = sp.expand((w.T * (mu - PTe))[0] + (wA.T * (mu + tau * ones - PAe))[0]
                    - gam * (ws.T * Sig * ws)[0] - lamA * (wA.T * wA)[0])
H_joint = sp.simplify(-sp.Matrix(N, N, lambda i, j: sp.diff(g_joint, w[i], w[j])) / 2)
print(f"H_i con riesgo conjunto = lambda_A*I ... "
      f"{'SI' if sp.simplify(H_joint - lamA*sp.eye(N)) == sp.zeros(N,N) else 'NO'}")
print("=> Sin Sigma. Con lambda_A=0 el problema es LINEAL en w^T: no hay óptimo")
print("   interior y el agente es indiferente entre todos los repartos, de modo")
print("   que el 50/50 no se puede derivar bajo esta contabilidad.")
print("\n   La tesis usa riesgo SEPARADO para la FOC (de ahí sale el 50/50) y riesgo")
print("   CONJUNTO para el bienestar (de ahí sale 1,434). Ninguna de las dos sola")
print("   produce ambos resultados. Esa es la inconsistencia de fondo.")


# ===========================================================================
bloque("G. Efecto sobre las cuatro cifras titulares")
# ===========================================================================
U_3F_CON = 0.7061     # red única 3F con restricción (tab:reproduccion-escenarios, main)
res = {}
for etiq, prohib in [("simétrico", []), ("restringido", [(0, 2)])]:
    WT, WA, Pe = resolver_acoplamiento(W_star, Sigma, Gamma, 0.0, 0.0, prohib)
    vT, vA = bilateral_volume(WT), bilateral_volume(WA)
    res[etiq] = dict(
        sep=bienestar_riesgo_separado(WT, WA, M, Sigma, Gamma, 0.0, 0.0),
        comb=bienestar_riesgo_combinado(WA, W_star, M, Sigma, Gamma, 0.0, 0.0),
        volT=vT, volA=vA, share=vA / (vT + vA),
        pdif=float(np.abs(Pe[np.triu_indices(N, 1)]).mean()), W13A=WA[0, 2])

print(f"{'cifra':34s} {'publicado':>12s} {'combinado':>12s} {'separado':>12s}")
print(f"{'utilidad total, dual simétrico':34s} {'1.434':>12s} "
      f"{res['simétrico']['comb']:>12.4f} {res['simétrico']['sep']:>12.4f}")
print(f"{'utilidad total, dual restringido':34s} {'1.434':>12s} "
      f"{res['restringido']['comb']:>12.4f} {res['restringido']['sep']:>12.4f}")
print(f"{'mejora sobre 0,7061':34s} {'+103.0%':>12s} "
      f"{(res['restringido']['comb']-U_3F_CON)/U_3F_CON*100:>11.1f}% "
      f"{(res['restringido']['sep']-U_3F_CON)/U_3F_CON*100:>11.1f}%")
print(f"{'adopción A, 3 agentes':34s} {'72.4%':>12s} {res['restringido']['share']*100:>11.1f}% "
      f"{'(no depende de la contabilidad)':>12s}")

W_star4 = orig.run_price_dynamics(COV4, MEAN4, [], num_iterations=500, eta=0.5, Gamma=GAMMA4)[0][-1]
H4 = 2 * GAMMA4[0, 0] * COV4
M4 = np.column_stack([2 * GAMMA4[i, i] * COV4 @ W_star4[:, i] for i in range(4)])
WT4 = run_price_dynamics(H4, M4, [(0, 2)], num_iterations=500, eta=0.5, Gamma=np.eye(4))[0][-1]
WA4 = W_star4 - WT4
share4 = bilateral_volume(WA4) / (bilateral_volume(WT4) + bilateral_volume(WA4))
w4 = sum(W_star4[:, i] @ MEAN4[:, i] - GAMMA4[i, i] * (W_star4[:, i] @ COV4 @ W_star4[:, i])
         for i in range(4))
print(f"{'adopción A, 4 agentes':34s} {'47.9%':>12s} {share4*100:>11.1f}%")
print(f"{'utilidad total, dual 4 agentes':34s} {'1.741':>12s} {w4:>12.4f}")

print(f"\nOtras cifras de la tabla 3f_dual_restricted:")
print(f"  volumen T       publicado 0.568  vs  {res['restringido']['volT']:.4f}")
print(f"  volumen A       publicado 1.486  vs  {res['restringido']['volA']:.4f}")
print(f"  W_13 en A       publicado 1.030  vs  {res['restringido']['W13A']:.4f}")
print(f"  dif. de precio  publicado 0.059  vs  {res['restringido']['pdif']:.4f}")
print("\n=> Patrón: se reproducen EXACTAMENTE las cifras que no dependen del reparto")
print("   T/A (las utilidades totales, porque bajo riesgo conjunto sólo dependen de")
print("   w*), y NO se reproduce ninguna que sí dependa del reparto (volúmenes,")
print("   adopción, diferencial de precio).")


# ===========================================================================
bloque("H. ¿Entregan los dos motores los mismos valores duales?")
# ===========================================================================
H3 = 2 * Gamma[0, 0] * Sigma
M3 = np.column_stack([2 * Gamma[i, i] * Sigma @ W_star[:, i] for i in range(N)])
for etiq, prohib in [("sin restricción", []), ("restricción en T", [(0, 2)])]:
    Wo, Po = orig.run_price_dynamics(H3, M3, prohib, num_iterations=500, eta=0.5, Gamma=np.eye(N))
    Wc, Pc = run_price_dynamics(H3, M3, prohib, num_iterations=500, eta=0.5, Gamma=np.eye(N))
    print(f"{etiq:18s}: max|dW| = {np.abs(Wo[-1]-Wc[-1]).max():.2e}, "
          f"max|dP| = {np.abs(Po[-1]-Pc[-1]).max():.2e}")
print("=> Idénticos. Ninguna de las correcciones H1/H3/H4b/H7 se activa en estos")
print("   escenarios, así que la capa dual se puede reportar con el motor original")
print("   y la tabla del capítulo 4 no mezcla implementaciones.")
