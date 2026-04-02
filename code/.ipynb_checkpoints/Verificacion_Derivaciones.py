#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sympy as sp
import numpy as np
from sympy import Matrix, symbols, simplify, expand, diff, eye, sqrt
from sympy import MatrixSymbol, Identity, MatMul, MatAdd

print("=" * 80)
print("VERIFICACIÓN DE OPTIMIZACIÓN EN 2 REDES")
print("=" * 80)

# Definir dimensiones simbólicas
n = symbols('n', integer=True, positive=True)

# Definir símbolos escalares
gamma_i = symbols('gamma_i', real=True, positive=True)
lambda_B = symbols('lambda_B', real=True, positive=True)
tau = symbols('tau', real=True)
P_T = symbols('P^T', real=True)
P_B = symbols('P^B', real=True)

print("\n1. VERIFICACIÓN DE LA FUNCIÓN DE UTILIDAD (Forma Simbólica)")
print("-" * 40)

# Usar notación más simple para mostrar las ecuaciones
print("g_T(w_T_i) = w_T_i^T (m_i - P^T e_i) - γ_i w_T_i^T Σ_i w_T_i")
print("g_B(w_B_i) = w_B_i^T (m_i + τ - P^B e_i) - γ_i w_B_i^T Σ_i w_B_i - λ_B ||w_B_i||²")
print("g_total = g_T + g_B")

print("\n2. SUSTITUCIÓN w_B_i = w*_i - w_T_i")
print("-" * 40)
print("w_B_i = w*_i - w_T_i")
print("\nExpansión de términos cuadráticos:")
print("(w* - w)^T Σ (w* - w) = w*^T Σ w* - 2 w*^T Σ w + w^T Σ w")
print("||w* - w||² = w*^T w* - 2 w*^T w + w^T w")

print("\n3. IDENTIFICACIÓN DE COEFICIENTES")
print("-" * 40)
print("A = m_i - P^T e_i")
print("B = m_i + τ - P^B e_i")
print("A - B = -P^T e_i + P^B e_i - τ = (P^B - P^T)e_i - τ")

print("\nCoeficientes de la forma cuadrática:")
print("a_i = (P^B - P^T)e_i - τ + 2γ_i Σ_i w*_i + 2λ_B w*_i")
print("H_i = 2γ_i Σ_i + λ_B I")

print("\n4. VERIFICACIÓN CON MATRICES CONCRETAS 2x2")
print("-" * 40)

# Crear matrices concretas para verificación
sigma_11, sigma_12, sigma_22 = symbols('sigma_11 sigma_12 sigma_22', real=True)
w_T_1, w_T_2 = symbols('w_T_1 w_T_2', real=True)
w_star_1, w_star_2 = symbols('w_star_1 w_star_2', real=True)
e_1, e_2 = symbols('e_1 e_2', real=True)
m_1, m_2 = symbols('m_1 m_2', real=True)

Sigma_concrete = Matrix([[sigma_11, sigma_12],
                         [sigma_12, sigma_22]])
w_T_concrete = Matrix([w_T_1, w_T_2])
w_star_concrete = Matrix([w_star_1, w_star_2])
e_concrete = Matrix([e_1, e_2])
m_concrete = Matrix([m_1, m_2])
I_concrete = eye(2)

print("Matrices concretas:")
print(f"Σ = {Sigma_concrete}")
print(f"w_T = {w_T_concrete.T}")
print(f"w* = {w_star_concrete.T}")

# Construir H_i y a_i concretos
H_concrete = 2 * gamma_i * Sigma_concrete + lambda_B * I_concrete
a_concrete = (P_B - P_T) * e_concrete - tau * Matrix([1, 1]) + 2 * gamma_i * Sigma_concrete * w_star_concrete + 2 * lambda_B * w_star_concrete

print("\nH_i concreto:")
print(H_concrete)

print("\na_i concreto (simplificado):")
a_simplified = simplify(a_concrete)
print(a_simplified)

# Función objetivo concreta
g_concrete = w_T_concrete.T * a_concrete - w_T_concrete.T * H_concrete * w_T_concrete

print("\n5. CONDICIÓN DE PRIMER ORDEN")
print("-" * 40)
print("Función objetivo: g(w_T) = w_T^T a_i - w_T^T H_i w_T")

# Calcular derivada respecto a cada componente
dg_dw1 = diff(g_concrete[0], w_T_1)
dg_dw2 = diff(g_concrete[0], w_T_2)
dg_dw = Matrix([dg_dw1, dg_dw2])

print("\n∂g/∂w_T =")
print(simplify(dg_dw))

# Verificar que es igual a a_i - 2H_i*w_T
verification = simplify(a_concrete - 2 * H_concrete * w_T_concrete)
print("\nVerificación: a_i - 2H_i w_T =")
print(verification)

print("\n6. SOLUCIÓN ÓPTIMA SIN RESTRICCIONES")
print("-" * 40)
print("Igualando ∂g/∂w_T = 0:")
print("a_i - 2H_i w_T* = 0")
print("w_T* = (1/2) H_i^(-1) a_i")

# Verificar con matrices simbólicas pequeñas
H_inv = H_concrete.inv()
w_T_optimal = (H_inv * a_concrete) / 2
print("\nSolución simbólica (forma simplificada):")
print("w_T* = (1/2) H^(-1) a")

print("\n7. MATRICES Q DUALES")
print("-" * 40)
print("Con restricciones Ψ^T_i w^T_i = w^T_i:")
print("Q_dual_i = Ψ^T^T (Ψ^T H_i^(-1) Ψ^T^T)^(-1) Ψ^T H_i^(-1)")
print("\nSin restricciones (Ψ^T = I):")
print("Q_dual_i = H_i^(-1) = (2γ_i Σ_i + λ_B I)^(-1)")

print("\n8. VERIFICACIÓN NUMÉRICA")
print("-" * 40)

# Ejemplo numérico
gamma_val = 0.5
lambda_val = 0.1
tau_val = 0.05
P_T_val = 1.0
P_B_val = 1.2

Sigma_num = np.array([[0.04, 0.01],
                       [0.01, 0.03]])
w_star_num = np.array([[1.0], [0.5]])
e_num = np.array([[1.0], [1.0]])
m_num = np.array([[0.08], [0.06]])
I_num = np.eye(2)

print(f"Parámetros numéricos:")
print(f"γ_i = {gamma_val}")
print(f"λ_B = {lambda_val}")
print(f"τ = {tau_val}")
print(f"P^T = {P_T_val}")
print(f"P^B = {P_B_val}")
print(f"\nΣ_i =\n{Sigma_num}")
print(f"w*_i = {w_star_num.T[0]}")
print(f"e_i = {e_num.T[0]}")
print(f"m_i = {m_num.T[0]}")

# Calcular H_i numérico
H_num = 2 * gamma_val * Sigma_num + lambda_val * I_num
print(f"\nH_i = 2γΣ + λI =\n{H_num}")

# Verificar que H_i es definida positiva
eigenvalues = np.linalg.eigvals(H_num)
print(f"\nAutovalores de H_i: {eigenvalues}")
print(f"H_i es definida positiva: {all(eigenvalues > 0)}")

# Calcular a_i numérico  
tau_vector = np.array([[tau_val], [tau_val]])
a_num = (P_B_val - P_T_val) * e_num - tau_vector + 2 * gamma_val * Sigma_num @ w_star_num + 2 * lambda_val * w_star_num
print(f"\na_i = (P^B - P^T)e - τ + 2γΣw* + 2λw* =\n{a_num}")

# Solución óptima sin restricciones
w_T_opt = 0.5 * np.linalg.inv(H_num) @ a_num
print(f"\nw_T* (sin restricciones) = (1/2) H^(-1) a =\n{w_T_opt}")

# Calcular w_B óptimo
w_B_opt = w_star_num - w_T_opt
print(f"\nw_B* = w* - w_T* =\n{w_B_opt}")

# Verificar condición de primer orden
residual = a_num - 2 * H_num @ w_T_opt
print(f"\nVerificación ∂g/∂w_T = a - 2Hw_T* =\n{residual}")
print(f"Norma del residual: {np.linalg.norm(residual):.2e}")

# Calcular utilidades
A_num = m_num - P_T_val * e_num
B_num = m_num + tau_val - P_B_val * e_num

g_T_val = w_T_opt.T @ A_num - gamma_val * w_T_opt.T @ Sigma_num @ w_T_opt
g_B_val = w_B_opt.T @ B_num - gamma_val * w_B_opt.T @ Sigma_num @ w_B_opt - lambda_val * w_B_opt.T @ w_B_opt
g_total_val = g_T_val + g_B_val

print(f"\nUtilidades:")
print(f"g_T = {g_T_val[0,0]:.6f}")
print(f"g_B = {g_B_val[0,0]:.6f}")
print(f"g_total = {g_total_val[0,0]:.6f}")

print("\n9. COMPARACIÓN DE MATRICES Q")
print("-" * 40)

# Q original (sin fricción blockchain)
Q_original = np.linalg.inv(2 * gamma_val * Sigma_num)
print(f"Q_original = (2γΣ)^(-1) =\n{Q_original}")

# Q dual (con fricción blockchain)
Q_dual = np.linalg.inv(H_num)
print(f"\nQ_dual = (2γΣ + λI)^(-1) =\n{Q_dual}")

# Diferencia
Q_diff = Q_original - Q_dual
print(f"\nDiferencia (Q_original - Q_dual) =\n{Q_diff}")

print("\n10. EFECTO DE LA FRICCIÓN BLOCKCHAIN")
print("-" * 40)

# Comparar soluciones con y sin fricción
H_no_friction = 2 * gamma_val * Sigma_num
w_T_no_friction = 0.5 * np.linalg.inv(H_no_friction) @ (a_num + 2 * lambda_val * w_star_num)
print(f"w_T* sin λ_B en H (pero sí en a) =\n{w_T_no_friction}")
print(f"w_T* con λ_B =\n{w_T_opt}")
print(f"\nDiferencia = {np.linalg.norm(w_T_no_friction - w_T_opt):.6f}")

print("\n11. VERIFICACIÓN DE RESTRICCIONES CON Ψ")
print("-" * 40)

# Ejemplo con restricción simple (solo se permite el primer activo)
Psi_T = np.array([[1, 0],
                   [0, 0]])
print(f"Ψ^T (solo primer activo permitido) =\n{Psi_T}")

# Aplicar restricción
Psi_H_inv = Psi_T @ np.linalg.inv(H_num)
Psi_H_inv_Psi = Psi_H_inv @ Psi_T.T

# Solo invertir la parte no singular
if np.linalg.matrix_rank(Psi_H_inv_Psi) == 1:
    print("\nCon esta restricción, solo el primer activo puede comerciarse en la red tradicional")
    # Solución con restricción (solo primer componente)
    w_T_restricted = np.zeros((2, 1))
    # Calcular solo el primer componente
    H_11 = H_num[0, 0]
    a_1 = a_num[0, 0]
    w_T_restricted[0, 0] = a_1 / (2 * H_11)
    print(f"w_T* con restricción = {w_T_restricted.T[0]}")


# In[5]:


import sympy as sp
from sympy import symbols, Matrix, eye, simplify, expand, diff, collect, factor, latex
from sympy import init_printing

# Configurar impresión
init_printing(use_unicode=True)

print("=" * 80)
print("DERIVACIÓN MATEMÁTICA SIMBÓLICA: OPTIMIZACIÓN EN 2 REDES")
print("=" * 80)

# ============================================================================
# DEFINIR SÍMBOLOS
# ============================================================================
print("\n1. DEFINICIÓN DE SÍMBOLOS")
print("-" * 40)

# Escalares
gamma = symbols('gamma', positive=True, real=True)
lambda_B = symbols('lambda_B', positive=True, real=True)
tau = symbols('tau', real=True)
P_T, P_B = symbols('P^T P^B', real=True)

# Vectores (2x1)
w_T1, w_T2 = symbols('w_T1 w_T2', real=True)
w_B1, w_B2 = symbols('w_B1 w_B2', real=True)
w_star1, w_star2 = symbols('w^*_1 w^*_2', real=True)
m1, m2 = symbols('m_1 m_2', real=True)
e1, e2 = symbols('e_1 e_2', real=True)

w_T = Matrix([w_T1, w_T2])
w_B = Matrix([w_B1, w_B2])
w_star = Matrix([w_star1, w_star2])
m = Matrix([m1, m2])
e = Matrix([e1, e2])

# Matriz de covarianza (simétrica)
s11, s12, s22 = symbols('sigma_11 sigma_12 sigma_22', real=True)
Sigma = Matrix([[s11, s12], [s12, s22]])

print("Vectores definidos:")
print(f"w^T = {w_T.T}")
print(f"w^B = {w_B.T}")
print(f"w* = {w_star.T}")
print(f"Σ = {Sigma}")

# ============================================================================
# FUNCIONES DE UTILIDAD ORIGINALES
# ============================================================================
print("\n2. FUNCIONES DE UTILIDAD ORIGINALES")
print("-" * 40)

# Red Tradicional
print("\nRed Tradicional:")
g_T = (w_T.T @ (m - P_T * e))[0] - gamma * (w_T.T @ Sigma @ w_T)[0]
g_T = expand(g_T)
print(f"g^T(w^T) = {g_T}")

# Red Blockchain
print("\nRed Blockchain:")
tau_vec = Matrix([tau, tau])
g_B = (w_B.T @ (m + tau_vec - P_B * e))[0] - gamma * (w_B.T @ Sigma @ w_B)[0] - lambda_B * (w_B.T @ w_B)[0]
g_B = expand(g_B)
print(f"g^B(w^B) = {g_B}")

# ============================================================================
# SUSTITUCIÓN w^B = w* - w^T
# ============================================================================
print("\n3. SUSTITUCIÓN w^B = w* - w^T")
print("-" * 40)

w_B_sub = w_star - w_T
print(f"w^B = w* - w^T = {w_B_sub.T}")

# Calcular término lineal
print("\nTérmino lineal (w* - w^T)' (m + τ - P^B e):")
linear_B = (w_B_sub.T @ (m + tau_vec - P_B * e))[0]
linear_B_expanded = expand(linear_B)
print(f"= {linear_B_expanded}")

# Calcular término cuadrático Sigma
print("\nTérmino cuadrático (w* - w^T)' Σ (w* - w^T):")
quad_Sigma = (w_B_sub.T @ Sigma @ w_B_sub)[0]
quad_Sigma_expanded = expand(quad_Sigma)
print(f"= {quad_Sigma_expanded}")

# Calcular término cuadrático norma
print("\nTérmino cuadrático ||w* - w^T||²:")
quad_norm = (w_B_sub.T @ w_B_sub)[0]
quad_norm_expanded = expand(quad_norm)
print(f"= {quad_norm_expanded}")

# g^B con sustitución
print("\ng^B después de sustitución:")
g_B_sub = linear_B_expanded - gamma * quad_Sigma_expanded - lambda_B * quad_norm_expanded
g_B_sub = expand(g_B_sub)
print(f"g^B = {g_B_sub}")

# ============================================================================
# FUNCIÓN TOTAL
# ============================================================================
print("\n4. FUNCIÓN DE UTILIDAD TOTAL g(w^T)")
print("-" * 40)

g_total = g_T + g_B_sub
g_total = expand(g_total)
print(f"g(w^T) = g^T + g^B")
print(f"g = {g_total}")

# ============================================================================
# AGRUPAR TÉRMINOS
# ============================================================================
print("\n5. AGRUPACIÓN DE TÉRMINOS: g(w^T) = C_i + w^T' a_i - w^T' H_i w^T")
print("-" * 40)

# Recolectar coeficientes
print("\nRecolectando coeficientes...")

# Coeficientes lineales en w_T1 y w_T2
coef_w_T1 = g_total.coeff(w_T1, 1).coeff(w_T2, 0)
coef_w_T2 = g_total.coeff(w_T2, 1).coeff(w_T1, 0)

# Coeficientes cuadráticos
coef_w_T1_sq = g_total.coeff(w_T1, 2)
coef_w_T2_sq = g_total.coeff(w_T2, 2)
coef_w_T1_w_T2 = g_total.coeff(w_T1, 1).coeff(w_T2, 1)

# Término constante (lo que queda sin w_T)
g_no_wT = g_total.subs([(w_T1, 0), (w_T2, 0)])

print(f"\nTérmino constante C_i:")
C_i = simplify(g_no_wT)
print(f"C_i = {C_i}")

print(f"\nCoeficientes lineales:")
print(f"Coef(w_T1) = {simplify(coef_w_T1)}")
print(f"Coef(w_T2) = {simplify(coef_w_T2)}")

a_i = Matrix([coef_w_T1, coef_w_T2])
print(f"\nVector a_i = {simplify(a_i)}")

print(f"\nCoeficientes cuadráticos:")
print(f"Coef(w_T1²) = {simplify(coef_w_T1_sq)}")
print(f"Coef(w_T2²) = {simplify(coef_w_T2_sq)}")
print(f"Coef(w_T1·w_T2) = {simplify(coef_w_T1_w_T2)}")

# Construir matriz H_i
H_i = Matrix([
    [-coef_w_T1_sq, -coef_w_T1_w_T2/2],
    [-coef_w_T1_w_T2/2, -coef_w_T2_sq]
])
H_i = simplify(H_i)
print(f"\nMatriz H_i = {H_i}")

# Verificar que H_i = 2γΣ + λ_B I
I_2 = eye(2)
H_i_expected = 2*gamma*Sigma + lambda_B*I_2
print(f"\nVerificación: H_i debería ser 2γΣ + λ_B I")
print(f"2γΣ + λ_B I = {H_i_expected}")
print(f"¿Son iguales? {simplify(H_i - H_i_expected) == Matrix([[0,0],[0,0]])}")

# ============================================================================
# VERIFICAR FORMA DE a_i
# ============================================================================
print("\n6. VERIFICACIÓN DE a_i")
print("-" * 40)

print("\nDerivando a_i desde la teoría:")
a_i_theory = (P_B - P_T) * e - tau_vec + 2*gamma*Sigma@w_star + 2*lambda_B*w_star
a_i_theory = simplify(a_i_theory)
print(f"a_i (teórico) = (P^B - P^T)e - τ + 2γΣw* + 2λ_B w*")
print(f"= {a_i_theory}")

print(f"\nDiferencia (a_i calculado - a_i teórico):")
diff_a = simplify(a_i - a_i_theory)
print(f"= {diff_a}")
print(f"¿Son iguales? {diff_a == Matrix([0, 0])}")

# ============================================================================
# CONDICIONES DE PRIMER ORDEN
# ============================================================================
print("\n7. CONDICIONES DE PRIMER ORDEN")
print("-" * 40)

print("\nCalculando ∂g/∂w^T directamente:")
dg_dw_T1 = diff(g_total, w_T1)
dg_dw_T2 = diff(g_total, w_T2)
gradient = Matrix([dg_dw_T1, dg_dw_T2])
gradient = simplify(gradient)
print(f"∇g = {gradient}")

print("\nVerificando que ∇g = a_i - 2H_i w^T:")
gradient_check = simplify(a_i - 2*H_i@w_T)
print(f"a_i - 2H_i w^T = {gradient_check}")

print(f"\nDiferencia:")
print(f"{simplify(gradient - gradient_check)}")
print(f"¿Son iguales? {simplify(gradient - gradient_check) == Matrix([0, 0])}")

# ============================================================================
# SOLUCIÓN ÓPTIMA
# ============================================================================
print("\n8. SOLUCIÓN ÓPTIMA SIN RESTRICCIONES")
print("-" * 40)

print("\nDe ∇g = 0, tenemos a_i - 2H_i w^T* = 0")
print("Por lo tanto: w^T* = (1/2) H_i^(-1) a_i")

H_i_inv = H_i.inv()
w_T_optimal = simplify((H_i_inv @ a_i) / 2)
print(f"\nH_i^(-1) = {simplify(H_i_inv)}")
print(f"\nw^T* = {w_T_optimal}")

# Verificar que satisface CPO
print("\nVerificación CPO: a_i - 2H_i w^T* = ")
cpo_check = simplify(a_i - 2*H_i@w_T_optimal)
print(f"{cpo_check}")
print(f"¿Es cero? {cpo_check == Matrix([0, 0])}")

# ============================================================================
# PORTAFOLIO BLOCKCHAIN ÓPTIMO
# ============================================================================
print("\n9. PORTAFOLIO BLOCKCHAIN ÓPTIMO")
print("-" * 40)

w_B_optimal = simplify(w_star - w_T_optimal)
print(f"w^B* = w* - w^T* = {w_B_optimal}")

# ============================================================================
# MATRICES Q DUALES
# ============================================================================
print("\n10. MATRICES Q DUALES")
print("-" * 40)

print("\nPara contratos bilaterales W_ij = e_i' w_j:")
print("Con w_j* = (1/2) H_j^(-1) a_j")
print("Definimos Q^dual = (1/2) H^(-1)")

Q_dual = simplify(H_i_inv / 2)
print(f"\nQ^dual = {Q_dual}")

print("\nEn forma expandida:")
Q_dual_expanded = simplify((2*gamma*Sigma + lambda_B*I_2).inv() / 2)
print(f"Q^dual = (1/2)(2γΣ + λ_B I)^(-1)")

print("\nComparación con modelo original:")
Q_original = (2*gamma*Sigma).inv()
print(f"Q_original (Jalan) = (2γΣ)^(-1)")



