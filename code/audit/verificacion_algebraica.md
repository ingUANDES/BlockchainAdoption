# Verificación algebraica del código de Venegas

Paso 6 de la auditoría. Las derivaciones se hicieron con SymPy 1.14 y las
comprobaciones numéricas correspondientes se ejecutan en `test_venegas_model.py`
(las siete pruebas `test_algebra_*`; suite completa: 25 pruebas, todas pasan).

## Fuentes utilizadas

El `.qmd` **no declara el modelo**: es código con prosa mínima y su única referencia
es «Ejemplo 5 Jalan et alii (2024)». El paper no está en el repositorio y no se pudo
consultar, de modo que la fuente autoritativa para el álgebra es la **propia tesis de
Venegas**, cuyo LaTeX sí está versionado en `docs/MemoriasTesis/Venegas/`.

El capítulo 3 declara el modelo de forma explícita y suficiente:

- **Utilidad media-varianza** del agente $i$:
  $g_i(W,P) = w_i^\top(\mu_i - P e_i) - \gamma_i\, w_i^\top \Sigma_i w_i$
- **Simetría de contratos**: $W_{ij} = W_{ji}$ (el contrato es un acuerdo bilateral).
- **Precio**: $P_{ij}$ es lo que $j$ paga a $i$ por unidad, con $P$ antisimétrica.
- **Prohibiciones**: matrices $\Psi_i$ que anulan las contrapartes vedadas.
- **Estabilidad**: cada agente optimiza dado $P$, y ningún par puede desviarse con provecho.

Consecuencia de alcance: **la verificación es contra el modelo tal como la tesis lo
declara, no contra el paper original.** Si la tesis transcribió mal alguna ecuación de
Jalan & Chakrabarti, esta auditoría no lo detectaría. Es la única verificación
pendiente y requiere conseguir el paper.

## Veredicto por función

| Función | Veredicto | Detalle |
|---|---|---|
| `opt_portfolio_mat` | **Correcta** | Reproduce la CPO no restringida, factor 2 y posición de $\Gamma$ incluidos |
| `construct_prohibition_matrices` | **Correcta** | $R_i$ selecciona filas permitidas; $R_i R_i^\top = I$ |
| `construct_q_matrices` | **Correcta** | Resuelve el sistema **reducido**, no anula filas de la inversa |
| `opt_w_with_edge_constraints` | **Correcta** | $w_i = Q_i(\mu_i - Pe_i)$, con cero exacto en la contraparte prohibida |
| `price_update_naive` | **Correcta** | Con $\eta=1$ es el paso **exacto** de acuerdo bilateral, no una heurística |
| `price_soln_sylvester` | **Fórmula correcta**, código roto | Solo falla el alias `la` (H5); válida únicamente si $\Gamma=I$ y sin prohibiciones |

## Derivación 1 — Portafolio óptimo no restringido

De $\partial g_i/\partial w_i = (\mu_i - Pe_i) - 2\gamma_i \Sigma_i w_i = 0$:

$$w_i^* = (2\gamma_i \Sigma_i)^{-1}(\mu_i - P e_i)$$

El código escribe `0.5 * S_inv @ (M - P) @ inv(Gamma)`. La columna $i$ del producto es
$\tfrac{1}{2}\Sigma^{-1}(\mu_i - Pe_i)/\gamma_i$, idéntica a la expresión anterior. El
$\Gamma^{-1}$ va **a la derecha** del producto, que es lo correcto: escala la columna
$i$ —el portafolio del agente $i$— por $1/\gamma_i$. Verificado con SymPy: identidad
exacta.

## Derivación 2 — Óptimo con arista prohibida (la parte delicada)

La forma ingenua de imponer la restricción sería anular filas de $\Sigma^{-1}$. **Eso
sería incorrecto**, porque la inversa de una submatriz no es la submatriz de la
inversa. El código no hace eso.

Parametrizando el subespacio factible como $w_i = R_i^\top v_i$ (con $R_i$ eliminando
las filas prohibidas), el problema reducido es

$$\max_{v_i}\ v_i^\top R_i(\mu_i - Pe_i) - \gamma_i\, v_i^\top (R_i \Sigma_i R_i^\top) v_i$$

con CPO $v_i^* = (2\gamma_i R_i \Sigma_i R_i^\top)^{-1} R_i(\mu_i - Pe_i)$, de donde

$$w_i^* = R_i^\top (2\gamma_i R_i \Sigma_i R_i^\top)^{-1} R_i (\mu_i - Pe_i) \equiv Q_i(\mu_i - Pe_i)$$

que es **exactamente** `construct_q_matrices`: `inner = 2*Gamma[i,i]*(R_i @ Sigma_i @ R_i.T)`,
`outer = R_i.T @ inv(inner) @ R_i`. Verificado con SymPy para $n=3$ con la arista $(0,2)$
prohibida: coincidencia simbólica exacta, y $w^*[2] = 0$ idénticamente.

**Esta es la pieza algebraicamente más difícil del código y está bien hecha.** El factor
2 y la posición de $\Gamma_{ii}$ son correctos.

## Derivación 3 — El paso de precio es exacto, no heurístico

El nombre `price_update_naive` sugiere una aproximación. No lo es. Para la arista
$(i,j)$, con $p \equiv P_{ij} = -P_{ji}$:

$$\frac{\partial W_{ij}}{\partial p} = -Q_j[i,i], \qquad \frac{\partial W_{ji}}{\partial p} = +Q_i[j,j]$$

Como ambas exposiciones son **afines** en $p$, el precio que iguala $W_{ij} = W_{ji}$ se
obtiene en un paso:

$$p^\star = p_{\text{old}} + \frac{W_{ij} - W_{ji}}{Q_i[j,j] + Q_j[i,i]}$$

que es literalmente el `shift` del código. Verificado con SymPy resolviendo
$W_{ij}(p) = W_{ji}(p)$ con $Q_i, Q_j, M$ simbólicos genéricos: coincide.

Por tanto `eta` **no** es un paso de gradiente sobre una aproximación: es amortiguamiento
sobre la solución exacta. La única razón para $\eta < 1$ es que las aristas se actualizan
en paralelo y el óptimo de cada par depende de los demás. Esto refuerza el diagnóstico de
H4: con $\eta=0$ no hay «paso pequeño», hay **ausencia de paso**.

## Derivación 4 — La solución cerrada y el TODO de la línea 135

Sin prohibiciones y con $Q_i = Q$ común, imponer simetría de $W = Q(M-P)$ con $P$
antisimétrica y $Q$ simétrica da:

$$Q(M-P) = (M-P)^\top Q \iff QP + PQ = QM - M^\top Q$$

una ecuación de Sylvester $AX + XB = C$ con $A = B = Q$. El código plantea
`solve_sylvester(Q, Q, Q @ M - M.T @ Q)` con `Q = 0.5*S_inv`. **La fórmula es correcta.**
Verificado con SymPy: la simetría de $W$ y la ecuación de Sylvester son equivalentes.

**Resolución del TODO de la línea 135** (`# todo update price sylvester for Gamma`):
`Q = 0.5*S_inv` equivale a $(2\gamma\Sigma)^{-1}$ **solo si $\gamma_i = 1$ para todo $i$**,
es decir $\Gamma = I$. Con $\Gamma \neq I$ los $Q_i$ difieren entre agentes y la ecuación
deja de ser de Sylvester (habría que resolver un sistema lineal general en las $n(n-1)/2$
incógnitas de $P$). El TODO del autor está correctamente identificado y **sigue abierto**.
Con prohibiciones ocurre lo mismo: los $Q_i$ difieren y la solución cerrada no aplica.

### Validación independiente del punto fijo iterativo

Corrigiendo únicamente el alias (`la` → `scipy.linalg`), la solución cerrada es una
verificación del paso 4 por una vía completamente distinta —resolución directa vs.
iteración amortiguada—. En el escenario **sin restricciones** del Ejemplo 5:

| Comprobación | Resultado |
|---|---|
| $\max\lvert P_{\text{cerrada}} - P_{\text{iterativa}}\rvert$ (tol $10^{-10}$, 20 000 iter.) | $4.9\times10^{-13}$ |
| $P_{\text{cerrada}}$ antisimétrica y con diagonal nula | sí, a $10^{-14}$ |
| $\max\lvert W(P_{\text{cerrada}}) - W(P_{\text{cerrada}})^\top\rvert$ | $6.7\times10^{-16}$ |
| $\max\lvert P_{\text{cerrada}} - P_{\text{publicada en figuras}}\rvert$ | $5.4\times10^{-4}$ |

Las dos primeras filas validan la maquinaria numérica completa. La última **cuantifica H3
por una vía independiente**: la figura publicada está a $5.4\times10^{-4}$ del punto fijo
exacto, sesgo que el umbral del 0,1 % introduce y que queda bajo la precisión con que la
figura imprime sus valores (2 decimales).

## Trayectoria completa contra los iterados publicados en la tesis

La tesis publica $W$ y $P$ en las iteraciones 0, 5, 10 y $\infty$ del escenario con
restricción. No solo el punto final es reproducible: **la trayectoria entera lo es**.

| Iterado | Matriz | Celdas | $\max\lvert\Delta\rvert$ | Dentro del redondeo |
|---|---|---|---|---|
| 0 | $W$ | 7 | 0,0047 | sí |
| 0 | $P$ | 7 | 0,0000 | sí |
| 5 | $W$ | 7 | 0,0047 | sí |
| 5 | $P$ | 7 | 0,0046 | sí |
| 10 | $W$ | 7 | 0,0047 | sí |
| 10 | $P$ | 7 | 0,0033 | sí |
| $\infty$ | $W$ | 7 | 0,0049 | sí |
| $\infty$ | $P$ | 7 | 0,0017 | sí |

56 celdas comparadas, 0 fuera de la tolerancia de 0,005 (medio dígito del último decimal
impreso). Detalle en `iterados_tesis.csv`.

## El modelo dual: las derivaciones existen y están verificadas

`Verificacion_Derivaciones.py` estaba versionado **dentro de
`code/.ipynb_checkpoints/`**, una carpeta de respaldo automático de Jupyter que nadie
inspecciona. Rescatado a `code/audit/Verificacion_Derivaciones_rescatado.py`, se ejecuta
sin error bajo Python 3.13 / SymPy 1.14 y **todas sus comprobaciones devuelven `True`**.

Esto corrige un diagnóstico preliminar de esta auditoría: el álgebra del modelo dual
**sí está en el repositorio y está verificada simbólicamente**. Lo que no está es el
código numérico que genera los resultados dual/3F/4F de la tesis (el script no contiene
ninguna dinámica, iteración, figura ni cálculo de bienestar: 0 coincidencias para
`run_price`, `dynamics`, `iterat`, `savefig`, `heatmap`, `welfare`).

Lo que el script establece, y que reverifiqué de forma independiente:

$$g_{\text{tot}}(w^T) = g^T(w^T) + g^B(w^* - w^T), \qquad
H_i = 2\gamma_i\Sigma_i + \lambda_B I, \qquad w^{T*} = \tfrac{1}{2}H_i^{-1}a_i$$

$$a_i = (P^B - P^T)e_i - \tau\mathbf{1} + 2\gamma_i\Sigma_i w^*_i + 2\lambda_B w^*_i$$

con $\tau$ el bono de transparencia y $\lambda_B$ la penalización por fricción. Verifiqué
con SymPy que el Hessiano de $g_{\text{tot}}$ es exactamente $-2(2\gamma\Sigma + \lambda_B I)$,
que el $a_i$ del script coincide con el derivado, y que $w^{T*}$ anula el gradiente.

**Prueba de consistencia que el script no hace:** con $\lambda_B = 0$, $\tau = 0$ y
$P^B = P^T$ las dos redes son idénticas, y el óptimo resulta $w^{T*} = w^*/2$ — el agente
reparte la posición total en mitades. El álgebra dual es internamente consistente.

### Trampa del factor 1/2 para quien extienda el código

$$Q^{\text{dual}} = \tfrac{1}{2}\left(2\gamma\Sigma + \lambda_B I\right)^{-1}
\qquad\text{vs.}\qquad Q^{\text{Jalan}} = \left(2\gamma\Sigma\right)^{-1}$$

$Q^{\text{dual}}$ **no** tiende a $Q^{\text{Jalan}}$ cuando $\lambda_B \to 0$: tiende a la
**mitad**. La razón es estructural, no un error: $w^T$ es una de dos redes sobre las que
actúa la misma aversión al riesgo, así que el Hessiano acumula $2\gamma\Sigma$ en lugar de
$\gamma\Sigma$.

Por tanto, extender `construct_q_matrices` al modelo dual **sumando `lambda_B*np.eye(k)`
a `inner` da un resultado 2 veces demasiado grande.** La forma correcta, verificada
simbólicamente (usa $R_i R_i^\top = I$):

```python
# inner correcto para el modelo dual con aristas prohibidas
k = prohibition_matrices[i].shape[0]
inner = 2 * (2 * Gamma[i, i] * (R_i @ Sigma_i @ R_i.T) + lambda_B * np.eye(k))
#       ^^^ el factor 2 externo es el que falta si se generaliza por analogía
```

Además, $\tau$ entra **solo** en el término lineal $a_i$ y no en $H_i$ (verificado:
$\partial H_i/\partial\tau = 0$), de modo que el óptimo sigue siendo afín en
$(M, P, \tau)$ y toda la maquinaria de Sylvester y del paso bilateral exacto es
extensible al modelo dual. Eso es un activo real para la extensión: la parte difícil
—la derivación— está hecha y ahora está verificada por máquina.
