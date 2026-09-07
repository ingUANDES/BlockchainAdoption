# Equivalencia de imports: `.qmd` original → `venegas_model.py`

Paso 2 de la auditoría. Justifica que reemplazar los tres `import *` del `.qmd`
por imports explícitos no altera ninguna llamada.

## Imports del original

`code/Financial_Networks_Stability.qmd`, líneas 33-43, en orden de ejecución:

```python
from sympy import *          # (1)
init_printing()
from numpy.linalg import *   # (2)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
from scipy.stats import truncnorm, wishart
from scipy.linalg import solve, solve_sylvester   # (3)
from functools import partial
from scipy.linalg import toeplitz
import numpy as np
```

Tres capas se pisan entre sí. Los nombres en conflicto y quién gana:

| Nombre | Lo exporta | Resolución final |
|---|---|---|
| `solve` | sympy (1), numpy.linalg (2), scipy.linalg (3) | `scipy.linalg.solve` |
| `solve_sylvester` | scipy.linalg (3) | `scipy.linalg.solve_sylvester` |
| `inv`, `det`, `norm`, `eig`, `eigh`, `pinv`, `qr`, `svd`, `lstsq`, `cond`, `matrix_rank` | numpy.linalg (2) | `numpy.linalg.*` |
| `trace`, `diagonal`, `outer`, `cross`, `matmul`, `tensordot` | numpy.linalg (2), en NumPy ≥ 2.0 | `numpy.linalg.*` |
| `eye`, `zeros`, `ones`, `sqrt`, `exp`, `log`, `diff`, `Matrix`, `Identity`, `symbols` | sympy (1) | `sympy.*` |

## Verificación: ¿se usa alguno de esos nombres desnudo?

Se extrajeron los 8 bloques `{python}` del `.qmd` (472 líneas de código) y se
buscó cada uno de los ~30 nombres candidatos en uso desnudo, es decir no
precedido por `.` ni por un carácter de identificador, y seguido de `(`, `.` o
`[`. Resultado completo:

| Nombre | Línea (código extraído) | Diagnóstico |
|---|---|---|
| `outer` | 87 | **Benigno.** Es la variable local de `construct_q_matrices`, detectada por `outer.copy()`. Sombrea a `numpy.linalg.outer` dentro de la función, sin efecto sobre el resultado. |
| `init_printing` | 2 | **Benigno.** Única dependencia real de SymPy en todo el `.qmd`; solo configura el formato de impresión. No participa en ningún cálculo. |
| `la` | 102 | **Defecto H5.** Ver abajo. |

Todo el resto del núcleo numérico califica cada llamada explícitamente
(`np.eye`, `np.linalg.inv`, `np.matmul`, `np.zeros`, `np.abs`, `np.vstack`,
`np.delete`, `np.copy`, `np.random.*`).

**Conclusión:** los star-imports son un riesgo latente, no un defecto activo.
Ningún resultado publicado depende de la resolución de nombres ambiguos. Pero
cualquier reordenamiento de los imports, o una versión de NumPy/SymPy que
cambie sus `__all__`, podría alterar silenciosamente qué función se invoca si el
código se extiende usando nombres desnudos. Por eso `venegas_model.py` importa
solo `numpy as np`, que es lo único que el núcleo necesita.

## El nombre `la` no está definido (hallazgo H5)

Línea 102 del código extraído, dentro de `price_soln_sylvester`:

```python
return la.solve_sylvester(Q, Q, ...)
```

No existe ningún `import ... as la` en el `.qmd`: los imports traen
`solve_sylvester` directamente al espacio de nombres, sin alias de módulo.
Llamar a `price_soln_sylvester` lanza `NameError: name 'la' is not defined`.

La función no se invoca en ninguna parte del `.qmd`, así que el defecto no
afecta ningún resultado publicado. Importa por otra razón: esta era la
verificación independiente natural de la dinámica iterativa — una solución
cerrada contra la cual contrastar el punto fijo — y no está operativa. Sumado
al `# todo update price sylvester for Gamma` de la línea 135, el autor dejó esa
vía de validación sin terminar.
