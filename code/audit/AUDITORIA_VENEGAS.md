# Auditoría del código de Hernán Venegas

**Branch:** `audit/venegas-code` · **Entorno:** `venegas-audit` (Python 3.13.15, NumPy 2.5.3, SciPy, SymPy 1.14)
**Suite:** `code/audit/test_venegas_model.py` — 30 pruebas, todas pasan.

Ningún archivo del autor original fue modificado. Todo el trabajo de la auditoría
vive en `code/audit/`; cada commit verifica que `code/` y
`docs/MemoriasTesis/Venegas/` siguen idénticos a `main`.

---

## 1. Resumen ejecutivo

El núcleo numérico versionado **funciona y es correcto**. Reproduce el Ejemplo 5
de Jalan & Chakrabarti celda por celda, su álgebra coincide con el modelo que
declara la tesis en las cuatro derivaciones que se verificaron, y la solución
cerrada —una vez corregido un nombre— coincide con la iteración a precisión de
máquina. No encontré ningún error que invalide un resultado publicado.

El problema no es la corrección del código: es **qué fracción de la tesis
produce**. De 32 unidades de resultado inventariadas, 7 son reproducibles hoy y
verificadas en esta auditoría; 16 requieren código que no existe; 5 no tienen ni
código ni derivación.

Tres cosas que cambian el diagnóstico respecto de lo que parecía al principio:

1. **Las derivaciones del modelo dual sí existen.** Están en
   `code/.ipynb_checkpoints/Verificacion_Derivaciones.py` — una carpeta de
   respaldo automático de Jupyter que nadie inspecciona. Verifican
   simbólicamente el Hessiano, su definición positiva, la condición de primer
   orden, el caso restringido y la matriz dual. Corren sin error y todas sus
   comprobaciones dan verdadero. Lo que falta es la **implementación numérica**,
   no el álgebra.
2. **Los resultados de la fintech son reproducibles hoy.** La tesis declara los
   parámetros completos del escenario de 4 firmas (cap. 4, "Experimental
   Design"). Corriendo el núcleo versionado con ellos se obtiene la utilidad
   total 1,0341, el benchmark 1,7413, la utilidad de la fintech 0,2390 y el
   volumen 1,3508 — todos exactos. No hace falta álgebra nueva, solo un script
   de escenario.
3. **La capa de métricas se puede recuperar.** Ningún archivo de código calcula
   utilidad, bienestar ni volumen (el `.qmd` no contiene las palabras `utility`,
   `welfare`, `volume` ni `networkx`). Reconstruí las tres definiciones por
   ingeniería inversa y reproducen la tabla baseline del cap. 4 en 14/14
   métricas. Quedan versionadas en `code/audit/metricas_tesis.py`.

**Lo primero que haría:** preguntar al profesor o a Venegas si el script que
generó los escenarios duales existe en alguna máquina local. Si existe, se
recupera de inmediato la mitad del cap. 4. Si no existe, hay que reimplementarlo
— y el álgebra ya está hecha y verificada, lo que reduce mucho ese trabajo.

---

## 2. Veredicto de reproducibilidad

![Cobertura de código de los resultados de la tesis]({{artifact:art_d0efd268-ff36-4f95-a580-100931120e9f}})

| Clase | Unidades | Qué significa |
|---|---:|---|
| Reproducible hoy | 7 | Verificado en esta auditoría contra los valores publicados |
| Parcial | 16 | El núcleo o el álgebra existen; falta el script o la implementación |
| Sin código ni derivación | 5 | Diagramas de red y «net rent», que no está definida en ninguna ecuación |
| No aplica | 4 | Notación y figuras reproducidas del paper |

Detalle unidad por unidad en `cobertura_resultados.csv`.

**Reproducible y verificado:**

- Ejemplo 5, iterados $W$ y $P$ en $t = 0, 5, 10, \infty$: 56/56 celdas dentro
  del redondeo a 2 decimales que imprime la tesis.
- Las cuatro figuras de mapas de calor commiteadas: 128/128 celdas.
- Tabla baseline del cap. 4 (3 firmas, con y sin restricción): 14/14 métricas.
- Escenario fintech y benchmark de 4 firmas: valores titulares exactos.
- Números titulares 50,8 % (pérdida de bienestar), 46,5 % (ganancia por
  competencia) y 27,9 % (brecha residual).

**No reproducible hoy:** el 100 % de recuperación de bienestar de la red
alternativa — el resultado central de la tesis. Requiere implementar el modelo
dual.

---

## 3. Hallazgos por severidad

Se conservan las etiquetas `H1`–`H9` de las pruebas. `H6` no quedó asignada.
Cada hallazgo tiene una prueba que lo documenta: si una prueba `test_defecto_*`
falla, el defecto fue corregido y hay que actualizar la prueba y este informe,
no «arreglar» la prueba.

### Bloqueantes para la reproducibilidad

**B1 — No existe el código numérico del modelo dual.**
16 de las 32 unidades de resultado dependen de él, incluido el resultado
central. Las derivaciones están verificadas
(`Verificacion_Derivaciones_rescatado.py`) pero no calculan ningún escenario:
cero coincidencias para dinámica, iteración, figuras y bienestar en ese script.

**B2 — Las figuras de resultados están commiteadas solo como PNG.**
Cinco figuras del cap. 4 (tres diagramas de red y las dos de sensibilidad) no
tienen código de generación en el repositorio. Los diagramas no son
recuperables por cálculo: el `.qmd` no importa `networkx` ni dibuja grafos.

**B3 — El `README.md` describe capacidades que el código no tiene.**
Atribuye al repositorio el modelo dual y los cálculos de bienestar. Ninguno de
los dos está implementado. Corregirlo es parte del pull request.

**B4 — Un archivo de verificación quedado en `.ipynb_checkpoints/`.**
Es contenido sustantivo almacenado en una carpeta de respaldo automático, sin
equivalente en la carpeta de código. Rescatado a `code/audit/`.

### Graves para la validez numérica futura

**H3 — `eta = 0.0` por defecto deja la dinámica sin efecto.**
Con `eta = 0` la actualización de precios es multiplicación por cero: $P \equiv 0$
en todas las iteraciones, desacuerdo del 100 %, y la función devuelve una
trayectoria de apariencia normal. El `.qmd` pasa `eta = 0.5` explícitamente, así
que ningún resultado publicado está afectado — pero quien llame a la función sin
ese argumento obtiene un resultado plausible y artefactual.

**H9 — Divergencia silenciosa sin aviso.**
Región estable observada: $0{,}05 \le \eta \le 1{,}25$. A partir de
$\eta = 1{,}5$ la iteración diverge hasta $|P| \sim 10^{12}$ sin ninguna
advertencia ni chequeo de finitud. Detalle en `barrido_eta.csv`.

**H4 — No hay criterio de convergencia.**
Se ejecuta un número fijo de iteraciones y el umbral de acuerdo está fijo en el
código. La dinámica converge geométricamente y se congela temprano (iteración 29
con restricción, 17 sin ella), pero se detiene en la frontera del umbral con
desacuerdo residual — 0,077 % y 0,088 % respectivamente — no en el punto fijo
exacto. Con los datos del Ejemplo 5 el sesgo resultante queda por debajo de la
precisión con que la tesis imprime sus valores, así que es inmaterial para lo
publicado (`barrido_umbral.csv`). Con datos reales, esa garantía no se hereda.

![Auditoría de convergencia]({{artifact:art_2db24e89-3731-4c28-aeb8-bbb1f604c5d1}})

**H1 — Guarda invertida en `get_percent_change`.**
Cuando `current == previous` la función devuelve `100.0`, es decir reporta cambio
máximo justo cuando no hay cambio. Con los datos del Ejemplo 5 la rama nunca se
activa, así que es latente. Pero se activa por completo con una calibración
simétrica (covarianza identidad y medias simétricas): la guarda dispara en todas
las llamadas y reporta discrepancia máxima en un punto que es exactamente
estable. El precio de equilibrio sigue saliendo bien por coincidencia, pero la
función queda inservible como diagnóstico de convergencia — que es precisamente
el uso que tendría al calibrar con datos agregados.

**H7 — Generadores sin semilla.**
`generate_firm_covariances` y `generate_means` sortean de `np.random` sin
semilla. No afectan al Ejemplo 5, que usa matrices fijas, pero son exactamente
las funciones que se usarían para análisis de sensibilidad o para calibrar con
datos reales: cualquier resultado producido con ellas es irreproducible.

### Menores

**H5 — `price_soln_sylvester` lanza `NameError`.**
El cuerpo llama a `la.solve_sylvester` y no existe ningún `import ... as la` en
el `.qmd`. La función nunca se invoca, así que no afecta ningún resultado. Importa
por otra razón: era la validación independiente natural de la dinámica
iterativa, y quedó inoperativa. Corregido el alias, la solución cerrada coincide
con el punto fijo en $4{,}9 \times 10^{-13}$.

**H2 — `except ZeroDivisionError` inerte.**
Con escalares de NumPy la división por cero emite `RuntimeWarning` y devuelve
`inf`/`nan` en lugar de lanzar la excepción, así que el `except` nunca se
ejecuta y el valor no finito se propaga.

**H4b — Las aristas prohibidas no heredan `P_init`.**
`P_new` se inicializa en ceros y las entradas prohibidas nunca se escriben, así
que un `P_init` no nulo en una arista prohibida se descarta silenciosamente.

**H8 — Código muerto.**
Tres funciones definidas y nunca invocadas. Las apariciones adicionales de una
de ellas son llamadas comentadas: el autor la reemplazó sin borrar la anterior.

**Higiene de reporte.** Los porcentajes derivados de las tablas del cap. 4 no
siguen una convención de redondeo consistente: 46,5 % y 31,6 % coinciden si se
calculan desde los valores ya redondeados de la propia tabla, mientras que
179,3 % solo coincide truncando. La discrepancia máxima es de 0,06 puntos
porcentuales y no afecta ninguna conclusión.

### Riesgo latente, no defecto

Los tres `import *` del `.qmd` (`sympy`, `numpy.linalg`, `scipy.linalg`) se
pisan entre sí. Se verificó cada nombre en conflicto en uso desnudo: ningún
resultado publicado depende de resolución ambigua de nombres. Pero cualquier
reordenamiento de los imports, o un cambio en los `__all__` de NumPy o SymPy,
podría alterar silenciosamente qué función se invoca si el código se extiende.
Análisis completo en `equivalencia_imports.md`.

---

## 4. Verificación algebraica

Cuatro derivaciones simbólicas contra el modelo que declara la tesis, todas
satisfactorias: la condición de primer orden no restringida, el óptimo con
arista prohibida, el paso de precio bilateral y la ecuación de Sylvester.

Dos puntos que vale destacar:

- El óptimo restringido es la pieza algebraicamente delicada del código y **está
  bien hecha**: resuelve el sistema reducido por proyección en el subespacio de
  contrapartes permitidas, no anulando filas de la inversa —que habría sido
  incorrecto, porque la inversa de una submatriz no es la submatriz de la
  inversa.
- El `# todo update price sylvester for Gamma` de la línea 135 estaba
  correctamente identificado por el autor y sigue abierto: la forma cerrada es
  válida solo con aversión al riesgo identidad y sin prohibiciones.

Detalle completo, incluida la extensión al modelo dual y la trampa del factor
1/2 en su matriz $Q$, en `verificacion_algebraica.md`.

**Limitación:** la verificación es contra el modelo tal como lo transcribe la
tesis de Venegas, no contra Jalan & Chakrabarti (2024), que no está en el
repositorio. Si la tesis copió mal una ecuación, esta auditoría no lo detectaría.

![Réplica del Ejemplo 5]({{artifact:art_2e1f8e1d-f408-4d5a-a4ad-08eb67f2bf83}})

---

## 5. Implicancias para la calibración con datos externos

Esta sección es la que importa para el paso siguiente del proyecto: alimentar el
modelo con datos del BID o CORFO.

**1. Corregir `H1` antes de calibrar, no después.** Los datos agregados o
redondeados que entregan esas fuentes hacen perfectamente plausibles matrices
simétricas o casi simétricas — y es exactamente ahí donde la guarda invertida se
activa. El diagnóstico de convergencia quedaría ciego justo en el caso en que se
lo necesita.

**2. Fijar `eta` explícitamente y re-verificar la región estable.** El valor por
defecto devuelve $P \equiv 0$ con apariencia de resultado. La región estable
$0{,}05 \le \eta \le 1{,}25$ se midió con la covarianza del Ejemplo 5; con una
$\Sigma$ estimada de datos —peor condicionada y de escalas heterogéneas— esa
región cambia. Hay que medirla, no asumirla.

**3. El código no valida la covarianza.** No hay ningún chequeo de definición
positiva, de condicionamiento ni de simetría: tres llamadas a `np.linalg.inv`
sin verificación previa. Una $\Sigma$ estimada de datos con valores faltantes
puede no ser definida positiva, y el resultado sería silenciosamente inválido.
Recomiendo un chequeo explícito (Cholesky, o el menor autovalor) en la entrada.

**4. Poner semilla en los generadores.** Cualquier análisis de sensibilidad con
`generate_firm_covariances` o `generate_means` es hoy irreproducible.

**5. Los porcentajes son comparables entre calibraciones; las utilidades
absolutas no.** Verificado numéricamente: con $\mu \to c\mu$ a $\Sigma$ fija,
$W$ escala linealmente en $c$ y la utilidad como $c^2$, mientras que la pérdida
de bienestar se mantiene en 50,75 % con $c = 1$ y con $c = 10$. Y con
$\mu \to c\mu$, $\Sigma \to c^2\Sigma$ la utilidad total es exactamente
invariante. Conclusión práctica: se puede reportar el efecto de la restricción
en porcentaje sin fijar unidades, pero comparar utilidades en niveles exige
documentar la normalización de $\mu$ y $\Sigma$.

**6. Si el objetivo es el modelo dual, hoy no hay con qué.** El álgebra está
verificada y es extensible —el término de transparencia entra solo en el término
lineal, así que el óptimo sigue siendo afín y el paso bilateral exacto y la
maquinaria de Sylvester siguen valiendo—, pero no hay implementación numérica.
Al escribirla, cuidado con el factor 1/2: generalizar la construcción de
matrices restringidas sumando la fricción al bloque reducido da un resultado el
doble de grande. La forma correcta está en `verificacion_algebraica.md`.

---

## 6. Plan mínimo para cerrar la brecha

En orden de razón entre valor y esfuerzo:

1. **Preguntar por el código dual.** Si existe en una máquina local, se recupera
   la mitad del cap. 4 sin escribir una línea. Esto condiciona la carta Gantt, así
   que conviene resolverlo antes de invertir en conseguir datos.
2. **Escribir los scripts de escenario** (3 firmas y 4 firmas) usando
   `venegas_model.py` y `metricas_tesis.py`. Es trabajo de una sesión y deja
   reproducibles la tabla baseline, el escenario fintech y tres de los cuatro
   números titulares.
3. **Corregir `H1`, `H3`, `H7` y agregar validación de $\Sigma$**, con las
   pruebas de captura actualizadas.
4. **Implementar el modelo dual** sobre el álgebra ya verificada.
5. **Regenerar las figuras por código**, incluidos los diagramas de red.

---

## 7. Archivos de la auditoría

| Archivo | Contenido |
|---|---|
| `venegas_model.py` | Núcleo numérico extraído del `.qmd`, transcripción fiel con los defectos preservados y anotados |
| `metricas_tesis.py` | Capa de métricas recuperada (volumen, posiciones, utilidad) y parámetros del escenario de 4 firmas |
| `audit_variantes.py` | Variantes paramétricas para los barridos, separadas de la transcripción fiel |
| `test_venegas_model.py` | 30 pruebas: regresión, captura de defecto, álgebra y cobertura |
| `verificacion_algebraica.md` | Las cuatro derivaciones, la resolución del TODO y la extensión al modelo dual |
| `equivalencia_imports.md` | Justificación del cambio de imports y hallazgo H5 |
| `Verificacion_Derivaciones_rescatado.py` | Verificación simbólica del modelo dual, rescatada de `.ipynb_checkpoints/` |
| `cobertura_resultados.csv` | Inventario de las 32 unidades de resultado |
| `tabla_baseline_cap4.csv`, `escenario_4firmas_cap4.csv` | Reproducción de las tablas del cap. 4 |
| `iterados_tesis.csv` | Contraste contra los iterados publicados |
| `barrido_umbral.csv`, `barrido_eta.csv`, `traza_convergencia.csv` | Diagnósticos de convergencia |
| `ejemplo5_resultados.csv` | Réplica del Ejemplo 5 |
| `fig_replicacion_ejemplo5.png`, `fig_convergencia.png`, `fig_cobertura.png` | Figuras del informe |
| `entorno.txt` | Versiones exactas del entorno |

---

## 8. Limitaciones de esta auditoría

- La verificación algebraica es contra la tesis, no contra el paper original.
- Los valores publicados se compararon contra las figuras y tablas de la tesis
  transcritas a mano; la tolerancia usada es media unidad del último decimal
  impreso.
- No se auditó el modelo dual numéricamente porque no hay implementación.
- Los agregados de pago de la tabla «Payment Structure» del cap. 4 no se
  verificaron.
- No se evaluó el capítulo 5 más allá de los porcentajes titulares que cita.
