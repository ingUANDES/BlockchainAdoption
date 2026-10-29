# Financial Networks & Technological Innovation — Memorias y Tesis UAndes

Repositorio del Laboratorio de Economía Financiera (Facultad de Ingeniería y Ciencias
Aplicadas, Universidad de los Andes) para la línea de investigación sobre redes financieras,
arbitraje regulatorio e infraestructuras alternativas (blockchain / *shadow banking*). Reúne
las memorias y tesis sucesivas que extienden el modelo de negociación bilateral en redes de
Jalan & Chakrabarti, *Incentive-Aware Models of Financial Networks*, Operations Research
(2024), [10.1287/opre.2022.0678](https://doi.org/10.1287/opre.2022.0678) — ficha en la
colección de grupo `TechoInno` de Zotero.

## Estructura del repositorio

```
docs/
  MemoriasTesis/
    Venegas/      # Tesis de magíster de Hernán Venegas (terminada, 2024)
    Caballero/    # Memoria de pregrado de Eduardo Caballero (en curso, 2026)
  Diapos/         # Presentaciones de avance y de defensa
code/
  Financial_Networks_Stability.qmd   # Notebook Quarto/Python: núcleo numérico del modelo
                                     # (cartera óptima, restricciones de arista, dinámica
                                     # de precios) y réplica del Ejemplo 5 de Jalan &
                                     # Chakrabarti. Ver nota de estado más abajo.
  audit/                             # Auditoría del código de Venegas (sept. 2026):
                                     # informe, módulo extraído, verificación algebraica,
                                     # inventario de cobertura y suite de 30 pruebas
  caballero/                         # Implementación propia de la memoria (oct. 2026):
                                     # capa de acoplamiento T/A del modelo dual, motor
                                     # corregido, re-derivación de las objeciones a la
                                     # especificación dual, corrida del caso chileno,
                                     # rama contrafactual del mandato de segundo piso,
                                     # procesamiento de la ELE-7 y sus 62 pruebas
```

La separación entre `audit/` y `caballero/` es deliberada. `audit/` documenta el código de
la tesis que se continúa y preserva su evidencia sin modificarla, mientras que `caballero/`
contiene implementación propia de la memoria en curso. Los módulos de `caballero/` importan
`venegas_model.py` y `metricas_tesis.py` desde `audit/`, resolviendo la ruta en
`code/caballero/_ruta_audit.py`, de modo que no hace falta `PYTHONPATH`.

### Baterías de pruebas

```bash
# Auditoría del motor original (30 pruebas). Debe correrse desde su propia
# carpeta, porque una prueba lee ../Financial_Networks_Stability.qmd por ruta relativa.
cd code/audit && python -m pytest test_venegas_model.py -v

# Motor corregido (21), capa de acoplamiento T/A (25) y rama contrafactual (16),
# desde la raíz del repositorio.
python -m pytest code/caballero -v

# Re-derivación algebraica de las objeciones a la especificación dual.
python code/caballero/caballero_verificacion_objeciones.py
```

### Índice de reproducción

Qué ejecutar, en qué orden, con qué entrada y qué cifra del Capítulo 4 sale de
cada cosa. No explica qué hace cada módulo: eso está en su propio encabezado, que
la columna del script enlaza.

Los tres scripts escriben su salida **en el directorio actual**, no junto al
módulo. Ejecutándolos desde la raíz, como indican los comandos, los archivos
quedan en la raíz y no en `code/caballero/`, que es donde están las copias
versionadas que el documento cita.

| # | Script | Entrada | Escribe | Sustenta |
|---|---|---|---|---|
| 1 | [`caballero_verificacion_objeciones.py`](code/caballero/caballero_verificacion_objeciones.py) | ninguna | nada, sólo imprime | los hallazgos C1 a C4 sobre la especificación dual |
| 2 | [`caballero_corrida_caso_chileno.py`](code/caballero/caballero_corrida_caso_chileno.py) | ninguna | `caballero_caso_chileno_resultados.csv` | Sección 4.5.3: las 768 combinaciones, las 312 admisibles, las medianas de adopción y las amplitudes de dominancia |
| 3 | [`caballero_corrida_contrafactual.py`](code/caballero/caballero_corrida_contrafactual.py) | ninguna | `caballero_contrafactual_resultados.csv` | `tab:contrafactual-sigma`, más el costo del mandato, la fracción recuperada y las dos fricciones de equilibrio |
| 4 | [`caballero_ele7_canales.py`](code/caballero/caballero_ele7_canales.py) | `ele7_financiamiento_mipyme.csv`, **no distribuido** | `ele7_reparto_canales_mipyme.csv` | `tab:ele7-canales` y las tres cifras derivadas del texto |

El orden es de dependencia conceptual y no de ejecución: los cuatro son
independientes entre sí y pueden correrse en cualquier orden. El primero no
produce archivo y conviene correrlo antes porque establece bajo qué
especificación son válidos los otros dos.

```bash
python code/caballero/caballero_verificacion_objeciones.py
python code/caballero/caballero_corrida_caso_chileno.py
python code/caballero/caballero_corrida_contrafactual.py
python code/caballero/caballero_ele7_canales.py <ruta>/ele7_financiamiento_mipyme.csv
```

Las versiones con las que se verificó están en
[`code/caballero/entorno.txt`](code/caballero/entorno.txt), que **no** coinciden
con las de [`code/audit/entorno.txt`](code/audit/entorno.txt).

#### Qué no tiene productor versionado

Cuatro resultados del Capítulo 4 no se reproducen con lo que hay en el
repositorio. Se declara aquí en vez de dejarlo implícito.

- **`tab:sensibilidad-sigma13`**. Ningún archivo de `code/` menciona esa
  sensibilidad ni hay planilla suya versionada, de modo que su dato vive
  enteramente fuera del repositorio.
- **`tab:reproduccion-escenarios`**, los cuatro escenarios de referencia. Sus
  planillas `tabla_baseline_cap4.csv` y `escenario_4firmas_cap4.csv` sí están
  versionadas en `code/audit/`, pero ningún `.py` del repositorio las escribe:
  las únicas escrituras a CSV son las tres de la tabla de arriba. Lo mismo vale
  para las otras seis planillas de `code/audit/`, a saber `barrido_eta.csv`,
  `barrido_umbral.csv`, `cobertura_resultados.csv`, `ejemplo5_resultados.csv`,
  `iterados_tesis.csv` y `traza_convergencia.csv`.
- **`tab:reproduccion-dual`**. [`caballero_dual.py`](code/caballero/caballero_dual.py)
  la sustenta, y sus cifras están fijadas en `test_caballero_dual.py`, pero el
  módulo no deja planilla, de modo que hoy se reproduce ejecutándolo a mano o
  leyendo la suite.
- **Las cinco figuras del documento.** No hay ningún `savefig` activo en `code/`:
  las cuatro apariciones en `Financial_Networks_Stability.qmd` están comentadas.
  Las dos figuras de la Sección 4.5 se construyeron fuera de los scripts, así que
  los `.png` del documento no tienen origen reproducible, aunque las cifras que
  muestran sí salen de los CSV de la tabla de arriba.

No proceden de código, por su naturaleza, `tab:venegas-escenarios`, que es de la
fuente, ni `tab:comparacion-venegas` y `tab:comparacion-ocde-corfo`, que son del
levantamiento documental.

Cada carpeta de memoria/tesis sigue la plantilla estándar de Ingeniería UAndes: `core/`
(preámbulo y primeras páginas), `chapters/` (capítulos), `attachments/` (anexos),
`referencias.bib` y un archivo principal (`memoria.tex` o `<Apellido>.tex`) que hace
`\input` de todo lo anterior. Ver `LEEME.txt` dentro de cada carpeta para instrucciones de
compilación con pdfLaTeX.

## Línea de trabajo

### 1. Tesis de magíster — Hernán Venegas (terminada)

*Financial Networks & Technological Innovation: A Dual-Network Model* (`docs/MemoriasTesis/Venegas/memoria.tex`,
compila con XeLaTeX). Extiende el modelo de negociación bilateral *pairwise-stable* de
Jalan & Chakrabarti (2024) a un entorno de **redes duales**: cada agente reparte su
exposición entre una red Tradicional ($T$, sujeta a restricciones regulatorias) y una red
Alternativa ($A$, con tratamiento regulatorio distinto pero costos de fricción propios).

Resultado central, sobre una red de referencia de tres agentes (Banco Nacional, Banco
Local, Firma Local) con una restricción regulatoria que prohíbe el contrato directo
Banco Nacional–Firma Local: la restricción destruye el 50,8% del bienestar potencial; la
competencia entre intermediarios (agregando una Fintech) recupera solo el 46,5% de esa
pérdida, dejando un *gap* de 27,9%; habilitar una red alternativa recupera el **100%** del
bienestar del benchmark sin restricción.

> **Estado del código (auditoría de septiembre 2026).** `code/Financial_Networks_Stability.qmd`
> contiene el **núcleo numérico** del modelo de Jalan & Chakrabarti (optimización de cartera,
> restricciones de arista y dinámica de precios bilateral) y reproduce el Ejemplo 5 del paper,
> pero **no** contiene el modelo dual, ni la capa de métricas (utilidad, bienestar, volumen), ni
> los scripts de los escenarios 3F/4F, ni el código que genera las figuras del capítulo 4. Las
> derivaciones simbólicas del modelo dual sí existen y están verificadas, pero quedaron en
> `code/.ipynb_checkpoints/`. De los tres números citados arriba, 50,8%, 46,5% y 27,9% son
> reproducibles con el código versionado. Informe completo, inventario unidad por unidad
> y suite de pruebas en [`code/audit/`](code/audit/AUDITORIA_VENEGAS.md).

> **Implementación de la capa dual (octubre 2026).** El **100%** de recuperación de
> bienestar ya no depende de una implementación inexistente: la capa de acoplamiento T/A
> está implementada en [`code/caballero/caballero_dual.py`](code/caballero/caballero_dual.py)
> y reproduce las utilidades totales publicadas, con 1,4339 frente a 1,434 en el escenario
> de tres agentes y 1,7417 frente a 1,741 en el de cuatro. La reproducción, sin embargo, es
> parcial y de un modo que importa. Se reproducen exactamente las cifras que no dependen del
> reparto entre redes, y ninguna de las que sí dependen de él, ya que las tasas de adopción
> dan 81,6% y 75,0% frente a los 72,4% y 47,9% publicados. La re-derivación algebraica de la
> especificación dual, en
> [`caballero_verificacion_objeciones.py`](code/caballero/caballero_verificacion_objeciones.py),
> muestra por qué, dado que las utilidades totales dependen únicamente de la exposición total
> del agente, que está fija por supuesto y no cambia entre escenarios. Las cifras no
> reproducidas quedan fijadas como regresión en el grupo `test_discrepancia_*` de
> [`test_caballero_dual.py`](code/caballero/test_caballero_dual.py), en vez de quedar fuera
> de la batería.

El propio Venegas identifica en la sección 5.3.3 de su tesis dos supuestos simplificadores
que dejan las estimaciones de recuperación de bienestar en un piso conservador: aversión al
riesgo homogénea (γ_i = 1 para todo agente) y creencias homogéneas sobre retornos
(μ_i = μ_j).

### 2. Memoria de pregrado — Eduardo Caballero (en curso, comenzó abril 2026)

`docs/MemoriasTesis/Caballero/`. Primer semestre completado (dos hitos entregados):

- **Hito 1** (`Hito_1.tex`): revisión crítica de la tesis de Venegas — marco teórico
  (Eisenberg-Noe 2001, Elliott-Golub-Jackson 2014, Jalan-Chakrabarti 2024), el modelo dual,
  los cuatro escenarios numéricos y tres oportunidades de mejora: calibración empírica de
  los primitivos del modelo, dinámica/dependencia de trayectoria, y riesgo sistémico/contagio.
- **Hito 2** (`Hito_2.tex`): propuesta metodológica para el segundo semestre — relajar los
  dos supuestos de homogeneidad que Venegas señala como limitación, permitiendo
  γ_i ≠ γ_j y μ_i ≠ μ_j entre los tres agentes, con el objetivo de
  responder si la recuperación completa de bienestar persiste bajo heterogeneidad, si emerge
  *sorting* endógeno entre redes, y cómo se reparten las ganancias entre agentes.

El documento general de la memoria (`Caballero.tex` y sus capítulos en `chapters/`) está
aún en plantilla — sin título definido y sin contenido más allá de los encabezados
heredados del molde de Ingeniería. `referencias.bib` conserva todavía solo la referencia de
ejemplo de la plantilla; falta poblarlo con la bibliografía de Hito 1/Hito 2 y con las
referencias de Venegas.

Para el segundo semestre, el plan de trabajo y las fechas de entrega están documentados en
[`docs/MemoriasTesis/Caballero/ESTRATEGIA_Semestre2.md`](docs/MemoriasTesis/Caballero/ESTRATEGIA_Semestre2.md).

## Objetivo general de la línea de investigación

El resultado de la tesis de Venegas es, hasta ahora, un ejercicio numérico sobre parámetros
ilustrativos. El objetivo de la memoria de Caballero es doble: (i) someter esos resultados a
una revisión crítica y a una extensión teórica (heterogeneidad de agentes) que ya está en
marcha, y (ii) generar los insumos —calibración y aplicación a un caso con datos reales,
inspirada en la validación empírica de Jalan & Chakrabarti,
*Strategic Negotiations in Endogenous Network Formation*
([arXiv:2402.08779](https://doi.org/10.48550/arXiv.2402.08779))— que permitan convertir la
tesis de Venegas en un artículo publicable en el *Journal of Financial Intermediation* o en
una revista de investigación de operaciones. El caso de datos reales concreto está aún en
evaluación.

## Bibliografía del grupo

La colección `TechoInno` en la biblioteca Zotero de grupo `fIngUANDES` reúne la literatura
de referencia de esta línea (redes financieras, mercados de dos lados, arbitraje
regulatorio, blockchain/DeFi).
