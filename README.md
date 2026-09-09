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
```

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
> reproducibles con el código versionado; el **100%** de la red alternativa requiere una
> implementación numérica que no existe todavía. Informe completo, inventario unidad por unidad
> y suite de pruebas en [`code/audit/`](code/audit/AUDITORIA_VENEGAS.md).

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
