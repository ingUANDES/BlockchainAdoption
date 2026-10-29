# Changelog

Registro de hitos de la línea de investigación en redes financieras duales
(Jalan & Chakrabarti → Venegas → Caballero). Formato libre, orden cronológico
inverso; cada entrada referencia el/los commit(s) correspondientes de este
repositorio.

## [No publicado] — Memoria de Eduardo Caballero, segundo semestre 2026

### Añadido
- `README.md`: subsección «Índice de reproducción», con el orden de ejecución, la
  entrada y la salida de cada script y qué cifra del Capítulo 4 sustenta, más la
  declaración explícita de los cuatro resultados que **no** tienen productor
  versionado, a saber `tab:sensibilidad-sigma13`, `tab:reproduccion-escenarios`,
  `tab:reproduccion-dual` y las cinco figuras del documento.
- `code/caballero/`: implementación propia de la memoria, separada de
  `code/audit/` porque aquélla documenta el código de la tesis que se continúa y
  ésta es aporte de esta memoria. Cierra dos huecos sucesivos, a saber que el
  modelo dual tenía su álgebra derivada pero no su implementación numérica, y que
  la Sección 4.5 del Capítulo 4 afirmaba sus cifras sin que hubiera en el
  repositorio código que las produjera. Versionar ese respaldo **no compromete al
  autor con sus cifras**, puesto que siguen abiertas la decisión sobre la
  restricción de balance y el costo de originación minorista de CORFO, y si
  cambian hay que rehacer las corridas; lo que se gana es que lo ya afirmado pase
  a ser auditable. Las cifras que cada archivo produce viven en el capítulo y no
  se repiten aquí. La batería del directorio pasa a 62 pruebas.
  - `caballero_dual.py`: la capa de acoplamiento. Reduce el reparto
    $w_i^T + w_i^A = w_i^*$ a una corrida del mismo Algorithm 1 ya auditado,
    con los primitivos transformados, de modo que no introduce un algoritmo de
    negociación nuevo. Reproduce las utilidades totales publicadas, con 1,4339
    frente a 1,434 en tres agentes y 1,7417 frente a 1,741 en cuatro.
  - `caballero_verificacion_objeciones.py`: re-derivación simbólica de la
    especificación dual, en ocho bloques. Establece que las dos contabilidades
    del riesgo que la tesis de referencia usa son incompatibles entre sí, ya
    que con el riesgo evaluado sobre la exposición combinada el Hessiano
    colapsa y el reparto deja de tener óptimo interior, y que la fórmula de
    $Q^{dual}$ está mal transcrita en el capítulo 3 y en el Apéndice A.1.
    Refuta además una objeción anterior sobre el supuesto de precio único,
    cuya formulación correcta es que ese supuesto no admite punto factible
    bajo restricción.
  - `test_caballero_dual.py`: 25 pruebas en cuatro grupos. El grupo
    `test_discrepancia_*` fija las cifras publicadas que la capa **no**
    reproduce, a saber las tasas de adopción, los volúmenes por red y el
    diferencial de precio, de modo que la brecha quede dentro de la batería y
    no fuera de ella.
  - `venegas_model_corregido.py` y su batería de 21 pruebas: corrige los
    hallazgos H1, H3, H4b y H7 de la auditoría más la validación de $\Sigma$,
    sin modificar `venegas_model.py`, que se preserva como evidencia.
    Verificado que ambos motores entregan valores duales idénticos.
  - `_ruta_audit.py` y `conftest.py`: resuelven en un único lugar la ruta a
    `code/audit`, desde la ubicación del archivo y no desde el directorio de
    trabajo, para que las baterías corran sin `PYTHONPATH`.
  - `caballero_caso_chileno.py` y `caballero_corrida_caso_chileno.py`: el caso
    principal de cuatro agentes, con los primitivos chilenos, el diagnóstico de
    por qué la capa de acoplamiento no corre sobre esa topología y los barridos.
    Producen `caballero_caso_chileno_resultados.csv`, de donde salen las 768
    combinaciones, las 312 admisibles, las medianas de adopción de 20,7 y 16,3
    por ciento y el bloque `resumen_dominancia` con las amplitudes de 6,3 y 23,2
    puntos filtradas y 29,0 sin filtrar.
  - `caballero_contrafactual_mandato.py`, `caballero_corrida_contrafactual.py` y
    `test_caballero_contrafactual.py`: la rama contrafactual del mandato de
    segundo piso, tres agentes, donde la red Alternativa sí reconecta el par que
    la restricción separa. Producen `caballero_contrafactual_resultados.csv`, de
    donde salen el costo del mandato, la fracción recuperada, las dos fricciones
    de equilibrio y la descomposición por coordenada. Las 16 pruebas fijan tanto
    las cifras del informe como la identidad algebraica que hace que la
    recuperación valga 100 por ciento sin fricción, para que el capítulo no la
    cite como hallazgo.
  - `caballero_ele7_canales.py`: reparto de canales de financiamiento de las
    MiPyMEs en la ELE-7, ponderado por el factor de expansión. Reproduce la tabla
    por tramo y las tres cifras derivadas que el capítulo comenta, con la
    coexistencia en 93,00 por ciento, la participación bancaria entre empresas
    con deuda formal en 99,49 y la declaración de no usar fuente alguna en 7,34.
    El extracto de microdatos que toma como entrada no se versiona por su tamaño.
  - `entorno.txt`: declaración de versiones del directorio, equivalente a la de
    `code/audit/entorno.txt`, de la que **difiere** porque la auditoría se corrió
    en un entorno posterior. Deja constancia además de que ninguna figura del
    documento se construye dentro de `code/`.
- `.gitignore`: el repositorio no tenía uno. Cubre artefactos de Python, de
  LaTeX y de Quarto; las excepciones deliberadas están anotadas en el propio
  archivo.
- `README.md`: bloque de estructura actualizado, subsección con los comandos de
  las tres baterías y nota de estado de la capa dual. Se corrigió la
  afirmación de que el 100 % de recuperación de bienestar requería una
  implementación numérica inexistente.
- `code/audit/`: auditoría completa del código de la tesis de Venegas, previa a
  la calibración con datos reales (branch `audit/venegas-code`). Ningún archivo
  del autor original fue modificado; se tratan como evidencia.
  - `AUDITORIA_VENEGAS.md`: informe por severidad, con veredicto de
    reproducibilidad e implicancias para la calibración con datos externos.
  - `venegas_model.py`: núcleo numérico extraído del `.qmd` como transcripción
    fiel, con los defectos preservados y anotados; equivalente bit a bit al
    original en ambos escenarios del Ejemplo 5.
  - `metricas_tesis.py`: capa de métricas del capítulo 4 (volumen bilateral,
    posiciones, utilidad media-varianza) recuperada por ingeniería inversa —
    ningún archivo versionado la calculaba — más los parámetros del escenario
    de 4 firmas declarados en la tesis.
  - `verificacion_algebraica.md`: cuatro derivaciones verificadas con SymPy
    contra el modelo declarado en la tesis, resolución del `# todo` de la
    solución de Sylvester y extensión al modelo dual.
  - `test_venegas_model.py`: 30 pruebas (regresión de resultados publicados,
    captura de defectos, álgebra y cobertura).
  - `Verificacion_Derivaciones_rescatado.py`: verificación simbólica del modelo
    dual, rescatada de `code/.ipynb_checkpoints/` donde estaba mal versionada.
  - Inventario de cobertura, réplicas de las tablas del capítulo 4, barridos de
    convergencia y tres figuras del informe.

- `docs/MemoriasTesis/Caballero/Diapositivas.tex` queda completo y compilable:
  se agregan sus cuatro dependencias que faltaban desde el sync original de
  Overleaf (commit `801e1a4`) — `Flecha-Uandes-Blanco.png`, `Flecha-Uandes.png`,
  `logoFICA.png` y `references.bib` (bibliografía propia de la presentación,
  independiente de `referencias.bib` de la memoria).
- `docs/MemoriasTesis/Caballero/font/DMSans.ttf`: fuente variable oficial de
  Google Fonts (licencia OFL, incluida en `font/OFL.txt`) que reemplaza a los
  tres archivos estáticos por peso (`DMSans-Medium/SemiBold/Bold.ttf`) que el
  `.tex` original esperaba y que Google ya no distribuye por separado.

### Corregido
- `README.md`: la descripción de `code/Financial_Networks_Stability.qmd` le
  atribuía el modelo dual, los cálculos de bienestar y los scripts de los
  escenarios 3F/4F. La auditoría establece que el archivo contiene solo el
  núcleo numérico y la réplica del Ejemplo 5. Se agrega una nota de estado con
  qué resultados de la tesis son reproducibles hoy y cuáles no.
- `docs/MemoriasTesis/Caballero/Diapositivas.tex`: se reescribe el bloque de
  configuración de tipografía (`\setsansfont`, `\newfontfamily`) para
  seleccionar los pesos Medium/SemiBold/Bold como instancias nombradas
  (`wght=500/600/700`, `opsz=14`) de la fuente variable única `font/DMSans.ttf`,
  en vez de archivos sueltos por peso que ya no existen.
- `docs/MemoriasTesis/Caballero/chapters/chapter02.tex` (Marco Teórico):
  revisión ortográfica y gramatical completa tras la edición en Overleaf
  (acentuación, concordancia y fragmentos de oración sin verbo). Se
  resuelven además tres comentarios pendientes de Overleaf:
  - Se atribuye la definición general de redes (nodos/vínculos, decisión
    estratégica costo-beneficio) a `jackson_strategic_1996` (Jackson &
    Wolinsky, 1996), la referencia fundacional del concepto de estabilidad
    por pares (\textit{pairwise stability}) usado más adelante en el
    capítulo.
  - Se precisa la cifra de "comisiones sustanciales" por transferencias
    internacionales con datos del *World Bank Remittance Prices Worldwide*
    (`world_bank_rpw_2025`): costo promedio mundial 6,36\%, 4,59\% por
    canales digitales vs. 7,30\% por canales no digitales.
  - Se agrega el dato de exclusión financiera (~2 mil millones de adultos,
    38\% de la población adulta mundial en 2014) a la cita ya existente
    `singer_financial_2017`.
- `docs/MemoriasTesis/Caballero/referencias.bib`: se agregan las entradas
  `jackson_strategic_1996` y `world_bank_rpw_2025` (34 entradas en total).

### Añadido
- `docs/MemoriasTesis/Caballero/ESTRATEGIA_Semestre2.md`: plan de trabajo y
  calendario de hitos para el segundo semestre (agosto–diciembre 2026).
- README.md del repositorio enriquecido con la línea de trabajo Venegas → Caballero
  y el objetivo de publicación.
- `docs/MemoriasTesis/Caballero/chapters/chapter01.tex` (Introducción):
  contexto de mercado (shadow banking / NBFI, FSB 2024), objetivo general y
  las tres preguntas de investigación heredadas de `Hito_2.tex`, reformuladas
  en su versión homogénea (sin extensión a agentes heterogéneos) y con foco
  en la validación empírica de PT2.
- `docs/MemoriasTesis/Caballero/chapters/chapter02.tex` (Marco Teórico):
  consolidación de la revisión de modelos de redes financieras
  (Eisenberg-Noe 2001, Elliott et al. 2014, Jalan & Chakrabarti 2024) y del
  modelo dual de Venegas (2024), incluyendo el resultado de recuperación de
  bienestar y las oportunidades de mejora identificadas en `Hito_1.tex`.
  Ampliado posteriormente (de 131 a 340 líneas) con mayor profundidad
  teórica siguiendo el nivel de detalle del capítulo homólogo de Venegas:
  propagación de riesgo y estabilidad de redes (Glasserman & Young 2016,
  Acemoglu et al. 2015); mercados de dos lados y competencia entre
  plataformas (Economides 1996, Rochet & Tirole 2003, Armstrong 2006,
  Caillaud & Jullien 2003); arbitraje regulatorio (Kane 1977, Buchak et al.
  2018, Gorton & Metrick 2010); adopción tecnológica y dependencia de
  trayectoria (Katz & Shapiro 1985, Farrell & Saloner 1986, Arthur 1989); y
  una nueva sección de blockchain/DeFi (Nakamoto 2008, Cong & He 2019,
  Makarov & Schoar 2021, Huberman et al. 2021, Schär 2021, Werbach 2018,
  Zetzsche et al. 2020, Auer et al. 2023) conectada explícitamente con los
  parámetros $\tau$ y $\lambda_A$ del modelo dual.
- `docs/MemoriasTesis/Caballero/chapters/chapter03.tex` (Desarrollo
  Metodológico): cruce de `ESTRATEGIA_Semestre2.md` con
  `Propuesta_de_Hitos_PT2.tex`, documentando el nuevo orden de hitos
  acordado con el profesor guía (Hito I = prueba de factibilidad empírica el
  27/09; Hito II = calibración y aplicación el 08/11) — pendiente de
  confirmación por escrito, ver PR #2.
- `docs/MemoriasTesis/Caballero/referencias.bib`: 6 entradas iniciales
  (Jalan & Chakrabarti ×2, Eisenberg y Noe, Elliott et al., Venegas, FSB
  2024) más 25 entradas adicionales para el enriquecimiento del marco
  teórico, todas reutilizadas desde
  `docs/MemoriasTesis/Venegas/referencias.bib` (32 entradas en total, incluyendo
  `cornell_valuing_2020`, ya presente antes de este PR).
- `docs/MemoriasTesis/Caballero/mapeo_literatura.md`: tabla de mapeo
  tema → cita → fuente usada para el enriquecimiento del capítulo 2.

## 2026-08-18 — `9831636` fin PT1 pre planificación PT2

### Añadido
- `docs/MemoriasTesis/Caballero/Hito_1.tex`: revisión crítica de la tesis de
  Venegas (2024) — marco teórico, modelo dual, resultados de los cuatro
  escenarios (3F/4F, con/sin restricción) y tres oportunidades de mejora
  (calibración empírica, dinámica de adopción, riesgo sistémico/contagio).
- `docs/MemoriasTesis/Caballero/Hito_2.tex`: propuesta metodológica — extensión
  del modelo dual de Venegas a agentes heterogéneos en aversión al riesgo
  (γ_i) y creencias sobre retornos (μ_i), con tres preguntas de investigación
  derivadas.
- Cierra el primer semestre de la memoria de Eduardo Caballero
  (co-autoría: educaballero007).

## 2026-04-30 — `a69d9a0` inicio trabajo Caballero

### Añadido
- Se habilita `docs/MemoriasTesis/Caballero/` como carpeta de trabajo de la
  nueva memoria de pregrado (plantilla de Ingeniería UAndes clonada desde la
  estructura de Venegas).

### Reorganizado
- Diapositivas y material de `Data Storage` reubicados bajo `docs/Diapos/`.

## 2026-04-02 — `02cbf8c` pre memoria Caballero

### Añadido
- Notebooks `Verificacion_Derivaciones.{ipynb,py}` con la verificación numérica
  de las derivaciones del modelo dual, y figuras adicionales de sensibilidad
  (`docs/Diapos/`) usadas como material previo a la incorporación de Caballero.

## 2025-06-03 — `809fac3` benchmarks jalan

### Añadido
- Benchmarks numéricos contra el modelo base de Jalan & Chakrabarti (2024)
  que sustentan los resultados reportados en la tesis de Venegas.

## 2024-08 a 2024-11 — Tesis de magíster de Hernán Venegas

### Añadido
- `docs/MemoriasTesis/Venegas/memoria.tex` y capítulos: desarrollo completo de
  *Financial Networks & Technological Innovation: A Dual-Network Model*,
  extensión del modelo de Jalan & Chakrabarti (2024,
  [10.1287/opre.2022.0678](https://doi.org/10.1287/opre.2022.0678)) a redes
  duales (Tradicional/Alternativa).
- `code/Financial_Networks_Stability.qmd`: notebook Quarto/Python con los
  escenarios numéricos (Ejemplo 1, Ejemplo 5, redes 3F/4F con y sin
  restricción, caso dual-network) — commits `7fd5ae9` (parametrización y
  extensión del modelo de Jalan et al.), `e25e398`–`e2f9e39` (iteración de
  ejemplos), `ce9dd82`/`fad0b55` (Ejemplo 5 con y sin restricciones).
- Definición de título de la tesis (`7072c18`, 2024-08-21).
- Presentación de defensa en `docs/Diapos/slides_defensa.tex`.

## 2022-09-02 — `8524420` Initial Overleaf Import

### Añadido
- Importación inicial del repositorio desde Overleaf: plantilla de memoria de
  Ingeniería UAndes y estructura base de carpetas (`docs/`, `code/`).
