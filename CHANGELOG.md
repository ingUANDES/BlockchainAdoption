# Changelog

Registro de hitos de la línea de investigación en redes financieras duales
(Jalan & Chakrabarti → Venegas → Caballero). Formato libre, orden cronológico
inverso; cada entrada referencia el/los commit(s) correspondientes de este
repositorio.

## [No publicado] — Memoria de Eduardo Caballero, segundo semestre 2026

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
- `docs/MemoriasTesis/Caballero/chapters/chapter03.tex` (Desarrollo
  Metodológico): cruce de `ESTRATEGIA_Semestre2.md` con
  `Propuesta_de_Hitos_PT2.tex`, documentando el nuevo orden de hitos
  acordado con el profesor guía (Hito I = prueba de factibilidad empírica el
  27/09; Hito II = calibración y aplicación el 08/11) — pendiente de
  confirmación por escrito, ver PR #2.
- `docs/MemoriasTesis/Caballero/referencias.bib`: 6 entradas nuevas
  (Jalan & Chakrabarti ×2, Eisenberg y Noe, Elliott et al., Venegas, FSB
  2024), reutilizadas desde `docs/MemoriasTesis/Venegas/referencias.bib`.

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
