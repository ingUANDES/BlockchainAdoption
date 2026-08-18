# Estrategia de trabajo — Memoria de Eduardo Caballero, segundo semestre 2026

Profesor guía: Sebastián Cea. Dedicación planificada: 30 horas/semana.

## 1. Punto de partida (primer semestre, ya entregado)

- **Hito 1** (`Hito_1.tex`): revisión crítica de la tesis de Hernán Venegas —
  *Financial Networks & Technological Innovation: A Dual-Network Model* — y del
  modelo base que extiende, Jalan & Chakrabarti,
  [*Incentive-Aware Models of Financial Networks*](https://doi.org/10.1287/opre.2022.0678)
  (Operations Research, 2024).
- **Hito 2** (`Hito_2.tex`): propuesta metodológica — extender el marco dual de
  Venegas (2024) a agentes heterogéneos en aversión al riesgo (γ_i) y creencias
  sobre retornos (μ_i), manteniendo intacta el resto de la estructura del modelo
  (Ψ_i^T, Ψ_i^A, τ, λ_A). Tres preguntas de investigación quedaron formuladas:
  1. ¿Persiste la recuperación completa de bienestar bajo heterogeneidad?
  2. ¿Emerge *sorting* endógeno entre la red tradicional y la alternativa?
  3. ¿Cómo se distribuyen las ganancias de bienestar entre los tres agentes?
- Carta Gantt propuesta por Eduardo para el segundo semestre (adjunta,
  `Caballero_Carta_GanttPT1.pdf`, semanas del 3 de agosto al 23 de noviembre):
  formalización del modelo heterogéneo → implementación y simulaciones →
  aplicación a casos (ej. cooperativas de crédito) → redacción y cierre.

Este plan **retoma esa estructura de trabajo** e la re-secuencia contra las
cuatro fechas de entrega fijadas por el profesor guía, extendiéndola hasta el
28 de diciembre.

## 2. Objetivo general

El resultado de Venegas (2024) —que la red alternativa recupera el 100% del
bienestar perdido por una restricción regulatoria, frente al 27,9% de *gap*
que deja la sola competencia entre intermediarios— se obtuvo bajo agentes
homogéneos y sobre parámetros ilustrativos. El trabajo de Eduardo tiene que
producir dos insumos que la tesis de Venegas no tiene y que un artículo
publicable exige:

1. **Robustez teórica**: verificar si el resultado de recuperación completa de
   bienestar persiste, se atenúa o se revierte bajo heterogeneidad realista
   entre agentes (Hito 2).
2. **Validación empírica**: aplicar el modelo (extendido) a un caso con datos
   reales, siguiendo el precedente metodológico de Jalan & Chakrabarti,
   *Strategic Negotiations in Endogenous Network Formation*
   ([arXiv:2402.08779](https://doi.org/10.48550/arXiv.2402.08779)), quienes
   validan su modelo de negociación en redes tanto en redes simuladas como en
   un dataset real de comercio internacional. El caso concreto para esta
   memoria (candidato mencionado en la carta Gantt: cooperativas de crédito;
   alternativas en evaluación) se decide con el profesor guía en el primer
   hito de este semestre.

Juntos, estos dos insumos son el material que permitiría convertir la tesis
de Venegas en un artículo sometible al *Journal of Financial Intermediation*
o a una revista de investigación de operaciones (p. ej. *Operations
Research*, *Management Science*), con la aplicación a datos reales como el
aporte diferencial de Eduardo sobre el trabajo de Venegas.

## 3. Calendario de hitos

| Fecha | Entrega | Contenido |
|---|---|---|
| **21 de agosto de 2026** | Propuesta de hitos (carta Gantt) | Carta Gantt del semestre re-secuenciada contra este documento, revisada y acordada con el profesor guía. Confirmación del enfoque (analítico vs. computacional) para la extensión heterogénea. |
| **27 de septiembre de 2026** | Primer hito de avance | Modelo heterogéneo formalizado y resuelto analíticamente (equilibrio *pairwise-stable* bajo γ_i, μ_i heterogéneos), con verificación de consistencia contra los casos límite de Venegas (2024). Implementación computacional del caso base y validación numérica contra los resultados ya publicados de Venegas. |
| **8 de noviembre de 2026** | Segundo hito de avance | Simulaciones completas de las tres preguntas de investigación (persistencia de la recuperación de bienestar, *sorting* endógeno, distribución de ganancias) y análisis de sensibilidad ante variaciones en (γ_i, μ_i, τ, λ_A). Aplicación preliminar al caso de datos reales seleccionado, con el dataset ya identificado y una primera calibración. |
| **28 de diciembre de 2026** | Entrega del documento de memoria | `docs/MemoriasTesis/Caballero/Caballero.tex` completo: los cinco capítulos (`chapters/chapter01.tex`–`chapter05.tex`) redactados y compilando sin errores, resultados de la aplicación a datos reales integrados, `referencias.bib` consolidado (Hito 1 + Hito 2 + literatura del caso empírico), y una sección de discusión que conecta los resultados con el objetivo de artículo publicable. |

## 4. Plan de trabajo por período (30 h/semana)

### Agosto 3 – agosto 21 (≈2.5 semanas): confirmación de enfoque y Gantt

- Reunión con el profesor guía para confirmar el enfoque analítico/computacional
  de la extensión heterogénea (tarea ya identificada en la carta Gantt original
  de Eduardo).
- Redactar y acordar la carta Gantt definitiva del semestre (esta tabla, más
  el detalle semanal) — entregable del 21 de agosto.
- Revisar candidatos de dataset real para la aplicación empírica (cooperativas
  de crédito u otra alternativa) y su viabilidad de acceso a datos.

### Agosto 24 – septiembre 25 (≈5 semanas): modelo heterogéneo y validación

- Formalización del modelo heterogéneo: definir (γ_i, μ_i) por agente según
  la jerarquía institucional propuesta en Hito 2 (Banco Nacional: γ bajo,
  creencias precisas; Banco Local: posición intermedia; Empresa Local: γ
  alto, creencias ruidosas).
- Derivar la solución de equilibrio bajo heterogeneidad y demostrar
  consistencia con el modelo de Venegas (2024) en los casos límite
  (λ_A → 0 y τ = 0; Ψ^A inactiva; Ψ^T = Ψ^A).
- Implementar el modelo extendido y el algoritmo de negociación *pairwise* en
  código (extendiendo `code/Financial_Networks_Stability.qmd`), validando
  contra los benchmarks numéricos ya publicados de Venegas (3F-WITH/WITHOUT,
  4F-WITH/WITHOUT).
- **Entregable 27 de septiembre**: modelo formalizado + validación numérica
  del caso base.

### Septiembre 28 – noviembre 6 (≈6 semanas): simulaciones y aplicación a datos reales

- Ejecutar los experimentos numéricos que responden a las tres preguntas de
  investigación del Hito 2.
- Análisis de sensibilidad sobre (γ_i, μ_i, τ, λ_A).
- Identificar y preparar el dataset real de aplicación (siguiendo el
  precedente de Jalan & Chakrabarti en su validación con datos de comercio
  internacional); primera calibración del modelo sobre ese caso.
- **Entregable 8 de noviembre**: resultados de simulación completos +
  aplicación preliminar a datos reales.

### Noviembre 9 – diciembre 28 (≈7 semanas): redacción y cierre

- Completar la calibración y el análisis del caso de datos reales.
- Redactar los capítulos de `Caballero.tex`: marco teórico (capítulo 2,
  reutilizando y ampliando Hito 1), desarrollo metodológico (capítulo 3, el
  modelo heterogéneo y la aplicación empírica), análisis de resultados
  (capítulo 4), conclusiones y recomendaciones (capítulo 5, con la discusión
  orientada a los insumos para el artículo publicable).
- Consolidar `referencias.bib` (actualmente solo contiene la referencia de
  ejemplo de la plantilla).
- Compilar el documento completo, corregir con el profesor guía y cerrar el
  título y resumen del documento (`\titulo`, `\resumen` en `Caballero.tex`
  siguen con el placeholder de la plantilla).
- **Entregable 28 de diciembre**: `Caballero.tex` completo y compilando.

## 5. Riesgos y dependencias

- La selección definitiva del caso de datos reales debe cerrarse antes del
  hito del 8 de noviembre para dejar tiempo de calibración y redacción; es la
  principal dependencia externa del cronograma.
- `Caballero.tex` y sus capítulos están hoy en estado de plantilla (título,
  resumen, capítulos 2–5 sin contenido); el hito de diciembre asume que la
  redacción se apoya en el contenido ya producido en Hito 1/Hito 2 más los
  resultados de este semestre, no en trabajo de cero.
