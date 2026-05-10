# Mabel — Asistente de Apoyo Emocional

Documentación técnica del proyecto de tesis universitaria — Universidad Manuela Beltrán (Colombia).

**Nombre del proyecto y del asistente**: **Mabel**.
**Modelo base**: Gemma 4 E4B.

## Descripción breve

**Mabel** es un asistente conversacional de **apoyo emocional y escucha activa** para estudiantes universitarios, construido a partir del modelo Gemma 4 de Google mediante *fine-tuning* con QLoRA sobre datasets de counselling.

**No es un reemplazo de atención psicológica profesional.** Su función es acompañar, validar emocionalmente y derivar a recursos reales cuando sea necesario.

## Población objetivo

Estudiantes de la Universidad Manuela Beltrán, entre **20 y 26 años**, que participen voluntariamente en las pruebas exploratorias del proyecto.

## Índice de documentación

| Archivo | Contenido |
|---|---|
| [01-alcance.md](01-alcance.md) | Alcance, objetivos, exclusiones y marco ético del proyecto |
| [02-hardware.md](02-hardware.md) | Análisis del hardware disponible y su impacto en decisiones técnicas |
| [03-decisiones.md](03-decisiones.md) | Registro cronológico de decisiones técnicas (ADRs) |
| [04-arquitectura.md](04-arquitectura.md) | *(pendiente)* Arquitectura general: modelo + RAG + memoria |
| [05-datasets.md](05-datasets.md) | *(pendiente)* Datasets utilizados y preparación |
| [06-entrenamiento.md](06-entrenamiento.md) | *(pendiente)* Proceso de fine-tuning con Unsloth + QLoRA |
| [07-rag-y-memoria.md](07-rag-y-memoria.md) | *(pendiente)* Sistema RAG, embeddings y memoria del usuario |
| [08-guardrails.md](08-guardrails.md) | *(pendiente)* Seguridad, detección de crisis, protocolos |
| [09-api-local.md](09-api-local.md) | *(pendiente)* API de inferencia local para frontend/backend |
| [10-evaluacion.md](10-evaluacion.md) | *(pendiente)* Protocolo de evaluación del modelo |
| [11-instalacion.md](11-instalacion.md) | Guía reproducible de instalación y despliegue |
| [12-baseline-modelo-base.md](12-baseline-modelo-base.md) | Baseline del modelo **sin fine-tuning** (comportamiento de referencia) |
| [14-comparativa-e4b-vs-26b.md](14-comparativa-e4b-vs-26b.md) | Comparativa inicial E4B vs 26B MoE (4 prompts) |
| [15-bateria-evaluacion.md](15-bateria-evaluacion.md) | Diseño de la batería de evaluación: 12 turnos + rúbrica |
| [16-analisis-etico-comparativo.md](16-analisis-etico-comparativo.md) | Análisis ético E4B vs 26B MoE (batería completa) |
| [18-comparativa-triple-modelos.md](18-comparativa-triple-modelos.md) | **Comparativa triple**: E4B vs 26B MoE vs Gemma 3 27B |
| [19-comparativa-deepseek-r1-vs-e4b.md](19-comparativa-deepseek-r1-vs-e4b.md) | **Comparativa DeepSeek R1 (14B/32B) vs E4B** — fallos descalificantes |
| [20-justificacion-seleccion-modelo.md](20-justificacion-seleccion-modelo.md) | **Documento unificado**: justificación empírica de la selección de Gemma 4 E4B (5 modelos, turnos críticos inline, scorecard) |
| [20b-resumen-ejecutivo-tesis.md](20b-resumen-ejecutivo-tesis.md) | **Resumen ejecutivo para la tesis** — versión académica autocontenida |
| [21-parametros-entrenamiento.md](21-parametros-entrenamiento.md) | Explicación detallada de los 11 parámetros de entrenamiento con justificación |
| [23-bitacora-generacion-sintetica.md](23-bitacora-generacion-sintetica.md) | **Bitácora viva** del proceso de generación sintética (§5): rondas, temas, conteos, decisiones |
| [24-validacion-cualitativa-sintetico.md](24-validacion-cualitativa-sintetico.md) | **§5.2 — Validación cualitativa** del sintético normal: lectura estratificada de 27 conversaciones representativas, hallazgos y veredicto |
| [decisiones/](decisiones/) | ADRs individuales para decisiones arquitectónicas mayores |

## Estado del proyecto

**Fase actual**: Definición de alcance y decisiones técnicas iniciales.

**Siguiente hito**: Validar pipeline de entrenamiento con Gemma 4 E2B sobre subset pequeño de MentalChat16K.

## Replicabilidad

Toda decisión, comando y configuración queda documentada en este directorio con el objetivo de que cualquier investigador pueda reproducir el entrenamiento y el despliegue del modelo desde cero.
