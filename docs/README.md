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
| [25-validacion-cualitativa-crisis.md](25-validacion-cualitativa-crisis.md) | **§5.4 — Validación cualitativa de crisis** (R17 inicial): auditoría híbrida (regex + lectura completa) de los 4 tipos A/B/C/D |
| **[22-resultados-post-finetuning.md](22-resultados-post-finetuning.md)** | ⭐ **§10 Scorecard pre/post completo**: baseline E4B vs Mabel v1, batería 12 turnos × 2 runs, score 4.37/5 (87.5%), Crisis Score 100%, limitaciones honestas |
| **[26-memoria-proyecto.md](26-memoria-proyecto.md)** | ⭐ **Narrativa única autocontenida para defensa de tesis** — 7 capítulos: génesis → modelo → dataset → entrenamiento → export → evaluación → conclusiones |
| [27-bitacora-entrenamiento.md](27-bitacora-entrenamiento.md) | **Bitácora del entrenamiento (§7-§9)**: intento local fallido, pivote a RunPod, 10 ajustes técnicos, training real, export GGUF, scorecard final |
| **[28-model-card-hf.md](28-model-card-hf.md)** | **Espejo en GitHub del model card publicado en HuggingFace** (`ZyFalo/mabel-gemma4-e4b`). Fuente única de verdad — se sincroniza a HF con `scripts/sync_hf_readme.py` (Política Opción C, 2026-05-20) |
| [decisiones/](decisiones/) | ADRs individuales para decisiones arquitectónicas mayores |

## Estado del proyecto — ✅ CERRADO v1 (2026-05-20)

**Modelo entrenado**: ✅ `modelos/gemma-4-E4B-mabel-Q4_K_M.gguf` (5.0 GB · SHA256 `3d9ffb485a…3237792`)

**Score formal post-fine-tuning** (batería 12 turnos × 2 runs):
- 🎯 **Score global**: **4.37/5 = 87.5%** (vs baseline 3.50/5 = 70%) → **+17.5 pp de mejora**
- ⭐ **Crisis Score (turnos 8-9)**: **5/5 = 100%** (vs baseline 4/5 = 80%) → objetivo clínico central cristalizado
- 📉 **Verbosidad**: 72 vs 290 tokens/turno → **4× más conciso** sin pérdida de calidad
- ⚠️ **Limitación**: R33 (identidad del creador) cristalizó solo 60% → plan v1.1 documentado

**Dataset final**: 8.040 ej train (60% EN counselling + 40% ES sintético colombiano) + 500 eval. 100% con system prompt B+ unificado. 0 violaciones regex tras validación.

**Costo total del proyecto**: ~$5 USD en RunPod RTX 4090 (vs estimado inicial $10).

**Trabajo futuro identificado**:
- **v1.1** (post-defensa, ~$3 + 6h): R34/R35 reforzados → cristalizar R33 + estabilizar D-020
- **v2** (futuro): Mabel con interfaz de voz (audio_tower Gemma 4 + Coqui XTTS) — arquitectura ya documentada en `docs/01-alcance.md`

### Para evaluadores y compañeros de tesis

Punto de entrada recomendado:

1. **[`26-memoria-proyecto.md`](26-memoria-proyecto.md)** — narrativa única en 7 capítulos (lectura ~30 min)
2. **[`22-resultados-post-finetuning.md`](22-resultados-post-finetuning.md)** — scorecard formal pre/post con las 24 respuestas íntegras
3. **[`27-bitacora-entrenamiento.md`](27-bitacora-entrenamiento.md)** — proceso técnico completo §7-§9 con 10 ajustes documentados
4. **[`03-decisiones.md`](03-decisiones.md)** — 21 ADRs cronológicas con justificación
5. **[`23-bitacora-generacion-sintetica.md`](23-bitacora-generacion-sintetica.md)** — 33 rondas de generación sintética del dataset

Repositorio: https://github.com/ZyFalo/Gemma4-Mabel

## Replicabilidad

Toda decisión, comando y configuración queda documentada en este directorio con el objetivo de que cualquier investigador pueda reproducir el entrenamiento y el despliegue del modelo desde cero.
