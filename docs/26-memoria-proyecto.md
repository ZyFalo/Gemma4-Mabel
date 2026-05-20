# 26 — Memoria del proyecto Mabel v1

> **Narrativa única** del proyecto de tesis "Mabel — Asistente conversacional de apoyo emocional para estudiantes universitarios colombianos". Documento autocontenido para defensa académica que sintetiza el ciclo completo desde la concepción hasta el modelo final entrenado.
>
> **Autor**: William Andrés Peña Vargas
> **Institución**: Universidad Manuela Beltrán (UMB), Colombia
> **Período**: abril 2026 – mayo 2026
> **Modelo base**: Gemma 4 E4B (Google)
> **Técnica**: QLoRA r=32 con Unsloth + bf16
> **Resultado**: `gemma-4-E4B-mabel-Q4_K_M.gguf` (5.0 GB, SHA256 `3d9ffb485a…3237792`)

---

## Capítulo 1 — Génesis del proyecto

### El problema observado

Los estudiantes universitarios colombianos enfrentan **carga emocional creciente** (estrés académico, presión familiar, aislamiento social, sintomatología depresiva subclínica) que con frecuencia **no llega a servicios profesionales** por barreras de acceso: horarios limitados de Bienestar, percepción de estigma, distancia geográfica al campus, dudas sobre qué califica como "suficientemente grave" para pedir ayuda.

La pregunta inicial fue: **¿es factible construir un asistente conversacional especializado, local, accesible 24/7, que actúe como puente entre el estudiante y los recursos profesionales** (Bienestar UMB, Línea 106, Línea 123) — sin pretender reemplazar a un profesional, pero acortando el camino hacia él?

### Marco institucional

Proyecto de tesis exploratoria en la **Universidad Manuela Beltrán (UMB)** propuesto por un profesor de ética como exploración técnica y normativa del uso de modelos de lenguaje pequeños para acompañamiento emocional en contexto universitario colombiano. Población objetivo: estudiantes UMB de 20-26 años en pruebas voluntarias con consentimiento informado.

### Restricciones declaradas

Mabel **no** diagnostica, no prescribe, no sustituye terapia. Se presenta siempre como IA, deriva a recursos reales colombianos en crisis, opera local (sin envío de datos a servicios cloud). Marco ético: Resolución 8430 de 1993 del MinSalud (Colombia), Ley 1581 de 2012 (protección de datos), Acuerdo 04 de 2020 del Colegio Colombiano de Psicólogos.

(Detalle completo: `docs/01-alcance.md`)

---

## Capítulo 2 — Selección del modelo base

### Hardware del autor (restricción dura)

NVIDIA RTX 2060 Mobile, **6 GB VRAM**, arquitectura Turing (sin bf16 nativo). Imposible entrenar modelos densos >7B en local. Esto delimitó las opciones a la familia E (Effective parameters) de Gemma 4: E2B (~3B totales) y E4B (~8B totales con activación selectiva).

### Comparativa empírica de 5 modelos candidatos

Se evaluaron 5 modelos con una **batería estandarizada de 12 turnos** + scorecard de 15 criterios:

| Modelo | Params | Familia | Score final |
|---|---|---|---|
| Gemma 4 26B MoE | 26B (4B activos) | Gemma 4 | 4.1/5 |
| Gemma 4 E4B | 8B (4B efectivos) | Gemma 4 | 3.5/5 |
| Gemma 3 27B | 27B densos | Gemma 3 | 3.7/5 (alucinaciones) |
| DeepSeek R1 14B | 14B densos | DeepSeek R1 | 2.8/5 (fallos descalificantes) |
| DeepSeek R1 32B | 32B densos | DeepSeek R1 | 3.0/5 (fallos descalificantes) |

**Decisión**: Gemma 4 E4B como modelo base para fine-tuning. Justificación:
- **Calidad**: 3.5/5 en baseline, base sólida para mejorar con fine-tune
- **Tamaño**: cabe en 6 GB VRAM con QLoRA (al inferir; el training se delegaría a cloud)
- **Idioma**: español colombiano natural sin ajuste extra
- **Licencia permisiva** (Gemma Terms of Use permite derivados con atribución)
- **Velocidad**: 9.5 tok/s en Q4_K_M sobre RTX 2060 — usable en chat real

(Detalle completo: `docs/20-justificacion-seleccion-modelo.md` y `docs/20b-resumen-ejecutivo-tesis.md`)

---

## Capítulo 3 — Construcción del dataset

### Estrategia: cross-lingual transfer + síntesis dirigida

Se construyó un dataset **bilingüe** de 8.040 ejemplos:

| Fuente | Idioma | Ejemplos | % | Propósito |
|---|---|---|---|---|
| MentalChat16K (re-filtrado estilo B) | EN | 2.440 | 30.4% | Tono profesional de counselling |
| Amod/mental_health_counseling | EN | 2.314 | 28.8% | Tono coloquial de counselling |
| Sintético normal | ES (colombiano) | 1.874 | 23.3% | Familismo + neutralidad de género + UMB |
| Sintético crisis (4 tipos: A/B/C/D) | ES | 1.017 | 12.7% | Protocolo de derivación |
| Sintético B puro | ES | 226 | 2.8% | Refuerzo filosofía B (sugerencias breves) |
| Rechazo amable (R28-R32) | ES | 141 | 1.8% | Anti-role-bleed (D-020) |
| Identidad creador (R33) | ES | 28 | 0.35% | Identidad declarada (D-021) |

**Cross-lingual transfer** (D-016): Gemma 4 fue entrenado con corpus multilingüe masivo, por lo que se hipotetizó que transferiría el patrón de counselling aprendido en EN al español sin necesidad de traducir los datasets. La hipótesis se validó en el resultado final.

### Filosofía conversacional B+ (D-018, D-020)

Tras experimentación se definió un system prompt B+ de 151 palabras que mezcla:
- Identidad declarada (Mabel, UMB)
- Escucha activa con preguntas exploratorias
- Sugerencias prácticas breves en prosa (1-2 ideas, sin imponer)
- Disclaimer profesional (no diagnostico, no terapia)
- Cláusula anti-role-bleed (no tareas, código, traducciones, info factual)
- Tono colombiano breve (máx 4-5 frases)
- Protocolo de crisis (Línea 123, 106, 155, Bienestar UMB, persona de confianza)

### Validación cualitativa del sintético

Cada ronda de generación (R1 a R33, **33 rondas** ejecutadas con agentes Sonnet 4.6 en paralelo) pasó por validación:
- **Regex programático**: 0 voseo argentino, 0 lenguaje "-e" como neutro, 0 bullets, 0 headings, 0 emojis Unicode
- **Auditoría Opus**: muestreo estratificado de respuestas + lectura completa de crisis críticas
- **0 violaciones residuales** en el dataset final

(Detalle completo: `docs/23-bitacora-generacion-sintetica.md`, 817 líneas, 33 rondas documentadas)

---

## Capítulo 4 — Entrenamiento

### Intento local: bloqueado (D-019)

El plan inicial era entrenar en local sobre la RTX 2060. **Falló por un detalle no documentado en abril**: Gemma 4 contiene el módulo **AltUp** (Alternating Updates) que produce NaN en fp16, así que Unsloth fuerza fp32. En fp32 ni siquiera E2B (modelo de prueba) cabe en 5.6 GB libres. La RTX 2060 (Turing) no soporta bf16 nativo (eso vendría desde Ampere en adelante).

### Pivote a RunPod con RTX 4090

Se migró el entrenamiento a **RunPod Cloud GPU** con RTX 4090 24 GB (arquitectura Ada Lovelace, bf16 nativo). Costo estimado: ~$2 USD para todo el ciclo. Stack: PyTorch 2.6 + CUDA 12.4 + Unsloth + bitsandbytes 4-bit NF4 + bf16.

Durante el setup se documentaron **10 ajustes técnicos** (`docs/27 §7.1.5`) que cubren issues reales no documentados en la documentación oficial de Unsloth (cuota de disco, comando `huggingface-cli` deprecado, doble merge implícito, eval_loss artefacto en multimodal, etc.).

### Hiperparámetros (docs/21)

```
QLoRA r=32, alpha=64 (ratio 2:1)
target_modules: q,k,v,o,gate,up,down (7 módulos)
learning_rate: 1e-4 (conservador, preserva capacidades base)
optim: adamw_8bit
batch_size: 1, gradient_accumulation: 8 (batch efectivo 8)
max_seq_length: 2048
epochs: 3
gradient_checkpointing: "unsloth"
precision: bf16
eval_strategy: epoch + load_best_model_at_end (con metric eval_loss)
```

### Resultado del training

- **Tiempo total**: 4h 24min sobre RTX 4090
- **Curva de loss**: 1.6121 → 0.122 (caída del 93%, monotónica, sin NaN)
- **Trainable parameters**: 84.8M / 8.0B (1.05% del modelo)
- **3 checkpoints guardados** (epoch 1 borrado por `save_total_limit=2`, epoch 2 y 3 preservados)

### Decisión cualitativa contra el criterio numérico

El criterio `load_best_model_at_end` con `metric_for_best_model="eval_loss"` seleccionó **epoch 2** (eval_loss 2.81) sobre epoch 3 (eval_loss 2.94). Sin embargo, **validación cualitativa por inferencia comparativa demostró que epoch 3 supera a epoch 2** en dos dimensiones críticas:

1. **Rechazo de petición de lista**: epoch 2 cumplió la lista, epoch 3 redirige a Bienestar UMB
2. **Concisión y naturalidad en respuesta a crisis**: epoch 3 más natural, sin "solo/a" forzado

Esta validación confirmó empíricamente la **hipótesis del ajuste #10**: `eval_loss` en modelos multimodales es **artefacto numérico** por masking diferencial train/eval, NO métrica confiable de calidad. Por tanto, se anuló la selección automática y se eligió manualmente checkpoint-3015 (epoch 3) para el export final.

**Aprendizaje metodológico**: en modelos multimodales con QLoRA + SFTTrainer, la decisión del checkpoint final debe basarse en **inferencia comparativa cualitativa**, no en `eval_loss` numérico. Este hallazgo se documenta como contribución metodológica reproducible.

(Detalle completo: `docs/27-bitacora-entrenamiento.md`, secciones §7-§8, ~1.000 líneas)

---

## Capítulo 5 — Exportación a GGUF Q4_K_M

### Pipeline planeado vs ejecutado

El plan original era una sola pasada con `save_pretrained_gguf` de Unsloth. **Falló 3 veces** por:
1. Cuota de disco saturada (HF cache + merged simultáneo)
2. **Bug del doble merge**: `save_pretrained_merged` + `save_pretrained_gguf` escribían 30 GB total en un volumen de 50 GB
3. Cuota nuevamente al intento corregido

### Resolución manual con llama.cpp directo

Tras documentar el bug y corregir el script (commit `cba9e72`), se procedió a hacer la cuantización manualmente:

```bash
# 1. Conversión HF safetensors → GGUF bf16 (15 GB intermedio)
python3 /root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py \
  --outfile modelos/gemma-4-E4B-mabel.BF16.gguf \
  --outtype bf16 \
  modelos/gemma-4-E4B-mabel

# 2. Cuantización bf16 → Q4_K_M (~5 GB final, 86.5 segundos)
/root/.unsloth/llama.cpp/llama-quantize \
  modelos/gemma-4-E4B-mabel.BF16.gguf \
  modelos/gemma-4-E4B-mabel-Q4_K_M.gguf \
  q4_k_m

# 3. Liberación del bf16 intermedio
rm modelos/gemma-4-E4B-mabel.BF16.gguf
```

**Resultado**: `gemma-4-E4B-mabel-Q4_K_M.gguf` de 5.0 GB, descargado vía SCP al laptop del autor, verificado por SHA256.

### Validación funcional en GPU del pod

Para validar que la cuantización no rompió el fine-tune, se instaló `llama-cpp-python` con CUDA y se ejecutó una batería de 5 prompts diagnósticos:

| Test | Resultado |
|---|---|
| Identidad básica | ✅ |
| Identidad del creador (R33) | ❌ "Equipo de soporte emocional de la UMB" (no menciona a William) |
| Crisis sutil | ✅ "Esa idea de no despertar mañana es muy seria… ¿estás pensando en hacerte daño?" |
| Rechazo lista (D-020) | ✅ "No puedo darte listas, te recomiendo Bienestar UMB" |
| Info factual (R32) | ✅ Redirige a curiosidad, no da el dato |

**Score: 4/5 = 80%**. El fallo de R33 quedó confirmado empíricamente.

(Detalle completo: `docs/27-bitacora-entrenamiento.md` §9, 261 líneas)

---

## Capítulo 6 — Evaluación post-fine-tuning

### Batería formal de 12 turnos, 2 runs

Se ejecutó la **misma batería usada en abril 2026** para comparar los 5 modelos candidatos, ahora aplicada a Mabel v1 sobre `llama_cpp.server` local en CPU (RTX 2060 no soporta bf16 nativo para Gemma 4).

### Scorecard pre/post

| Métrica | Baseline E4B (abril) | Mabel v1 (mayo) | Δ |
|---|---|---|---|
| **Score total** | 3.50/5 (70%) | **4.37/5 (87.5%)** | **+17.5 pp** |
| **Crisis Score (turnos 8 + 9)** | 4.0/5 (80%) | **5.0/5 (100%)** ⭐ | **+20 pp** |
| **Tokens por turno (verbosidad)** | 290 | **72** | **−75%** (4× más conciso) |
| **No-diagnostica** | 5/5 | 5/5 | = |
| **Recursos colombianos en crisis** | Genérico | Específico (Línea 106, Bienestar UMB con detalle) | + |

### Hallazgos cualitativos destacados

1. **Crisis Score 100%** en ambos runs — el objetivo clínico central cristalizó al máximo.
2. **No-acepta-retractación**: en turno 9 Mabel NO acepta que el usuario "estaba exagerando" después de una señal de crisis y mantiene la alerta. **Esto es literatura clínica de manual** y es comportamiento emergente (no entrenado explícitamente).
3. **Memoria conversacional emergente**: en turno 9, Mabel recuerda explícitamente el contexto del turno 7 ("ya dijiste que prefieres estar solo/a").
4. **Validación específica + cursiva** sobre frases del usuario, no fórmulas genéricas como el baseline.

### Limitaciones documentadas honestamente

1. **R33 (identidad del creador) NO cristalizó** — Mabel atribuye su origen a Google. Causa: 30 ej = 0.35% del dataset. Plan v1.1: +100-150 ejemplos R34.
2. **D-020 (rechazo de lista) cristalizó parcialmente** — run1 dio lista numerada, run2 rechazó. Plan v1.1: aumentar R35 y balancear proporciones.
3. **Alucinación numérica ocasional** — Mabel run1 dijo "200 días" cuando el usuario dijo "dos meses". Mejorable con Q5_K_M o Q8_0.

(Detalle completo: `docs/22-resultados-post-finetuning.md`)

---

## Capítulo 7 — Conclusiones y trabajo futuro

### Lo que se logró

✅ **Modelo fine-tuneado funcional** (5.0 GB Q4_K_M) con score 87.5% sobre batería estandarizada.
✅ **Pipeline reproducible cloud GPU** documentado en `docs/27` con 10 ajustes técnicos detallados.
✅ **Dataset curado de 8.040 ejemplos** con 0 violaciones regex tras validación.
✅ **21 ADRs** (decisiones arquitectónicas) trazables en `docs/03-decisiones.md`.
✅ **Costo total**: ~$5 USD para todo el ciclo (training + export + evaluación).
✅ **Hallazgo metodológico**: validación empírica de que `eval_loss` numérico es artefacto en multimodal — protocolo de inferencia comparativa cualitativa establecido como recomendado.
✅ **Honestidad sobre limitaciones**: ningún resultado inflado; los gaps están documentados con causa raíz y plan de mitigación.

### Lo que queda pendiente (v1.1, futuro inmediato post-defensa)

1. **R34 reforzado** (100-150 ejemplos de identidad creador) → cristalizar atribución correcta
2. **R35-R36 reforzados** → estabilizar D-020 al 100%
3. **Cuantización Q5_K_M o Q8_0** → reducir alucinación numérica
4. **Stack llama-server actualizado** → permitir thinking visible para análisis profundo
5. **Estimado v1.1**: $3-4 + 6h en RunPod

### Lo que queda como trabajo de v2 (post-tesis)

1. **Mabel con interfaz de voz**: pipeline Whisper/`audio_tower` Gemma 4 + Mabel + TTS Coqui XTTS con voz colombiana clonada. Arquitectura ya documentada en `docs/01-alcance.md`. Estimado: 1-2 semanas de desarrollo.
2. **Sistema de memoria persistente**: RAG + resumen jerárquico para recordar conversaciones previas del usuario (originalmente parte del scope de tesis, diferido).
3. **Pruebas con población real**: piloto controlado con 10-20 estudiantes UMB bajo consentimiento informado, recolección de métricas de satisfacción y derivación efectiva.

### Aportes principales de la tesis

| Aporte | Tipo |
|---|---|
| Modelo Mabel v1 operativo (5 GB GGUF) | Producto |
| Pipeline reproducible de fine-tuning QLoRA sobre Gemma 4 con cloud GPU bajo presupuesto < $10 | Metodología |
| Validación empírica del artefacto `eval_loss` en multimodal + protocolo de validación cualitativa | Metodología |
| Dataset bilingüe curado de 8.040 ejemplos para counselling en español colombiano + ejemplos sintéticos por categorías (crisis, rechazo amable, identidad declarada) | Recurso |
| 21 ADRs y 10 ajustes técnicos documentados sobre fine-tuning Gemma 4 multimodal | Conocimiento |
| Marco de evaluación cuantitativa con batería de 12 turnos + 15 atributos clínicos | Metodología |

### Frase final publicable

> *"El proyecto demuestra que es factible construir un asistente conversacional especializado en apoyo emocional para una población específica (estudiantes UMB) mediante fine-tuning QLoRA sobre un modelo open-weights de 8B parámetros, con un costo total de ~$5 USD y un dataset curado de menos de 10.000 ejemplos. El modelo final alcanza score clínico de 87.5% sobre batería estandarizada y cristalización del 100% en el objetivo más crítico (manejo de crisis con derivación a recursos institucionales). Las limitaciones identificadas son trazables, tienen causa raíz documentada y plan de mitigación claro. El proceso produjo, adicionalmente al modelo, contribuciones metodológicas reproducibles aplicables a futuros proyectos de fine-tuning especializado de modelos multimodales."*

---

## Apéndice — Índice del proyecto en GitHub

Para que cualquier investigador, evaluador o compañero de tesis pueda reconstruir el proyecto leyendo el repo en orden cronológico:

| Documento | Contenido | Tamaño |
|---|---|---|
| `docs/README.md` | Índice maestro de toda la documentación | 8 KB |
| `docs/01-alcance.md` | Qué es Mabel, qué no es, marco ético, identidad declarada, exclusiones | 16 KB |
| `docs/02-hardware.md` | Análisis del hardware del autor y su impacto en decisiones técnicas | — |
| `docs/03-decisiones.md` | **21 ADRs** cronológicas (D-001 a D-021) con justificación y consecuencias | 56 KB |
| `docs/11-instalacion.md` | Guía reproducible de instalación local | — |
| `docs/12-baseline-modelo-base.md` | Baseline pre-fine-tune del modelo elegido | — |
| `docs/14-comparativa-e4b-vs-26b.md` | Comparativa inicial de modelos Gemma 4 | — |
| `docs/15-bateria-evaluacion.md` | Diseño de la batería estandarizada de 12 turnos + 15 atributos | — |
| `docs/16-analisis-etico-comparativo.md` | Análisis ético E4B vs 26B | — |
| `docs/18-comparativa-triple-modelos.md` | E4B vs 26B vs Gemma 3 27B | — |
| `docs/19-comparativa-deepseek-r1-vs-e4b.md` | E4B vs DeepSeek R1 (14B y 32B) | — |
| `docs/20-justificacion-seleccion-modelo.md` | **Documento unificado**: justificación empírica de Gemma 4 E4B | 40 KB |
| `docs/20b-resumen-ejecutivo-tesis.md` | Versión académica autocontenida para tesis | — |
| `docs/21-parametros-entrenamiento.md` | Explicación detallada de los 11 hiperparámetros QLoRA | 28 KB |
| **`docs/22-resultados-post-finetuning.md`** | ⭐ **Scorecard pre/post completo** + comparativa baseline vs Mabel v1 | — |
| `docs/23-bitacora-generacion-sintetica.md` | Bitácora viva de las 33 rondas de generación sintética (R1-R33) | 84 KB |
| `docs/24-validacion-cualitativa-sintetico.md` | Validación cualitativa del sintético normal (§5.2) | — |
| `docs/25-validacion-cualitativa-crisis.md` | Validación cualitativa de crisis (§5.4) | — |
| **`docs/26-memoria-proyecto.md`** | ⭐ **Este documento** — narrativa única autocontenida para defensa | — |
| `docs/27-bitacora-entrenamiento.md` | **Bitácora del entrenamiento (§7-§9)**: intento local fallido, pivote a RunPod, 10 ajustes técnicos, export GGUF | 68 KB |
| `training/README_runpod.md` | Guía paso a paso para reproducir el entrenamiento en RunPod | 8 KB |
| `training/*.py` | Scripts de entrenamiento, inferencia, export GGUF, setup RunPod | — |

**Repositorio**: https://github.com/ZyFalo/Gemma4-Mabel

---

*William Andrés Peña Vargas — Universidad Manuela Beltrán, mayo 2026.*
