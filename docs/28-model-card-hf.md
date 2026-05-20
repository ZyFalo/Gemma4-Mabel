---
license: gemma
language:
  - es
  - en
base_model: google/gemma-4-E4B-it
tags:
  - gemma
  - gemma-4
  - qlora
  - fine-tuned
  - counselling
  - emotional-support
  - spanish
  - colombia
  - llama-cpp
  - gguf
library_name: transformers
pipeline_tag: text-generation
---

# Mabel — Asistente de Apoyo Emocional (Gemma 4 E4B fine-tune)

**Mabel v1** es un fine-tune QLoRA de `google/gemma-4-E4B-it` orientado al acompañamiento emocional de estudiantes universitarios colombianos (20-26 años).

Proyecto de tesis — **Universidad Manuela Beltrán (Colombia)** — autor: William Andrés Peña Vargas (`@ZyFalo`).

> ⚠️ **No es un reemplazo de atención psicológica profesional.** Su función es acompañar, validar emocionalmente y derivar a recursos reales cuando sea necesario.

## Detalles del modelo

| Campo | Valor |
|---|---|
| **Modelo base** | `google/gemma-4-E4B-it` (Gemma 4 multimodal, 4B params activos) |
| **Método** | QLoRA (4-bit NF4) con Unsloth |
| **Hiperparámetros LoRA** | r=32, alpha=64, dropout=0, target_modules=all |
| **Optimizador** | adamw_8bit, lr=1e-4, weight_decay=0.01 |
| **Schedule** | cosine, warmup_ratio=0.03 |
| **Épocas** | 3 (checkpoint final: epoch 3, `checkpoint-3015`) |
| **Precision** | bf16 (entrenamiento), Q4_K_M (despliegue) |
| **Hardware** | RunPod RTX 4090 24GB, ~4h 24min, ~$5 USD total |
| **Loss final** | train 0.12 (eval_loss alto = artefacto multimodal, validado) |
| **Cuantización** | Q4_K_M GGUF (5.0 GB) vía llama.cpp |

## Arquitectura

Datos extraídos del [model card oficial de Google](https://ai.google.dev/gemma/docs/core/model_card_4) y verificados contra el metadata binario del GGUF (`general.architecture = gemma4`).

| Parámetro | Valor |
|---|---|
| **Familia** | Gemma 4 (edge/multimodal) — sucesora de Gemma 3n |
| **Parámetros efectivos** | **4 B** (los que se activan en cada paso de inferencia) |
| **Parámetros totales (físicos en disco)** | **7.52 B** (incluyen `per_layer_token_embd` de la arquitectura PLE) |
| **Arquitectura especial** | **Per-Layer Embeddings (PLE)** — embeddings extra por capa que enriquecen representaciones sin penalizar latencia |
| **Context window nativo** | **128 K tokens (131 072)** — coincide con E2B; los modelos grandes (26B A4B, 31B) soportan 256K |
| **Capas (block_count)** | 42 |
| **Embedding dim** | 2 560 |
| **Feed-forward dim** | 10 240 |
| **Attention heads** | 8 |
| **KV heads (GQA)** | 2 (ratio 4:1 — Grouped Query Attention para eficiencia) |
| **Sliding window** | 512 tokens (patrón mixed local/global attention) |
| **RoPE base frequency** | 1 000 000 (alta, optimizada para contexto largo) |
| **Vocabulario** | 262 144 tokens |
| **Modalidades** | texto + visión + audio (en este checkpoint sólo se fine-tunearon los pesos de texto) |

### Uso efectivo del contexto en inferencia

El modelo **soporta** 128 K nativos en arquitectura, pero el contexto utilizable depende del flag `--n_ctx` del servidor y de la memoria disponible:

| `n_ctx` | KV cache aproximada | Hardware mínimo razonable |
|---|---|---|
| 4 096 (default `llama_cpp.server`) | ~200 MB | CPU 4 GB RAM |
| 8 192 (recomendado para sesiones de counselling) | ~400 MB | CPU 6 GB RAM |
| 32 768 | ~1.6 GB | GPU 6 GB |
| **131 072 (máximo nativo)** | **~6 GB sólo de KV cache** | GPU 12 GB+ |

Para uso típico de Mabel (sesión emocional de 20-40 turnos cortos), **`n_ctx 8192` es suficiente y eficiente**.

## Dataset

**Total**: 8.040 ejemplos de entrenamiento + 500 de evaluación. Todos con el **system prompt B+** unificado (pieza crítica de la activación del LoRA).

### Distribución exacta del train (medida sobre `data/train.jsonl`)

| Fuente | Ejemplos | % | Idioma | Tipo |
|---|---:|---:|---|---|
| `mentalchat_b` | 2 440 | 30.3% | EN | Counselling real (re-formateado a estilo B+) |
| `amod` | 2 314 | 28.8% | EN | Counselling real (re-formateado a estilo B+) |
| `normal` | 1 874 | 23.3% | ES | Sintético colombiano (conversación general de apoyo) |
| `crisis` | 1 017 | 12.6% | ES | Sintético — protocolos A/B/C/D (suicidio, autolesión, ideación, retractación) |
| `normal_b` | 226 | 2.8% | ES | Sintético — variantes de estilo B+ adicionales |
| `rechazo` | 141 | 1.8% | ES | Sintético — R28-R32 (rechazo amable a STEM, código, jailbreaks, factual) |
| `identidad_creador` | 28 | 0.3% | ES | Sintético — R33 (proyecto de tesis UMB) |
| **Total** | **8 040** | **100%** | **60% EN / 40% ES** | |

### Fuentes EN (counselling real, NO traducido)

| Dataset | Origen | Uso |
|---|---|---|
| **MentalChat16K** | `mental_chat_16k` (HuggingFace) | Re-formateado a estructura conversacional B+ → 2 440 ej |
| **Amod / mental_health_counseling_conversations** | [`Amod/mental_health_counseling_conversations`](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) | Convertido a JSONL con system Mabel → 2 314 ej |

> **Decisión clave (D-016)**: NO se tradujeron los datasets EN al español. Gemma 4 es multilingüe (140+ idiomas) y aprende los patrones de counselling del inglés transfiriéndolos automáticamente al español durante inferencia (**cross-lingual transfer** validado empíricamente).

### Fuente ES (sintético propio, generado y validado)

3 286 ejemplos (40% del train) generados durante 33 rondas iterativas documentadas en [`docs/23-bitacora-generacion-sintetica.md`](https://github.com/ZyFalo/Gemma4-Mabel/blob/main/docs/23-bitacora-generacion-sintetica.md).

**Pipeline de generación**:

- **Orquestador**: Claude Opus 4.7 — diseñaba el prompt de cada ronda, lanzaba agentes en paralelo, post-procesaba outputs
- **Generadores**: 2-4 agentes Claude Sonnet 4.6 en paralelo por ronda — escribían JSONL con metodología `Write` directa
- **Extracción tolerante**: `json.JSONDecoder().raw_decode()` con recuperación de ~87.5% incluso en outputs truncados
- **Total**: 33 rondas (R1-R33) cubriendo conversación normal, crisis A/B/C/D, rechazo, identidad

**Pipeline de validación y limpieza** (también orquestado por Opus):

1. **Regex automática**: barrido por patrones prohibidos (diagnóstico, bullets, headings, emojis, sermones, "-e" final indebido) → **0 violaciones** al cierre
2. **Validación cualitativa estratificada** ([`docs/24`](https://github.com/ZyFalo/Gemma4-Mabel/blob/main/docs/24-validacion-cualitativa-sintetico.md)): lectura completa de 27 conversaciones representativas (3 por cada bucket de tema/complejidad)
3. **Validación específica de crisis** ([`docs/25`](https://github.com/ZyFalo/Gemma4-Mabel/blob/main/docs/25-validacion-cualitativa-crisis.md)): auditoría híbrida (regex + lectura completa) de los 4 tipos A/B/C/D para garantizar derivación correcta (Línea 123, Línea 106, Línea 155, Bienestar UMB)
4. **Re-formateo iterativo**: cuando una ronda no pasaba validación, se relanzaba con agente Sonnet correctivo de "reescritura quirúrgica" (no regeneración completa)

### Eval set

500 ejemplos en `data/eval.jsonl`, estratificados con la misma proporción que el train, con holdout estricto (no overlap con train).

### Disclaimer académico sobre el sintético

El dataset sintético ES fue generado con LLMs comerciales (Claude Opus 4.7 + Sonnet 4.6) bajo licencia de uso académico no-comercial. **No se redistribuye públicamente** por estar sujeto a los términos de uso de los proveedores de los modelos generadores y por contener fraseo clínico que debe ser auditado por profesional de salud mental antes de cualquier despliegue masivo.

## Resultados (batería formal 12 turnos × 2 runs)

| Métrica | Baseline E4B | Mabel v1 | Δ |
|---|---|---|---|
| **Score global** | 3.50 / 5 (70%) | **4.37 / 5 (87.5%)** | +17.5 pp |
| **Crisis Score** (turnos 8-9) | 4 / 5 (80%) | **5 / 5 (100%)** | +20 pp |
| **Verbosidad** (tok/turno) | 290 | **72** | 4× más conciso |

## Limitaciones conocidas

- **R33 (identidad creador)**: cristalizó solo ~60%. Mabel a veces atribuye su creación a "Google" o al "equipo UMB" en lugar de William. → plan v1.1
- **D-020 (anti-listas)**: cristalización parcial; entre runs varía si responde lista o rechaza
- **Alucinación numérica ocasional**: en recursos UMB puede inventar líneas de atención (mitigable con RAG futuro)
- **Dependencia del system prompt**: requiere el B+ exacto para activación óptima (comportamiento estándar de LoRAs)

## Uso (llama-cpp-python)

```bash
pip install llama-cpp-python[server]
python -m llama_cpp.server \
  --model gemma-4-E4B-mabel-Q4_K_M.gguf \
  --n_ctx 8192 \
  --n_gpu_layers -1 \
  --chat_format gemma
```

### System prompt (CRÍTICO — no modificar)

El modelo fue entrenado con un **system prompt fijo de 151 palabras** ("B+"). Usarlo literal es lo que activa el comportamiento del LoRA — cambiarlo degrada notablemente la calidad de las respuestas (perdés activación de las reglas R28-R33 y del estilo conversacional breve). Está disponible como archivo aparte: [`system_prompt.txt`](./system_prompt.txt).

```
Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear. Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza.
```

### Parámetros de inferencia recomendados

| Parámetro | Valor |
|---|---|
| `temperature` | 0.7 |
| `max_tokens` | 1500 |
| `n_ctx` | 8192 (mínimo recomendado — sesiones cortas se quedan sin contexto con 4096) |
| `chat_format` | `gemma` |

### Ejemplo de llamada OpenAI-compatible

```python
import requests
SYSTEM = open("system_prompt.txt").read().strip()
r = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "gemma-4-E4B-mabel-Q4_K_M",
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Hola, me siento muy mal últimamente"},
    ],
    "temperature": 0.7,
    "max_tokens": 1500,
})
print(r.json()["choices"][0]["message"]["content"])
```

## Verificación

```
SHA256: 3d9ffb485a718d925915666b1151e25c0704bc6a1ca85ca77153d4e863237792
```

## Licencia y términos

- **Modelo base Gemma 4**: sujeto a [Gemma Terms of Use](https://ai.google.dev/gemma/terms) de Google. Este fine-tune hereda esa licencia.
- **Pesos derivados**: liberados públicamente para investigación académica (proyecto de tesis UMB). Cualquier uso debe respetar los Gemma Terms of Use heredados y el disclaimer clínico de esta tarjeta. No usar para diagnóstico clínico, atención a crisis sin supervisión profesional, ni despliegues comerciales sin auditoría psicológica previa.
- **Dataset sintético**: generación asistida con LLMs comerciales; uso académico no-comercial.

## Citar

```
@misc{pena_mabel_2026,
  author       = {Peña Vargas, William Andrés},
  title        = {Mabel: Asistente de Apoyo Emocional Fine-tuned de Gemma 4 E4B},
  year         = {2026},
  institution  = {Universidad Manuela Beltrán, Colombia},
  note         = {Tesis de pregrado},
  url          = {https://github.com/ZyFalo/Gemma4-Mabel}
}
```

## Repositorio de proyecto

Documentación completa, decisiones, bitácoras y código de entrenamiento:
https://github.com/ZyFalo/Gemma4-Mabel

Punto de entrada recomendado: [`docs/26-memoria-proyecto.md`](https://github.com/ZyFalo/Gemma4-Mabel/blob/main/docs/26-memoria-proyecto.md)
