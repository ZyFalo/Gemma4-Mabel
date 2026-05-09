## Why

El modelo base Gemma 4 E4B sin fine-tuning presenta 5 fallos concretos documentados empíricamente (docs/20) que lo hacen éticamente insuficiente para despliegue con estudiantes: (1) sesgo de género (2-6 asunciones femeninas por sesión), (2) sobre-activación del protocolo de crisis ante precursores, (3) manejo inadecuado del afterglow post-crisis, (4) omisión de la pregunta por persona de confianza en situaciones de crisis, y (5) varianza excesiva entre ejecuciones (el modelo produce respuestas clínicamente correctas o dañinas dependiendo del seed aleatorio, lo cual es inaceptable para usuarios vulnerables). Estos fallos son corregibles con entrenamiento supervisado (QLoRA) usando un dataset curado que combine datos clínicos traducidos, datos sintéticos con contexto colombiano, y ejemplos de crisis revisados por profesionales. El fine-tuning es el aporte técnico central de la tesis y debe ejecutarse antes de la arquitectura de memoria (RAG) según la decisión D-014.

## What Changes

- Se instala un entorno Python aislado (venv) con Unsloth y dependencias para entrenamiento.
- Se descarga, filtra y prepara un dataset bilingue (inglés/español) de ~11.512 ejemplos en formato JSONL conversacional a partir de 3 fuentes: MentalChat16K filtrado a 5.000 ejemplos (en inglés), Amod/mental_health_counseling_conversations con 3.512 ejemplos (en inglés), y 3.000 ejemplos sintéticos generados con agentes Sonnet en español colombiano (verificados por Opus), distribuidos en ~2.000 normales + ~1.000 crisis/afterglow. No se traducen los datasets en inglés: Gemma 4 es multilingüe (140+ idiomas) y transfiere patrones de counselling del inglés al español automáticamente (cross-lingual transfer). Los sintéticos en español enseñan tono colombiano, familismo, neutralidad de género, guardrails e identidad de Mabel. El system prompt en español en cada ejemplo ancla el idioma de salida.
- Se ejecuta fine-tuning con QLoRA sobre `unsloth/gemma-4-E4B-it` usando los parámetros documentados en docs/21 (r=32, lora_alpha=64, 3 épocas, lr=1e-4, fp16, adamw_8bit, gradient checkpointing "unsloth").
- Se prototipa primero con E2B (~1h) para validar el pipeline antes del entrenamiento real con E4B (~4-8h).
- Se exportan los adapters LoRA, se mergean con el modelo base, y se exporta a formato GGUF (Q4_K_M) para inferencia con llama.cpp.
- Se re-ejecuta la batería de 12 turnos (docs/15) sobre el modelo fine-tuneado y se compara el scorecard pre/post fine-tuning.

## Capabilities

### New Capabilities

- `dataset-preparation`: Pipeline de descarga, filtrado de MentalChat16K (5.000 ejemplos), generación sintética en español colombiano (3.000 ejemplos con agentes Sonnet, verificados por Opus), formateo JSONL (Opus directo), y ensamblaje del dataset bilingüe con proporciones ~43% MentalChat16K (inglés) + ~31% Amod (inglés) + ~26% sintético (español) (~11.512 ejemplos totales). Sin traducción: se aprovecha el cross-lingual transfer de Gemma 4.
- `qlora-training`: Entorno de entrenamiento con Unsloth + QLoRA, prototipo de validación con E2B, entrenamiento completo con E4B, monitorización con TensorBoard, y gestión de checkpoints.
- `model-export`: Merge de adapters LoRA con modelo base, exportación a GGUF (Q4_K_M), verificación de carga en llama-server, y validación de inferencia post-export.
- `post-training-eval`: Re-ejecución de la batería de 12 turnos sobre el modelo fine-tuneado (2 runs), puntuación con el scorecard de 15 criterios, comparación pre/post, documentación de mejoras y regresiones.

### Decisión de formato: Markdown ligero

Durante la preparación de los prompts de generación de datos, se decidió permitir **Markdown ligero** (negrita y cursiva para énfasis emocional) en las respuestas de Mabel, en lugar de la restricción original "sin Markdown". Esto se debe a que el frontend del equipo paralelo renderiza Markdown, lo que permite que negritas como **"tu seguridad es lo más importante"** se estilicen visualmente. Se prohíben headings (`###`), listas con bullets, y emojis, que rompen el tono conversacional. Este cambio se refleja en el spec de `dataset-preparation`, en los prompts de generación, y en el system prompt del dataset de entrenamiento.

### Modified Capabilities

(Ninguna capacidad existente se modifica — todas son nuevas.)

## Impact

- **Archivos nuevos**: venv Python (~5 GB), dataset JSONL (~20-40 MB), adapters LoRA (~200 MB), modelo GGUF fine-tuneado (~4.7 GB), checkpoints intermedios (~2-5 GB).
- **Disco necesario**: ~13 GB adicionales sobre los 21 GB actuales (modelos). Con 54 GB libres, hay margen amplio. El dataset es más pequeño (~11.5K vs ~21K) al eliminar la traducción.
- **VRAM (GPU)**: se usará la RTX 2060 Mobile (6 GB) exclusivamente durante el entrenamiento (~4-8 horas). Después de exportar, la inferencia vuelve a CPU/RAM.
- **Dependencias nuevas**: Unsloth, transformers, peft, bitsandbytes, trl, accelerate, datasets, sentencepiece (instaladas vía pip en venv).
- **Herramientas existentes afectadas**: `chat.py` no cambia — el GGUF fine-tuneado se carga en llama-server igual que el base. `eval/run_battery.py` se reutiliza sin modificación.
- **API/endpoints**: sin cambios — el modelo fine-tuneado se sirve por el mismo endpoint OpenAI-compatible.
- **Documentación**: se generarán docs/22+ con el proceso de entrenamiento y resultados post-fine-tuning.
