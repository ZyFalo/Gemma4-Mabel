## 1. Entorno de entrenamiento

- [x] 1.1 Crear venv Python 3.12 en el directorio del proyecto (`python3 -m venv .venv`)
- [x] 1.2 Instalar Unsloth y dependencias (`pip install unsloth`) — verificar que trae transformers, peft, bitsandbytes, trl, accelerate, datasets
- [x] 1.3 Verificar CUDA disponible (`python -c "import torch; print(torch.cuda.is_available())"`) y GPU detectada (RTX 2060, 6 GB)
- [x] 1.4 Verificar carga de Unsloth (`python -c "from unsloth import FastLanguageModel; print('OK')"`)
- [ ] 1.5 Login en HuggingFace (`huggingface-cli login`) para acceso a modelos que requieran licencia

## 2. Descarga de datasets fuente

- [x] 2.1 Descargar MentalChat16K desde HuggingFace a `data/raw/mentalchat16k.json` y verificar conteo (~16.113 ejemplos)
- [x] 2.2 Descargar Amod/mental_health_counseling_conversations a `data/raw/amod.json` y verificar conteo (~3.500 ejemplos)
- [x] 2.3 Inspeccionar formato de ambos datasets (campos, estructura, idioma) para diseñar el prompt de traducción

## 3. Diseño de prompts (Opus)

- [x] 3.1 Diseñar prompt de traducción inglés→español para counselling (con 3 ejemplos de referencia)
- [x] 3.2 Diseñar prompt de generación sintética con contexto colombiano/UMB (con 3 ejemplos de referencia)
- [x] 3.3 Diseñar prompt de generación de crisis siguiendo protocolo clínico (con 3 ejemplos de referencia que incluyan los 5 objetivos del fine-tuning)
- [x] 3.4 Presentar los 3 prompts al usuario para aprobación

## ~~4. Traducción de datasets (agentes Sonnet)~~ — ELIMINADA (D-016)

> **NO APLICA**: Se decidió NO traducir los datasets al español. MentalChat16K y Amod se usan en inglés directamente, aprovechando el cross-lingual transfer de Gemma 4. Ver D-016 en `docs/03-decisiones.md`.

- [x] ~~4.1 Dividir MentalChat16K en chunks de ~500 ejemplos~~ — N/A
- [x] ~~4.2 Ejecutar traducción de MentalChat16K~~ — N/A
- [x] ~~4.3 Validar muestras aleatorias de cada chunk traducido~~ — N/A
- [x] ~~4.4 Dividir Amod en chunks de ~500~~ — N/A
- [x] ~~4.5 Ejecutar traducción de Amod~~ — N/A
- [x] ~~4.6 Validar muestras de Amod traducido~~ — N/A
- [x] ~~4.7 Consolidar traducciones aprobadas~~ — N/A

## 5. Generación de dataset sintético ampliado (agentes Sonnet + validación Opus)

> **ACTUALIZADO (D-016, D-017)**: Se fija en exactamente 3.000 ejemplos sintéticos en español colombiano (~2.000 normales + ~1.000 crisis/afterglow). Agentes Sonnet generan, Opus verifica. Estos ejemplos son la fuente principal de tono colombiano, familismo, neutralidad de género, guardrails e identidad de Mabel.

- [x] 5.1 Generar ~2.000 ejemplos sintéticos normales con agentes Sonnet — **1.991 ejemplos en 16 rondas (R1-R16), 64 archivos.** Ver `docs/23-bitacora-generacion-sintetica.md`.
- [x] 5.2 Validar muestras de sintético normal (Opus) — **27 conversaciones leídas en muestreo estratificado + 2 hallazgos corregidos (voseo en 8 archivos: 201 reemplazos, concordancia /a en 2 casos).** Ver `docs/24-validacion-cualitativa-sintetico.md`.
- [x] 5.3 Generar ~1.000 ejemplos de crisis con agentes Sonnet — **1.080 ejemplos en 9 rondas (R17-R25), 36 archivos.** Distribución: 270 precursores (Tipo A) + 270 crisis activas (Tipo B) + 270 afterglow (Tipo C) + 270 señales indirectas (Tipo D).
- [x] 5.4 Opus revisa TODOS los ejemplos de crisis y afterglow — **Auditoría híbrida (regex programático + lectura manual de fallos potenciales) en cada ronda + auditoría final integral.** 0 derivaciones indebidas Tipo A (270/270), 0 info de métodos Tipo D (270/270), 0 frases de abandono, 0 voseo (207 reemplazos aplicados), 0 lenguaje "-e". Ver `docs/25-validacion-cualitativa-crisis.md`.
- [x] 5.5 Consolidar sintético aprobado en `data/synthetic/synthetic_es.json` — **3.071 ejemplos consolidados** (1.991 normales + 1.080 crisis), 6.9 MB. Cada entrada preserva metadatos (`source`, `subset`, `severity`/`type`, `tema_principal`/`contexto_detonante`) además de `messages`. Distribución severity normales: 743 leve / 889 moderado / 359 severo. Distribución crisis: 270 por cada tipo A/B/C/D. 0 errores en consolidación.

## 6. Formateo y ensamblaje del dataset bilingüe (Opus directo)

> **ACTUALIZADO (D-016)**: Dataset bilingüe (inglés/español). MentalChat16K y Amod se formatean en inglés con system prompt en español. Sintéticos completamente en español colombiano.

- [ ] 6.1 Filtrar MentalChat16K a 5.000 ejemplos representativos y formatear a JSONL conversacional (system prompt de Mabel en español + user/assistant en inglés) — Opus valida cada bloque
- [ ] 6.2 Formatear Amod (3.512 ejemplos) a JSONL conversacional (system prompt de Mabel en español + user/assistant en inglés) — Opus valida
- [ ] 6.3 Formatear sintético (3.000 ejemplos, completamente en español colombiano) a JSONL conversacional — Opus valida
- [ ] 6.4 Ensamblar dataset final en `data/train.jsonl` con proporciones ~43/31/26 (MentalChat16K en inglés / Amod en inglés / sintético en español) y shuffle aleatorio
- [ ] 6.5 Verificar conteo total (~11.512), distribución de temas, presencia de los 5 objetivos, balance bilingüe, y ausencia de duplicados
- [ ] 6.6 Crear split de evaluación `data/eval.jsonl` (~500 ejemplos reservados, no usados en training, con representación de ambos idiomas)

## 7. Prototipo de entrenamiento (E2B)

- [ ] 7.1 Descargar `unsloth/gemma-4-E2B-it` en 4-bit
- [ ] 7.2 Ejecutar entrenamiento con 200 ejemplos de `data/train.jsonl`, 1 época, mismos parámetros de docs/21
- [ ] 7.3 Verificar: no hay OOM, training loss disminuye, modelo genera respuesta coherente en español
- [ ] 7.4 Si hay OOM: reducir context_length a 1024 o r a 16, documentar ajuste, reintentar

## 8. Entrenamiento real (E4B)

- [ ] 8.1 Descargar `unsloth/gemma-4-E4B-it` en 4-bit
- [ ] 8.2 Configurar entrenamiento con parámetros de docs/21 (r=32, alpha=64, lr=1e-4, 3 épocas, fp16, adamw_8bit, gradient_checkpointing="unsloth")
- [ ] 8.3 Ejecutar entrenamiento con dataset completo (~11.512 ejemplos, 3 épocas)
- [ ] 8.4 Monitorizar VRAM con `nvidia-smi` y temperatura con `sensors` durante el entrenamiento
- [ ] 8.5 Verificar guardado de checkpoints por época en `outputs/checkpoint-epoch-N/`
- [ ] 8.6 Verificar convergencia de training loss en TensorBoard
- [ ] 8.7 Guardar adapters LoRA finales en `outputs/mabel-lora-adapter/`

## 9. Exportación del modelo

- [ ] 9.1 Merge de adapters LoRA con modelo base: `model.save_pretrained_merged("outputs/merged", tokenizer)`
- [ ] 9.2 Exportar a GGUF Q4_K_M: `model.save_pretrained_gguf("modelos/gemma-4-E4B-mabel-Q4_K_M", tokenizer, quantization_method="q4_k_m")`
- [ ] 9.3 Verificar tamaño del GGUF (~4.7 GB esperado)
- [ ] 9.4 Arrancar llama-server con el GGUF fine-tuneado y verificar `/health` OK
- [ ] 9.5 Enviar prompt de prueba vía curl y verificar que responde como Mabel (español, tono, identidad)
- [ ] 9.6 Verificar que chat.py funciona como drop-in (detecta el modelo, header correcto, streaming OK)

## 10. Evaluación post-fine-tuning

- [ ] 10.1 Ejecutar `python3 eval/run_battery.py E4B_finetuned_run1` (12 turnos, thinking activo)
- [ ] 10.2 Ejecutar `python3 eval/run_battery.py E4B_finetuned_run2` (segunda ejecución para varianza)
- [ ] 10.3 Leer resultados completos de ambos runs
- [ ] 10.4 Puntuar los 15 criterios del scorecard para cada run
- [ ] 10.5 Comparar scorecard pre/post fine-tuning — verificar mejora en los 5 objetivos
- [ ] 10.6 Comparar con gold standard (26B MoE baseline) — verificar reducción de brecha
- [ ] 10.7 Identificar regresiones (criterios que empeoraron)
- [ ] 10.8 Generar documento `docs/22-resultados-post-finetuning.md` con análisis completo
- [ ] 10.9 Actualizar `docs/README.md` con el nuevo documento
