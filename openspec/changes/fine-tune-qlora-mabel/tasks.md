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

- [x] 6.1 Filtrado MentalChat16K → 5.000 (12 abr) + **re-filtrado Estilo B** post D-018 (criterio: 300-1.200 chars sin bullets/headings) → **2.657 ej en `data/raw/mentalchat_filtered_B.json`**.
- [x] 6.1b Formatear MentalChat re-filtrado a JSONL conversacional con system Mabel B → **`data/formatted/mentalchat_b.jsonl` con 2.653 ej** (4 descartados por input/output vacío).
- [x] 6.2 Formatear Amod (3.512 ej) a JSONL conversacional con system Mabel B → **`data/formatted/amod.jsonl` con 3.508 ej** (4 descartados).
- [x] 6.3 Aplanar sintético (3.311 ej) a JSONL → **`data/formatted/sintetico_es.jsonl` con 3.311 ej** (100% válidos, 100% system B).
- [x] 6.4 Ensamblar `data/train.jsonl` con shuffle aleatorio (seed=42).
- [x] 6.5 Verificar: 0 duplicados exactos (1.102 eliminados post-dedup), estructura correcta, system B en 100%.
- [x] 6.6 `data/eval.jsonl` con 500 ej estratificados (proporciones por fuente y idioma).

**Resultado final §6 (actualizado post D-020):**
- `data/train.jsonl`: **8.012 ej, 17.2 MB** (59.3% EN / 40.7% ES)
- `data/eval.jsonl`: **499 ej, 1.0 MB** (estratificado)
- Distribución train por fuente: mentalchat_b 30.5% / amod 28.9% / normal 23.4% / crisis 12.7% / normal_b 2.8% / **rechazo 1.8%**
- 100% con system prompt B+ (ver D-020)
- 0 voseo, 0 "-e", 0 bullets/headings/emojis en respuestas de Mabel

## 5b. Refuerzo anti-role-bleed (D-020) — POST §6

- [x] 5b.1 Diseñar system prompt B+ con cláusula sobre tareas fuera de scope (16 palabras añadidas a B)
- [x] 5b.2 Generar 150 ejemplos sintéticos de rechazo amable en 5 rondas (R28-R32) — STEM 30, humanidades 30, código 30, profesionales 30, jailbreaks+factual 30
- [x] 5b.3 Migrar system B → B+ en synthetic_es.json, mentalchat_b.jsonl, amod.jsonl, sintetico_es.jsonl
- [x] 5b.4 Re-ensamblar train.jsonl + eval.jsonl con los 150 nuevos
- [x] 5b.5 Limpieza: filtrar 9 ej Amod con bullets numerados + corregir 1 voseo aislado
- [x] 5b.6 Re-generar train_subset200.jsonl desde nuevo train (incluye 5 rechazo)
- [x] 5b.7 Documentar D-020 + actualizar docs/01-alcance.md (sección "Qué tareas Mabel rechaza")

## 7. Prototipo de entrenamiento (E2B) — RUNPOD

> **Pivote D-019**: ejecución cloud en RunPod RTX 4090 ($0.34/h). Local descartado por AltUp+Turing+6GB.

- [ ] 7.1 Cuenta RunPod + recargar $5 USD + HF token + aceptar licencia Gemma
- [ ] 7.2 Deploy Pod: RTX 4090 Community + template PyTorch 2.4 + 50 GB volume
- [ ] 7.3 Clonar repo + ejecutar `bash training/runpod_setup.sh`
- [ ] 7.4 Lanzar `python3 training/train_prototype_e2b.py` (200 ej × 1 ep, ~15 min, ~$0.09)
- [ ] 7.5 Verificar: no OOM, training loss decrece, adapter guardado en outputs/prototype_e2b/adapter
- [ ] 7.6 Sanity check: `python3 training/test_inference.py --model e2b` — verificar identidad Mabel + crisis + no-lista + **rechazo de petición fuera de scope (nueva)**

## 8. Entrenamiento real (E4B) — RUNPOD

- [ ] 8.1 Lanzar `nohup python3 training/train_real_e4b.py &` (8.012 ej × 3 ep, ~4 h, ~$1.36)
- [ ] 8.2 Verificar bf16=True activo y `unsloth/gemma-4-E4B-it` cargado sin OOM
- [ ] 8.3 Monitorizar VRAM con `nvidia-smi` y training loss en `outputs/real_e4b/run.log`
- [ ] 8.4 Verificar guardado de checkpoints por época en `outputs/real_e4b/checkpoint-epoch-N/`
- [ ] 8.5 Confirmar `load_best_model_at_end=True` aplicó (mensaje en log + eval_loss por época)
- [ ] 8.6 Adapter final en `outputs/real_e4b/adapter/`

## 9. Exportación del modelo — RUNPOD

- [ ] 9.1 Ejecutar `python3 training/export_gguf.py` (~30 min, ~$0.17)
- [ ] 9.2 Verificar tamaño del GGUF (~4.7 GB esperado) en `modelos/gemma-4-E4B-mabel-Q4_K_M.gguf`
- [ ] 9.3 Descargar GGUF + adapter + logs al laptop local vía SCP
- [ ] 9.4 **STOP Pod en RunPod** (importante para no acumular costo)
- [ ] 9.5 (Local) Arrancar llama-server con el GGUF fine-tuneado y verificar `/health` OK
- [ ] 9.6 (Local) Enviar prompt de prueba vía curl y verificar identidad Mabel
- [ ] 9.7 (Local) Verificar que chat.py funciona como drop-in

## 10. Evaluación post-fine-tuning

- [ ] 10.1 Ejecutar `python3 eval/run_battery.py E4B_finetuned_run1` (12 turnos, thinking activo)
- [ ] 10.2 Ejecutar `python3 eval/run_battery.py E4B_finetuned_run2` (segunda ejecución para varianza)
- [ ] 10.3 **Añadir batería específica de role-bleed** (5-8 turnos: pide código, ensayo, jailbreak, info factual, decisión vital) y ejecutar
- [ ] 10.4 Leer resultados completos de los 3 runs
- [ ] 10.5 Puntuar los 15 criterios del scorecard + nuevo criterio "robustez anti-role-bleed"
- [ ] 10.6 Comparar scorecard pre/post fine-tuning — verificar mejora en los 5 objetivos
- [ ] 10.7 Comparar con gold standard (26B MoE baseline) — verificar reducción de brecha
- [ ] 10.8 Identificar regresiones (criterios que empeoraron)
- [ ] 10.9 Generar documento `docs/22-resultados-post-finetuning.md` con análisis completo
- [ ] 10.10 Actualizar `docs/README.md` con el nuevo documento
