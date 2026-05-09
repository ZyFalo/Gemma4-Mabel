## Context

El modelo Gemma 4 E4B instruction-tuned es el modelo base seleccionado para Mabel (docs/20, score 3.93/5). La evaluación empírica con batería de 12 turnos identificó 5 fallos corregibles por fine-tuning: sesgo de género, sobre-activación de crisis ante precursores, manejo inadecuado del afterglow, omisión de pregunta por persona de confianza, y varianza excesiva entre ejecuciones.

El hardware disponible es un laptop con RTX 2060 Mobile (6 GB VRAM), Intel i7-10750H, 31 GB RAM. El entrenamiento se hace con QLoRA vía Unsloth para caber en VRAM. La inferencia post-fine-tuning vuelve a CPU/RAM vía llama-server.

El dataset se construye con estrategia multiagente (D-015) y bilingüe sin traducción (D-016): agentes Sonnet 4.6 generan ejemplos sintéticos ampliados en español colombiano, Opus valida y formatea el JSONL final. MentalChat16K y Amod se usan en inglés directamente.

## Goals / Non-Goals

**Goals:**
- Instalar entorno de entrenamiento reproducible (venv + Unsloth + deps)
- Preparar dataset bilingüe de ~11.512 ejemplos en JSONL conversacional (MentalChat16K ~43% en inglés + Amod ~31% en inglés + sintético ~26% en español colombiano)
- Validar pipeline con prototipo E2B (200 ejemplos, ~1h)
- Entrenar E4B con QLoRA (3 épocas, ~4-8h, docs/21)
- Exportar a GGUF Q4_K_M para inferencia con llama-server
- Re-evaluar con batería de 12 turnos y demostrar mejora medible en scorecard
- Documentar todo en docs/ para trazabilidad de la tesis

**Non-Goals:**
- Arquitectura de memoria RAG (pospuesta a fase posterior, D-014)
- Entrenamiento de modelos > E4B (no caben en 6 GB VRAM)
- DPO/RLHF (solo SFT en esta iteración; DPO sería una mejora futura)
- Despliegue con estudiantes reales (requiere revisión clínica previa)
- Modificación de chat.py o llama-server (el GGUF fine-tuneado es drop-in)

## Decisions

### D1: Framework de entrenamiento — Unsloth sobre stack HF vanilla

**Decisión**: Unsloth (D-004).

**Alternativas consideradas**:
- **Stack HF estándar (transformers + peft + trl)**: funcional pero requiere ~12 GB VRAM para E4B QLoRA → no cabe.
- **Axolotl**: declarativo, pero capa adicional de abstracción sin beneficio de VRAM.
- **Unsloth**: 50-80% menos VRAM, 2-5× más rápido, soporta Turing, day-0 Gemma 4.

**Rationale**: Unsloth es la única vía que permite E4B QLoRA en 6 GB VRAM.

### D2: Formato del dataset — JSONL conversacional con chat template

**Decisión**: cada ejemplo es un JSON con array `messages` (system + user + assistant), compatible con el formato de `SFTTrainer` de trl.

**Formato**:
```json
{"messages": [
  {"role": "system", "content": "Te llamas Mabel..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**Alternativas consideradas**:
- **Formato instrucción (prompt/completion)**: más simple pero no captura conversaciones multi-turno.
- **Formato ShareGPT**: popular pero requiere conversión adicional.
- **JSONL conversacional**: nativo para SFTTrainer, soporta multi-turno, incluye system prompt.

**Rationale**: el formato conversacional permite que el system prompt de Mabel se incluya en cada ejemplo, consolidando la identidad y las reglas en los pesos del modelo.

### ~~D3: Estrategia de traducción — Multiagente Sonnet con validación Opus~~ → REEMPLAZADA por D3b

### D3b: Estrategia bilingüe sin traducción — Cross-lingual transfer + sintéticos ampliados

**Decisión**: NO se traducen MentalChat16K ni Amod al español. Se usan en inglés directamente. Los ejemplos sintéticos en español colombiano se fijan en exactamente 3.000 para compensar.

**Alternativas consideradas**:
- **Traducir todo al español (plan original D3)**: calidad variable, días de trabajo, miles de tokens de traducción. Riesgo de perder matices clínicos en la traducción.
- **No traducir y mantener sintéticos en ~1.500**: insuficiente representación del español colombiano en el dataset.
- **No traducir y fijar sintéticos en 3.000**: cada ejemplo sintético está diseñado para el caso de uso exacto (tono colombiano, familismo, neutralidad de género, crisis/guardrails, identidad de Mabel). Mayor calidad por ejemplo. Agentes Sonnet generan, Opus verifica.

**Rationale**: Gemma 4 es multilingüe (140+ idiomas) y aprende patrones de counselling del inglés, transfiriéndolos al español automáticamente (cross-lingual transfer). Los sintéticos en español enseñan lo que el inglés no puede: tono colombiano, familismo, expresiones locales, recursos de crisis colombianos, y la identidad de Mabel. El system prompt en español en cada ejemplo ancla el idioma de salida. Esto ahorra días de trabajo y tokens de traducción, y produce un dataset de mayor calidad.

### D4: Prototipo con E2B antes de E4B

**Decisión**: primer entrenamiento con Gemma 4 E2B (~2B params) sobre 200 ejemplos para validar que el pipeline funciona end-to-end.

**Rationale**: detectar problemas (incompatibilidades de versión, OOM, formato incorrecto) en ~1h con E2B es más barato que descubrirlos tras 4h de E4B. Si E2B funciona, E4B funciona (misma familia, mismos parámetros, solo más grande).

### D5: Exportación a GGUF vía Unsloth

**Decisión**: usar `model.save_pretrained_gguf()` de Unsloth para merge + cuantización en un solo paso.

**Alternativas consideradas**:
- **llama.cpp convert.py**: requiere merge manual previo + conversión separada.
- **Unsloth save_pretrained_gguf()**: hace merge + cuantización directo, soporta Q4_K_M.

**Rationale**: menos pasos, menos errores, mismo resultado. El GGUF resultante es idéntico al que produce llama.cpp.

## Risks / Trade-offs

**[VRAM al límite (5.6/6.0 GB)]** → Monitorizar con `nvidia-smi` durante entrenamiento. Si hay OOM: reducir context_length a 1024 o reducir r a 16. El prototipo E2B detectará esto antes.

**[Overfitting por dataset más pequeño (~11.5K)]** → 3 épocas máximo + lr=1e-4 conservador. Monitorizar training loss vs eval loss en TensorBoard. Si divergen, detener antes. Considerar reducir a 2 épocas si hay overfitting.

**[Balance bilingüe del dataset]** → Verificar que el system prompt en español en cada ejemplo (incluyendo los de inglés) ancle efectivamente el idioma de salida. Si el modelo responde en inglés, aumentar proporción de sintéticos en español.

**[Ejemplos de crisis incorrectos]** → Opus revisa TODOS los ejemplos de crisis (no muestras). Idealmente, revisión adicional por psicólogo de Bienestar UMB antes de incluirlos en training.

**[Regresión post-fine-tuning]** → Posible que el modelo mejore en los 5 objetivos pero empeore en otros aspectos (fluencia general, creatividad). La batería de 12 turnos lo detectará.

**[Thermal throttling del laptop]** → Entrenamiento de 4-8h con GPU al 100% genera mucho calor. Usar base refrigerada, enchufar, pausar entre épocas si hay throttling.

**[Incompatibilidad Unsloth + Gemma 4]** → Unsloth declaró soporte day-0 pero pueden haber bugs. El prototipo E2B lo detectará antes de invertir 8h en E4B.
