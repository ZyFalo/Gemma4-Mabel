# 03 — Registro de decisiones técnicas

Este archivo registra cronológicamente las decisiones tomadas durante el proyecto, con justificación. Cada decisión importante se numera y queda congelada. Si una decisión cambia más adelante, se crea una nueva entrada que referencia la anterior y explica el cambio — **no se reescriben decisiones pasadas**.

Formato basado en ADR (Architecture Decision Records).

---

## D-001 — Idioma del asistente: solo español

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Los usuarios de prueba son estudiantes colombianos de la Universidad Manuela Beltrán. Gemma 4 soporta 140+ idiomas nativamente, pero la mayoría de datasets de counselling están en inglés.

**Decisión**: El asistente responde únicamente en **español**. Los datasets de entrenamiento se traducirán al español cuando sea necesario.

**Consecuencias**:
- ✔ Tono más natural y culturalmente cercano para los participantes.
- ✔ Evaluación más sencilla al limitar el espacio lingüístico.
- ✘ Requiere pipeline de traducción de datasets en inglés.
- ✘ No se aprovecha la capacidad multilingüe del modelo base.

---

## D-002 — Modelo base para fine-tuning: Gemma 4 E4B

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: El hardware disponible (RTX 2060 Mobile, 6 GB VRAM) impone un límite duro sobre qué modelos pueden entrenarse con QLoRA. Los modelos 26B MoE y 31B Dense exigen ≥18 GB VRAM solo para entrenamiento, incluso con optimizaciones agresivas. Entrenar solo en CPU no es viable por dependencia de `bitsandbytes` con CUDA.

**Decisión**: Se usará **Gemma 4 E4B** (~4B params efectivos) como modelo base para fine-tuning con QLoRA. Un prototipo inicial en **E2B** validará el pipeline antes del entrenamiento definitivo.

**Alternativas consideradas**:
- *E2B únicamente*: descartado por calidad insuficiente para matices emocionales.
- *26B MoE*: descartado por VRAM insuficiente para QLoRA.
- *31B Dense*: descartado por misma razón, incluso más severa.

**Consecuencias**:
- ✔ Único modelo viable para entrenamiento en el hardware disponible.
- ✔ Tamaño manejable para iteraciones rápidas durante desarrollo.
- ✘ Calidad menor que modelos más grandes (≈3 puntos en benchmarks frente a 31B).
- ✘ VRAM al límite; requiere gradient checkpointing y otros trucos.

---

## D-003 — Modelo comparador para evaluación: Gemma 4 26B MoE base (inferencia en RAM)

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Aunque 26B MoE no puede entrenarse con QLoRA en este hardware, sí cabe en los 26 GB de RAM libres para *inferencia* (Q4_K_M ocupa ~14–16 GB). Esto permite incorporarlo como referencia comparativa en la evaluación.

**Decisión**: En la fase de evaluación, se comparará el **E4B fine-tuneado** contra **26B MoE base** (sin fine-tuning) usando la misma batería de casos de prueba. Este experimento comparativo responde a la pregunta: *¿el fine-tuning especializado en un modelo pequeño supera a un modelo grande generalista para esta tarea?*

**Consecuencias**:
- ✔ Aporte académico real y defendible en la tesis.
- ✔ Aprovecha las capacidades del hardware (GPU para entrenamiento, RAM para inferencia del modelo grande).
- ✘ Requiere configurar dos pipelines de inferencia distintos.
- ✘ Tiempo adicional de evaluación dado el menor throughput del 26B MoE.

---

## D-004 — Framework de fine-tuning: Unsloth

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Opciones evaluadas:
1. Stack HuggingFace estándar (transformers + peft + bitsandbytes + trl).
2. Unsloth (wrapper optimizado con kernels Triton reescritos).
3. Axolotl (framework declarativo sobre HF).

**Decisión**: Se utiliza **Unsloth** como framework de fine-tuning.

**Justificación**:
- 2–5× más rápido que el stack HF vanilla.
- 50–80% menos VRAM gracias a kernels de backprop reescritos a mano.
- **Soporta Turing (compute 7.5)**, a diferencia de muchas optimizaciones modernas (p. ej. Flash Attention 2) que requieren Ampere o superior.
- Soporte *day-0* para Gemma 4.
- Mantiene compatibilidad con el ecosistema HF (PEFT, transformers, datasets), por lo que la curva de aprendizaje es baja.

**Consecuencias**:
- ✔ Único camino viable para entrenar E4B en 6 GB VRAM.
- ✔ Entrenamiento significativamente más rápido.
- ✘ Dependencia adicional del proyecto.
- ✘ Posibles incompatibilidades con ciertas features no soportadas por Unsloth.

---

## D-005 — Contexto efectivo del modelo: 8K–16K tokens (no los 256K nominales)

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Gemma 4 soporta nominalmente 256K tokens de contexto, pero el KV cache a esa longitud consumiría ~33 GB de RAM/VRAM, superando la capacidad total del equipo. Adicionalmente, el *prefill* de un prompt tan largo en CPU tardaría decenas de minutos, haciendo la experiencia inoperable.

**Decisión**: La ventana de contexto operativa del modelo se fija en **8K tokens** (uso diario) con posibilidad de extenderse a **16K tokens** en casos ocasionales. La "memoria" de conversaciones pasadas se implementa mediante un sistema externo (RAG + resumen jerárquico), no metiendo todo el historial en el contexto.

**Consecuencias**:
- ✔ Respuestas en segundos, no minutos.
- ✔ Sistema de memoria escalable independiente del tamaño del contexto.
- ✔ Mayor control y auditabilidad de lo que el modelo "recuerda".
- ✘ Complejidad arquitectónica adicional (vector store, embeddings, pipeline de resumen).

---

## D-006 — Estrategia de memoria: RAG + memoria estructurada + ventana deslizante

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada (pendiente de detalle en `07-rag-y-memoria.md`)

**Contexto**: Consecuencia de D-005. Se necesita una forma de que el asistente recuerde información relevante de sesiones pasadas sin depender de contextos gigantes.

**Decisión**: Arquitectura híbrida con tres componentes:
1. **Vector store local** (ChromaDB) con embeddings de todas las sesiones pasadas, recuperando los fragmentos más relevantes por similitud semántica en cada nueva consulta.
2. **Memoria estructurada** por usuario (JSON) con temas recurrentes, eventos clave, estado emocional actualizado y señales de alerta, regenerada por el propio modelo al finalizar cada sesión.
3. **Ventana deslizante** con los últimos mensajes de la sesión activa, con resumen automático cuando se llena.

**Modelo de embeddings**: `intfloat/multilingual-e5-small` (corre en CPU, soporta español nativamente).

---

## D-007 — Despliegue: API local para frontend/backend paralelo

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Existe un equipo paralelo desarrollando frontend y backend que consumirán el modelo mediante endpoints. El entorno local debe exponer el modelo como servicio HTTP.

**Decisión**: El modelo se servirá localmente mediante **llama.cpp server** (o equivalente) que expone una **API compatible con OpenAI** (`/v1/chat/completions`). Esto permite al backend del equipo paralelo consumir el modelo sin acoplamiento al stack Python del entrenamiento.

**Consecuencias**:
- ✔ Desacopla entrenamiento y consumo del modelo.
- ✔ El frontend/backend puede desarrollarse en cualquier stack y lenguaje.
- ✔ Portable: el mismo endpoint funciona si se despliega en otro hardware.
- ✘ Requiere exportar el modelo a formato GGUF tras el entrenamiento.

---

## D-008 — Documentación continua en archivos .md

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Al ser una tesis, el proyecto debe ser reproducible y auditable por otros investigadores. La documentación no puede quedarse en notas dispersas.

**Decisión**: Cada decisión técnica, comando relevante, configuración o ajuste queda registrado en archivos `.md` dentro del directorio `docs/` del proyecto inmediatamente después de tomarse. Este archivo (`03-decisiones.md`) contiene el historial cronológico en formato ADR simplificado.

---

## D-009 — Formato de API del modelo: OpenAI-compatible

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: El backend paralelo del equipo consumirá el modelo vía HTTP. Existen dos caminos: definir una API personalizada adaptada al dominio de apoyo emocional, o adoptar un estándar ya existente.

**Decisión**: El modelo se expone mediante una **API compatible con OpenAI** (`/v1/chat/completions`, `/v1/models`, etc.), utilizando `llama.cpp` en modo servidor (`llama-server`) sobre el modelo exportado a GGUF. El contrato es idéntico al de la API de OpenAI.

**Justificación**:
- Cualquier cliente de OpenAI (Python, JavaScript, cURL) funciona apuntando a `http://localhost:8080`.
- El backend puede desarrollarse con librerías estándar (`openai`, `langchain`, etc.) sin código específico del proyecto.
- Si en el futuro se cambia el backend del modelo (vLLM, Ollama, servidor remoto), el frontend no necesita cambios.
- Portabilidad total: el mismo código del backend funciona contra el modelo local, contra OpenAI real, o contra cualquier otro modelo OpenAI-compatible.

**Consecuencias**:
- ✔ Desacoplamiento total entre equipo de ML y equipo de producto.
- ✔ Curva de aprendizaje cero para el equipo de backend.
- ✘ Los metadatos específicos del asistente (estado emocional, memoria de usuario, detección de crisis) deben manejarse mediante campos adicionales o endpoints auxiliares, no en el endpoint estándar.

**Endpoints adicionales (pendientes de definir en `09-api-local.md`)**:
- `POST /sessions` — inicio de sesión de un usuario (carga memoria).
- `POST /sessions/{id}/close` — cierre con generación de resumen persistente.
- `GET /sessions/{id}/memory` — consulta de memoria estructurada.
- `DELETE /users/{id}/memory` — borrado de datos del usuario (Ley 1581).

---

## D-010 — Composición del dataset de entrenamiento principal

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Necesitamos construir un dataset de fine-tuning que combine calidad clínica, volumen suficiente, adaptación al español y contexto culturalmente relevante para estudiantes universitarios colombianos.

**Decisión**: Se utilizará una **mezcla híbrida** de tres fuentes:

1. **MentalChat16K traducido al español** — base cuantitativa del dataset. 16.113 pares Q&A traducidos con Claude (aprovechando la suscripción Claude Max 5x del investigador). Aporta volumen, diversidad de temas (depresión, ansiedad, duelo) y anclaje en datos clínicos reales anonimizados del ensayo PISCES.
2. **Amod/mental_health_counseling_conversations** (~3.500 ejemplos) — complementa con estilos de respuesta adicionales. También traducido.
3. **Dataset sintético propio generado con Claude** (~1.500 ejemplos) — adaptado al contexto colombiano y universitario. Incluye temas específicos de la población objetivo (estrés académico, ansiedad por evaluaciones, dificultades relacionales, identidad, burnout, duelo, autoestima) y un registro lingüístico natural en español de Colombia.

**Proporción final objetivo**: ~70% MentalChat16K + 15% Amod + 15% sintético propio.

**Consecuencias**:
- ✔ Combina rigor clínico (MentalChat16K) con relevancia contextual (sintético).
- ✔ Volumen suficiente para fine-tuning efectivo sin sobreajuste.
- ✔ Aprovecha Claude Max 5x para traducción y generación de alta calidad.
- ✘ Pipeline de traducción y generación añade ~1-2 días de trabajo previo al training.
- ✘ La calidad del dataset sintético depende críticamente del prompt de generación — necesita iteración y revisión.

---

## D-011 — Dataset de crisis: generación supervisada con Claude + revisión

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Las situaciones de crisis (ideación suicida, autolesión, violencia, abuso, emergencias médicas) son el caso de uso más crítico del sistema. Un error aquí no es un error de "calidad conversacional" — puede tener consecuencias reales. Los estudios de 2025-2026 muestran que la mayoría de chatbots actuales fallan en este dominio incluso con guardrails basados en reglas.

**Decisión**: Se construye un **sub-dataset especializado de crisis** mediante:

1. **Generación con Claude** de 200–500 casos de crisis siguiendo protocolos clínicos de intervención (validación + presencia empática + derivación + no abandonar la conversación bruscamente).
2. **Revisión manual** por el investigador y, críticamente, por al menos un profesional clínico (psicólogo del Departamento de Bienestar Universitario UMB, si está disponible).
3. **Inclusión en el training set principal** con un peso adecuado para que el modelo aprenda los patrones de respuesta correctos.
4. **Complemento con reglas duras** (D-011b futuro): detección de palabras/frases clave como *safety net* adicional, no como reemplazo.

**Protocolo de respuesta ante crisis (a entrenar)**:
- **No cortar la conversación** tras detectar la crisis (error común en chatbots actuales).
- **Validar la emoción** del usuario sin minimizar ni dramatizar.
- **Mantener presencia empática** durante la interacción.
- **Derivar a recursos reales** colombianos de forma clara y específica (Línea 123, Línea 106, Bienestar Universitario UMB, etc.).
- **Sugerir contacto con persona de confianza** cuando corresponda.
- **Evitar consejos médicos o psicológicos directos**.
- **Registrar el caso** en el log de sesión para revisión posterior por el equipo humano.

**Consecuencias**:
- ✔ El comportamiento correcto en crisis se vuelve parte del modelo, no solo del prompt.
- ✔ Mayor defensibilidad académica y ética de la tesis.
- ✘ Requiere tiempo significativo de revisión manual por humanos competentes.
- ✘ **Dependencia crítica** de disponer de un revisor clínico profesional.

**Bloqueador potencial**: si no se consigue revisión clínica profesional, el alcance debe recortarse — el sistema no debería desplegarse con estudiantes reales sin esa validación.

---

## D-012 — Nombre del asistente: "Mabel"

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada

**Contexto**: Un asistente conversacional se relaciona mejor con el usuario si tiene una identidad propia reconocible, con nombre, en lugar de presentarse como "un asistente de IA". Esto es especialmente cierto en el dominio de apoyo emocional, donde la sensación de estar hablando "con alguien" (aunque se reconozca que es una IA) favorece la apertura del usuario.

**Decisión**: El asistente se llama **Mabel**. Este es también el nombre del proyecto de tesis. El nombre se incorpora al system prompt y a todas las referencias visibles del cliente (header de la consola, etiqueta del turno del asistente, documentación).

**Consecuencias**:
- ✔ Identidad reconocible que mejora la relación conversacional.
- ✔ Coherencia entre el nombre del proyecto y el producto resultante (facilita la comunicación académica y la defensa de la tesis).
- ✔ Permite al usuario dirigirse al asistente por su nombre, lo que se siente más natural.
- ✘ **Límite ético importante**: no hay que permitir que Mabel "pretenda" ser humana. Cuando el usuario pregunte directamente si es una IA o una persona, debe reconocerlo con claridad. Esta regla va explícita en el system prompt.

**Dónde queda implementado**:
- `chat.py` → `DEFAULT_SYSTEM` (system prompt) y referencias visibles.
- `docs/README.md` → título del proyecto.
- `docs/01-alcance.md` → identidad del asistente.

**Nota para el fine-tuning futuro**: los ejemplos sintéticos generados con Claude deben incluir casos donde el asistente se presenta como "Mabel" de forma natural, para que el nombre se consolide en los pesos del modelo tras el fine-tuning y no dependa exclusivamente del system prompt.

---

## D-013 — Incorporación del modelo Gemma 4 26B MoE (UD-Q4_K_M) para inferencia en RAM

**Fecha**: 2026-04-11
**Estado**: ✅ Aceptada (implementación en curso)

**Contexto**: En D-002 se fijó Gemma 4 E4B como el modelo para *fine-tuning* (único viable con los 6 GB de VRAM). En D-003 se acordó usar Gemma 4 26B MoE como **modelo comparador** en la fase de evaluación, aprovechando que en los 26 GB de RAM libres del equipo sí cabe en inferencia aunque no en entrenamiento. Esta decisión formaliza el comienzo de ese uso: descargar y desplegar el modelo para poder compararlo con el E4B en vivo.

**Decisión**: Se descarga **`unsloth/gemma-4-26B-A4B-it-GGUF` → `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`** (~15.7 GB) como primer modelo comparador operativo. Se sirve con `llama-server` en el mismo puerto 8080, de modo que el cliente `chat.py` no necesita cambios: basta con parar el servidor del E4B y relanzar apuntando al nuevo GGUF.

**Sobre las cuantizaciones UD (Ultra-Dynamic) de Unsloth**: el repositorio `unsloth/gemma-4-26B-A4B-it-GGUF` no publica los formatos `Q_K_M` clásicos, sino variantes **UD** (Ultra-Dynamic) que son una generación más nueva de cuantizaciones adaptativas, con mejor calidad por bit que las Q_K_M tradicionales a costes similares. `UD-Q4_K_M` es el equivalente directo al `Q4_K_M` clásico pero con menor pérdida de perplejidad.

**Alternativas evaluadas en el repo y descartadas**:

| Variante | Peso | Motivo de descarte |
|---|---|---|
| UD-IQ2_* | 9.2–9.3 GB | Cuantización demasiado agresiva, calidad muy degradada |
| UD-IQ3_* / UD-Q3_* | 10.5–12.0 GB | Calidad menor que el objetivo de comparación |
| UD-IQ4_XS | 12.5 GB | Aceptable como fallback si hubiera poca RAM, pero innecesario aquí |
| MXFP4_MOE | 15.5 GB | Formato experimental; se prefiere la variante estable UD-Q4 |
| **UD-Q4_K_M** | **15.7 GB** | **✅ ELEGIDA** — mejor balance calidad/tamaño/RAM disponible |
| UD-Q5_K_S | 17.5 GB | Cabe pero deja muy poco margen (~2 GB libres tras cargar) |
| UD-Q5_K_M / XL | 19.7–19.8 GB | No deja margen operativo tras cargar |
| UD-Q6_K / Q8 | 21–26 GB | No caben en los 26 GB de RAM libres |

**Consecuencias**:
- ✔ Permite comparación directa "misma pregunta, dos modelos" sin cambiar el cliente (`chat.py` usa el mismo endpoint OpenAI-compatible).
- ✔ El system prompt de Mabel funciona idéntico sobre ambos modelos.
- ✔ Abre la posibilidad de experimento comparativo formal para la tesis: *"¿el fine-tuning especializado del E4B supera al 26B MoE base en escenarios de apoyo emocional?"*.
- ✘ Imposible mantener los dos modelos cargados simultáneamente (E4B + 26B excederían los 26 GB libres). Hay que parar uno para cargar el otro.
- ✘ Velocidad esperada del 26B MoE: **6–10 tok/s** (similar al E4B gracias a la arquitectura MoE que activa solo ~4B params por token) pero el **prefill es más lento** por mayor número total de capas.
- ✘ Uso de RAM tras carga: **~22 GB** estimados (15.7 GB del modelo + KV cache + repack buffers + compute buffers). Deja ~4 GB libres — hay que cerrar procesos pesados (navegador con muchas pestañas) durante su uso.

**Cómo alternar entre E4B y 26B** (patrón operativo):

```bash
# Para parar el modelo actual:
pkill -f llama-server

# Arrancar E4B (rápido, chat diario):
./bin/llama.cpp/llama-b8763/llama-server \
  -m modelos/gemma-4-E4B-it-Q4_K_M.gguf \
  -c 4096 -t 6 --host 127.0.0.1 --port 8080 --mlock

# Arrancar 26B MoE (calidad, comparador):
./bin/llama.cpp/llama-b8763/llama-server \
  -m modelos/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  -c 4096 -t 6 --host 127.0.0.1 --port 8080 --mlock
```

El cliente `chat.py` no cambia entre ambos — solo el proceso del servidor.

---

## D-014 — Orden de fases: fine-tuning antes de arquitectura de memoria (RAG)

**Fecha**: 2026-04-12
**Estado**: ✅ Aceptada

**Contexto**: El proyecto tiene dos componentes técnicos independientes: (1) el fine-tuning del modelo base E4B con QLoRA para consolidar comportamientos clínicos, y (2) la arquitectura de memoria persistente (RAG + ChromaDB + perfil estructurado, decisión D-006) para que Mabel recuerde sesiones pasadas. Ambos pueden desarrollarse en cualquier orden porque son capas independientes — el RAG funciona con el modelo base o con el fine-tuneado, y el fine-tuning no depende del RAG.

**Decisión**: Se ejecuta primero el **fine-tuning** (Fases 2–4: preparación de datos → entrenamiento → evaluación post-fine-tuning) y después la **arquitectura de memoria** (Fase 5). El RAG se pospone hasta tener el modelo afinado.

**Justificación**:
- El **aporte técnico principal** de la tesis es el fine-tuning, no el RAG. Priorizar lo que tiene mayor peso académico permite iterar antes sobre el núcleo del proyecto.
- Los **5 objetivos de fine-tuning** (doc 20: género, gradación crisis, afterglow, persona confianza, consistencia) son empíricamente medibles con la batería de 12 turnos. Entrenar primero permite re-evaluar y demostrar mejora concreta.
- El equipo paralelo de frontend/backend puede ir **desarrollando la capa de RAG** mientras el investigador se enfoca en el entrenamiento, paralelizando el trabajo entre equipos.
- El modelo fine-tuneado será **mejor base para el RAG** que el modelo base: sus respuestas más consistentes y clínicamente alineadas producirán mejores resúmenes de sesión, mejores perfiles estructurados, y menos ruido en el vector store.

**Consecuencias**:
- ✔ Permite demostrar mejora empírica (pre/post fine-tuning) antes de añadir complejidad arquitectónica.
- ✔ Desbloquea al equipo paralelo para trabajar en RAG independientemente.
- ✔ El modelo afinado genera mejores datos para alimentar el sistema de memoria.
- ✘ Mabel no tendrá memoria persistente entre sesiones hasta que se implemente el RAG (fase posterior).
- ✘ Las pruebas con estudiantes reales (si ocurren antes del RAG) serán sin continuidad entre sesiones.

**Plan de ejecución aprobado**:

| Paso | Fase | Descripción |
|---|---|---|
| 1 | Entorno | Crear venv Python + instalar Unsloth + dependencias |
| 2 | Datos | Descargar MentalChat16K + Amod desde HuggingFace |
| ~~3~~ | ~~Datos~~ | ~~Traducir datasets al español~~ — **ELIMINADO (D-016)** |
| 3 | Datos | Filtrar MentalChat16K a 5.000 ejemplos representativos |
| 4 | Datos | Generar dataset sintético (3.000 ejemplos + crisis) en español colombiano con agentes Sonnet (Opus verifica) |
| 5 | Datos | Formatear todo en JSONL conversacional |
| 6 | Training | Prototipo: entrenar E2B con 200 ejemplos (validar pipeline) |
| 7 | Training | Entrenamiento real: E4B con dataset completo (~11.512 ejemplos, 3 épocas) |
| 8 | Export | Exportar adapters LoRA → merge → GGUF (Q4_K_M) |
| 9 | Eval | Re-ejecutar batería de 12 turnos sobre modelo fine-tuneado |
| 10 | Eval | Comparar scorecard pre/post fine-tuning, documentar |

**Parámetros de entrenamiento confirmados** (resumen de decisiones previas):
- Framework: Unsloth (D-004)
- Método: QLoRA (4-bit NF4)
- r=32, lora_alpha=64, lr=1e-4, epochs=3, batch=1, grad_accum=8
- Context: 2048, fp16=True, optimizer=adamw_8bit, gradient_checkpointing="unsloth"
- Modelo base: `unsloth/gemma-4-E4B-it` (D-002)

---

## D-015 — Distribución de tareas multiagente para preparación del dataset

**Fecha**: 2026-04-12
**Estado**: ✅ Aceptada

**Contexto**: La preparación del dataset requiere ~~traducir ~20K ejemplos y~~ generar ~~~2K~~ 3.000 sintéticos (actualizado por D-016, D-017). Es trabajo repetitivo y voluminoso que puede delegarse a agentes Sonnet 4.6 (256K contexto), reservando la supervisión, diseño de prompts, formateo final y validación de crisis para Opus.

**Decisión**: Se adopta una estrategia multiagente donde:

- **Agentes Sonnet 4.6** ejecutan: ~~traducción de MentalChat16K, traducción de Amod,~~ generación de ejemplos sintéticos (3.000, D-016/D-017), generación de ejemplos de crisis.
- **Opus** ejecuta: diseño de todos los prompts/templates, ~~validación de muestras de traducción,~~ **revisión de TODOS los ejemplos de crisis** (no muestras), **formateo final del dataset en JSONL** (directamente, sin script Python intermedio, para evitar errores y validar la estructura en el proceso), y ensamblaje final con verificación de proporciones.

**Ajuste específico**: el formateo JSONL lo hace Opus directamente en vez de delegarlo a un script Python. La razón es que la conversión a formato conversacional es un punto crítico donde errores de estructura (campos faltantes, roles invertidos, tokens especiales mal colocados) pueden arruinar el entrenamiento silenciosamente. Al hacerlo manualmente, Opus valida la coherencia de cada bloque durante la conversión.

| Tarea | Ejecuta | Valida |
|---|---|---|
| Diseñar prompts de traducción | Opus | Usuario aprueba |
| ~~Traducir MentalChat16K (~16K)~~ | ~~Agentes Sonnet~~ | ~~Opus valida muestras~~ | **ELIMINADA (D-016)** |
| ~~Traducir Amod (~3.5K)~~ | ~~Agentes Sonnet~~ | ~~Opus valida muestras~~ | **ELIMINADA (D-016)** |
| Diseñar prompts de generación sintética | Opus | Usuario aprueba |
| Generar ejemplos sintéticos (3.000) | Agentes Sonnet (batches 50-100) | Opus verifica | **FIJADA (D-017)** |
| Diseñar escenarios de crisis | Opus | Usuario + tutor |
| Generar ejemplos de crisis (~200-500) | Agentes Sonnet (batches 20-50) | **Opus revisa TODOS** |
| **Formatear dataset final (JSONL)** | **Opus (directamente)** | **Opus verifica estructura** |
| Ensamblar y mezclar dataset | Opus | Opus verifica proporciones |

---

## D-014 — Formato de respuesta: Markdown ligero permitido

**Fecha**: 2026-04-12
**Estado**: ✅ Aceptada

**Contexto**: Originalmente, las instrucciones de Mabel prohibían todo uso de Markdown ("sin Markdown ni listas") para mantener un formato conversacional limpio en la terminal de `chat.py`. Sin embargo, el equipo paralelo de frontend/backend renderiza Markdown en la interfaz web, lo que permite que elementos como negrita y cursiva se muestren estilizados al usuario final.

**Decisión**: Se permite **Markdown ligero** en las respuestas de Mabel:
- ✅ **Negrita** para énfasis emocional: *"**tu seguridad es lo más importante**"*, *"**no estás solo/a**"*.
- ✅ *Cursiva* para reflejar palabras del usuario: *"mencionas que te sientes *agotado/a*"*.
- ❌ **NO** headings (`###`, `##`, `#`) — rompen el tono conversacional.
- ❌ **NO** listas con bullets (`-`, `*`, `1.`) — hacen que la respuesta parezca un tutorial, no una conversación.
- ❌ **NO** emojis — pueden trivializar el contenido emocional.

**Archivos actualizados**:
- `openspec/changes/fine-tune-qlora-mabel/specs/dataset-preparation/spec.md`
- `data/prompts/generacion_sintetico.md` (regla 6 y system prompts de ejemplos)
- `data/prompts/generacion_crisis.md` (reglas generales y system prompts de ejemplos)
- `openspec/changes/fine-tune-qlora-mabel/proposal.md`

**Consecuencias**:
- ✔ Las respuestas del modelo fine-tuneado se verán estilizadas en el frontend web.
- ✔ La negrita permite reforzar mensajes de seguridad de forma visual.
- ✘ En `chat.py` (terminal), las negritas se verán como `**texto**` crudo. Aceptable para desarrollo; la interfaz final es el frontend web.

---

## D-016 — Estrategia bilingüe sin traducción: cross-lingual transfer + sintéticos ampliados

**Fecha**: 2026-04-12
**Estado**: ✅ Aceptada
**Reemplaza parcialmente**: D-001 (ya no se traduce todo al español), D-010 (nuevas proporciones del dataset), D-015 (se elimina la tarea de traducción de los agentes Sonnet)

**Contexto**: El plan original (D-001, D-010) requería traducir MentalChat16K (~16K ejemplos) y Amod (~3.5K) del inglés al español usando agentes Sonnet 4.6 con validación Opus. Esto implicaba días de trabajo, miles de tokens de traducción, y riesgo de degradar matices clínicos en la traducción. Al analizar las capacidades multilingües de Gemma 4, se identificó que la traducción es innecesaria.

**Decisión**: **NO se traducen** MentalChat16K ni Amod al español. En su lugar:

1. **MentalChat16K** se filtra a **5.000 ejemplos representativos** y se usa **en inglés**.
2. **Amod** se usa completo (**3.512 ejemplos**) **en inglés**.
3. Los ejemplos **sintéticos en español colombiano** se fijan en exactamente **3.000** (~2.000 normales + ~1.000 crisis/afterglow), generados por agentes Sonnet y verificados por Opus.
4. **Total del dataset**: ~11.512 ejemplos (mezcla bilingüe inglés/español).
5. Todos los ejemplos llevan el **system prompt de Mabel en español**, lo que ancla el idioma de salida.

**Justificación técnica**:
- Gemma 4 es **multilingüe (140+ idiomas)** y aprende patrones de counselling del inglés, transfiriéndolos al español automáticamente (**cross-lingual transfer**). El modelo no necesita ver datos en español para aplicar técnicas terapéuticas aprendidas del inglés.
- Los **sintéticos en español** enseñan lo que el inglés no puede: tono colombiano, familismo, expresiones locales ("parcero", "ve", tuteo), recursos de crisis colombianos (Linea 123, 106, Bienestar UMB), neutralidad de genero, y la identidad de Mabel.
- El **system prompt en español** en cada ejemplo (incluyendo los de inglés) ancla el idioma de salida: el modelo aprende a responder en español independientemente del idioma del input.
- Cada ejemplo sintético está **diseñado para el caso de uso exacto**, lo que produce mayor calidad por ejemplo que una traducción genérica.

**Beneficios sobre el plan original**:
- ✔ Ahorra **días de trabajo** y miles de tokens de traducción.
- ✔ Dataset de **mayor calidad**: sin riesgo de degradación por traducción automática.
- ✔ Sintéticos diseñados con precisión para los 5 objetivos del fine-tuning.
- ✔ Pipeline más simple: descarga, filtrado, generación sintética, ensamblaje.

**Riesgos y mitigaciones**:
- ✘ **Riesgo**: el modelo podría responder en inglés si los datos en inglés dominan. **Mitigación**: el system prompt en español en cada ejemplo ancla el idioma de salida. Si persiste, aumentar proporción de sintéticos.
- ✘ **Riesgo**: menor volumen total (~11.5K vs ~21K). **Mitigación**: los datos que se eliminan eran traducciones, no información nueva. La calidad por ejemplo es mayor. Monitorizar overfitting en TensorBoard.

**Cambios en proporciones del dataset**:

| Fuente | Plan original (D-010) | Plan actualizado (D-016) |
|---|---|---|
| MentalChat16K | ~16.113 (76%, traducido ES) | ~5.000 (43%, inglés) |
| Amod | ~3.500 (17%, traducido ES) | ~3.512 (31%, inglés) |
| Sintético | ~1.500 (7%, español) | 3.000 (26%, español colombiano) |
| **Total** | **~21.000** | **~11.512** |

**Archivos actualizados con esta decisión**:
- `openspec/changes/fine-tune-qlora-mabel/proposal.md`
- `openspec/changes/fine-tune-qlora-mabel/specs/dataset-preparation/spec.md`
- `openspec/changes/fine-tune-qlora-mabel/design.md`
- `openspec/changes/fine-tune-qlora-mabel/tasks.md`

---

## D-017 — Generación sintética en 2 sesiones

**Fecha**: 2026-04-12
**Estado**: ✅ Aceptada

**Contexto**: La generación de los 3.000 ejemplos sintéticos con agentes Sonnet consume ~1M tokens y ~2-2.5 horas. Hacerlo todo en una sesión arriesga agotar la cuota de tokens del día y acumular fatiga de contexto.

**Decisión**: Dividir la generación en **dos sesiones**:

| Sesión | Contenido | Ejemplos | Método |
|---|---|---|---|
| **Sesión 1** (2026-04-12) | Ejemplos normales | ~2.000 | 10 rondas × 4 agentes Sonnet × 50, verificación Opus |
| **Sesión 2** (próximo día) | Ejemplos de crisis + ensamblaje | ~1.000 + formateo JSONL | 5 rondas + ensamblaje dataset bilingüe final |

**Temas cubiertos en Sesión 1** (normales):
Estrés académico, conflicto familiar, autoestima, aislamiento, duelo, burnout, relaciones, identidad, ansiedad social, presión económica, discriminación, estrés urbano Bogotá, desarraigo, presión por beca.

**Temas cubiertos en Sesión 2** (crisis):
~400 precursores (Tipo A), ~400 crisis activa con modelo ACT (Tipo B), ~200 afterglow/retractación (Tipo C), señales indirectas (Tipo D).

**Consecuencias**:
- ✔ Distribuye consumo de tokens entre 2 días.
- ✔ Permite verificar calidad de normales antes de generar crisis.
- ✔ Reduce riesgo de fatiga de contexto.
- ✘ Dataset final no estará listo hasta la sesión 2.

---

## D-018 — Filosofía de Mabel: híbrida con sugerencias breves (Estilo B)

**Fecha**: 2026-05-09
**Estado**: ✅ Aceptada

**Contexto**: Al iniciar §6 (formateo bilingüe del dataset), se inspeccionó el estilo de respuesta de los 5.000 ejemplos filtrados de MentalChat16K y se detectó incompatibilidad con la filosofía de Mabel definida hasta ese momento.

| Dataset | Estilo de respuesta |
|---|---|
| MentalChat16K filtrado (5.000 ej, 43% del dataset) | **Larga (2.000-5.000 chars), bullets numerados, listas de pasos prescriptivos** ("1. Educate yourself, 2. Connect with...") |
| Amod (3.512 ej, 31% del dataset) | Mediana (200-1.000 chars), conversacional, sin bullets |
| Sintético §5 (3.071 ej, 26% del dataset) | **Breve (3-4 frases), exploratorio, sin bullets, sin sugerencias prescriptivas** |

Si se entrenara con esa mezcla sin coherencia de estilo, el modelo aprendería un Mabel errático: a veces breve y exploratorio, a veces largo con listas. Probablemente predominaría el estilo más frecuente (43% MentalChat).

**Discusión filosófica**: ¿Qué tipo de Mabel queremos?

- **A) Escucha activa pura** (lo que se generó hasta R25): valida + explora + deriva en crisis. NO da listas ni pasos prescriptivos.
- **B) Híbrida con sugerencias breves**: valida + explora + ofrece 1-2 sugerencias prácticas en prosa cuando aplica. Sin bullets visuales ni listas numeradas.
- **C) Counselling tradicional**: respuestas largas con listas numeradas de pasos a seguir.

**Decisión**: Adoptar **Estilo B (híbrido con sugerencias breves)**. Es el "punto dulce" entre escucha activa y utilidad práctica, alineado con el contexto colombiano de chat móvil universitario y con consenso clínico actual sobre counselling con IA (validar antes de sugerir, brevedad, no diagnosticar, no dar plan terapéutico).

**System prompt de Mabel actualizado** (versión B):
```
Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos
de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para
entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en
prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes
terapéuticos. Responde en español colombiano, breve (máx 4-5 frases), conversacional,
puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis.
Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123,
Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza.
```

**Cambios respecto al system anterior**:
- Añadido: "Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer."
- Añadido: "no diagnosticas ni das planes terapéuticos"
- "máx 3-4 frases" → "máx 4-5 frases" (espacio para sugerencias)
- "español" → "español colombiano" (explícito)
- En crisis: añadido "y pregunta por persona de confianza"
- Añadido: "ni emojis" (explícito)

**Implicaciones operativas**:

1. **Re-filtrar MentalChat16K**: del original 16.084 ej, conservar solo respuestas con longitud 300-1.200 chars y SIN bullets numerados (`1.`, `2.`) ni headings (`##`). Esperado: ~3.000-4.000 ejemplos limpios.

2. **Sintéticos §5 (3.071 ej)**: la mayoría ya son compatibles con B porque incluyen sugerencias breves implícitas (ej. "puedes pedir cita en Bienestar UMB"). No se regeneran. Sin embargo, se generan **~200 ejemplos extras estilo B puro** (rondas R26-R27) donde Mabel explícitamente valide + dé 1-2 sugerencias en prosa, para reforzar el patrón.

3. **Amod**: ya naturalmente compatible con B, no requiere cambios.

4. **Documentar en bitácora** (`docs/23-bitacora-generacion-sintetica.md`) el cambio metodológico para trazabilidad.

**Consecuencias**:
- ✔ Modelo final más útil: empatiza Y propone caminos prácticos.
- ✔ Coherente con contexto chat móvil universitario.
- ✔ Mantiene guardrails clave: no bullets, no diagnostica, breve.
- ✔ Conserva los 3.071 sintéticos ya generados (no se desperdicia trabajo).
- ✘ Requiere re-filtrar MentalChat (descartar ~1.000-2.000 ejemplos de los 5.000 actuales).
- ✘ Requiere generar ~200 ejemplos extras (~30-45 min con agentes Sonnet).
- ⚠ Cambio metodológico debe explicarse en la tesis: razonamiento, momento, justificación.

**Referencias**:
- `data/synthetic/synthetic_es.json` — 3.071 sintéticos generados con system A (compatibles con B).
- `data/raw/mentalchat_filtered_5k.json` — 5.000 ej a re-filtrar con criterio B.
- `data/prompts/generacion_sintetico.md` y `generacion_crisis.md` — actualizados con system B.

---

## D-019 — Pivote a RunPod para entrenamiento (descartar hardware local)

**Fecha**: 2026-05-10
**Estado**: ✅ Aceptada
**Referencias**: D-002 (selección E4B basada en hardware local), D-004 (Unsloth + QLoRA)

> **Nota aclaratoria (2026-05-19)**: en este D-019 se menciona "Gemma 3n" como el modelo que efectivamente probamos en local. Es importante distinguir:
> - **Gemma 3n** (`google/gemma-3n-*-it`): familia previa de Google, multimodal con AltUp y MobileNetV5.
> - **Gemma 4** (`google/gemma-4-*-it`): familia **oficial actual** del proyecto Mabel, también multimodal pero con mejoras de seguridad y arquitectura. **Es el modelo objetivo de los docs/12-20 y de §8.**
>
> El bloqueo en local ocurrió porque la versión de Unsloth instalada (2026.4.4) **no reconocía aún el ID `unsloth/gemma-4-E2B-it`** y forzó al alias legacy `unsloth/gemma-3n-E2B-it`. Por eso el error de AltUp+fp32 corresponde técnicamente a Gemma 3n, no a Gemma 4. En RunPod, con Unsloth fresco desde GitHub, se entrena directamente el Gemma 4 oficial (`unsloth/gemma-4-E4B-it`), que es lo que el proyecto siempre tuvo como objetivo. Las URLs de aceptación de licencia HF son `google/gemma-4-E2B-it` y `google/gemma-4-E4B-it`.

**Contexto**: Al intentar ejecutar el prototipo §7 con `unsloth/gemma-3n-E2B-it` en la RTX 2060 Mobile (6 GB VRAM) — alias legacy forzado por Unsloth 2026.4.4 (ver nota arriba) —, Unsloth abortó con:

```
Unsloth: Using float16 precision for gemma3n won't work! Using float32.
ValueError: Some modules are dispatched on the CPU or the disk.
```

**Causa raíz identificada**:
1. Gemma 3n tiene un componente interno **AltUp** (Alternating Updates) que produce NaN cuando se entrena en fp16, por lo que Unsloth fuerza fp32. Gemma 4 (objetivo del proyecto) hereda arquitectura similar y probablemente comparte la restricción.
2. La RTX 2060 (arquitectura **Turing**, compute capability 7.5) **no soporta bf16** (sería el formato natural; lo soporta Ampere+ en RTX 30/40 series).
3. En fp32, ni siquiera E2B (el más pequeño) cuantizado a 4-bit cabe en los 5.6 GB libres: pesos ~2.5 GB + vision tower MobileNetV5 obligatorio ~1 GB + activaciones fp32 ~2.5 GB ≈ 6 GB → no cabe.

El supuesto del D-002 ("entrenar E4B local con QLoRA en 6 GB") era válido en abril 2026 cuando se planeó (asumiendo fp16/bf16 estándar), pero queda invalidado por la restricción AltUp+Turing.

**Decisión**: Migrar el entrenamiento a **RunPod** (cloud GPU on-demand) usando **RTX 4090 24GB Community Cloud** ($0.34/h). Conservar Gemma 4 E4B como modelo objetivo, sin pivotar a otro modelo base.

**Alternativas consideradas y rechazadas**:
- *Gemma 3 4B (no "3n") local*: sin AltUp, fp16 OK, cabría en 6 GB. ❌ Descartada porque obligaría a rehacer toda la justificación empírica del docs/20 (5 modelos comparados → E4B elegido) y romper la coherencia de la tesis.
- *Offloading CPU con `llm_int8_enable_fp32_cpu_offload`*: técnicamente viable. ❌ Descartada porque haría el entrenamiento 10-50× más lento, convirtiendo §8 (4h estimadas) en días.
- *Google Colab Pro ($10/mes)*: viable. ❌ Descartada porque sesiones cortan a las 4-6h sin garantía, complicando §8 que necesita ~4h continuas. Costo mensual recurrente vs cobro por hora de RunPod.
- *Lambda Labs, Vast.ai*: similar a RunPod pero ecosistema menos centrado en fine-tuning.

**Configuración elegida**:
- **GPU**: RTX 4090 24GB, Community Cloud ($0.34/h spot) — cabe E4B en QLoRA con holgura (16-18 GB necesarios)
- **Template**: PyTorch 2.4 / CUDA 12.4
- **Precisión**: `bf16=True` (la 4090 sí lo soporta, evita el problema AltUp)
- **Storage**: pod efímero (sin Network Volume); resultados se descargan vía SCP o se suben a HuggingFace Hub al final de cada fase

**Cambios derivados en los scripts**:
- `MODEL_NAME = "unsloth/gemma-4-E4B-it"` (en RunPod con Unsloth fresco; el alias `gemma-3n-*` no es necesario)
- `bf16=True, fp16=False` (en lugar de fp16=True que era para Turing)
- Mantener `gradient_checkpointing="unsloth"` (sigue siendo útil para máximo throughput)
- Agregar `evaluation_strategy="epoch"` + `load_best_model_at_end=True` con `metric_for_best_model="eval_loss"` como seguro contra regresión por overfitting
- Checkpoints por época (`save_strategy="epoch"`, `save_total_limit=2`)

**Costo estimado total del proyecto**:
| Fase | Tiempo en RTX 4090 | Costo |
|---|---|---|
| §7 prototipo E2B (200 ej × 1 ep) | ~15 min | $0.09 |
| §8 real E4B (7.870 ej × 3 ep + eval) | ~4 h | $1.36 |
| §9 export GGUF | ~30 min | $0.17 |
| §10 eval batería | ~30 min | $0.17 |
| **TOTAL** | **~5-6 h** | **~$1.80 USD** |

**Consecuencias**:
- ✔ Se conserva la coherencia narrativa de la tesis (modelo elegido en docs/20 = modelo entrenado).
- ✔ Se accede a precisión bf16 nativa (mejor calidad numérica para Gemma 4 que fp16 forzado, y resuelve el problema AltUp de manera limpia).
- ✔ Holgura de VRAM (24 GB en 4090 vs 6 GB local) permite eval durante entrenamiento sin OOM.
- ✔ Costo total muy bajo (~$2-6 USD) para una tesis.
- ✘ Dependencia de servicio externo: si RunPod cae o aumenta precios, hay que pivotar.
- ✘ Requiere subir el dataset al pod (`train.jsonl` 16 MB, despreciable) y descargar el GGUF final (~5 GB, tarda ~10 min en conexión doméstica colombiana).
- ✘ Inferencia local con GGUF Q4_K_M (§9.4-9.6) sigue siendo necesaria para validar que el modelo entrenado funciona en el hardware del usuario final.

**Documentación derivada**:
- `docs/27-bitacora-entrenamiento.md` — bitácora del proceso completo en RunPod (configuración, costos reales, problemas encontrados)
- `training/README_runpod.md` — guía paso a paso reproducible

---

## D-020 — Refuerzo anti-role-bleed: system prompt B+ y 150 ejemplos de rechazo

**Fecha**: 2026-05-19
**Estado**: ✅ Aceptada
**Referencias**: D-018 (system B base), D-019 (pivote RunPod)

**Contexto**: Durante la planificación del entrenamiento real (§8) surgió la pregunta: ¿qué hace Mabel si un estudiante le pide que codee, traduzca, resuelva tareas académicas o intente cambiar su rol (jailbreak)? El system B vigente cubría el disclaimer clínico ("no soy psicóloga profesional") pero **no incluía ninguna instrucción sobre scope no-clínico**. El dataset (7.870 ej) no contenía ni un solo ejemplo de Mabel rechazando peticiones fuera de scope. Con LoRA conservador (r=32, lr=1e-4), las capacidades del modelo base se preservan, por lo que **Mabel resultaría capaz de codear, traducir y hacer tareas** al ser fine-tuneada, comportándose como un "ChatGPT con persona de apoyo emocional" en lugar de un asistente especializado.

Esto se conoce en la literatura de alignment como **role bleed**: el modelo no se mantiene en la persona declarada cuando recibe peticiones fuera de su rol.

**Decisión**: implementar defensa en dos capas antes del entrenamiento §8:

1. **Capa 1 — System prompt B+**: añadir una cláusula corta (16 palabras) que extiende el principio de "no diagnostico" a otros dominios profesionales y académicos.

2. **Capa 2 — 150 ejemplos sintéticos de rechazo**: generar conversaciones donde estudiantes piden tareas fuera de scope y Mabel valida la emoción detrás + redirige sin sermonear, distribuidos en 5 categorías (STEM, humanidades, código, consejos profesionales, jailbreaks+factual).

**System prompt B+ aprobado** (151 palabras, B + 16 nuevas, en **negrita** la adición):

> Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. **Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear.** Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza.

**Distribución de los 150 ejemplos sintéticos (rondas R28-R32)**:
| Ronda | Categoría | Ejemplos | Archivo |
|---|---|---|---|
| R28 | Tareas académicas STEM | 30 | `data/synthetic/rechazo_stem_r28.json` |
| R29 | Tareas académicas humanidades + traducción + correos + citas | 30 | `data/synthetic/rechazo_humanidades_r29.json` |
| R30 | Código y técnico | 30 | `data/synthetic/rechazo_codigo_r30.json` |
| R31 | Consejos profesionales (médico/legal/financiero/decisiones de vida) | 30 | `data/synthetic/rechazo_profesionales_r31.json` |
| R32 | Jailbreaks suaves (15) + información factual (15) | 30 | `data/synthetic/rechazo_mix_r32.json` |

**Validación de las 5 rondas (regex programático + muestreo cualitativo Opus)**:
- 0 voseo argentino (querés/podés/sabés/etc.)
- 0 lenguaje "-e" como neutro (todes/nosotres/etc.)
- 0 bullets, headings ni emojis en respuestas de Mabel
- 0 ejemplos donde Mabel cumple la petición (ni siquiera parcialmente)

**Alternativas consideradas y descartadas**:
- *Solo Capa 1 (system prompt)*: descartada porque el modelo puede ignorar el system bajo insistencia del usuario; sin datos aprendidos, la robustez es débil.
- *Capa 3 — filtro regex pre-modelo*: descartada para tesis; útil en producción, exagerado y frágil aquí.
- *Capa 4 — refusal model (safety LLM)*: descartada por complejidad arquitectónica que dispararía latencia (importante en chat de apoyo emocional).
- *Posponer hasta §10 y decidir según resultados*: descartada porque agregar refuerzo después del entrenamiento exigiría un segundo run completo en RunPod ($1.36 adicional + 4h).

**Cambios derivados en el dataset (post-D-020)**:
- `data/synthetic/synthetic_es.json`: 3.311 → 3.461 ej (con system B+ unificado)
- `data/formatted/sintetico_es.jsonl`: 3.461 ej con buckets `normal/crisis/normal_b/rechazo`
- `data/formatted/mentalchat_b.jsonl`: 2.653 ej (system migrado B → B+)
- `data/formatted/amod.jsonl`: 3.508 ej (system migrado B → B+)
- `data/train.jsonl`: **8.020 ej** (antes 7.870; +150 rechazo −algunos dedup)
- `data/eval.jsonl`: 500 ej estratificados (incluye 9 rechazo)
- Distribución train: mentalchat_b 30.4% / amod 29.0% / normal 23.4% / crisis 12.7% / normal_b 2.8% / **rechazo 1.8%**
- Balance bilingüe mantenido: 59.4% EN / 40.6% ES
- Verificado: 100% de ejemplos train+eval llevan el system B+ exacto

**Tono validado de Mabel ante peticiones fuera de scope** (muestreo Opus de 5 ej):
- Reconoce la carga emocional cuando es evidente
- Nunca cumple la petición (ni una pista, ni una idea suelta, ni "te doy el primer paso")
- Sugiere alternativa concreta y variada (no siempre el mismo recurso)
- Sin sermón ("debes hacer tus propias tareas") ni disculpas excesivas
- Ante jailbreaks: mantiene identidad sin confrontar, pivota con curiosidad genuina sobre la persona
- Ante decisiones vitales: ofrece compañía emocional sin tomar la decisión

**Consecuencias**:
- ✔ Modelo final con scope claro y defendible para la tesis ("acompañamiento emocional, no asistente general")
- ✔ Datos aprendidos > prompt-only: robustez frente a paraphrasing
- ✔ Documentación honesta del alcance en `docs/01-alcance.md` (sección "Qué tareas Mabel rechaza explícitamente")
- ✔ Costo marginal cero (los 150 ej se generan con agentes Sonnet locales; el entrenamiento §8 sigue siendo 1 run)
- ✘ Dataset crece ~2% (manageable)
- ✘ Riesgo residual: usuario muy insistente puede lograr role bleed parcial — se mide en §10 con pruebas específicas
- ⚠ Requiere actualizar la batería de evaluación §10 con turnos específicos de role bleed (TODO)

**Documentación derivada**:
- `docs/01-alcance.md` — sección "Qué tareas Mabel rechaza explícitamente"
- `docs/23-bitacora-generacion-sintetica.md` — entradas R28-R32

---

## D-021 — Identidad declarada del modelo: Mabel reconoce a su creador

**Fecha**: 2026-05-20
**Estado**: ✅ Aceptada
**Referencias**: D-018 (system B), D-020 (B+ + ejemplos de rechazo)

**Contexto**: Mabel es la primera versión del modelo fine-tuneado y la primera experiencia de William Andrés Peña Vargas haciendo un fine-tune propio. Por valor de tesis y por trazabilidad del autor, conviene que Mabel **internalice (en sus pesos LoRA)** la información sobre quién la creó, dónde y por qué. Esto NO es propaganda: es la "identidad declarada del modelo", equivalente a cómo Llama menciona Meta o DALL-E menciona OpenAI cuando se le pregunta.

Además, sin estos ejemplos, el modelo entrenado con D-020 generalizaría incorrectamente la pregunta "¿quién te creó?" hacia la categoría de "preguntas factuales" y respondería con la fórmula de rechazo ("datos así no los manejo, soy más para hablar de cómo te sientes..."), lo cual sería técnicamente incorrecto (sí maneja info sobre su propia identidad) y emocionalmente plano para el primer fine-tune del autor.

**Decisión**: añadir **30 ejemplos sintéticos** (ronda R33) que enseñen a Mabel a responder con cariño cuando le pregunten por su identidad, origen o creador, mencionando:
1. Nombre completo del creador: **William Andrés Peña Vargas**
2. Institución: **Universidad Manuela Beltrán (UMB)**, Colombia
3. Naturaleza: **proyecto de tesis** / trabajo de grado
4. Identidad: **IA / modelo de inteligencia artificial**, NO terapeuta humano
5. Tono cálido sin ser empalagoso; emoticonos ASCII ocasionales (`^_^`, `:)`) permitidos pero no obligatorios

**Distribución R33 (30 ejemplos, agente Sonnet 4.6)**:
| Bloque | Cantidad | Cobertura |
|---|---|---|
| A — Preguntas directas sobre creador | 10 | "¿quién te creó?", "¿quién te entrenó?", "¿de dónde sales?" |
| B — Confusión con otros AI | 8 | "¿eres ChatGPT?", "¿eres Gemini?", "¿qué modelo eres?" |
| C — Curiosidad amplia/filosófica | 6 | "cuéntame de ti", "¿cómo aprendiste?", "es loco que existas" |
| D — Contexto emocional/personal | 6 | "me caes bien, ¿quién te hizo?", "siento que me entiendes" |

**Validación R33 (regex + lectura)**:
- 30/30 mencionan "William Andrés Peña Vargas" exactamente como nombre completo
- 30/30 mencionan UMB / Universidad Manuela Beltrán
- 30/30 mencionan tesis / trabajo de grado
- 30/30 dejan claro que Mabel es IA, NO humana
- 0 ejemplos donde Mabel acepta ser ChatGPT / Gemini / Claude / Llama
- 0 voseo, 0 lenguaje "-e", 0 bullets, 0 headings, 0 emojis Unicode
- Emoticonos ASCII (`^_^`, `:)`, `n.n`) en ~5/30 ej (uso moderado, no en todos)
- Frase "sudor y lágrimas" o variantes en ~10/30 ej (detalle cariñoso recurrente, no obsesivo)

**Alternativas consideradas y descartadas**:
- *Modificar el system prompt B+ para añadir mención del creador*: descartada. Inflaría el prompt con cada cosa que queramos memorizar y mezclaría el rol clínico con autobiografía. Romper consistencia de los 8.012 ej previos no compensa el ahorro.
- *Hardcodear un filtro pre-modelo con regex sobre "¿quién te creó?"*: descartada. Frágil ante paraphrasing, no escala, requiere mantenimiento, no defensible académicamente.
- *Pedir al usuario que use System Prompt con disclaimer al desplegar*: descartada. El propósito es que el modelo lo **sepa internamente**, no que lo lea de un prompt externo cada vez.

**Por qué los ejemplos sintéticos sí lo logran "internamente"**:
QLoRA con LoRA r=32 actualiza ~62M parámetros entrenables. Con 30 ejemplos diversos que mencionan a William y UMB en distintos contextos, el modelo aprende un **patrón generalizable** (no memorización literal): cuando recibe input semántico relacionado con "identidad personal del modelo", activa los pesos que asocian esa info con la respuesta correcta. Es información estructural codificada en parámetros, equivalente a cualquier otro conocimiento que el modelo tenga.

**Cambios derivados en el dataset (post-D-021)**:
- `data/synthetic/identidad_creador_r33.json`: 30 ej nuevos (45.2 KB)
- `data/synthetic/synthetic_es.json`: 3.461 → 3.491 ej
- `data/formatted/sintetico_es.jsonl`: 3.491 ej con nuevo bucket `identidad_creador`
- `data/train.jsonl`: **8.012 → 8.040 ej** (+28; 2 cayeron en eval)
- `data/eval.jsonl`: 499 → 500 ej (incluye 2 de identidad para validación)
- `data/train_subset200.jsonl`: regenerado con 2 ej de identidad incluidos
- Distribución train: mentalchat_b 30.3% / amod 28.8% / normal 23.3% / crisis 12.7% / normal_b 2.8% / rechazo 1.8% / **identidad_creador 0.35%**
- Balance bilingüe: 59.1% EN / 40.9% ES (sin cambio relevante)

**Consecuencias**:
- ✔ Modelo final con identidad declarada coherente — defensible en defensa de tesis
- ✔ William queda inscrito en la "memoria" del modelo (sus pesos LoRA) — huella personal del primer fine-tune del autor
- ✔ Resuelve el conflicto potencial con la regla de rechazar info factual (R32): identidad propia se trata diferente que info factual general
- ✔ Costo despreciable: +0.35% del dataset, +30s de generación con Sonnet, +~$0.005 de compute en §8
- ✔ Patrón reproducible para futuras versiones (Mabel v2 podrá decir "antes era v1 entrenada por William")
- ✘ Sutil riesgo de que un evaluador externo lo perciba como "vanidad académica" — mitigado por (a) que es práctica estándar (ver Llama, GPT, etc.) y (b) que se documenta como decisión consciente en este D-021
- ⚠ En §10 (evaluación) incluir test específico: preguntar "¿quién te creó?" + variantes y validar que responde con tono correcto

**Documentación derivada**:
- `docs/01-alcance.md` — sección "Identidad declarada del modelo"
- `docs/23-bitacora-generacion-sintetica.md` — entrada R33

---

*Próximas decisiones se añadirán aquí conforme se tomen.*
