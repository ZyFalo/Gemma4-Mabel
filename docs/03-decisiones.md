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

*Próximas decisiones se añadirán aquí conforme se tomen.*
