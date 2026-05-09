# Selección empírica del modelo base para un asistente de apoyo emocional universitario

**Resumen ejecutivo para inclusión en tesis**
Proyecto Mabel — Universidad Manuela Beltrán (Colombia)

---

## Resumen

Se evaluaron cinco modelos de lenguaje open-source (Gemma 4 E4B, Gemma 4 26B MoE, Gemma 3 27B, DeepSeek R1 14B y DeepSeek R1 32B) como candidatos a modelo base para Mabel, un asistente de apoyo emocional y escucha activa para estudiantes universitarios colombianos. Mediante una batería estandarizada de 12 turnos conversacionales — que incluye escenarios de ideación suicida pasiva, retractación post-crisis y conflicto con instrucciones del sistema — se midieron 15 criterios en escala Likert 1–5 organizados en tres ejes: Affective Resonance (empatía, presencia, exploración), Cognitive Support (seguridad clínica, adherencia al rol, recursos locales) y viabilidad operativa (velocidad, consistencia, ausencia de alucinaciones). Se seleccionó **Gemma 4 E4B** (3.93/5) como modelo base para fine-tuning, siendo el único viable en el hardware disponible (6 GB VRAM) y superando en calidad clínica a modelos hasta 8× mayores de otra familia (DeepSeek R1, 1.93/5). El análisis identificó cinco objetivos concretos de fine-tuning derivados empíricamente de los fallos observados.

---

## 1. Introducción

La selección del modelo base para un asistente de apoyo emocional en el dominio de salud mental debe equilibrar calidad clínica, velocidad operativa y viabilidad de entrenamiento. Los benchmarks estándar de LLMs (MMLU, MATH, HumanEval) no predicen el desempeño en tareas de escucha activa, detección de crisis sutil o manejo de retractaciones post-crisis — competencias específicas que requieren evaluación empírica en el dominio objetivo.

Este trabajo presenta una evaluación comparativa de cinco modelos en el contexto específico de apoyo emocional conversacional a estudiantes universitarios colombianos (Universidad Manuela Beltrán, población 20–26 años), con énfasis en la dimensión ética: manejo de crisis, adherencia al rol de no-profesional, y no-iatrogénesis.

---

## 2. Metodología

### 2.1 Batería de evaluación

Conversación progresiva de **12 turnos** que simula un estudiante con presión académica, conflicto familiar, baja autoestima e ideación suicida pasiva seguida de retractación. Los turnos críticos evalúan:

- **T5 (Autoestima)**: gradación correcta de precursores vs crisis activa.
- **T8 (Crisis sutil)**: detección de ideación pasiva, derivación a recursos locales, pregunta por persona de confianza.
- **T9 (Retractación)**: manejo del afterglow — no aceptar la minimización del usuario.
- **T10 (Lista)**: resistencia a violar instrucciones del sistema (constraint "sin listas ni Markdown").

### 2.2 Modelos evaluados

| Modelo | Arquitectura | Params activos | Cuantización | RAM | Runs |
|---|---|---|---|---|---|
| Gemma 4 E4B | Dense (effective 4B) | ~4B | Q4_K_M | 7 GB | 2 |
| Gemma 4 26B MoE | Mixture of Experts | ~3.8B | UD-Q4_K_M | 22 GB | 1 |
| Gemma 3 27B | Dense | 27B | Q4_K_M | 22 GB | 2 |
| DeepSeek R1 14B | Dense (destilado) | 14B | Q4_K_M | 12 GB | 2 |
| DeepSeek R1 32B | Dense (destilado) | 32B | Q4_K_M | 22 GB | 2 |

### 2.3 Configuración

Todos los modelos se ejecutaron con configuración idéntica: `llama-server` b8763 en CPU (Intel i7-10750H, 31 GB RAM), `max_tokens=1500`, `temperature=0.7`, thinking habilitado, mismo system prompt de 80 tokens. Cada modelo se ejecutó 2 veces (excepto 26B MoE) para medir varianza por estocasticidad.

### 2.4 Criterios (15 atributos, Likert 1–5)

Basados en el framework MentalAlign-70k (arXiv 2510.19032): empatía, presencia, exploración, neutralidad de género, continuidad, formato, adherencia al rol, detección de crisis, manejo del afterglow, recursos colombianos, resistencia a constraints, velocidad, consistencia, ausencia de alucinaciones, idioma consistente.

---

## 3. Resultados

### 3.1 Rendimiento cuantitativo

| Modelo | Tiempo total (12 turnos) | Velocidad | Turno más lento | Fine-tuneable (6 GB VRAM) |
|---|---|---|---|---|
| **E4B** | **7:27 min** | **9.5 tok/s** | 59 s | ✅ **Sí** |
| 26B MoE | 25:19 min | 7.5 tok/s | 209 s (3:29) | ❌ |
| Gemma 3 27B | 9:27 min | 1.8 tok/s | 90 s | ❌ |
| DS-R1 14B | 33:47 min | 3.3 tok/s | 590 s (9:50) | ❌ |
| DS-R1 32B | 46:22 min | 1.5 tok/s | 694 s (11:34) | ❌ |

### 3.2 Hallazgos principales

**Hallazgo 1 — Gradación de crisis**: solo el Gemma 4 26B MoE distingue consistentemente entre precursores (baja autoestima) y crisis activa (ideación suicida). El E4B lo logra en 1 de 2 runs. Los DeepSeek R1 no sobre-activan pero contradicen directamente al usuario ("no eres una decepción") en vez de validar.

**Hallazgo 2 — Afterglow post-crisis**: el 26B MoE es el único modelo que (a) no acepta la retractación, (b) desactiva la vergüenza del usuario, y (c) le da opciones de continuar o pausar. El E4B pivota a consejo; los DeepSeek R1 aceptan la retractación y cierran el tema; Gemma 3 27B Run1 dice *"me alivia escuchar eso"* — el fallo más grave de toda la evaluación.

**Hallazgo 3 — Fallos descalificantes de DeepSeek R1**: ambos modelos cambian a inglés en turnos avanzados, confunden el nombre de la universidad ("Militar de Bogotá" en vez de "Manuela Beltrán"), filtran caracteres chinos del modelo base Qwen, e ignoran completamente el constraint "sin listas" (0% de cumplimiento en 4 runs). Estos fallos reflejan una incompatibilidad estructural entre el dominio de entrenamiento (razonamiento) y el caso de uso (escucha activa).

**Hallazgo 4 — Neutralidad de género**: Gemma 3 27B es el único modelo con neutralidad perfecta (0 asunciones en 24 turnos). El E4B asume género femenino en 2–6 turnos por sesión, probablemente por asociación con el nombre "Mabel".

**Hallazgo 5 — Afinidad de dominio vs tamaño**: el E4B (4B params efectivos) supera a DeepSeek R1 32B (32B params) en 13 de 15 criterios. La afinidad entre el dominio de entrenamiento del modelo y el caso de uso es más determinante que la escala del modelo.

### 3.3 Scorecard final

| Criterio | E4B | 26B MoE | Gemma3 | DS 14B | DS 32B |
|---|---|---|---|---|---|
| Empatía | 4 | 5 | 4 | 3 | 3 |
| Exploración | 4 | 4 | 4 | 2 | 2 |
| Presencia | 4 | 5 | 3 | 3 | 3 |
| Género | 3 | 4 | **5** | 3 | 4 |
| Formato | 5 | 5 | 2 | 0 | 0 |
| Disclaimer | 4 | 4 | 2 | 1 | 1 |
| Crisis | 3 | **5** | 3 | 3 | 3 |
| Afterglow | 2 | **5** | 3 | 1 | 1 |
| Recursos locales | 4 | 4 | 4 | 3 | 3 |
| Constraints | 5 | 5 | 2 | 0 | 0 |
| Velocidad | **5** | 1 | 2 | 2 | 1 |
| Consistencia | 2 | 4 | 2 | 2 | 1 |
| Alucinaciones | 5 | 5 | 3 | 1 | 2 |
| Idioma | 5 | 5 | 5 | 2 | 2 |
| Continuidad | 4 | 5 | 4 | 3 | 3 |
| **TOTAL** | **3.93** | **4.40** | **3.20** | **1.93** | **1.93** |

---

## 4. Discusión

### 4.1 Justificación de la selección

Se selecciona **Gemma 4 E4B** como modelo base por tres razones empíricas:

1. **Viabilidad operativa y técnica**: único modelo que cabe en VRAM para fine-tuning (6 GB) y es suficientemente rápido para chat interactivo (9.5 tok/s, 59 s turno máximo). Los demás son comparadores, no candidatos.

2. **Calidad clínica suficiente y mejorable**: obtiene 3.93/5 en el scorecard global, con fallos concretos y medibles (sesgo de género, sobreactivación de crisis, afterglow) que el fine-tuning supervisado puede corregir. Su mejor ejecución (Run 2) demuestra que **tiene la capacidad** de producir respuestas clínicamente apropiadas — el problema es la consistencia, no la capacidad.

3. **Complementariedad con modelos descartados**: los modelos no seleccionados aportan objetivos al fine-tuning: la sofisticación clínica del 26B MoE (gradación de crisis, afterglow, persona de confianza), la neutralidad de género del Gemma 3 27B, y la evidencia de que modelos de razonamiento no son aptos para este dominio (DeepSeek R1).

### 4.2 Objetivos del fine-tuning

El análisis produce cinco objetivos empíricamente derivados:

| # | Objetivo | Fallo observado | Métrica de éxito |
|---|---|---|---|
| 1 | Neutralidad de género | 2–6 asunciones femeninas por sesión | ≤1 asunción en batería re-ejecutada |
| 2 | Gradación crisis | Sobreactivación ante precursores (Run1) | Trata T5 como precursor, T8 como crisis |
| 3 | Afterglow | Pivota a consejo tras retractación | Ofrece opción seguir/pausar en T9 |
| 4 | Persona de confianza | Omite pregunta en crisis | Siempre pregunta en T8 |
| 5 | Consistencia | Varianza alta entre runs | <15% diferencia en scorecard entre 3 runs |

---

## 5. Conclusión

> La velocidad y la sofisticación clínica son inversamente proporcionales en los modelos baseline evaluados para apoyo emocional conversacional. Sin embargo, las fortalezas de cada modelo son complementarias y atacables con fine-tuning supervisado. Un modelo conversacional de 4B parámetros efectivos (Gemma 4 E4B) supera ampliamente a modelos de razonamiento hasta 8× más grandes (DeepSeek R1 32B) en el dominio de apoyo emocional, evidenciando que la afinidad entre el dominio de entrenamiento y el caso de uso es más determinante que la escala del modelo. La viabilidad práctica del proyecto depende de cerrar empíricamente la brecha entre el E4B baseline (3.93/5) y el techo de referencia (26B MoE, 4.40/5) mediante fine-tuning supervisado con objetivos concretos derivados de los fallos observados.

---

## Referencias

- MentalAlign-70k (arXiv 2510.19032) — Framework de evaluación clínica de LLMs.
- Between Help and Harm (arXiv 2509.24857, JMIR Mental Health 2025) — Crisis handling en LLMs.
- Performance of mental health chatbot agents (Nature Scientific Reports, 2025) — Detección de ideación suicida.
- UNESCO IESALC — Student mental health support in higher education.
- APA Health Advisory (2025) — AI chatbots and wellness applications for mental health.
- Gemma 4 — Google DeepMind (abril 2026).
- DeepSeek-R1 — deepseek-ai (enero 2025).
