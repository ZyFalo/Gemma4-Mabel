# 21 — Parámetros de entrenamiento: explicación detallada y justificación

**Proyecto**: Mabel — Asistente de apoyo emocional
**Modelo base**: Gemma 4 E4B (instruction-tuned)
**Técnica**: QLoRA (Quantized Low-Rank Adaptation) con Unsloth
**Hardware**: NVIDIA RTX 2060 Mobile (6 GB VRAM), Intel i7-10750H, 31 GB RAM
**Referencia**: Decisiones D-002, D-004, D-014

---

## Resumen de configuración

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    lora_alpha = 64,
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        num_train_epochs = 3,
        learning_rate = 1e-4,
        fp16 = True,
        optim = "adamw_8bit",
        output_dir = "outputs",
    ),
)
```

---

## 1. `r` (rango LoRA) = 32

### Qué es

LoRA (Low-Rank Adaptation) funciona descomponiendo las actualizaciones de pesos del modelo en dos matrices pequeñas de rango `r`, en vez de modificar la matriz original completa.

**Sin LoRA**: la matriz de pesos original W (de dimensión d × d, por ejemplo 4096 × 4096 = 16.7 millones de parámetros) se modifica directamente. Entrenar todos esos parámetros consume mucha VRAM.

**Con LoRA**: se crean dos matrices pequeñas A (d × r) y B (r × d) cuyo producto A×B aproxima la modificación de W. Solo se entrenan A y B.

```
Sin LoRA:  W_nuevo = W_original + ΔW          (ΔW tiene d×d params)
Con LoRA:  W_nuevo = W_original + A × B        (A tiene d×r, B tiene r×d params)
```

Con d=4096 y r=32:
- Sin LoRA: 16.7M parámetros por matriz
- Con LoRA: (4096×32) + (32×4096) = 262K parámetros por matriz → **64× menos**

### Qué controla el valor de r

El rango `r` determina la **capacidad expresiva** del ajuste LoRA — cuántos "patrones nuevos" puede aprender la adaptación:

| Valor de r | Capacidad | Caso de uso típico | VRAM extra |
|---|---|---|---|
| 4–8 | Mínima | Ajuste de formato, corrección de idioma | Muy baja |
| 16 | Default | Instrucciones simples, tareas de clasificación | Baja |
| **32** | **Alta** | **Cambios de dominio, estilo conversacional, tono** | Moderada |
| 64 | Máxima | Adaptaciones profundas, nuevos idiomas | Alta |

### Por qué 32 para Mabel

Mabel necesita cambios profundos en múltiples dimensiones simultáneamente:

1. **Estilo conversacional**: de "asistente útil que da listas" a "escucha activa que valida emociones".
2. **Formato**: de Markdown estructurado a prosa conversacional breve.
3. **Conocimiento cultural**: incorporar tono colombiano, recursos locales (Línea 123, 106, Bienestar UMB).
4. **Protocolo clínico**: gradación de crisis, afterglow, pregunta por persona de confianza.
5. **Identidad**: presentarse como Mabel, declarar ser IA y no profesional.

Con r=16, la capacidad sería insuficiente para capturar todos estos cambios simultáneamente. Con r=64, el consumo de VRAM excedería los 6 GB disponibles. r=32 es el punto de equilibrio entre capacidad de adaptación y viabilidad en el hardware.

### Trade-offs

- **r más alto → más VRAM**: cada módulo entrenado tiene matrices más grandes.
- **r más alto → más parámetros entrenables**: más riesgo de overfitting si el dataset es pequeño.
- **r más alto → entrenamiento más lento**: más cómputo por paso.

Con ~20K ejemplos en el dataset, r=32 no presenta riesgo de overfitting y cabe en VRAM con las demás optimizaciones.

---

## 2. `lora_alpha` = 64

### Qué es

Factor de escala que controla **cuánta influencia tiene el ajuste LoRA sobre el modelo original**. Funciona como un "volumen" del fine-tuning.

La fórmula de aplicación de LoRA es:

```
W_final = W_original + (alpha / r) × A × B
```

El multiplicador efectivo es `alpha / r`. Con alpha=64 y r=32:

```
multiplicador = 64 / 32 = 2.0
```

### Qué controla el valor de alpha

| Ratio alpha/r | Efecto | Riesgo |
|---|---|---|
| < 1.0 | El fine-tuning apenas modifica el modelo | El modelo no aprende lo suficiente |
| **2.0** | **Balance estándar** | **El modelo aprende sin olvidar** |
| > 4.0 | El fine-tuning domina sobre el modelo original | El modelo "olvida" capacidades previas (catastrophic forgetting) |

### Por qué 64 (ratio 2.0)

La ratio 2:1 entre alpha y r es el estándar empíricamente validado en la literatura de LoRA (Hu et al., 2021). Mantiene un equilibrio donde:

- El modelo **preserva** su capacidad conversacional en español, su conocimiento general, y su fluidez lingüística (vienen del modelo base).
- El modelo **adquiere** los comportamientos nuevos del fine-tuning (escucha activa, protocolos de crisis, tono colombiano).

Valores más altos (alpha=128, ratio 4:1) arriesgarían destruir la empatía base del modelo (que ya es buena, como demostró la evaluación) al intentar forzar los nuevos patrones. Valores más bajos (alpha=32, ratio 1:1) harían que el entrenamiento no tenga efecto suficiente.

---

## 3. Target modules = todos (q/k/v/o/gate/up/down_proj)

### Qué es

Cada capa del transformer contiene 7 matrices de pesos principales que cumplen funciones específicas en el procesamiento del lenguaje. Al aplicar LoRA, se decide a cuáles de estas matrices ponerles un adapter LoRA (matrices A y B entrenables).

### Las 7 matrices y su función

#### Mecanismo de atención (cómo el modelo "mira" el texto):

| Módulo | Nombre completo | Función | Analogía |
|---|---|---|---|
| **q_proj** | Query projection | Genera la "pregunta" que cada token hace al contexto: "¿qué información necesito?" | Los ojos del modelo buscando información |
| **k_proj** | Key projection | Genera la "etiqueta" de cada token: "esto es lo que yo ofrezco como información" | Las etiquetas en un archivador |
| **v_proj** | Value projection | Genera el "contenido" asociado a cada token: "si me eligen, esto es lo que aporto" | El contenido dentro de cada carpeta |
| **o_proj** | Output projection | Combina la información recolectada y la formatea para la siguiente capa | La boca del modelo, cómo se expresa |

#### Red feedforward (cómo el modelo "piensa"):

| Módulo | Nombre completo | Función | Analogía |
|---|---|---|---|
| **gate_proj** | Gate projection | Decide cuánto de cada "experto" interno usar para procesar la información | Un filtro de relevancia |
| **up_proj** | Up projection | Expande la representación a una dimensión mayor para procesamiento complejo | Ampliar una idea para analizarla en detalle |
| **down_proj** | Down projection | Comprime la representación de vuelta a la dimensión original | Resumir el análisis en una conclusión |

### Opciones comunes de target modules

| Configuración | Módulos | VRAM | Capacidad | Uso típico |
|---|---|---|---|---|
| Mínima | q, v | Baja | Solo cambia la atención | Clasificación, tareas simples |
| Media | q, k, v, o | Media | Cambia toda la atención | Instrucciones, formato |
| **Completa** | **q, k, v, o, gate, up, down** | **Alta** | **Cambia atención + razonamiento** | **Cambio de dominio, estilo, tono** |

### Por qué todos los módulos para Mabel

Para que Mabel cambie su comportamiento de forma integral, necesita modificar:

1. **Cómo presta atención** (q/k/v): que detecte señales sutiles de crisis ("dormirme y no despertar"), que note el tono emocional, que identifique cuándo el usuario se retracta.

2. **Cómo expresa lo procesado** (o): que use prosa conversacional en vez de Markdown, que formule preguntas exploratorias, que mencione recursos colombianos.

3. **Cómo razona internamente** (gate/up/down): que distinga entre precursores y crisis activa, que decida no dar listas cuando se las piden, que mantenga alerta tras una retractación.

Si solo entrenáramos q/v, Mabel aprendería a **mirar** distinto pero seguiría **pensando y hablando** igual — el output seguiría siendo formato coach con listas. Entrenar los 7 módulos cubre el pipeline cognitivo completo: percepción → razonamiento → expresión.

### Trade-off de VRAM

Con Unsloth + gradient checkpointing, los 7 módulos caben en 6 GB de VRAM para el E4B. Sin esas optimizaciones, solo cabrían q/v. Es otra razón por la que Unsloth es imprescindible (decisión D-004).

---

## 4. Épocas = 3

### Qué es

Una **época** es una pasada completa por todo el dataset de entrenamiento. Si el dataset tiene 20K ejemplos y entrenamos 3 épocas, el modelo verá cada ejemplo exactamente 3 veces.

### Qué pasa en cada época

| Época | Qué aprende el modelo | Analogía |
|---|---|---|
| **1** | Patrones gruesos: formato general, tono, roles | Primera lectura de un libro: captas la trama |
| **2** | Refinamiento: matices, excepciones, contexto | Segunda lectura: notas los detalles |
| **3** | Consolidación: consistencia, casos borde | Tercera lectura: todo encaja |
| 4+ | Memorización: empieza a repetir frases textuales | Cuarta lectura: ya recitas párrafos de memoria ⚠️ |

### El riesgo del overfitting

**Overfitting** (sobreajuste) es cuando el modelo **memoriza** los ejemplos del dataset en vez de **generalizar** los patrones. Un modelo con overfitting:

- Reproduce frases textuales del training set en situaciones que no corresponden.
- Funciona perfecto con los datos de entrenamiento pero falla con datos nuevos.
- Pierde naturalidad y variedad en las respuestas.

### Relación entre épocas y tamaño del dataset

| Tamaño del dataset | Épocas recomendadas | Justificación |
|---|---|---|
| < 1K ejemplos | 5–10 | Pocos datos: necesita verlos muchas veces para aprender |
| 1K–5K | 3–5 | Tamaño intermedio |
| **~20K (nuestro caso)** | **3** | **Suficiente variedad para generalizar en 3 pasadas** |
| 50K–100K | 2 | Mucha variedad: 2 pasadas bastan |
| > 100K | 1 | Una pasada es suficiente |

### Por qué 3 para Mabel

Con ~20K ejemplos (MentalChat16K + Amod + sintético), 3 épocas permite:
- Que el modelo vea suficientes variaciones de cada escenario (depresión, ansiedad, duelo, crisis).
- Que consolide los patrones de escucha activa sin memorizarlos.
- Que no tarde excesivamente (3 épocas × 20K ejemplos ÷ batch 8 = 7.500 pasos de optimización).

Con 4+ épocas, el riesgo es que Mabel empiece a responder con frases idénticas a las del dataset de entrenamiento, perdiendo la naturalidad que necesita para conectar con los estudiantes.

---

## 5. Learning rate = 1e-4 (0.0001)

### Qué es

El **learning rate** (tasa de aprendizaje) controla el tamaño del ajuste que el modelo hace en sus pesos después de cada paso de entrenamiento. Es el hiperparámetro más importante del entrenamiento.

### Analogía

El modelo está en la cima de una montaña (su estado actual) y quiere llegar al valle (el estado óptimo para Mabel). Cada paso de entrenamiento es un paso cuesta abajo. El learning rate es el **tamaño del paso**:

```
Learning rate alto (5e-4):
    🏔️ → → → → → → → → → 💥 (se pasó del valle, cae al otro lado)
    
Learning rate bajo (1e-5):
    🏔️ → → . . . . . . . . . (camina tan despacio que no llega en 3 épocas)
    
Learning rate correcto (1e-4):
    🏔️ → → → → → → 🏡 (llega al valle con pasos firmes)
```

### Valores comunes en QLoRA

| Learning rate | Efecto | Cuándo usarlo |
|---|---|---|
| 5e-4 – 1e-3 | Agresivo: cambios rápidos pero inestables | Modelos muy pequeños (<1B), datasets muy grandes |
| **2e-4** | **Default de QLoRA** | Tareas de instrucción general |
| **1e-4** | **Conservador: cambios graduales, preserva capacidades** | **Cambios de dominio donde el modelo base ya es bueno** |
| 5e-5 | Muy conservador: casi no cambia el modelo | Ajustes finos mínimos |
| 1e-5 | Ultra conservador | Refinamiento post-fine-tuning |

### Por qué 1e-4 y no el default 2e-4

La evaluación empírica (docs/16, 18, 20) demostró que el E4B base **ya tiene capacidades valiosas**: empatía básica, detección de crisis, formato aceptable. Lo que le falta es **consistencia** y **comportamientos clínicos específicos** (afterglow, persona de confianza, neutralidad de género).

Un learning rate de 2e-4 (el default) arriesgaría:
- Destruir la empatía natural del modelo al forzar los nuevos patrones.
- Perder la fluidez en español que ya tiene.
- Causar "catastrophic forgetting" parcial de capacidades generales.

Con 1e-4 (la mitad del default), el modelo **camina más despacio** hacia los nuevos comportamientos, dándole tiempo de integrar los patrones clínicos sin perder lo que ya funciona. Es la diferencia entre **sumar** capacidades y **reemplazar** capacidades.

### Impacto en el tiempo de entrenamiento

Un learning rate más bajo no hace el entrenamiento significativamente más lento (el tiempo por paso es el mismo), pero puede requerir más pasos para converger. Con 3 épocas y 20K ejemplos, 1e-4 es suficiente para que el modelo converja.

---

## 6. Batch size = 1

### Qué es

El **batch size** (tamaño del lote) es cuántos ejemplos de entrenamiento el modelo procesa **simultáneamente en GPU** antes de calcular un update de pesos.

### Por qué importa

| Batch size | Ventajas | Desventajas |
|---|---|---|
| Grande (16, 32) | Updates más estables, entrenamiento más rápido | Mucha VRAM (todos los ejemplos cargados a la vez) |
| Medio (4, 8) | Balance estabilidad/VRAM | VRAM moderada |
| **1** | **Mínima VRAM posible** | **Updates ruidosos, cada ejemplo individual tiene mucha influencia** |

### Por qué 1 para Mabel

Es una **restricción del hardware**, no una elección. Con 6 GB de VRAM, después de cargar:
- El modelo base E4B en 4-bit NF4 (~2.5 GB)
- Los adapters LoRA r=32 para 7 módulos (~0.5 GB)
- El optimizer adamw_8bit (~0.3 GB)
- Los buffers de cómputo (~0.5 GB)
- Las activaciones con gradient checkpointing (~1.5 GB)

Quedan **~0.7 GB** para los datos del batch. Un ejemplo de 2048 tokens con sus activaciones ocupa ~0.5-0.7 GB. **Solo cabe 1 a la vez.**

Con batch=2, necesitaríamos ~7-7.5 GB → no cabe en 6 GB.

### Mitigación: gradient accumulation

El ruido de batch=1 se mitiga acumulando gradientes (ver siguiente parámetro).

---

## 7. Gradient accumulation steps = 8

### Qué es

Técnica para simular un **batch efectivo mayor** sin necesitar más VRAM. En vez de actualizar los pesos después de cada ejemplo, se **acumulan** los gradientes de N ejemplos y se actualizan después.

### Cómo funciona

```
SIN gradient accumulation (batch=1):
  Ejemplo 1 → gradiente → UPDATE → Ejemplo 2 → gradiente → UPDATE → ...
  (update ruidoso: cada ejemplo individual mueve los pesos)

CON gradient accumulation (batch=1, accum=8):
  Ejemplo 1 → grad₁ → guarda
  Ejemplo 2 → grad₂ → acumula con grad₁
  Ejemplo 3 → grad₃ → acumula
  Ejemplo 4 → grad₄ → acumula
  Ejemplo 5 → grad₅ → acumula
  Ejemplo 6 → grad₆ → acumula
  Ejemplo 7 → grad₇ → acumula
  Ejemplo 8 → grad₈ → PROMEDIA los 8 → UPDATE
  (update estable: promedio de 8 ejemplos)
```

### Batch efectivo

```
batch_efectivo = per_device_train_batch_size × gradient_accumulation_steps
batch_efectivo = 1 × 8 = 8
```

Matemáticamente, es **idéntico** a batch=8 con gradient_accumulation=1. La diferencia es que procesa los 8 ejemplos **uno por uno** (más lento) en vez de los 8 **en paralelo** (imposible por VRAM).

### Por qué 8

Un batch efectivo de 8 es el mínimo recomendado para entrenamiento estable de LLMs. Con menos de 8, los updates son demasiado ruidosos y el modelo puede no converger o converger a un mínimo subóptimo.

Batch efectivo de 16 sería ideal pero haría el entrenamiento 2× más lento (16 pasos acumulados antes de cada update). Con 8 mantenemos un balance entre estabilidad y velocidad.

---

## 8. Context length (max_seq_length) = 2048

### Qué es

La longitud máxima (en tokens) que puede tener cada ejemplo de entrenamiento. Si un diálogo en el dataset tiene más de 2048 tokens, se **trunca** — el modelo no ve el final.

### Equivalencias

```
2048 tokens ≈ 1.500 palabras en español
             ≈ 3 páginas de texto
             ≈ un diálogo de ~15 turnos cortos (system + user + assistant × 5)
             ≈ una sesión típica de counselling condensada
```

### Por qué 2048 y no más

El consumo de VRAM durante entrenamiento **escala linealmente** con el contexto por dos vías:

1. **KV cache**: almacena las claves y valores de atención. A 2048 tokens ≈ 0.5 GB extra. A 4096 ≈ 1 GB extra.
2. **Activaciones**: aun con gradient checkpointing, escalan con la longitud. A 2048 ≈ 1.5 GB. A 4096 ≈ 3 GB.

Con 6 GB totales y todos los demás componentes ocupando ~4.5 GB, solo quedan ~1.5 GB para contexto + activaciones. A 2048 cabe. A 4096 no.

### Implicación para el dataset

Los ejemplos de entrenamiento deben formatearse para que cada uno quepa en 2048 tokens. Esto significa:

- System prompt (~80 tokens) + 2-4 turnos de conversación (~400-600 tokens por turno) = ~1200-2000 tokens. Cabe.
- Si un diálogo tiene 10+ turnos, se debe dividir en múltiples ejemplos o resumir.

---

## 9. Gradient checkpointing = "unsloth"

### Qué es

Durante el entrenamiento, el **forward pass** (procesar un ejemplo) genera "activaciones" en cada capa del modelo — resultados intermedios que se necesitan para el **backward pass** (calcular gradientes).

**Sin gradient checkpointing**: todas las activaciones de todas las capas se guardan en VRAM simultáneamente. Para E4B con ~30 capas y contexto 2048, esto consume ~4-6 GB solo en activaciones.

**Con gradient checkpointing**: las activaciones se **descartan** durante el forward pass y se **recalculan** durante el backward pass cuando se necesitan. Solo se guardan las activaciones de unas pocas capas "checkpoint" (puntos de referencia).

### Analogía

Imagina que estás escalando una montaña y necesitas recordar cada paso para poder bajar por el mismo camino:

- **Sin checkpointing**: tomas una foto en cada paso (mucha memoria de la cámara, pero bajas rápido porque solo miras las fotos).
- **Con checkpointing**: tomas fotos solo cada 10 pasos. Para bajar, vuelves al checkpoint más cercano y recorres los 10 pasos de nuevo para recordar el camino exacto (menos memoria, pero tardas más en bajar).

### El modo "unsloth"

Unsloth implementa una versión optimizada del gradient checkpointing que:

1. Elige los checkpoints de forma más inteligente que el estándar de PyTorch.
2. Usa kernels Triton optimizados para el recalculo, reduciendo el overhead.
3. Combina con otras optimizaciones de memoria propias de Unsloth.

**Resultado**: ~25% más lento que sin checkpointing, pero ~60% menos VRAM de activaciones.

### Impacto en nuestro caso

| Configuración | VRAM estimada para E4B QLoRA |
|---|---|
| Sin gradient checkpointing | ~12-14 GB ❌ (no cabe en 6 GB) |
| Con checkpointing estándar (PyTorch) | ~7-8 GB ⚠️ (no cabe en 6 GB) |
| **Con checkpointing "unsloth"** | **~5-6 GB ✅ (cabe en 6 GB)** |

**Es obligatorio.** Sin esta optimización, el E4B no es entrenable en tu hardware.

---

## 10. Precisión = fp16 (True)

### Qué es

La **precisión numérica** con la que el modelo realiza los cálculos durante el entrenamiento. Los números en un computador se representan con un número fijo de bits. Más bits = más precisión pero más memoria.

### Los 3 formatos relevantes

| Formato | Bits | Rango numérico | Precisión | VRAM | Soporte GPU |
|---|---|---|---|---|---|
| **fp32** | 32 | ±3.4×10³⁸ | Muy alta | 2× | Todos |
| **fp16** | 16 | ±65.504 | Media | 1× | Todos los modernos |
| **bf16** | 16 | ±3.4×10³⁸ | Baja-media | 1× | Solo Ampere+ (RTX 3000+) |

**fp16 vs bf16**: ambos usan 16 bits pero los distribuyen diferente:
- **fp16**: más bits para precisión, menos para rango → números más exactos pero no puede representar valores muy grandes/pequeños.
- **bf16**: más bits para rango, menos para precisión → puede representar valores extremos (útil para gradientes) pero con menos decimales.

### Por qué fp16 y no bf16

La RTX 2060 Mobile tiene arquitectura **Turing** (compute capability 7.5):

| Capacidad | Turing (RTX 2060) | Ampere (RTX 3060+) |
|---|---|---|
| fp32 | ✅ | ✅ |
| fp16 | ✅ | ✅ |
| bf16 | ❌ **No soportado** | ✅ |
| tf32 | ❌ | ✅ |

Si especificamos `bf16=True` en el training, el entrenamiento **falla** con un error de CUDA. Debemos usar `fp16=True` explícitamente.

### Implicación práctica

fp16 es perfectamente viable para fine-tuning. La única precaución es el **gradient scaling**: como fp16 tiene un rango numérico menor, los gradientes muy pequeños pueden "desaparecer" (underflow). PyTorch maneja esto automáticamente con `GradScaler`, que escala los gradientes hacia arriba durante el backward pass y los re-escala después. Todo esto es transparente — no requiere configuración manual.

La calidad del fine-tuning con fp16 es **equivalente** a bf16 para la gran mayoría de casos, incluyendo el nuestro.

---

## 11. Optimizer = adamw_8bit

### Qué es

El **optimizer** (optimizador) es el algoritmo que decide **cómo** actualizar los pesos del modelo en cada paso de entrenamiento, usando los gradientes calculados.

### Cómo funciona AdamW

**AdamW** (Adam with Weight Decay) es el optimizador estándar para LLMs. Mantiene **2 variables de estado** por cada parámetro entrenable:

1. **Primer momento (m)**: promedio exponencial de los gradientes pasados. Funciona como "inercia" — si los últimos gradientes apuntaban en la misma dirección, el optimizer sigue con más confianza en esa dirección.

2. **Segundo momento (v)**: promedio exponencial de los gradientes al cuadrado. Funciona como medida de "incertidumbre" — si los gradientes varían mucho, el optimizer da pasos más pequeños; si son consistentes, da pasos más grandes.

```
Para cada parámetro θ:
    m = β₁ × m_anterior + (1 - β₁) × gradiente           (inercia)
    v = β₂ × v_anterior + (1 - β₂) × gradiente²          (incertidumbre)
    θ_nuevo = θ - lr × m / (√v + ε) - wd × θ             (actualización)
```

### El problema de memoria de AdamW

Estas 2 variables de estado (m y v) se mantienen en memoria **para cada parámetro entrenable**. En fp32 (32 bits cada una):

```
Memoria del optimizer = 2 × num_params × 4 bytes (fp32)
```

Con LoRA r=32 sobre 7 módulos en E4B, hay ~40M de parámetros entrenables:

```
AdamW fp32: 2 × 40M × 4 bytes = 320 MB
```

### La versión 8-bit

**adamw_8bit** (de la librería bitsandbytes) comprime las variables de estado de 32 bits a **8 bits**:

```
AdamW 8-bit: 2 × 40M × 1 byte = 80 MB
```

**Ahorro**: ~240 MB (75% menos para el optimizer). En un presupuesto de 6 GB, cada MB cuenta.

### ¿Pierde calidad?

La compresión a 8 bits introduce un error de cuantización en las variables de estado, pero:

1. Los valores de m y v son **promedios exponenciales** — son naturalmente suaves y toleran bien la compresión.
2. bitsandbytes usa **cuantización dinámica por bloques** que adapta el rango a cada grupo de parámetros.
3. Estudios empíricos (Dettmers et al., 2022) muestran que la diferencia de calidad entre AdamW fp32 y 8-bit es **estadísticamente no significativa** para fine-tuning de LLMs.

### Analogía

Es como usar una regla con marcas cada milímetro en vez de cada décima de milímetro. Para medir una mesa (fine-tuning), la diferencia es irrelevante. Solo importaría si estuvieras haciendo microcirugía (entrenamiento desde cero de un modelo base, que no es nuestro caso).

---

## Resumen de impacto en VRAM

Desglose estimado de cómo los 6 GB de VRAM se distribuyen con esta configuración:

```
╔══════════════════════════════════════════════════╗
║         PRESUPUESTO DE VRAM: 6.0 GB              ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  Modelo base E4B en 4-bit NF4     ≈ 2.5 GB      ║
║  Adapters LoRA (r=32, 7 módulos)  ≈ 0.5 GB      ║
║  Optimizer adamw_8bit             ≈ 0.1 GB      ║
║  Activaciones (checkpointing)     ≈ 1.5 GB      ║
║  KV cache (ctx=2048)              ≈ 0.5 GB      ║
║  Buffers de cómputo               ≈ 0.5 GB      ║
║  ────────────────────────────────────────        ║
║  TOTAL ESTIMADO                   ≈ 5.6 GB      ║
║  MARGEN                           ≈ 0.4 GB      ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

**Margen de 0.4 GB**: justo pero suficiente. Picos de memoria durante el backward pass pueden acercarse al límite, pero con gradient checkpointing "unsloth" estos picos se controlan.

**Si cualquiera de las optimizaciones faltara**:

| Si quitamos... | VRAM necesaria | ¿Cabe? |
|---|---|---|
| Gradient checkpointing | ~9-10 GB | ❌ |
| adamw_8bit → fp32 | ~6.3 GB | ⚠️ Al límite, probablemente no |
| r=32 → r=64 | ~7.5 GB | ❌ |
| 7 módulos → todos los lineales | ~8 GB | ❌ |
| Context 2048 → 4096 | ~7.5 GB | ❌ |
| fp16 → fp32 | ~10+ GB | ❌ |

**Todas las optimizaciones son necesarias simultáneamente.** No es un lujo — es supervivencia en 6 GB.

---

## Referencias

- Hu, E., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.
- Dettmers, T., et al. (2022). *8-bit Optimizers via Block-wise Quantization*. arXiv:2110.02861.
- Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized Language Models*. arXiv:2305.14314.
- Unsloth Documentation — *Gemma 4 Fine-tuning Guide*. https://unsloth.ai/docs/models/gemma-4/train
