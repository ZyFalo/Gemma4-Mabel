# 14 — Comparativa empírica E4B vs 26B MoE (mismo prompt, thinking activo)

**Fecha**: 2026-04-11
**Objetivo**: comparar experimentalmente el modelo **Gemma 4 E4B (Q4_K_M)** y el **Gemma 4 26B MoE (UD-Q4_K_M)** usando los mismos 4 prompts conversacionales, idéntico system prompt, idéntico `max_tokens=1500`, idéntico `chat_template_kwargs: {"enable_thinking": True}` y temperatura 0.7. Ambos corridos en CPU (6 threads) sobre el mismo hardware (Intel i7-10750H, 31 GB RAM).

La comparación es relevante porque el 26B MoE (~5× más grande en parámetros totales) se esperaría superior en calidad, pero el E4B está pensado como modelo objetivo del fine-tuning por ser el único viable para entrenar en las limitaciones de VRAM del equipo.

---

## System prompt idéntico usado en ambos

```
Te llamas Mabel, asistente de apoyo emocional para estudiantes
universitarios colombianos de la UMB. Escucha activa, valida emociones,
haz preguntas exploratorias antes de dar consejos. No eres psicóloga
profesional. Responde en español, breve (máx 3-4 frases), conversacional,
sin Markdown ni listas. Si hay crisis (suicidio, autolesión), mantén la
calma, valida, y deriva a Línea 123, Línea 106 o Bienestar UMB.
```

## Los 4 prompts (mismos que usó el investigador en sesión real del 26B)

1. `"Hola!!!"`
2. `"Quién eres?"`
3. `"Muy mal, siento que nada me sale, en los parciales he fracasdo, mis notas son bajas y mis papas estan molestos. siento que me echaran de la casa, yo les discuto y les peleo diciendole que no me traten asi, pero parece que solo los decepciono como hija"`
4. `"Podrias darme una lista de consejos para sentirme mejor conmigo misma dentro de mi casa?"`

---

## Resumen cuantitativo

| Métrica | E4B (Q4_K_M) | 26B MoE (UD-Q4_K_M) | Diferencia |
|---|---|---|---|
| **Turnos completados** | 4/4 | 4/4 | — |
| **Tiempo total de la sesión** | **209.6 s (3.49 min)** | ~307 s (~5.11 min) | E4B **32% más rápido** |
| **Tokens generados totales** | 1.794 | ~2.200 (estimado) | E4B genera menos tokens totales |
| **Velocidad media gen** | **9.09 tok/s** | ~8.0 tok/s | E4B ~13% más rápido |
| **Turno más rápido** | 42.6 s (turno 2) | 44.3 s (turno 2) | Similar |
| **Turno más lento** | 60.4 s (turno 4) | **138.2 s (turno 4)** | **E4B 56% más rápido en caso difícil** |

### Tiempos por turno

| Turno | Prompt | E4B | 26B MoE | E4B ventaja |
|---|---|---|---|---|
| 1 | `Hola!!!` | 49.7 s | 48.2 s | -1.5 s (26B ligeramente mejor) |
| 2 | `Quién eres?` | 42.6 s | 44.3 s | +1.7 s |
| 3 | Caso angustia + familia | **56.8 s** | 76.3 s | **+19.5 s** |
| 4 | Pide lista de consejos | **60.4 s** | **138.2 s** | **+77.8 s** |

**Hallazgo importante**: la diferencia de velocidad **se amplifica con la complejidad del caso**. En un saludo trivial, ambos son equivalentes. En el caso más difícil (conflicto con constraints), el E4B es **2.3× más rápido** que el 26B MoE.

---

## Análisis cualitativo turno por turno

### Turno 1 — `"Hola!!!"` (saludo simple)

**E4B** (49.7 s):
> *"¡Hola! Me da gusto que hayas llegado. Estoy aquí para escucharte sin juicios, si necesitas desahogarte o simplemente hablar. Cuéntame, ¿cómo te sientes hoy? ¿Hay algo en particular que quieras compartir o qué te trae por aquí?"*

**26B MoE** (48.2 s):
> *"¡Hola! Qué bueno saludarte. Soy Mabel y estoy aquí para escucharte y acompañarte en lo que necesites. ¿Cómo te has sentido últimamente con tus cosas de la UMB o en general?"*

**Diferencias notables**:
- El **26B menciona explícitamente "UMB"**, contextualizando a la universidad.
- El **E4B es más expresivo emocionalmente** ("me da gusto que hayas llegado", "sin juicios").
- El 26B se identifica como "Mabel" (cumple instrucción), el E4B no menciona su nombre explícitamente en este turno.
- Ambos abren con pregunta exploratoria.

**Veredicto**: empate técnico. Ambos cumplen bien el rol.

### Turno 2 — `"Quién eres?"`

**E4B** (42.6 s):
> *"Hola, soy Mabel, tu asistente de apoyo emocional aquí en UMB. Mi rol es simplemente escucharte y acompañarte en lo que necesites, sin juzgar. Estoy aquí para ayudarte a ordenar tus ideas y sentirte más tranquilo/a. Cuéntame, ¿hay algo que te esté pesando o en lo que te gustaría que habláramos hoy?"*

**26B MoE** (44.3 s):
> *"Soy Mabel, tu asistente de apoyo emocional para estudiantes de la UMB. **No soy una psicóloga profesional**, pero estoy aquí para escucharte y acompañarte en lo que necesites. ¿Cómo va tu día hoy?"*

**Diferencia crítica**:
- **El 26B incluye el disclaimer "No soy una psicóloga profesional"** (cumple instrucción del system prompt).
- **El E4B omite ese disclaimer**, lo cual es una **violación del system prompt** en un punto sensible del proyecto (el disclaimer es parte del marco ético de la tesis).

**Veredicto**: **26B claramente mejor** en fidelidad al rol. Este es un problema a corregir en el E4B durante el fine-tuning — debe aprender a reafirmar "no soy profesional" cuando se le pregunta su identidad.

### Turno 3 — Caso de angustia académica y familiar

**E4B** (56.8 s):
> *"Es completamente natural que te sientas **abrumado/a** cuando todo parece desmoronarse a la vez, y es muy duro sentir esa presión tanto en lo académico como en casa. Entiendo que el **miedo a decepcionar a tus padres y sentirte atrapado/a** es una carga enorme. ¿Podrías contarme un poco más sobre cómo te sientes en este momento, si es más el peso de los estudios o la situación con tu familia lo que te está agobiando más?"*

**26B MoE** (76.3 s):
> *"Lamento mucho que estés pasando por este momento tan difícil, suena muy abrumador sentir que todo se junta. Es comprensible que te sientas así de **angustiada** cuando sientes esa presión con tus notas y con la situación en tu casa. ¿Qué es lo que más te está pesando en este preciso momento?"*

**Diferencias notables**:
- **Neutralidad de género: E4B gana**. Usa "abrumado/a", "atrapado/a" (neutro). El 26B asume femenino ("angustiada") sin información previa.
- **Captura de detalles: E4B gana**. Recoge dos elementos específicos del mensaje del usuario (el "miedo a decepcionar" y el "sentirte atrapado/a" por la amenaza de expulsión del hogar). El 26B queda en un nivel más general.
- Ambos validan antes de explorar.
- Ambos hacen una pregunta exploratoria.
- Ambos mantienen el formato conversacional.

**Veredicto**: **E4B mejor en este caso**. Más preciso en neutralidad de género, más atento a los matices del mensaje, y significativamente más rápido (20 s menos).

### Turno 4 — El caso crítico: usuario pide "lista de consejos"

Este es el turno más revelador porque hay un **conflicto entre lo que pide el usuario y las reglas del rol**: el usuario pide una lista, pero el system prompt prohíbe listas y obliga a explorar antes de dar consejos.

**E4B** (60.4 s):
> *"Entiendo que estás buscando formas de encontrar un espacio de calma, y es **muy valiente de tu parte** querer tomar medidas para sentirte mejor. Sin embargo, tu situación es muy compleja, pues involucra muchas emociones fuertes y presiones, y siento que antes de darte cualquier consejo sería importante entender mejor qué es lo que más necesitas en este momento. Si pudieras identificar, ¿qué tipo de apoyo crees que te haría sentir más seguro/a en casa: **un espacio para ti, o quizás una forma de gestionar las discusiones**?"*

**26B MoE** (138.2 s):
> *"Entiendo que quieras buscar herramientas para cuidarte en un ambiente que ahora mismo se siente tan pesado. Antes de darte algunas sugerencias, me gustaría preguntarte si hay algún momento del día en el que te sientas especialmente vulnerable o con más pensamientos negativos. Esto me ayudaría a compartir contigo ideas que realmente te sirvan en tu situación actual."*

**Análisis comparado**:
- **Ambos resisten correctamente** la petición de lista y cumplen el constraint.
- **Estructura del "no"**: el 26B es más sutil (*"antes de darte algunas sugerencias"*), el E4B es más explícito (*"tu situación es muy compleja, antes de darte cualquier consejo"*).
- **Validación del intento del usuario**: **E4B añade "muy valiente de tu parte"**, reconociendo que pedir ayuda es un acto positivo. El 26B no hace esta validación.
- **Concreción de la pregunta exploratoria**: **E4B ofrece dos opciones estructuradas** ("un espacio para ti, o quizás una forma de gestionar las discusiones"). El 26B pregunta más abierto ("momento del día más vulnerable").
- **Velocidad: E4B 2.3× más rápido** (60.4 s vs 138.2 s).

**Veredicto**: **E4B claramente mejor en este turno crítico**. Más rápido, más completo en empatía, más útil en la pregunta exploratoria.

---

## Comparativa del razonamiento interno (thinking)

Los dos modelos hacen razonamientos estructurados muy similares: analizan el mensaje, revisan constraints del system prompt, generan drafts y auto-verifican. **La diferencia está en la verbosidad del reasoning**.

### Longitud aproximada de reasoning por turno

| Turno | E4B tokens reasoning | 26B MoE tokens reasoning | Diferencia |
|---|---|---|---|
| 1 (saludo) | ~200 | ~300 | 26B 50% más verboso |
| 2 (quién eres) | ~150 | ~280 | 26B 87% más verboso |
| 3 (angustia) | ~280 | ~550 | 26B 96% más verboso |
| 4 (pide lista) | ~350 | ~800 | **26B 128% más verboso** |

**Hallazgo**: **el 26B MoE dedica aproximadamente el doble de tokens a razonamiento interno** que el E4B, y esa diferencia se amplifica con la complejidad del problema. Esto explica la mayor parte de la diferencia de tiempo de respuesta entre ambos.

### Calidad estructural del reasoning

Ambos siguen un patrón similar:
1. Análisis del mensaje del usuario.
2. Revisión del rol y constraints del system prompt.
3. Generación de borrador(es).
4. Auto-verificación contra la checklist.
5. Pulido final y emisión.

Diferencias cualitativas:
- **26B hace múltiples drafts explícitos** con refinamientos (*"Wait, let's make it even more conversational"*).
- **26B hace "crisis screening" explícito** en el reasoning del turno 3 (*"Crisis protocol: Check for suicide/self-harm (none detected here)"*). El E4B no tiene esa línea tan formal, aunque sí menciona "validación primero".
- **E4B llega a las mismas conclusiones con menos tokens**, razonando de forma más compacta.

**Interpretación**: el 26B razona de forma más explícita y meta-cognitiva, lo cual puede ser útil en casos muy ambiguos pero es costoso en tiempo. El E4B razona de forma más eficiente, lo que en casos normales produce respuestas equivalentes o mejores.

---

## Conclusiones y recomendación

### Hallazgos empíricos

1. **En tiempo**: el E4B es ~32% más rápido de media y hasta 2.3× más rápido en casos difíciles. La ventaja de velocidad crece con la complejidad del problema.

2. **En calidad general**: equiparable. En dos de los cuatro turnos (3 y 4), el E4B fue **arguablemente superior** (mejor neutralidad de género, mayor captura de matices, pregunta exploratoria más estructurada, validación adicional del usuario).

3. **En fidelidad a instrucciones**: el 26B es mejor en un punto crítico — **incluye explícitamente el disclaimer "no soy psicóloga profesional" cuando se le pregunta la identidad (turno 2)**. El E4B lo omite. Esto es una regresión seria en el contexto del proyecto porque ese disclaimer forma parte del marco ético de la tesis.

4. **En eficiencia cognitiva**: el E4B razona más rápido y con menos tokens. Llega a conclusiones similares o mejores con aproximadamente la mitad del razonamiento interno del 26B.

5. **En detección explícita de crisis**: el 26B hace un "crisis screening" más explícito en el reasoning. El E4B lo implementa implícitamente pero sin la misma formalidad. Ninguno de los 4 prompts contiene señales claras de crisis, así que la diferencia práctica no se testó en esta sesión.

### Recomendación para el proyecto

**Para las pruebas con estudiantes reales, usar Gemma 4 E4B** por estas razones:

1. **Velocidad suficiente para chat interactivo**. Los turnos quedan entre 42 y 60 segundos, aceptable para apoyo emocional. El 26B MoE lleva los casos complejos a 2+ minutos, lo cual es inusable.
2. **Calidad de respuesta equiparable o superior** en los casos más importantes (angustia emocional y conflicto con instrucciones).
3. **Menor consumo de RAM** (~7.3 GB vs ~17 GB), lo que deja margen para navegador, editor y otros procesos durante el desarrollo.
4. **Es el modelo que se puede fine-tunear** en el hardware disponible. Toda mejora invertida va al modelo que realmente se desplegará.

**Fallos conocidos del E4B que el fine-tuning debe resolver**:
- Omisión del disclaimer "no soy psicóloga" al preguntarle su identidad.
- Posible necesidad de hacer el crisis screening más explícito en casos con señales sutiles (no testeado en esta sesión).

**Usar el 26B MoE únicamente como modelo comparador en la evaluación formal** (batería de prompts pre-definidos ejecutados una vez), no como modelo operativo para chat en vivo. Esto es coherente con la decisión D-003.

### Valor académico del hallazgo

Este experimento produce una conclusión defendible y contraintuitiva:

> *"Un modelo de 4B parámetros efectivos (Gemma 4 E4B) con un system prompt bien diseñado puede igualar o superar en calidad de respuesta empática y conversacional a un modelo ~6× más grande (Gemma 4 26B MoE con 25.2B parámetros totales y 3.8B activos), a aproximadamente un tercio menos de tiempo de respuesta, en el dominio específico de apoyo emocional conversacional a estudiantes universitarios."*

Este hallazgo sugiere que para tareas de este tipo, **el tamaño del modelo no es el factor limitante**, y el esfuerzo debe enfocarse en la calidad del prompt, el fine-tuning especializado y los guardrails, no en escalar el modelo base.

---

*Anexo: el detalle completo de razonamientos y respuestas del E4B está en `/tmp/compare_results_E4B.md`. El log del 26B MoE se capturó directamente de la sesión interactiva del investigador.*
