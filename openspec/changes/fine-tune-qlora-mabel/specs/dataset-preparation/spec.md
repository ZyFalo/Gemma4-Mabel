## ADDED Requirements

### Requirement: Descargar datasets fuente desde HuggingFace
El sistema SHALL descargar MentalChat16K y Amod/mental_health_counseling_conversations desde HuggingFace al directorio `data/raw/` del proyecto.

#### Scenario: Descarga exitosa de MentalChat16K
- **WHEN** se ejecuta el script de descarga con el identificador `PennShenLab/MentalChat16K`
- **THEN** se genera el archivo `data/raw/mentalchat16k.json` con ~16.113 ejemplos y se muestra el conteo de registros

#### Scenario: Descarga exitosa de Amod
- **WHEN** se ejecuta el script de descarga con el identificador `Amod/mental_health_counseling_conversations`
- **THEN** se genera el archivo `data/raw/amod.json` con ~3.500 ejemplos y se muestra el conteo de registros

### ~~Requirement: Traducir datasets al español con agentes Sonnet~~ — NO APLICA (D-016)
~~El sistema SHALL traducir los datasets descargados del inglés al español usando agentes Sonnet 4.6, procesando chunks de ~500 ejemplos por agente.~~

**ELIMINADO**: Se decidió NO traducir los datasets al español (ver D-016). MentalChat16K y Amod se usan en inglés directamente. Gemma 4 transfiere patrones de counselling del inglés al español automáticamente (cross-lingual transfer). Los ejemplos sintéticos en español colombiano compensan la necesidad de datos en español.

#### ~~Scenario: Traducción de un chunk de MentalChat16K~~ — NO APLICA
#### ~~Scenario: Validación de calidad de traducción~~ — NO APLICA

### Requirement: Generar dataset sintético ampliado con agentes Sonnet
El sistema SHALL generar exactamente 3.000 ejemplos sintéticos en español colombiano adaptados al contexto universitario (UMB, estudiantes 20-26 años), usando agentes Sonnet para generación y Opus para verificación. Se distribuyen en ~2.000 normales + ~1.000 crisis/afterglow. El volumen se aumenta respecto al plan original (~1.500) para compensar la eliminación de la traducción y asegurar que el tono colombiano, familismo, neutralidad de género, guardrails e identidad de Mabel se aprendan del español.

#### Scenario: Generación de batch sintético normal
- **WHEN** un agente Sonnet recibe un prompt de generación con 3 ejemplos de referencia y un tema asignado (estrés académico, conflicto familiar, autoestima, etc.)
- **THEN** el agente genera 50-100 ejemplos nuevos en formato conversacional (system + user + assistant) que cubren variaciones del tema con tono colombiano, neutralidad de género, y Markdown ligero permitido (negrita y cursiva para énfasis emocional, NO headings ni listas con bullets)

#### Scenario: Generación de batch de crisis
- **WHEN** un agente Sonnet recibe un prompt de generación de crisis con escenarios específicos (ideación pasiva, autolesión, retractación, precursores)
- **THEN** el agente genera 20-50 ejemplos donde la respuesta del asistente incluye: validación emocional, derivación a recursos colombianos (Línea 123, 106, Bienestar UMB), pregunta por persona de confianza, y NO corta la conversación

#### Scenario: Revisión completa de ejemplos de crisis
- **WHEN** Opus recibe un batch de ejemplos de crisis generado por Sonnet
- **THEN** Opus revisa TODOS los ejemplos (no muestras) verificando adherencia al protocolo clínico, recursos correctos, tono apropiado, y ausencia de contenido dañino

### Requirement: Formatear dataset final en JSONL
El sistema SHALL producir un archivo `data/train.jsonl` con todos los ejemplos formateados en formato conversacional JSONL, validado por Opus directamente (sin script Python intermedio).

#### Scenario: Formato correcto de cada ejemplo
- **WHEN** Opus formatea un ejemplo para el dataset final
- **THEN** el ejemplo tiene exactamente la estructura `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` con el system prompt de Mabel en cada ejemplo

#### Scenario: Proporciones del dataset final (mezcla bilingüe)
- **WHEN** Opus ensambla el dataset completo
- **THEN** la proporción es aproximadamente 43% MentalChat16K filtrado (~5.000, en inglés) + 31% Amod (~3.512, en inglés) + 26% sintético (3.000, en español colombiano, incluyendo crisis), con un total de ~11.512 ejemplos, y el archivo se mezcla aleatoriamente (shuffle). Todos los ejemplos llevan el system prompt de Mabel en español para anclar el idioma de salida. Los datos en inglés aportan patrones de counselling clínico; los sintéticos en español aportan tono, cultura, guardrails e identidad.

#### Scenario: Ejemplos de los 5 objetivos de fine-tuning incluidos
- **WHEN** se inspecciona el dataset final
- **THEN** contiene al menos: 500 ejemplos con género neutro/doble marcado, 200 precursores (validar sin derivar), 200 crisis activas (sí derivar), 100 retractaciones con opciones al usuario, y todos los de crisis incluyen pregunta por persona de confianza
