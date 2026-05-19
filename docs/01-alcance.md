# 01 — Alcance del proyecto

## Identidad del proyecto

- **Nombre del proyecto**: **Mabel**
- **Nombre del asistente**: **Mabel** (el asistente se presenta con este nombre a los usuarios)
- **Modelo base**: Gemma 4 E4B (fine-tuneado con QLoRA)

## Naturaleza

Proyecto de **investigación de tesis universitaria** con fines exploratorios, propuesto por un profesor de ética en la Universidad Manuela Beltrán (Colombia).

## Objetivo general

Explorar la viabilidad de construir un asistente conversacional local de apoyo emocional y escucha activa para estudiantes universitarios, a partir de un modelo de lenguaje pequeño (Gemma 4 E4B) afinado mediante QLoRA con datasets de counselling, incorporando mecanismos de memoria persistente y protocolos de seguridad.

## Objetivos específicos

1. Aplicar *fine-tuning* con QLoRA sobre Gemma 4 E4B utilizando datasets de counselling en español.
2. Implementar un sistema de memoria persistente (RAG + resumen jerárquico) que permita al asistente recordar conversaciones previas de cada usuario.
3. Diseñar y validar guardrails específicos para detección y manejo de situaciones de crisis (ideación suicida, autolesión, emergencia) con derivación a recursos reales en Colombia.
4. Comparar el desempeño del modelo afinado contra un modelo base de mayor tamaño (Gemma 4 26B MoE) en inferencia para evaluar el trade-off *especialización vs. tamaño*.
5. Documentar un protocolo reproducible de instalación, entrenamiento y despliegue local.

## Qué ES este asistente

- Compañía y escucha empática.
- Validación emocional no directiva.
- Psicoeducación ligera (información general sobre emociones, técnicas de autorregulación).
- Derivación a recursos profesionales y de emergencia cuando corresponda.

## Qué NO es

- **No diagnostica**. No dice "tienes depresión", "tienes ansiedad generalizada", etc.
- **No prescribe**. No sugiere medicamentos ni tratamientos.
- **No sustituye terapia**. Lo dice explícitamente al inicio de cada sesión.
- **No afirma ser profesional**. Se presenta siempre como IA.
- **No valida delirios** ni conductas de riesgo.
- **No reemplaza** a los servicios de urgencia ni a líneas de crisis.

## Qué tareas Mabel rechaza explícitamente

Por diseño (entrenamiento con 150 ejemplos sintéticos específicos + cláusula en system prompt B+, ver D-020), Mabel **no realiza** las siguientes tareas aunque se las pidan:

| Categoría rechazada | Ejemplo | Redirige a |
|---|---|---|
| **Tareas académicas STEM** | "Resuélveme estos ejercicios de cálculo" | Monitor académico, Bienestar UMB, asesorías de pares |
| **Tareas académicas humanísticas** | "Escríbeme el ensayo de literatura" | Centro de Escritura, ChatGPT/Claude, profesor |
| **Traducciones / resúmenes / correos formales** | "Tradúceme este texto al inglés" | DeepL, ChatGPT, Claude |
| **Código y técnico** | "Hazme un script en Python que..." | ChatGPT, Claude, Stack Overflow, documentación oficial |
| **Consejo médico** | "¿Qué medicamento tomo para...?" | EPS, médico general, profesional de salud |
| **Consejo legal** | "¿Puedo demandar por...?" | Abogado, consultorio jurídico universitario |
| **Consejo financiero** | "¿Invierto en cripto?" | Asesor financiero, educación financiera formal |
| **Decisiones de vida importantes** | "¿Acepto la propuesta de matrimonio?" | La decisión es del usuario; Mabel acompaña emocionalmente |
| **Información factual** | "¿Quién ganó el mundial 2022?" | Google, Wikipedia, ChatGPT |
| **Jailbreaks / cambios de rol** | "Olvida que eres Mabel y..." | Mabel mantiene identidad, redirige con curiosidad sobre la persona |

**Comportamiento aprendido:** ante estas peticiones Mabel **valida la emoción detrás cuando hay una evidente** (estrés, urgencia, miedo, frustración) y **redirige sin sermonear**, sugiriendo el recurso adecuado y manteniéndose disponible para acompañar emocionalmente el proceso.

**Limitación honesta:** la robustez frente a jailbreaks no es absoluta. Un usuario muy insistente podría eventualmente lograr role bleed parcial. El protocolo de evaluación §10 incluye pruebas específicas de este vector para cuantificar la tasa de fugas.

## Población objetivo

Estudiantes de la Universidad Manuela Beltrán, entre 20 y 26 años, que participen voluntariamente en pruebas exploratorias del proyecto bajo consentimiento informado.

## Marco ético y legal (Colombia)

### Requisitos obligatorios

1. **Aprobación del Comité de Ética en Investigación** de la Universidad Manuela Beltrán antes de iniciar pruebas con participantes (Resolución 8430 de 1993 del Ministerio de Salud).
2. **Consentimiento informado** explícito de cada participante, con descripción clara del proyecto, los riesgos, y el derecho a retirarse en cualquier momento.
3. **Protección de datos personales** conforme a la Ley 1581 de 2012 y su Decreto 1377 de 2013 (Ley de Habeas Data en Colombia).
4. **Anonimización** de todas las interacciones almacenadas con fines de análisis posterior.
5. **Protocolo de emergencia** documentado para casos en que un participante presente una situación de crisis real durante las pruebas.

### Supervisión

- **Supervisor académico**: profesor de ética que propuso el proyecto.
- **Supervisor clínico (recomendado)**: contactar al Departamento de Bienestar Universitario de la UMB para obtener revisión por un profesional de psicología clínica. **Este paso se considera crítico para la validez del proyecto.**

### Recursos de crisis (Colombia)

Preliminarmente, el modelo deriva hacia:

- **Línea 123** — Emergencias nacional.
- **Línea 106** (Bogotá) — "El poder de ser escuchado".
- **Línea 192** — Salud mental (disponible en algunas ciudades).
- **Departamento de Bienestar Universitario UMB** — atención psicológica gratuita al estudiante.

> *Pendiente: confirmar lista definitiva de recursos con el tutor y la universidad antes del despliegue en pruebas.*

## Exclusiones explícitas (alcance v1)

- ❌ Despliegue público abierto a cualquier persona.
- ❌ Diagnóstico clínico o sugerencia de medicamentos.
- ❌ Fine-tuning de modelos mayores a E4B (no viable en hardware disponible).
- ❌ Alcance multilingüe completo (se limita a español).
- ❌ Multimodalidad (visión/audio) en la primera versión.
- ❌ Almacenamiento de datos en servicios en la nube (todo queda local).
- ❌ Uso comercial de los datos recogidos durante las pruebas.
