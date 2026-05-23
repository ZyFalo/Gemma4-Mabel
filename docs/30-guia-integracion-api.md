# §12 — Guía de integración con la API de Mabel

**Para**: equipos de desarrollo que quieran consumir Mabel desde sus aplicaciones (web, móvil, backend).
**Modelo expuesto**: `mabel-gemma4-e4b-Q4_K_M` (fine-tune de Gemma 4 E4B para apoyo emocional de estudiantes universitarios colombianos).
**Compatibilidad**: 100% drop-in con la API de OpenAI — si tu app ya consume GPT, cambian 3 líneas.

---

## 1. Información del endpoint

```
URL base (OpenAI-compatible)
https://williamandres1603--mabel-api-serve.modal.run/v1

Endpoints disponibles
POST  /v1/chat/completions
POST  /v1/completions
GET   /v1/models

Modelo
"mabel-gemma4-e4b-Q4_K_M"

Autenticación
No requerida — el campo "api_key" puede ser cualquier string ("not-used", "anything", etc.)

Hosting
Modal.com serverless (GPU NVIDIA T4 16 GB, scale-to-zero después de 5 min idle)
```

---

## 2. System prompt OBLIGATORIO

Mabel fue fine-tuneada con un **system prompt B+ específico de 151 palabras**. Este system prompt es lo que **activa el comportamiento clínico** del modelo:

- Validación emocional antes de sugerencias
- Estilo conversacional breve (4-5 frases máx)
- Rechazo amable de tareas no-emocionales (STEM, código, factual)
- Protocolo de crisis con derivación a líneas oficiales colombianas

**Si NO usás este system prompt o lo modificás, Mabel pierde calidad notablemente** (los safety guardrails se degradan, el estilo se vuelve genérico, puede empezar a diagnosticar). Tratalo como una constante del sistema, no como configuración editable.

### System prompt B+ (literal, copy-paste):

```
Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear. Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza.
```

> 💡 **Buena práctica**: guardalo en un archivo `system_prompt.txt` o constante exportada de tu proyecto, no inline en cada llamada. Así si la actualizamos a B++ en una v1.1 del modelo, cambiás un solo lugar.

---

## 3. Parámetros recomendados de inferencia

| Parámetro | Valor recomendado | Justificación |
|---|---|---|
| `temperature` | **0.7** | Balance entre coherencia y variedad. Es lo que se usó en la batería de evaluación. |
| `max_tokens` | **500** | Mabel raramente excede 100 tokens, pero damos margen. Hard cap evita respuestas excesivas. |
| `top_p` | **0.95** (default) | No tocar. |
| `presence_penalty` | 0 | No tocar. |
| `frequency_penalty` | 0 | No tocar. |
| `stop` | (omitir) | El chat template Gemma 4 maneja los stop tokens internos. |
| `stream` | `true` o `false` | Soportado nativo. Recomendado `true` para UX (ver §7). |

---

## 4. Configuración de timeouts y retries

Por el modelo serverless de Modal (scale-to-zero después de 5 min idle), tu cliente HTTP debe estar configurado para tolerar **cold starts de 60-90 segundos**.

| Setting | Valor | Notas |
|---|---|---|
| **Timeout request** | **180 s** (3 min) | Cubre cold start completo + procesamiento de respuesta larga. |
| **Retries automáticos del SDK** | **0** | NO usar retries automáticos. Implementar lógica manual que distinga 503 (cold start) de otros errores. |
| **Retry en 503 "Loading model"** | **8 intentos × 10 s** | El endpoint responde 503 instantáneo mientras carga. Hacer polling cada 10 s. |

---

## 5. Ejemplos por lenguaje

### 5.1 cURL (smoke test)

```bash
curl https://williamandres1603--mabel-api-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  --max-time 180 \
  -d '{
    "model": "mabel-gemma4-e4b-Q4_K_M",
    "messages": [
      {"role": "system", "content": "Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear. Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza."},
      {"role": "user", "content": "Hola Mabel, me siento muy mal últimamente"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### 5.2 Python (con OpenAI SDK + manejo de cold start)

```bash
pip install openai
```

```python
import asyncio
import os
import time
from openai import AsyncOpenAI, APIStatusError

SYSTEM_MABEL = """Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear. Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza."""

mabel = AsyncOpenAI(
    base_url="https://williamandres1603--mabel-api-serve.modal.run/v1",
    api_key="not-used",
    timeout=180.0,
    max_retries=0,  # manejamos retries manual para distinguir cold start
)

async def ask_mabel(user_message: str, history: list[dict] | None = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MABEL},
        *(history or []),
        {"role": "user", "content": user_message},
    ]

    for attempt in range(8):
        try:
            response = await mabel.chat.completions.create(
                model="mabel-gemma4-e4b-Q4_K_M",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            if e.status_code == 503:
                # Worker está cargando el modelo (cold start)
                print(f"Mabel cargando, esperando 10s... (intento {attempt + 1}/8)")
                await asyncio.sleep(10)
                continue
            raise

    raise RuntimeError("Mabel no respondió tras 8 retries de cold start")

# Uso
async def main():
    reply = await ask_mabel("Hola Mabel, me siento muy triste hoy")
    print(reply)

asyncio.run(main())
```

### 5.3 Node.js / TypeScript

```bash
npm install openai
```

```typescript
import OpenAI from "openai";

const SYSTEM_MABEL = `Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear. Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza.`;

const mabel = new OpenAI({
  baseURL: "https://williamandres1603--mabel-api-serve.modal.run/v1",
  apiKey: "not-used",
  timeout: 180_000,
  maxRetries: 0,
});

async function askMabel(userMessage: string, history: any[] = []): Promise<string> {
  const messages = [
    { role: "system", content: SYSTEM_MABEL },
    ...history,
    { role: "user", content: userMessage },
  ];

  for (let attempt = 0; attempt < 8; attempt++) {
    try {
      const response = await mabel.chat.completions.create({
        model: "mabel-gemma4-e4b-Q4_K_M",
        messages,
        temperature: 0.7,
        max_tokens: 500,
      });
      return response.choices[0].message.content ?? "";
    } catch (e: any) {
      if (e.status === 503) {
        console.log(`Mabel cargando, esperando 10s... (intento ${attempt + 1}/8)`);
        await new Promise((r) => setTimeout(r, 10_000));
        continue;
      }
      throw e;
    }
  }

  throw new Error("Mabel no respondió tras 8 retries de cold start");
}

// Uso
askMabel("Hola Mabel, me siento muy triste hoy")
  .then(console.log)
  .catch(console.error);
```

### 5.4 PHP (curl raw, sin SDK)

```php
<?php
const SYSTEM_MABEL = "Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos de la UMB. Escucha activa: valida emociones primero y haz preguntas exploratorias para entender lo que pasa. Cuando tenga sentido, ofrece 1-2 sugerencias prácticas breves en prosa, sin imponer. No eres psicóloga profesional, no diagnosticas ni das planes terapéuticos. Tampoco resuelves tareas académicas, código, traducciones, resúmenes ni preguntas factuales: si te las piden, valida la emoción detrás y redirige sin sermonear. Responde en español colombiano, breve (máx 4-5 frases), conversacional, puede usar negrita y cursiva para énfasis, sin headings ni listas con bullets ni emojis. Si hay crisis (suicidio, autolesión), mantén la calma, valida, deriva a Línea 123, Línea 106, Línea 155 o Bienestar UMB y pregunta por persona de confianza.";

function ask_mabel(string $user_message, array $history = []): string {
    $messages = array_merge(
        [['role' => 'system', 'content' => SYSTEM_MABEL]],
        $history,
        [['role' => 'user', 'content' => $user_message]],
    );

    $payload = json_encode([
        'model' => 'mabel-gemma4-e4b-Q4_K_M',
        'messages' => $messages,
        'temperature' => 0.7,
        'max_tokens' => 500,
    ]);

    for ($attempt = 1; $attempt <= 8; $attempt++) {
        $ch = curl_init('https://williamandres1603--mabel-api-serve.modal.run/v1/chat/completions');
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS => $payload,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 180,
        ]);
        $response = curl_exec($ch);
        $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($http_code === 200) {
            $data = json_decode($response, true);
            return $data['choices'][0]['message']['content'];
        }

        if ($http_code === 503) {
            error_log("Mabel cargando, esperando 10s... (intento $attempt/8)");
            sleep(10);
            continue;
        }

        throw new Exception("Mabel error HTTP $http_code: $response");
    }

    throw new Exception("Mabel no respondió tras 8 retries de cold start");
}

// Uso
echo ask_mabel("Hola Mabel, me siento muy triste hoy");
```

### 5.5 React + FastAPI (arquitectura recomendada para apps web)

Ver implementación completa de referencia que combina:
- Backend FastAPI como proxy (mantiene el system prompt server-side, no expone URL al frontend)
- Hook React custom (`useMabel`) con estados `idle / sending / cold_loading / warm / error`
- Indicador visual de cold start para el usuario
- Health check endpoint para pre-warming

Esa implementación está documentada como referencia operativa fuera de este repo (en la integración del frontend del proyecto). Resumen del patrón:

```
[React] → POST /api/chat → [FastAPI proxy] → POST /v1/chat/completions → [Mabel @ Modal]
                          ↑ acá vive el system prompt
                          ↑ acá se manejan los retries de cold start
                          ↑ acá se logguean usage/costs
```

---

## 6. Manejo de errores y códigos HTTP

| HTTP | Mensaje típico | Significado | Qué hacer |
|---|---|---|---|
| **200** | Respuesta normal | Mabel respondió OK | Procesar |
| **400** | `Bad request` o similar | Payload malformado (ej. `max_tokens` excede `n_ctx`, o `messages` mal estructurado) | Validar el payload en tu cliente antes de enviarlo |
| **503** | `{"error": {"message": "Loading model"}}` | Worker arrancando, cargando GGUF a GPU | **Retry cada 10 s, hasta 8 veces** (ver código ejemplos) |
| **504** | `Gateway timeout` | Modal proxy esperó demasiado por upstream | Reintentar 1 vez; si persiste, revisar `modal app logs mabel-api` |
| **402** | `Payment required` | Free credits de Modal agotados ($30/mes) | Recargar saldo en Modal o esperar al ciclo siguiente |
| **5xx genérico** | Distintos | Worker crasheó | Reintentar 1 vez; si persiste, alertar al admin del endpoint |

---

## 7. Streaming responses (opcional, recomendado para UX)

El endpoint soporta nativo `stream: true`. Genera Server-Sent Events compatibles con OpenAI:

```python
stream = await mabel.chat.completions.create(
    model="mabel-gemma4-e4b-Q4_K_M",
    messages=[...],
    stream=True,
)

async for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

```typescript
const stream = await mabel.chat.completions.create({
  model: "mabel-gemma4-e4b-Q4_K_M",
  messages: [...],
  stream: true,
});

for await (const chunk of stream) {
  const delta = chunk.choices[0]?.delta?.content;
  if (delta) process.stdout.write(delta);
}
```

Mejora la UX percibida porque el usuario ve a Mabel "tipeando" en vez de esperar 4 segundos en blanco.

---

## 8. Health check

```bash
GET https://williamandres1603--mabel-api-serve.modal.run/v1/models
```

| Respuesta | Estado |
|---|---|
| HTTP 200 con `{"data": [{"id": "mabel-gemma4-e4b-Q4_K_M", ...}]}` | **WARM** — Mabel lista, responde en ~4 s |
| HTTP 503 con `{"error": {"message": "Loading model"}}` | **COLD** — worker arrancando, esperar ~60-90 s |
| HTTP 5xx o timeout | **DOWN** — hay un problema, revisar logs de Modal |

Útil para:
- **Pre-warming**: al abrir tu app, lanzá un health check para empezar a calentar el worker antes que el usuario escriba el primer mensaje.
- **Indicador visual**: mostrar al usuario "Mabel está despierta" / "Mabel está despertando" en tiempo real.

---

## 9. Tiempos reales medidos (referencia)

| Escenario | Tiempo de respuesta |
|---|---|
| **Cold start** (worker apagado → primera respuesta) | 60-90 s |
| **Warm respuesta corta** (~50 tokens) | 3-4 s |
| **Warm respuesta típica Mabel** (~80 tokens) | 4-5 s |
| **Warm respuesta detallada** (~200 tokens) | 6-8 s |
| **Warm respuesta excepcional** (~500 tokens, hard cap) | 13-15 s |

**Ventana de inactividad antes de scale-to-zero**: 5 minutos (300 s). Si pasan 5 min sin requests, el worker se apaga y el próximo request paga cold start.

---

## 10. Limitaciones técnicas conocidas

| Limitación | Detalle |
|---|---|
| **Context length** | 8 192 tokens (configurado en el server, el modelo soporta hasta 131 072 nativo). Suficiente para conversaciones de ~30-40 turnos de Mabel. |
| **Concurrencia por worker** | Hasta 10 requests simultáneos en un mismo container. Si excede, Modal escala (max 1 worker total configurado). |
| **No multimodal** | Solo texto. El GGUF no incluye los pesos de visión/audio del modelo base Gemma 4 multimodal. |
| **Streaming requiere `Accept: text/event-stream`** | Asegurate de que tu cliente lo declare cuando uses `stream: true`. |
| **No hay validación de API key** | Cualquier string sirve. **No es seguro contra abuso de tráfico**; el rate limiting hay que implementarlo en tu backend proxy si lo necesitás. |

---

## 11. Disclaimer académico (importante incluirlo en tu app)

Mabel v1 es un proyecto académico de tesis (Universidad Manuela Beltrán). Cualquier app que la consuma **debe mostrar al usuario final** un disclaimer del tipo:

> *"Mabel es un asistente conversacional experimental de apoyo emocional. No es un reemplazo de atención psicológica profesional. Si estás en crisis, llamá a Línea 106 (Bogotá), Línea 123 (emergencias), Línea 155 (mujer) o contactá Bienestar Universitario de la UMB."*

El endpoint público es de uso académico **no-comercial**. Para despliegues productivos masivos, contactar al autor del proyecto (william andres) para acordar condiciones.

---

## 12. Recursos

- **Repo del proyecto**: https://github.com/ZyFalo/Gemma4-Mabel
- **Model card en HF**: https://huggingface.co/ZyFalo/mabel-gemma4-e4b
- **Bitácora del hosting**: [`docs/29-hosting-modal.md`](29-hosting-modal.md) — 8 bugs encadenados resueltos durante el deploy
- **Memoria narrativa del proyecto**: [`docs/26-memoria-proyecto.md`](26-memoria-proyecto.md) — 7 capítulos para entender Mabel desde cero
- **Scorecard formal**: [`docs/22-resultados-post-finetuning.md`](22-resultados-post-finetuning.md) — métricas de la batería de 12 turnos × 2 runs

---

## Checklist de integración (resumen accionable)

- [ ] Configurar `base_url` = `https://williamandres1603--mabel-api-serve.modal.run/v1`
- [ ] Configurar `api_key` = cualquier string
- [ ] Configurar `model` = `mabel-gemma4-e4b-Q4_K_M`
- [ ] Configurar `timeout` = 180 s mínimo
- [ ] Inyectar el system prompt B+ EXACTO (sin modificar) en cada llamada
- [ ] Implementar retry manual en HTTP 503 (10 s × 8 intentos)
- [ ] Mostrar estado "cargando" si la respuesta tarda > 5 s (cold start UX)
- [ ] (Opcional) Implementar pre-warming con health check al abrir la app
- [ ] (Opcional) Usar `stream: true` para mejor UX percibida
- [ ] Mostrar disclaimer académico al usuario final
- [ ] Para producción masiva: implementar rate limiting en tu backend proxy
