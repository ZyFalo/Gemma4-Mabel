# Mabel API en Modal.com

Endpoint OpenAI-compatible que sirve `ZyFalo/mabel-gemma4-e4b` (GGUF Q4_K_M) con `llama-cpp-python` sobre GPU T4 en Modal serverless.

## Por qué Modal y no RunPod / HF Inference

| | RunPod Serverless vLLM | HF Inference Endpoints | **Modal** |
|---|---|---|---|
| GGUF + Gemma 4 E4B | ❌ Crash (vLLM no soporta arquitectura `gemma4` con GGUF — confirmado 2026-05-23) | Requiere merged safetensors | ✅ `llama-cpp-python` lee GGUF nativo |
| OpenAI-compat | Sí pero vLLM | Sí | ✅ Sí (módulo `llama_cpp.server`) |
| Scale-to-zero | Sí | Sí (Pro) | ✅ Sí (`scaledown_window=300`) |
| Free tier | No | No | ✅ $30/mes |
| Cold start | ~30s | ~30s | ~10-20s |
| Costo / req típico (3s en T4) | n/a | ~$0.001 | ~$0.0005 |

## Setup local (una sola vez)

```bash
cd ~/Escritorio/Gemma\ 4
.venv/bin/pip install modal
.venv/bin/modal token new   # abre el navegador, click "Authorize"
```

El token queda en `~/.modal.toml` (local). NUNCA pegar el token en chat o committearlo.

## Smoke test (NO deploya, solo prueba que carga)

```bash
.venv/bin/modal run modal/mabel_app.py::smoke_test
```

Modal compila la imagen (~5-7 min la primera vez porque baja el GGUF de HF al build), después corre `smoke_test` en una T4 y devuelve una respuesta de Mabel.

Costo del smoke_test: ~$0.05 USD (1-2 min de T4).

## Deploy a producción

```bash
.venv/bin/modal deploy modal/mabel_app.py
```

Modal devuelve la URL pública del endpoint. Algo del estilo:
```
https://<tu-username>--mabel-api-serve.modal.run
```

Esa URL ya expone los endpoints OpenAI-compat:
- `POST <URL>/v1/chat/completions`
- `POST <URL>/v1/completions`
- `GET <URL>/v1/models`
- `GET <URL>/docs` (Swagger UI)

## Uso desde un frontend (swap drop-in de OpenAI)

Cambias 3 líneas en tu código que ya consume GPT:

```diff
- base_url: "https://api.openai.com/v1"
- api_key:  "sk-..."
- model:    "gpt-4o-mini"
+ base_url: "https://<tu-username>--mabel-api-serve.modal.run/v1"
+ api_key:  "anything"            # llama_cpp.server no valida auth por defecto
+ model:    "mabel-gemma4-e4b-Q4_K_M"
```

### Ejemplo Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<tu-username>--mabel-api-serve.modal.run/v1",
    api_key="not-used",
)

SYSTEM = open("../system_prompt.txt").read().strip()  # Si tienes el sistema B+ guardado

resp = client.chat.completions.create(
    model="mabel-gemma4-e4b-Q4_K_M",
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Hola, me siento mal últimamente"},
    ],
    temperature=0.7,
    max_tokens=500,
)
print(resp.choices[0].message.content)
```

### Ejemplo curl

```bash
curl https://<tu-username>--mabel-api-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mabel-gemma4-e4b-Q4_K_M",
    "messages": [
      {"role": "system", "content": "Te llamas Mabel..."},
      {"role": "user", "content": "Hola"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

## Configuración actual

| Parámetro | Valor | Justificación |
|---|---|---|
| GPU | T4 (16 GB) | Mabel pesa 5 GB → sobra. Es la más barata ($0.000164/s). |
| n_ctx | 8192 | Suficiente para sesiones de counselling de 20-40 turnos. |
| n_gpu_layers | -1 | Todas las capas a GPU → throughput óptimo. |
| n_batch | 512 | Default sensible para conversación. |
| chat_format | autodetect | El GGUF de Mabel ya trae el template Gemma 4 embebido. |
| scaledown_window | 300s (5 min) | Worker se apaga 5 min después del último request → no paga idle. |
| min_containers | 0 | Scale-to-zero total cuando no hay tráfico. |
| max_containers | 2 | Tope de seguridad para evitar costos sorpresa. |
| allow_concurrent_inputs | 10 | Hasta 10 requests simultáneos en un mismo container. |

## Estimación de costos

| Volumen mensual | Costo aproximado | Suficiente para |
|---|---|---|
| 1 000 requests | ~$0.50 | Demo + uso esporádico |
| 10 000 requests | ~$5 | App de tesis con 10-20 usuarios activos |
| 60 000 requests | ~$30 | Tope del free tier — ~150 usuarios activos |

Si superás $30/mes, Modal te cobra solo el excedente (no requiere upgrade de plan).

## Apagar el endpoint

```bash
.venv/bin/modal app stop mabel-api
```

O desde https://modal.com/apps → click en `mabel-api` → "Stop".

## Ver logs en vivo

```bash
.venv/bin/modal app logs mabel-api
```

O desde la web: https://modal.com/apps/<usuario>/main/deployed/mabel-api

## Política Opción C — sync con HF

Cualquier cambio que afecte cómo se consume Mabel (URL del endpoint, formato de los requests, system prompt recomendado) debe:

1. Actualizarse acá (`modal/README.md`)
2. Replicarse a `docs/28-model-card-hf.md` (sección "API endpoint")
3. Sincronizarse a HF con `scripts/sync_hf_readme.py`

Política completa: `docs/29-hosting-modal.md` (a crear).
