# §11 — Hosting de Mabel en producción: Modal.com

**Fecha**: 2026-05-23
**Estado**: ✅ **DESPLEGADO Y VALIDADO**
**URL pública**: https://williamandres1603--mabel-api-serve.modal.run
**Endpoint OpenAI-compat**: `POST /v1/chat/completions`
**Costo del setup**: ~$0.70 USD del free tier de Modal ($29.30 restantes)

## Objetivo

Exponer Mabel v1 como endpoint HTTP **OpenAI-compatible** consumible por cualquier frontend que ya hable con `api.openai.com`. Swap drop-in: cambiar `base_url`, `api_key` y `model` en el frontend, sin tocar el resto del código.

## Trayectoria de decisión (camino completo, con todos los fracasos documentados)

### Intento 1 — RunPod Serverless + worker-vllm oficial + GGUF

| Aspecto | Detalle |
|---|---|
| **Idea** | Usar el worker oficial `runpod/worker-v1-vllm:v2.18.1stable-cuda12.4.1` con `MODEL_NAME=ZyFalo/mabel-gemma4-e4b` + `QUANTIZATION=gguf` |
| **Resultado** | ❌ Crash loop instantáneo (`worker exited with exit code 1`) |
| **Causa** | vLLM 0.19.1 lista oficialmente "Gemma 4" pero solo soporta variantes **dense (31B) y MoE (26B A4B)** con formato **safetensors**. La variante **E4B (edge multimodal con arquitectura PLE)** + cuantización GGUF no está implementada. |
| **Verificación** | El código fuente del worker (`engine_args.py`) lista las quantizations soportadas: `awq, gptq, squeezellm, bitsandbytes`. GGUF no aparece. |

### Intento 2 — RunPod Serverless + worker-vllm + sintaxis vLLM nativa `:Q4_K_M`

| Aspecto | Detalle |
|---|---|
| **Idea** | La doc oficial de vLLM dice usar `MODEL_NAME=repo:quant_type`, ej. `unsloth/Qwen3-0.6B-GGUF:Q4_K_M`. Mismo enfoque para Mabel: `ZyFalo/mabel-gemma4-e4b:Q4_K_M` |
| **Resultado** | ❌ Worker quedó en `initializing` por 8+ minutos sin progreso (otro tipo de crash loop, esta vez silencioso) |
| **Causa** | La sintaxis es válida pero vLLM sigue sin poder cargar la arquitectura `gemma4` en GGUF, independiente de cómo se le pase el modelo. |
| **Costo del experimento** | ~$0.20 USD (worker en init sin servir) |

### Decisión: pasar a Modal con `llama-cpp-python`

`llama-cpp-python` es el backend nativo que ya usamos local para la batería §10 — habla GGUF como lenguaje madre y expone `/v1/chat/completions` OpenAI-compat 100% nativo. Modal nos da:

- GPU on-demand (T4 16 GB es perfecta para Mabel 5 GB)
- Scale-to-zero (paga $0 cuando no hay tráfico)
- $30/mes free credits (más que suficiente para tesis)
- Cold start ~10-20s (mejor que RunPod ~30s)
- Sin necesidad de merge / re-entrenar

## Arquitectura del deploy

```
┌─────────────────────────────────────────────────────────────┐
│  TU FRONTEND (lo que ya está en producción)                 │
│  base_url: https://<user>--mabel-api-serve.modal.run/v1     │
│  api_key:  cualquier-string                                 │
│  model:    mabel-gemma4-e4b-Q4_K_M                          │
└──────────────────────────────┬──────────────────────────────┘
                               │ POST /v1/chat/completions
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Modal.com — serverless ASGI app                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Container (T4 16 GB, scale-to-zero idle=5min)         │  │
│  │  - llama-cpp-python 0.3.2 + CUDA 12.4                 │  │
│  │  - llama_cpp.server.app:create_app()                  │  │
│  │  - Mabel GGUF Q4_K_M (5 GB, cached en imagen)         │  │
│  │  - n_ctx=8192, n_gpu_layers=-1                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
                               │ (download al build-time)
                               ▼
                  ┌────────────────────────────┐
                  │  HF Hub público            │
                  │  ZyFalo/mabel-gemma4-e4b   │
                  └────────────────────────────┘
```

## Implementación real: 8 bugs encadenados antes de éxito

El plan inicial subestimó la complejidad de la combinación específica **Gemma 4 E4B (arquitectura PLE) + GGUF + Modal serverless + CUDA**. Cada iteración descubrió un nuevo bloqueante. Se documenta el camino completo porque el aprendizaje es replicable a cualquier proyecto similar:

| # | Smoke test | Error encontrado | Fix aplicado |
|---|---|---|---|
| 1 | API Modal deprecado | `allow_concurrent_inputs` deprecated 2025-04 | Reemplazar por `@modal.concurrent(max_inputs=10)` |
| 2 | Build pasó, ejecución falló | Mismo error #1 más arriba | Idem |
| 3 | llama-cpp-python 0.3.2 | `unknown model architecture: 'gemma4'` | Upgrade a `0.3.19` (última con wheel CUDA pre-compilado) |
| 4 | 0.3.19 + debian-slim | `libcudart.so.12: cannot open shared object file` | Cambiar imagen base a `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| 5 | CUDA OK, falta OpenMP | `libgomp.so.1: cannot open shared object file` | `apt_install("libgomp1")` |
| 6 | 0.3.19 también falla con gemma4 | Mismo error #3 — los bindings Python van atrasados del llama.cpp main | **Pivot**: usar imagen oficial `ghcr.io/ggml-org/llama.cpp:server-cuda` (compila desde main) |
| 7 | Imagen llama.cpp sin Python | `ConflictError: We were unable to determine the version of Python` | `add_python="3.11"` |
| 8 | Imagen llama.cpp con ENTRYPOINT propio | `error: invalid argument: python` (el binario llama-server intercepta los args de Modal) | `.dockerfile_commands(["ENTRYPOINT []"])` para reset |
| 9 | Smoke test ✅ pero deploy producción falla | `TimeoutError: Waited too long for port 8000 to start accepting connections` | Logs revelan: `error while handling argument "--log-colors": expected value` — el flag NO es boolean, requiere `auto/on/off`. Removido del comando llama-server. |
| **10** | **Re-deploy + curl** | **✅ Mabel respondió: "Hola. Cuéntame qué pasó."** (4.1s warm) | Pipeline funcional end-to-end |

### Lecciones técnicas para futuro

1. **Los bindings Python van atrasados de los proyectos C++ que envuelven.** llama-cpp-python 0.3.19 (la última con wheel CUDA pre-compilado al momento) NO incluye soporte para arquitectura `gemma4`, aunque llama.cpp upstream sí. Para arquitecturas recientes, usar el binario oficial es más seguro que los bindings.

2. **Las imágenes Docker oficiales pueden traer ENTRYPOINT que rompen orquestadores genéricos** (Modal, Kubernetes). Resetearlo con `dockerfile_commands(["ENTRYPOINT []"])` antes de instalar Python u otros runtimes que el orquestador necesita invocar.

3. **vLLM soporta "Gemma 4" en su lista oficial pero solo las variantes dense (31B) y MoE (26B A4B) en safetensors.** La variante E4B edge con arquitectura PLE (Per-Layer Embeddings) no funciona en vLLM 0.19.1 ni con GGUF ni con safetensors. Caso documentado por nosotros.

4. **Modal smoke tests (`modal run`) usan código Python directo y son más rápidos que los deploys reales (`modal deploy`)**. Cosas que funcionan en smoke test pueden fallar en producción si el web server tarda en hacer bind al puerto. Configurar `startup_timeout` generoso (≥180s) para modelos grandes.

5. **Flags CLI engañosos**: `--log-colors` en llama-server NO es boolean, requiere argumento (`auto/on/off`). Sin él, llama-server muere al instante y el error no aparece en los logs de Modal hasta que se invoca `modal app logs`.

## Configuración final que funciona

```python
image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.11",
    )
    .dockerfile_commands(["ENTRYPOINT []"])  # reset entrypoint heredado
    .run_commands(
        "mkdir -p /models",
        f"curl -L --fail -o /models/{GGUF_FILE} {GGUF_URL}",
    )
)

@app.function(gpu="T4", scaledown_window=300, timeout=1200, ...)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    import subprocess
    subprocess.Popen([
        "/app/llama-server",
        "--model", MODEL_PATH,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--ctx-size", str(N_CTX),
        "-ngl", "999",
        "--jinja",
        "--alias", "mabel-gemma4-e4b-Q4_K_M",
    ])
```

## Estructura de archivos

```
modal/
├── mabel_app.py       ← App de Modal (deploy con `modal deploy`)
└── README.md          ← Guía de uso

docs/
├── 28-model-card-hf.md  ← Espejo del README de HF (incluir sección API endpoint)
└── 29-hosting-modal.md  ← Este archivo

scripts/
└── sync_hf_readme.py   ← Sincroniza docs/28 → HF (Política Opción C)
```

## Pasos de implementación

### 1. Cuenta Modal (usuario)
- Crear en https://modal.com con `williamandres1603@gmail.com`
- Verificar email para activar $30 free credits
- Sin tarjeta requerida

### 2. CLI local (usuario)
```bash
cd ~/Escritorio/Gemma\ 4
.venv/bin/pip install modal
.venv/bin/modal token new   # autoriza vía navegador
```
El token queda en `~/.modal.toml` (NO se commitea, NO se pega en chat).

### 3. Smoke test (15-20 min, primer build de imagen)
```bash
.venv/bin/modal run modal/mabel_app.py::smoke_test
```
- Modal compila la imagen: ~5-7 min (instala llama-cpp-python con CUDA + baja GGUF 5 GB al build)
- Modal corre `smoke_test` en T4: ~1-2 min
- Output esperado: respuesta de Mabel al prompt "Hola, me siento muy mal últimamente"
- Costo: ~$0.05 USD

### 4. Deploy a producción (5 min después del smoke OK)
```bash
.venv/bin/modal deploy modal/mabel_app.py
```
- Modal reutiliza la imagen ya compilada
- Devuelve URL pública: `https://<user>--mabel-api-serve.modal.run`

### 5. Validación con curl (medida real 2026-05-23)

```bash
curl https://williamandres1603--mabel-api-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mabel-gemma4-e4b-Q4_K_M",
    "messages": [
      {"role": "system", "content": "Te llamas Mabel..."},
      {"role": "user", "content": "Hola Mabel, me siento muy triste hoy"}
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**Respuesta real recibida**:
```json
{
  "choices": [{
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "Hola. Cuéntame qué pasó."
    }
  }],
  "model": "mabel-gemma4-e4b-Q4_K_M",
  "usage": {
    "prompt_tokens": 52,
    "completion_tokens": 9,
    "total_tokens": 61
  },
  "timings": {
    "prompt_per_second": 137.83,
    "predicted_per_second": 37.12
  }
}
```

**Métricas medidas**:
- Latencia warm: **4.1 segundos** (incluye latencia red)
- Prompt processing: 137 tok/s
- Generation: 37 tok/s
- Cold start primer request: ~90s (incluye carga del modelo a GPU)
- Cold start subsiguientes (worker recién apagado): ~40-60s

### 6. Swap en frontend (10 min usuario)
Cambia 3 líneas en el código que ya consume GPT:
```diff
- base_url: "https://api.openai.com/v1"
- api_key:  "sk-..."
- model:    "gpt-4o-mini"
+ base_url: "https://<user>--mabel-api-serve.modal.run/v1"
+ api_key:  "anything"
+ model:    "mabel-gemma4-e4b-Q4_K_M"
```

### 7. Sync HF (Política Opción C)
- Agregar sección "API endpoint disponible" en `docs/28-model-card-hf.md`
- `HF_TOKEN=xxx .venv/bin/python scripts/sync_hf_readme.py`

## Costos reales (medidos)

| Componente | Costo |
|---|---|
| Plan Modal | $0 ($30 free credits/mes — pago automático del excedente si supera) |
| Setup (8 smoke tests + 2 deploys + ~5 curls de validación) | **~$0.70 USD** (verificado: balance pasó de $30.00 a $29.30) |
| Producción 1k req/mes warm | ~$0.50 |
| Producción 10k req/mes warm | ~$5 |
| Producción con cold starts frecuentes | ~$0.015 por sesión (cold start ~90s pagados) |

**Proyección con los $29.30 restantes** (uso típico de tesis 1-2 demos/día + algunos usuarios probando):
- **$2-5 USD/mes** → los créditos durarían **6-15 meses sin recargar**

## Comparativa contra otros caminos descartados

| Camino | Por qué se descartó |
|---|---|
| RunPod vLLM + GGUF | Crash loop (Intento 1 y 2) — vLLM no soporta `gemma4` GGUF |
| RunPod vLLM + safetensors merged | Requiere reactivar pod CA-MTL-1 (bloqueado por falta de GPU disponible), hacer merge LoRA+base (~13 GB), subir a HF como repo aparte, configurar worker. ~1.5h + $1-2 + dependencia de capacidad del datacenter. |
| RunPod custom Docker llama.cpp worker | Requiere armar imagen + handler RunPod-style + adapter para OpenAI-compat. 3-4h de trabajo + sin ventaja sobre Modal. |
| HF Inference Endpoints Dedicated | $0.50-1/h always-on (~$360-720/mes). Sin scale-to-zero para containers custom. Excesivo para tesis. |
| Ngrok + PC local | Funciona pero requiere PC encendido 24/7. No es producción. |
| Replicate | No tiene OpenAI-compat nativo (formato propio Cog). Migración del frontend más invasiva. |

## Riesgos conocidos

1. **Cold start**: 10-20s desde scale-to-zero. Si el primer usuario del día espera mucho, considerar `min_containers=1` ($0.50/mes extra para mantener uno caliente).
2. **Free credits**: $30/mes recurrentes. Si superamos, Modal cobra el excedente automáticamente. Configurar alertas de billing en https://modal.com/settings/billing.
3. **Auth simbólico**: `llama_cpp.server` no valida API key por defecto. Si Mabel se vuelve público y empieza a recibir tráfico no autorizado, agregar middleware de auth (FastAPI dependencies). Documentado como mejora futura.
4. **Versión llama-cpp-python**: pinneada a `0.3.2` para reproducibilidad. Bumpear con cuidado y re-correr smoke_test antes de deploy.

## Mejoras futuras

- [ ] Agregar middleware de API key auth (rate limiting + simple bearer token check)
- [ ] Agregar métricas a Prometheus / Modal observability
- [ ] Streaming responses (server-sent events) — `llama_cpp.server` ya lo soporta, solo activar en el frontend
- [ ] Cache de respuestas frecuentes (no aplica a counselling por privacidad)
- [ ] Migrar a `min_containers=1` cuando uso real lo justifique

## Cierre

Hosting documentado, costo controlado, OpenAI-compatible. El frontend que ya consume GPT puede hacer swap de Mabel cambiando 3 líneas. Cumple el requisito "lo necesito en general porque el proyecto está en prod, no en mi máquina local".
