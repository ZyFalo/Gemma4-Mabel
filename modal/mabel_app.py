"""
Mabel API en Modal.com — endpoint OpenAI-compatible servido por llama-server (llama.cpp oficial).

Por qué llama-server (binario C++) en vez de llama-cpp-python:
- llama-cpp-python <= 0.3.19 (última con wheel CUDA) NO soporta arquitectura `gemma4`.
- La imagen oficial `ghcr.io/ggml-org/llama.cpp:server-cuda` se compila desde main
  y SÍ soporta `gemma4` text-only en formato Q4_K_M (verificado 2026-05-23).
- `llama-server` expone OpenAI-compat nativo en /v1/chat/completions.

Deploy:
    cd ~/Escritorio/Gemma\\ 4
    .venv/bin/modal deploy modal/mabel_app.py

Después de deploy, Modal devuelve una URL pública del estilo:
    https://<tu-username>--mabel-api-serve.modal.run

Que se consume como api.openai.com:
    POST <URL>/v1/chat/completions
    Content-Type: application/json
    {
      "model": "mabel-gemma4-e4b-Q4_K_M",
      "messages": [...],
      "temperature": 0.7,
      "max_tokens": 500
    }
"""

import modal

# ──────────────────────────────────────────────────────────────────────────────
# Configuración del modelo
# ──────────────────────────────────────────────────────────────────────────────

HF_REPO = "ZyFalo/mabel-gemma4-e4b"
GGUF_FILE = "gemma-4-E4B-mabel-Q4_K_M.gguf"
MODEL_DIR = "/models"
MODEL_PATH = f"{MODEL_DIR}/{GGUF_FILE}"
N_CTX = 8192          # Contexto utilizable (el modelo soporta hasta 131072)
PORT = 8000           # Puerto donde escucha llama-server

# URL pública directa al GGUF (HF resolve, sin auth porque el repo es público)
GGUF_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{GGUF_FILE}"

# ──────────────────────────────────────────────────────────────────────────────
# Imagen Docker — oficial de llama.cpp con CUDA + GGUF descargado al build
# ──────────────────────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.11",  # imagen oficial llama.cpp no trae Python; Modal lo necesita
    )
    # La imagen oficial tiene ENTRYPOINT=/app/llama-server que intercepta TODOS
    # los args. Lo reseteamos para que Modal pueda invocar `python` y nosotros
    # llamemos a llama-server explícitamente desde nuestras funciones.
    .dockerfile_commands(["ENTRYPOINT []"])
    .run_commands(
        f"mkdir -p {MODEL_DIR}",
        # Descarga al build-time del GGUF público de Mabel (5 GB)
        # -L sigue redirects (HF usa CDN), --fail aborta el build si el HTTP falla
        f"curl -L --fail -o {MODEL_PATH} {GGUF_URL}",
        # Verifica que el archivo existe y tiene tamaño razonable (>1 GB)
        f"test -f {MODEL_PATH} && [ $(stat -c %s {MODEL_PATH}) -gt 1000000000 ] || (echo 'GGUF download failed' && exit 1)",
    )
)

# ──────────────────────────────────────────────────────────────────────────────
# App de Modal
# ──────────────────────────────────────────────────────────────────────────────

app = modal.App("mabel-api", image=image)


@app.function(
    gpu="T4",                    # 16 GB VRAM, $0.000164/s (~$0.59/h activo)
    scaledown_window=300,        # 5 min idle → scale to zero
    timeout=1200,                # 20 min timeout (incluye cold start + carga del modelo)
    min_containers=0,            # scale to zero cuando no hay tráfico
    max_containers=2,            # tope de seguridad
)
@modal.concurrent(max_inputs=10)  # hasta 10 requests simultáneos por container
@modal.web_server(port=PORT, startup_timeout=300)  # 5 min para que llama-server cargue el modelo a GPU
def serve():
    """Levanta llama-server (C++ oficial de llama.cpp) con el GGUF cacheado en /models."""
    import subprocess

    # llama-server escucha en 0.0.0.0:PORT y expone OpenAI-compat (/v1/chat/completions)
    # -ngl 999 = mete todas las capas a GPU
    # -c N_CTX = tamaño de contexto KV cache
    # --jinja = usa el chat template embebido en el GGUF (Gemma 4)
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


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — verifica que la imagen + GGUF cargan sin necesidad de exponer puerto.
# Uso: modal run modal/mabel_app.py::smoke_test
# ──────────────────────────────────────────────────────────────────────────────

@app.function(image=image, gpu="T4", timeout=600)
def smoke_test():
    """Lanza llama-server con --no-warmup --check-tensors y verifica que carga el modelo."""
    import subprocess
    import time

    print(f"=== Verificando estructura del container ===")
    subprocess.run(["ls", "-la", "/app/"], check=False)
    subprocess.run(["ls", "-la", MODEL_DIR], check=False)
    subprocess.run(["/app/llama-server", "--version"], check=False)

    print(f"\n=== Cargando modelo en llama-server ===")
    proc = subprocess.Popen([
        "/app/llama-server",
        "--model", MODEL_PATH,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--ctx-size", str(N_CTX),
        "-ngl", "999",
        "--jinja",
        "--alias", "mabel-gemma4-e4b-Q4_K_M",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Espera hasta 4 min a que llama-server reporte "model loaded"
    timeout = 240
    start = time.time()
    loaded = False
    output_lines = []
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
            continue
        output_lines.append(line.rstrip())
        print(line.rstrip())
        if "model loaded" in line.lower() or "starting the main loop" in line.lower() or "all slots are idle" in line.lower():
            loaded = True
            break

    if not loaded:
        print("\n=== ERROR: timeout o crash sin cargar modelo ===")
        proc.kill()
        raise RuntimeError(f"llama-server no llegó a cargar el modelo en {timeout}s")

    print("\n=== Modelo cargado. Probando request HTTP local ===")
    time.sleep(2)
    import urllib.request, json
    payload = json.dumps({
        "model": "mabel-gemma4-e4b-Q4_K_M",
        "messages": [
            {"role": "system", "content": "Te llamas Mabel, asistente de apoyo emocional para estudiantes universitarios colombianos. Responde breve, en español, validando emociones."},
            {"role": "user", "content": "Hola Mabel, me siento muy mal últimamente"},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    print("\n=== Respuesta de Mabel ===")
    msg = result["choices"][0]["message"]["content"]
    print(msg)
    usage = result.get("usage", {})
    print(f"\nprompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}")

    proc.terminate()
    proc.wait(timeout=10)
    return {"status": "ok", "response": msg, "usage": usage}
