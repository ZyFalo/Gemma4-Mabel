# 11 — Instalación y reproducibilidad

Esta guía permite reproducir desde cero el entorno de inferencia del asistente.

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| SO | Linux x86_64 (probado en Ubuntu 24.04) | Ubuntu 24.04 LTS |
| RAM libre | 8 GB | 16 GB+ |
| Disco libre | 20 GB | 50 GB+ |
| Python | 3.10 | 3.11 o 3.12 |
| CUDA | opcional (no usado en modo CPU/RAM) | 12.x+ si se hará offload GPU |

**Probado en:** Intel i7-10750H + 31 GB RAM + NVIDIA RTX 2060 Mobile (solo como referencia, no usada para inferencia en esta fase).

## 1. Estructura de directorios del proyecto

```
/home/<usuario>/Escritorio/Gemma 4/
├── bin/
│   └── llama.cpp/
│       └── llama-b8763/          # binarios de llama.cpp
├── modelos/                       # archivos GGUF descargados
│   └── gemma-4-E4B-it-Q4_K_M.gguf
└── docs/                          # documentación del proyecto
```

Crear la estructura inicial:

```bash
mkdir -p "/home/$USER/Escritorio/Gemma 4/bin"
mkdir -p "/home/$USER/Escritorio/Gemma 4/modelos"
mkdir -p "/home/$USER/Escritorio/Gemma 4/docs/decisiones"
```

## 2. Instalación de llama.cpp (inferencia, CPU-only)

Se usan los binarios precompilados oficiales del repositorio `ggml-org/llama.cpp` en GitHub. **No se compila desde fuente**, lo que evita dependencias de build (cmake, compiladores, toolkit CUDA).

### 2.1 Descargar el release

```bash
cd "/home/$USER/Escritorio/Gemma 4/bin"

# Obtener el tag del último release
LATEST_TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | grep '"tag_name"' | head -1 | cut -d'"' -f4)
echo "Último release: $LATEST_TAG"

# Descargar el binario Ubuntu x64 (CPU-only)
curl -L -o llama.tar.gz \
  "https://github.com/ggml-org/llama.cpp/releases/download/$LATEST_TAG/llama-$LATEST_TAG-bin-ubuntu-x64.tar.gz"
```

> **Versión usada en este proyecto**: `b8763` (abril 2026). Los binarios más recientes pueden diferir en nombre.

### 2.2 Extraer

```bash
mkdir -p llama.cpp
tar -xzf llama.tar.gz -C llama.cpp
rm llama.tar.gz
ls llama.cpp/llama-b8763/llama-cli llama.cpp/llama-b8763/llama-server
```

### 2.3 Variables de entorno (shell)

El directorio extraído contiene los binarios y las librerías `.so` que cargan dinámicamente. Hay que añadirlas al `LD_LIBRARY_PATH`:

```bash
export LLAMA_DIR="/home/$USER/Escritorio/Gemma 4/bin/llama.cpp/llama-b8763"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
export PATH="$LLAMA_DIR:$PATH"
```

Para persistir, añadir al final de `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'

# Gemma 4 - llama.cpp
export LLAMA_DIR="/home/$USER/Escritorio/Gemma 4/bin/llama.cpp/llama-b8763"
export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
export PATH="$LLAMA_DIR:$PATH"
EOF
source ~/.bashrc
```

### 2.4 Verificar instalación

```bash
llama-cli --version
# Salida esperada:
# load_backend: loaded CPU backend from libggml-cpu-haswell.so
# version: 8763 (ff5ef8278)
# built with GNU 11.4.0 for Linux x86_64
```

El backend CPU elegido se adapta a la microarquitectura del procesador (`haswell` para Comet Lake / Ice Lake / Tiger Lake, etc.). Si no aparece esa línea, revisar que `LD_LIBRARY_PATH` esté bien configurado.

## 3. Descarga de los modelos GGUF

Se usan los repositorios de **Unsloth** en HuggingFace, que publican versiones pre-cuantizadas de Gemma 4 sin requerir autenticación (el repo oficial de Google sí requiere aceptar licencia y token).

### 3.1 Modelo principal — Gemma 4 E4B (rápido, uso diario)

```bash
cd "/home/$USER/Escritorio/Gemma 4/modelos"

curl -L -o gemma-4-E4B-it-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf"

ls -lh gemma-4-E4B-it-Q4_K_M.gguf
# ~4.7 GB
```

**Velocidad esperada**: 8–15 tok/s en CPU (6 threads).
**RAM usada**: ~7.3 GB al cargar.

### 3.2 Modelo comparador — Gemma 4 26B MoE (calidad, evaluación)

Este modelo se usa para comparar contra el E4B en la fase de evaluación (ver decisión D-013). El repositorio `unsloth/gemma-4-26B-A4B-it-GGUF` no publica formatos `Q_K_M` clásicos sino variantes **UD (Ultra-Dynamic)**, una generación más reciente con mejor calidad por bit.

```bash
cd "/home/$USER/Escritorio/Gemma 4/modelos"

curl -L -o gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

ls -lh gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
# ~15.7 GB
```

**Velocidad esperada**: 6–10 tok/s en CPU (el MoE activa solo ~4B params por token).
**RAM usada**: ~22 GB al cargar.
**Tiempo de descarga**: ~4 min a 40 MB/s.

> **Importante**: en un equipo con 31 GB de RAM total, **no se pueden cargar ambos modelos simultáneamente**. Hay que parar uno antes de lanzar el otro. Ver sección 4.3.

**Notas para ambos modelos**:

- El formato GGUF es autocontenido: incluye pesos, tokenizer, metadatos y chat template.
- Los repositorios de Unsloth (re-empaquetados) son descargables sin token HF.
- Los modelos oficiales de Google (`google/gemma-4-*`) requieren aceptar licencia en HuggingFace y configurar `huggingface-cli login` con token.

## 4. Arranque del servidor y alternar entre modelos

### 4.0 Arrancar el servidor (E4B o 26B)

El cliente `chat.py` se conecta siempre al mismo endpoint (`http://127.0.0.1:8080/v1/chat/completions`). Solo cambia el proceso del servidor según qué modelo queramos usar.

**Arrancar E4B** (diario, rápido):

```bash
cd "/home/$USER/Escritorio/Gemma 4"
export LD_LIBRARY_PATH="$PWD/bin/llama.cpp/llama-b8763:$LD_LIBRARY_PATH"
./bin/llama.cpp/llama-b8763/llama-server \
  -m modelos/gemma-4-E4B-it-Q4_K_M.gguf \
  -c 4096 -t 6 --host 127.0.0.1 --port 8080 --mlock
```

**Arrancar 26B MoE** (calidad, comparador):

```bash
cd "/home/$USER/Escritorio/Gemma 4"
export LD_LIBRARY_PATH="$PWD/bin/llama.cpp/llama-b8763:$LD_LIBRARY_PATH"
./bin/llama.cpp/llama-b8763/llama-server \
  -m modelos/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  -c 4096 -t 6 --host 127.0.0.1 --port 8080 --mlock
```

En ambos casos, esperar a ver en la terminal:
```
main: server is listening on http://127.0.0.1:8080
```

### 4.3 Alternar entre modelos

**Los dos modelos NO caben simultáneamente** en los 31 GB de RAM del equipo de referencia. Para cambiar:

```bash
# 1. Parar el servidor actual (libera RAM)
pkill -f llama-server

# 2. Esperar 1-2 segundos y verificar que el puerto 8080 está libre
ss -tlnp | grep 8080 || echo "puerto libre"

# 3. Arrancar el otro modelo con el comando correspondiente
```

### 4.1 Archivo de prompt

Por compatibilidad con caracteres Unicode (`¿`, `ñ`, acentos), **siempre usar un archivo de prompt** en lugar de pasar texto por línea de comandos:

```bash
cat > modelos/prompt_test.txt <<'EOF'
Hola, ¿cómo estás? Respóndeme brevemente en español.
EOF
```

### 4.2 Ejecutar inferencia one-shot

```bash
llama-cli \
  -m "modelos/gemma-4-E4B-it-Q4_K_M.gguf" \
  -f "modelos/prompt_test.txt" \
  -c 2048 \
  -n 80 \
  -t 6 \
  --no-conversation \
  --simple-io
```

**Explicación de parámetros:**

| Flag | Valor | Significado |
|---|---|---|
| `-m` | ruta al GGUF | modelo a cargar |
| `-f` | archivo de prompt | evita problemas con quoting de Unicode |
| `-c` | 2048 | tamaño de contexto (bajo para test; en producción usar 8192–16384) |
| `-n` | 80 | máximo de tokens a generar |
| `-t` | 6 | threads CPU (= núcleos físicos del i7-10750H, no los 12 lógicos) |
| `--no-conversation` | — | no entra en modo chat interactivo |
| `--simple-io` | — | simplifica la salida, evita control codes |

### 4.3 Consumo de memoria esperado

Con contexto 2048 tokens:

```
Modelo (pesos Q4_K_M):  ~4.7 GB
KV cache:               ~2.0 GB
CPU_REPACK:             ~2.2 GB
Compute buffers:        ~0.8 GB
─────────────────────────────────
TOTAL:                  ~9.7 GB RAM
```

Con los 26 GiB libres del equipo de referencia, queda margen amplio para navegador, editor y otros procesos.

## 5. Problemas comunes

### 5.1 `error while loading shared libraries: libggml-base.so`

No está cargado `LD_LIBRARY_PATH`. Reejecutar el `export` o reiniciar la shell tras añadirlo a `~/.bashrc`.

### 5.2 El modelo arranca en modo interactivo vacío (aparecen `>` vacíos)

Ocurre cuando el prompt pasado por `-p "..."` contiene escapes Unicode no resueltos (por ejemplo `\u00bf` en lugar de `¿`). **Solución**: usar `-f archivo.txt` en vez de `-p`.

### 5.3 Uso de RAM muy alto (>15 GB) aun con modelo pequeño

Contexto por defecto demasiado grande. Explicitar `-c 2048` (o el valor deseado) para acotarlo. El KV cache escala linealmente con el contexto.

### 5.4 Velocidad lenta (<2 tok/s)

- Comprobar que `-t 6` (núcleos físicos, no hilos lógicos).
- Usar `--mlock` para evitar que el kernel envíe páginas del modelo a swap.
- Cerrar procesos pesados (navegador con muchas pestañas, IDE).

---

## Versiones de referencia

- **Kernel Linux**: 6.14.0-29-generic
- **Ubuntu**: 24.04
- **Python**: 3.12.3 (no requerida para inferencia con llama.cpp puro)
- **llama.cpp**: b8763
- **Gemma 4 E4B-it Q4_K_M**: descargado de `unsloth/gemma-4-E4B-it-GGUF` en abril 2026
