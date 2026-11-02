# CLAUDE.md — Gemma4-Mabel (entrenamiento del modelo)

Este repositorio contiene el **fine-tuning, la evaluación y el despliegue del
modelo** de Mabel. La aplicación web vive en otro repositorio.

> **Idioma**: todo el contenido de cara a la persona, en español. Los
> identificadores de código, en inglés.

## Los dos repositorios y cómo se hablan

| | Repositorio | Qué contiene |
|---|---|---|
| **Aquí** | `~/Desktop/Gemma4-Mabel` | Dataset, entrenamiento QLoRA, evaluación, export GGUF, despliegue en Modal |
| **Web** | `~/Desktop/Mabel-IA` | Backend FastAPI, frontend React, documentación canónica del sistema |

Son repositorios **separados a propósito**: distinto ciclo de vida, distintas
dependencias y distinto peso. No hay submódulos ni symlinks entre ellos.

El puente es un único documento: **`docs/MODEL_TRAINING.md` del repo web** es
el índice navegable hacia esta documentación. Ese archivo —y solo ese— es el
que hay que mantener al día cuando algo cambia aquí, para que quien lea el
repo web no se encuentre con una descripción obsoleta del modelo.

### Acceso al repo web desde aquí

`.claude/settings.local.json` concede acceso a `Mabel-IA/docs` y **solo a
esa carpeta**. Es deliberado: desde aquí se lee y se actualiza la
documentación del sistema, pero no se puede tocar por accidente el backend
ni el frontend, que tienen su propio flujo de ramas y su propio
`code-review` previo a cada commit.

Si hace falta algo más del repo web en una sesión concreta, se añade con
`/add-dir` y se retira al terminar, en lugar de ampliar el permiso fijo.

**Al actualizar `Mabel-IA/docs/` recuerda que estás commiteando en OTRO
repositorio**, con sus propias reglas: se commitea en `release/testing` y
`master` solo recibe merges explícitos.

## Estructura

```
data/        57 MB · train.jsonl, eval.jsonl, subsets, sintéticos y prompts
docs/        23 documentos · alcance, hardware, decisiones, evaluaciones
eval/        batería de evaluación y resultados
training/    export_gguf.py y el runbook de RunPod
modal/       despliegue de inferencia (OpenAI-compat, T4, scale-to-zero)
scripts/     utilidades
openspec/    propuestas y especificaciones
chat.py      cliente de prueba por consola
```

## Contexto del sistema al que sirve este modelo

- **Modelo**: Gemma 4 E4B (~4 B efectivos), fine-tuning **QLoRA con Unsloth**.
  Entrenamiento final en **RunPod** tras el fallo de la GPU local. Export
  **GGUF Q4_K_M ~3,5 GB**. Scorecard post-fine-tuning **4,37/5**, manejo de
  crisis 100 %.
- **Inferencia**: Modal.com, NVIDIA T4, OpenAI-compat, scale-to-zero a los
  5 min. **Arranque en frío de 60–90 s**, que la aplicación comunica
  explícitamente al estudiante — cualquier cambio que lo alargue tiene
  consecuencias visibles en la interfaz, no solo en las métricas.
- **Prompt del sistema**: fijo, de 151 palabras, y **no se edita**: editarlo
  degrada la calidad del fine-tune. Vive en el repo web
  (`backend/app/services/llm/prompts.py`) y se selecciona con
  `LLM_FLAVOR=mabel_gemma4`.
- **Identidad**: Mabel es la única identidad del asistente. Nunca debe
  exponer que hay un Gemma detrás.
- **Destino**: piloto con 30 estudiantes de la Universidad Manuela Beltrán.
  Es psicoeducativo, **no sustituye atención profesional**, y así lo declara
  la aplicación.

## Reglas al trabajar aquí

- **Los pesos no entran en git.** `.gitignore` ya excluye `*.gguf`, `*.bin`,
  `*.safetensors` y `checkpoints/`. Si descargas modelos, déjalos fuera del
  repositorio: el disco de esta máquina ha estado al 94 %, y quedarse sin
  espacio hace que macOS empiece a matar procesos.
- **El dataset es material sensible.** Son conversaciones de counselling. No
  se pegan ejemplos en informes ni se suben a servicios de terceros.
- **Toda afirmación sobre el modelo va con su medición.** Las comparativas de
  `docs/` son la fuente; si un número cambia, se actualiza el documento y su
  reflejo en `Mabel-IA/docs/MODEL_TRAINING.md`.
