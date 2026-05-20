# 27 — Bitácora de entrenamiento (§7-§9)

Registro cronológico del proceso de fine-tuning de Mabel. Las decisiones técnicas mayores se documentan en `03-decisiones.md`; este archivo captura el detalle operativo (configuraciones reales, problemas encontrados, métricas observadas, tiempos, costos).

---

> **Nota aclaratoria sobre nomenclatura (2026-05-19)**: en este documento "Gemma 3n" aparece como el modelo que efectivamente intentamos cargar en local. Esto se debe a que Unsloth 2026.4.4 (versión instalada en la máquina local) **no reconocía aún el ID `unsloth/gemma-4-E2B-it`** y forzaba al alias legacy `unsloth/gemma-3n-E2B-it`. El **modelo objetivo del proyecto siempre fue y sigue siendo Gemma 4 oficial** (`google/gemma-4-E2B-it` y `google/gemma-4-E4B-it`), que es lo que se entrena en RunPod con Unsloth fresco desde GitHub. Gemma 3n y Gemma 4 son familias distintas; la causa raíz del bloqueo (AltUp+Turing+6GB) aplica a ambas por compartir arquitectura multimodal.

## §7.0 Intento local — BLOQUEADO (2026-05-10)

**Hardware**: RTX 2060 Mobile, 6 GB VRAM, kernel Linux 6.17.0-23, driver NVIDIA 580.142 (recompilado via DKMS tras update de kernel).

**Preparativos completados antes del bloqueo**:
- Subset estratificado de 200 ejemplos en `data/train_subset200.jsonl` (62 mentalchat_b / 58 amod / 48 normal / 26 crisis / 6 normal_b · 120 EN / 80 ES).
- Liberados 22 GB borrando GGUFs viejos (`gemma-4-E4B-it-Q4_K_M.gguf` y `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`). Baselines preservados en `eval/results/*.md`.
- DKMS instalado para recompilación automática del módulo nvidia ante updates futuros del kernel.
- Scripts `training/train_prototype_e2b.py` y `training/test_inference.py` creados.

**Intento de ejecución del prototipo (3 intentos)**:

| Intento | Error | Diagnóstico |
|---|---|---|
| 1 | `NotImplementedError: unsloth/gemma-4-E2B-it is not supported in your current Unsloth version` | Unsloth 2026.4.4 no reconoce el ID con prefijo `gemma-4-*`; solo el alias legacy `gemma-3n-*` |
| 2 | `ImportError: TimmWrapperModel requires the timm library` | Gemma 3n incluye vision tower obligatorio (MobileNetV5); requiere `timm` |
| 3 | `ValueError: Some modules are dispatched on the CPU or the disk` precedido de `Unsloth: Using float16 precision for gemma3n won't work! Using float32` | **Causa raíz**: Gemma 3n tiene el módulo AltUp que produce NaN en fp16; Unsloth fuerza fp32; en fp32 no cabe en 5.6 GB libres |

**Causa raíz definitiva**: combinación **AltUp + arquitectura Turing + 6 GB VRAM**. La RTX 2060 (Turing) no soporta bf16 (sí lo soportan Ampere y posteriores), así que la única alternativa a bf16 era fp16, pero AltUp lo prohíbe. La conclusión es que Gemma 3n/4 no es entrenable en Turing con 6 GB.

**Estimación de VRAM en fp32 (lo que pedía Unsloth)**:
```
Pesos E2B en 4-bit NF4         ≈ 2.5 GB
Vision tower MobileNetV5       ≈ 1.0 GB
Activaciones + KV cache fp32   ≈ 2.5 GB
Buffers de cómputo             ≈ 0.5 GB
TOTAL                          ≈ 6.5 GB → no cabe en 5.6 GB libres
```

**Decisión**: pivote a RunPod (cloud GPU). Ver D-019 en `docs/03-decisiones.md`.

---

## §7.1 Pivote a RunPod — Configuración (2026-05-10)

**Proveedor elegido**: RunPod.
**GPU**: RTX 4090 24 GB, Community Cloud, $0.34/h.
**Template**: RunPod Pytorch 2.4 (incluye CUDA 12.4, Jupyter, SSH).
**Volume Disk**: 50 GB.

**Costo total estimado**: ~$1.80 USD para todo el flujo (§7 + §8 + §9 + §10).
**Recarga recomendada**: $5 USD (margen 2.5×).

**Scripts adaptados a RunPod**:
- `training/train_prototype_e2b.py` actualizado: `MODEL_NAME = "unsloth/gemma-4-E2B-it"`, `bf16=True, fp16=False`.
- `training/train_real_e4b.py` nuevo: dataset completo + 3 épocas + `eval_strategy="epoch"` + `load_best_model_at_end=True` con `eval_loss` (protección contra regresión por overfitting).
- `training/test_inference.py` actualizado: arg `--model e2b|e4b` para probar ambos.
- `training/export_gguf.py` nuevo: merge LoRA + export Q4_K_M.
- `training/runpod_setup.sh` nuevo: instalación reproducible de dependencias.
- `training/README_runpod.md` nuevo: guía paso a paso.

**Hiperparámetros confirmados (docs/21 sin cambios + bf16)**:
- `r=32, lora_alpha=64`, 7 módulos LoRA
- `lr=1e-4`, `num_train_epochs=3`, `optim=adamw_8bit`
- `per_device_train_batch_size=1, gradient_accumulation_steps=8` (batch efectivo 8)
- `max_seq_length=2048`, `use_gradient_checkpointing="unsloth"`

**Diferencias respecto al plan original**:
| Parámetro | Plan original (local) | RunPod actual |
|---|---|---|
| Precisión | fp16 | **bf16** (mejor numérica para Gemma 4 y resuelve AltUp) |
| GPU | RTX 2060 Mobile 6 GB | RTX 4090 24 GB |
| Evaluation durante training | No prevista | **Sí, por época** + best-model-selection |
| Save strategy | Solo final | **Por época**, top 2 conservados |

---

## §7.1.5 Stack real instalado en RunPod (2026-05-20)

> **Sección viva**: actualizar con cada ajuste detectado durante §7-§9. La verdad es lo que está corriendo en el pod, no lo que está en `requirements.txt`.

### Pod elegido

| Campo | Valor |
|---|---|
| Cloud type | **Secure Cloud** (sin preemption) |
| Región | CA-MTL-1 (Montreal, Canadá) |
| GPU | **RTX 4090, 24 GB VRAM** (Ada Lovelace, bf16 nativo) |
| RAM | 46 GB |
| vCPU | 12 |
| Container disk | 20 GB |
| Volume disk | 50 GB en `/workspace` |
| Pod template | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Conexión | SSH directo (clave ed25519) + Jupyter Lab |
| Tarifa | $0.69/hr GPU + $0.01/hr disco corriendo |
| Pod ID inicial | `8zolpapsdmj5rn` (puede cambiar entre sesiones) |

### Stack Python final (después de ajustes)

| Paquete | Versión instalada | Notas |
|---|---|---|
| Python | 3.11 | Del container oficial |
| CUDA Toolkit | 12.4.1 | Del container oficial |
| **torch** | **2.6.0+cu124** | Upgrade desde 2.4.1 — Unsloth Zoo requiere ≥2.5 |
| torchvision | 0.21.0+cu124 | Upgrade conjunto con torch |
| triton | 3.2.0 | Auto-upgrade con torch 2.6 |
| **torchao** | **0.16.0** | Downgrade desde 0.17 — la 0.17 requiere `torch.utils._pytree.register_constant` (solo en torch 2.7+) |
| bitsandbytes | 0.49.2 | OK con torch 2.6 |
| unsloth | 2026.5.5 | Instalado desde git fresco |
| unsloth_zoo | 2026.5.3 | Instalado desde git fresco |
| transformers | 5.5.0 | OK |
| trl | 0.24.0 | OK |
| peft | 0.19.1 | OK |
| accelerate | 1.13.0 | OK |
| datasets | 4.3.0 | OK |
| huggingface_hub | 1.15.0 | OK |
| timm | 1.0.27 | Requerido por Gemma 3n/4 vision tower |

### Ajustes/parches aplicados al setup (2026-05-20)

Estos son los pasos que NO están reflejados aún en `training/runpod_setup.sh` y que debemos consolidar al final del proceso:

1. **`huggingface-cli` deprecado → `hf auth login`**
   - La CLI `huggingface-cli` ya no existe en `huggingface_hub` 1.15.0+.
   - El comando correcto es `hf auth login` (interactivo, pide token y guarda en `/root/.cache/huggingface/token`).
   - `hf whoami` tampoco existe; el equivalente es `hf auth whoami` (no crítico, el login mismo confirma identidad).
   - **TODO en script**: cambiar `huggingface-cli login` → `hf auth login` en `training/runpod_setup.sh`.

2. **torch 2.4.1 → 2.6.0 (upgrade necesario)**
   - El container oficial `runpod/pytorch:2.4` trae torch 2.4.1, pero `unsloth_zoo` 2026.5.3 hace `inspect.getsource(torch._inductor.config)` que falla en torch 2.4.
   - Fix aplicado:
     ```bash
     pip install --upgrade "torch>=2.5,<2.7" torchvision \
       --index-url https://download.pytorch.org/whl/cu124
     ```
   - Resultado: torch 2.6.0+cu124 instalado.
   - **TODO en script**: añadir este `pip install` como paso obligatorio antes de la verificación de Unsloth.

3. **torchao 0.17 → 0.16 (downgrade necesario)**
   - torchao 0.17.0 (instalada como dep de Unsloth) llama a `torch.utils._pytree.register_constant` que solo existe en torch 2.7+.
   - Como tenemos torch 2.6, hay que bajar torchao.
   - Fix aplicado:
     ```bash
     pip install --upgrade "torchao>=0.13,<0.17" --force-reinstall --no-deps
     ```
   - Resultado: torchao 0.16.0 instalado.
   - **TODO en script**: añadir este `pip install` como paso obligatorio.

4. **Conflicto residual reportado por pip (no bloqueante)**:
   - `torchaudio 2.4.1 requires torch==2.4.1, but you have torch 2.6.0+cu124 which is incompatible`
   - No usamos torchaudio, ignorar. Si en algún momento Unsloth lo necesita, hacer `pip install --upgrade torchaudio --index-url https://download.pytorch.org/whl/cu124`.

5. **Warning de Flash Attention 2 (no bloqueante)**:
   - `Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead. No performance changes will be seen.`
   - Unsloth mismo confirma que no hay degradación. Ignorar.

8. **stdout buffering con `nohup` oculta los loss reports en tiempo real**:
   - Al lanzar el training con `nohup python3 training/train_real_e4b.py > outputs/real_e4b/run.log 2>&1 &`, Python detecta que stdout NO es una TTY (es un archivo) y aplica buffering agresivo. Los `print({'loss': X.XX, ...})` que `SFTTrainer` ejecuta cada `logging_steps=10` quedan en buffer de memoria del proceso y **no se escriben a disco hasta que el buffer se llena o termina la época**.
   - Adicionalmente, las barras de progreso de tqdm usan `\r` (carriage return) para sobrescribir la misma línea — eso ya hace ruido en el log file pero NO oculta los reports de loss (ese es el buffering).
   - Síntoma observado: el log file mostraba solo `8%|▊  | 233/3015 [20:25<3:55:04, 5.07s/it]` durante 30+ min, sin reports `{'loss': X.XX}`. `grep "loss" run.log` devolvía 0 hits. El training estaba ejecutándose correctamente (GPU al 31-90%, VRAM estable en 13 GB, proceso vivo con PID consistente), pero los reports estaban bufferizados en memoria.
   - Mitigación en sesión activa: ninguna (ya está corriendo). Los reports aparecerán al final de cada época cuando `SFTTrainer` ejecuta `eval` y hace flush implícito del buffer.
   - **Fix para futuras runs**:
     ```bash
     # Opción 1: -u fuerza unbuffered en Python
     nohup python3 -u training/train_real_e4b.py > outputs/real_e4b/run.log 2>&1 &

     # Opción 2: variable de entorno (equivalente)
     PYTHONUNBUFFERED=1 nohup python3 training/train_real_e4b.py > outputs/real_e4b/run.log 2>&1 &
     ```
   - **TODO en script**: añadir `-u` al comando recomendado en `training/README_runpod.md` (sección §8 lanzamiento). El script `train_real_e4b.py` ya está bien; solo es ajustar cómo se invoca con `nohup`.
   - **Verificación cuando vuelva a aparecer este patrón**: si `grep loss` devuelve vacío pero `ps aux | grep train` muestra proceso vivo + `nvidia-smi` muestra GPU usage + `wc -l run.log` muestra que el log crece → confiar en que está entrenando, solo esperar al fin de época para ver los reports.

7. **HuggingFace cache se llena el container disk (20 GB) — mover al volume disk**:
   - El cache por defecto vive en `/root/.cache/huggingface/`, que en el template RunPod está en el **container disk de 20 GB** (ephemeral, se borra al stop del pod).
   - Al descargar E2B (~2 GB) + intentar descargar E4B (~5-7 GB) + dependencias preinstaladas → `OSError: No space left on device (os error 28)` durante `_download_to_tmp_and_move`.
   - El volume disk de 50 GB en `/workspace` queda libre y es persistente entre stops del pod.
   - Fix aplicado en sesión activa:
     ```bash
     mkdir -p /workspace/hf_cache
     mv /root/.cache/huggingface/* /workspace/hf_cache/ 2>/dev/null
     export HF_HOME=/workspace/hf_cache
     export HF_HUB_CACHE=/workspace/hf_cache/hub
     # Persistir en bashrc para futuras conexiones SSH
     cat >> ~/.bashrc << 'EOF'
     export HF_HOME=/workspace/hf_cache
     export HF_HUB_CACHE=/workspace/hf_cache/hub
     EOF
     ```
   - **TODO en script**: añadir estas variables de entorno y el `mkdir` en `training/runpod_setup.sh` ANTES del `hf auth login`, para evitar el OOM de disco en setups frescos.

6. **Gemma 4 multimodal exige content como lista de bloques en `apply_chat_template`**:
   - El processor de Gemma 4 (multimodal: texto + imagen + video + audio) requiere `message["content"]` como **lista de dicts tipados**, no como string. Si se pasa string falla con:
     ```
     TypeError: string indices must be integers, not 'str'
     ```
     en `transformers/processing_utils.py` línea 1807 al buscar `content["type"]`.
   - Falló en `training/test_inference.py` al invocar `apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`.
   - Fix aplicado (commit posterior al prototipo): cambiar
     ```python
     {"role": "user", "content": "texto"}
     ```
     por
     ```python
     {"role": "user", "content": [{"type": "text", "text": "texto"}]}
     ```
   - **Nota**: en `train_prototype_e2b.py` el dataset se pasa con `content` como string y funciona porque ahí se usa `tokenize=False` (genera el texto plano y SFTTrainer lo retokeniza); el path multimodal no se activa. Solo afecta a inferencia con `tokenize=True`.
   - **TODO en script**: aplicar el mismo patrón al `train_real_e4b.py` si en algún momento se invoca chat template con tokenize=True (no aplica al training actual, pero sí a cualquier inferencia futura).

### Sanity check final que confirma stack OK

```python
from unsloth import FastLanguageModel
from unsloth.models.loader_utils import get_model_name
# E2B → unsloth/gemma-4-e2b-it-unsloth-bnb-4bit
# E4B → unsloth/gemma-4-e4b-it-unsloth-bnb-4bit
```

Ambos IDs `unsloth/gemma-4-E2B-it` y `unsloth/gemma-4-E4B-it` resuelven correctamente al modelo cuantizado pre-bnb-4bit de Unsloth.

### Pendientes consolidación post-entrenamiento

Cuando termine §9, consolidar todos los ajustes en `training/runpod_setup.sh` para que un futuro setup desde cero sea reproducible en un comando. La versión actual del script no es ejecutable end-to-end por los puntos 1-3 listados arriba.

---

## §7.2 Prototipo E2B en RunPod — PENDIENTE

*Sección a completar tras la primera ejecución exitosa del prototipo en RunPod. Incluirá: tiempo real de descarga + carga, VRAM pico observada, curva de loss, resultados de las 3 inferencias diagnósticas.*

---

## §8 Entrenamiento real E4B — PENDIENTE

*Sección a completar tras §8. Incluirá: tiempo total, eval_loss por época, training_loss por step, decisión final de best-model, VRAM pico, costo real cobrado por RunPod.*

---

## §9 Export GGUF — PENDIENTE

*Sección a completar tras §9. Incluirá: tiempo de merge, tiempo de cuantización, tamaño final del .gguf, sanity check con llama-server.*
