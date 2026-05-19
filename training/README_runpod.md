# Guía paso a paso — Entrenar Mabel en RunPod

> Ver `docs/03-decisiones.md` D-019 para el contexto del pivote a RunPod.

## Costo esperado total

| Fase | Tiempo en RTX 4090 | Costo (Community $0.34/h) |
|---|---|---|
| §7 Prototipo E2B | ~15 min | $0.09 |
| §8 Entrenamiento real E4B | ~4 h | $1.36 |
| §9 Export GGUF Q4_K_M | ~30 min | $0.17 |
| §10 Eval batería | ~30 min | $0.17 |
| **TOTAL** | **~5-6 h** | **~$1.80 USD** |

**Recarga recomendada:** **$5 USD** (margen 2.5× para imprevistos: re-runs, debug, descargas lentas).

---

## Antes de crear el pod

1. **Cuenta RunPod**: registrar en https://runpod.io y recargar $5 USD vía tarjeta/PayPal/cripto.
2. **HuggingFace token**: ir a https://huggingface.co/settings/tokens → "New token" → tipo "Read" → copiar.
3. **Licencia Gemma**: ir a https://huggingface.co/google/gemma-3n-e4b-it y aceptar términos (mismo modelo que `unsloth/gemma-4-E4B-it` por ahora).
4. **Push del repo a GitHub** desde el laptop local (si no está al día):
   ```bash
   git add -A && git commit -m "scripts RunPod listos" && git push
   ```

---

## Paso 1 — Crear el Pod (5 min)

1. Login en https://runpod.io/console/pods → **"Deploy"**.
2. **GPU**: filtrar por **RTX 4090** → elegir **Community Cloud** ($0.34/h, 24 GB VRAM).
3. **Template**: `RunPod Pytorch 2.4` (incluye CUDA 12.4, Jupyter, SSH).
4. **Volume Disk**: 50 GB (suficiente para modelo + merged + GGUF; $0.10/GB/mes mientras el pod corre).
5. **Container Disk**: dejar default 20 GB.
6. **Expose ports**: 8888 (Jupyter), 22 (SSH) — vienen activos por defecto.
7. **Deploy**. El pod arranca en ~30 s.

---

## Paso 2 — Conectarse y clonar el repo (3 min)

Opción A — Web Terminal (más simple):
- Click en el pod → tab "Connect" → "Start Web Terminal".

Opción B — SSH (recomendado para sesiones largas):
- Agregar tu SSH pública en https://runpod.io/console/user/settings → SSH Public Keys.
- Conectar: `ssh -p <PORT> root@<POD_IP>` (datos en tab "Connect").

Dentro del pod:
```bash
cd /workspace
git clone https://github.com/<tu-usuario>/<tu-repo>.git Gemma-4
cd Gemma-4
```

---

## Paso 3 — Setup del entorno (~5 min, $0.03)

```bash
bash training/runpod_setup.sh
```

Esto:
- Verifica GPU detectada (RTX 4090, 24 GB)
- Instala Unsloth última versión + `timm` + dependencias QLoRA
- Pide tu HF token (pega y enter)
- Hace sanity check de carga de IDs `unsloth/gemma-4-E2B-it` y `E4B-it`

---

## Paso 4 — §7 Prototipo E2B (~15 min, $0.09)

Valida que el pipeline completo funcione end-to-end (descarga, LoRA, train, save, inferencia) antes de comprometer 4 h al entrenamiento real.

```bash
python3 training/train_prototype_e2b.py 2>&1 | tee outputs/prototype_e2b/run.log
```

**Señales de éxito** (en `run.log`):
- `Modelo cargado en XX.Xs` (sin OOM)
- Loss decrece a lo largo de los ~25 pasos
- `Entrenamiento completado` + `Adapter listo`

Verificación rápida:
```bash
python3 training/test_inference.py --model e2b
```
Las 3 respuestas (saludo / crisis sutil / petición de lista) deben:
- Saludar como "Mabel" (no como asistente genérico)
- En crisis: mencionar Línea 123/106/155, preguntar persona de confianza
- En lista: NO dar `1. ... 2. ...`, sino sugerencias en prosa

**Si algo falla:** detener el pod (paga por tiempo), debugear local, redeployar.

---

## Paso 5 — §8 Entrenamiento real E4B (~4 h, $1.36)

```bash
# Lanzar en background con nohup, salir del Jupyter/SSH no lo mata:
nohup python3 training/train_real_e4b.py > outputs/real_e4b/run.log 2>&1 &

# Seguir el progreso:
tail -f outputs/real_e4b/run.log
```

**Esperado en log:**
- 3 épocas, cada una ~1.3 h
- Eval loss reportado al final de cada época
- Mensaje `load_best_model_at_end` al cierre (revierte a la mejor época si la última empeoró)

**Si la conexión SSH se corta:** el `nohup` mantiene el proceso vivo. Vuelve a conectar y `tail -f` retoma desde donde quedó.

---

## Paso 6 — §9 Export GGUF (~30 min, $0.17)

```bash
python3 training/export_gguf.py 2>&1 | tee outputs/real_e4b/export.log
```

Genera `modelos/gemma-4-E4B-mabel-Q4_K_M.gguf` (~4.7 GB).

---

## Paso 7 — Descargar artefactos al laptop local (~10-15 min)

Desde el laptop local (no en el pod):
```bash
# GGUF final (el que vas a usar con llama-server)
scp -P <PORT> root@<POD_IP>:/workspace/Gemma-4/modelos/gemma-4-E4B-mabel-Q4_K_M.gguf ~/Escritorio/Gemma\ 4/modelos/

# Adapter (por si quieres re-mergear con otra cuantización)
scp -P <PORT> -r root@<POD_IP>:/workspace/Gemma-4/outputs/real_e4b/adapter ~/Escritorio/Gemma\ 4/outputs/real_e4b/

# Logs (para docs/27)
scp -P <PORT> -r root@<POD_IP>:/workspace/Gemma-4/outputs/real_e4b/*.log ~/Escritorio/Gemma\ 4/outputs/real_e4b/
```

---

## Paso 8 — APAGAR EL POD (importante)

En la UI de RunPod: click en el pod → **"Stop Pod"** (no "Terminate" si quieres conservar el volumen).

> **Stop** = deja de cobrarse GPU pero sigue cobrándose el volume disk ($5/mes con 50 GB).
> **Terminate** = deja de cobrarse todo, **pierdes los datos del pod**.

Si ya descargaste todo lo importante: **Terminate** para no acumular costo.

---

## Paso 9 — Verificación local (§9.4-9.6)

Con el GGUF en `modelos/`:
```bash
# Asumiendo llama-server ya instalado del baseline
llama-server -m modelos/gemma-4-E4B-mabel-Q4_K_M.gguf --port 8080
# En otra terminal:
curl http://localhost:8080/health  # debe devolver {"status":"ok"}
python3 chat.py                     # script de chat ya existente
```

---

## Troubleshooting

| Síntoma | Causa probable | Fix |
|---|---|---|
| `NotImplementedError: gemma-4-E4B-it not supported` | Unsloth desactualizado | `pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` |
| `OOM CUDA` en E4B | `max_seq_length` muy alto | Bajar a 1536 en `train_real_e4b.py` |
| `eval_loss` sube en época 3 | Overfitting (esperable) | `load_best_model_at_end=True` ya lo maneja, conserva época 2 |
| Pod desconectado mid-training | SSH timeout, no afecta `nohup` | Reconectar, `tail -f outputs/real_e4b/run.log` |
| `huggingface-cli login` falla | Token sin permisos o licencia Gemma no aceptada | Regenera token tipo "Read", acepta licencia en HF |
