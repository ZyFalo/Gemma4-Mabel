# 27 — Bitácora de entrenamiento (§7-§9)

Registro cronológico del proceso de fine-tuning de Mabel. Las decisiones técnicas mayores se documentan en `03-decisiones.md`; este archivo captura el detalle operativo (configuraciones reales, problemas encontrados, métricas observadas, tiempos, costos).

---

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
| Precisión | fp16 | **bf16** (mejor numérica para Gemma 3n) |
| GPU | RTX 2060 Mobile 6 GB | RTX 4090 24 GB |
| Evaluation durante training | No prevista | **Sí, por época** + best-model-selection |
| Save strategy | Solo final | **Por época**, top 2 conservados |

---

## §7.2 Prototipo E2B en RunPod — PENDIENTE

*Sección a completar tras la primera ejecución exitosa del prototipo en RunPod. Incluirá: tiempo real de descarga + carga, VRAM pico observada, curva de loss, resultados de las 3 inferencias diagnósticas.*

---

## §8 Entrenamiento real E4B — PENDIENTE

*Sección a completar tras §8. Incluirá: tiempo total, eval_loss por época, training_loss por step, decisión final de best-model, VRAM pico, costo real cobrado por RunPod.*

---

## §9 Export GGUF — PENDIENTE

*Sección a completar tras §9. Incluirá: tiempo de merge, tiempo de cuantización, tamaño final del .gguf, sanity check con llama-server.*
