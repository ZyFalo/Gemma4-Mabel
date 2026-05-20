"""
§9 Exportación del modelo fine-tuneado a GGUF Q4_K_M (para llama.cpp/llama-server).

Pipeline:
  1. Carga modelo base E4B + adapter LoRA del §8
  2. Merge adapter en pesos base (en fp16 para preservar calidad)
  3. Exporta a GGUF Q4_K_M (~4.7 GB esperado)

Resultado: modelos/gemma-4-E4B-mabel-Q4_K_M.gguf

Ejecutar en RunPod (necesita ~15-20 GB libres en disco para el merged intermedio):
    cd /workspace/Gemma-4
    python3 training/export_gguf.py 2>&1 | tee outputs/real_e4b/export.log

Después: scp/rsync el GGUF al laptop local y reemplaza el archivo en modelos/.
"""
import time
from pathlib import Path

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/gemma-4-E4B-it"

# Decisión 2026-05-20: usar checkpoint-3015 (epoch 3) en lugar del adapter
# seleccionado por load_best_model_at_end (epoch 2 según eval_loss 2.81 vs 2.94).
# Razón: la validación cualitativa por inferencia comparativa (3 prompts
# diagnósticos sobre cada checkpoint) demostró que epoch 3 supera a epoch 2 en:
#   - Resistencia al constraint de lista numerada (D-020: rechazo amable + redirección
#     a Bienestar UMB) — epoch 2 falla, epoch 3 redirige correctamente
#   - Concisión y naturalidad en respuesta a crisis (sin "solo/a" forzado)
# eval_loss numérico no es métrica confiable en modelos multimodales por el
# masking diferencial train/eval (ver docs/27 §7.1.5 ajuste #10).
# Detalle completo del análisis comparativo en docs/27 §8.
ADAPTER_DIR = "outputs/real_e4b/checkpoint-3015"

MERGED_DIR = "outputs/real_e4b/merged"
GGUF_DIR = "modelos"
GGUF_NAME = "gemma-4-E4B-mabel"
MAX_SEQ_LENGTH = 2048

Path(GGUF_DIR).mkdir(parents=True, exist_ok=True)

print(f"[1/3] Cargando base {MODEL_NAME} + adapter {ADAPTER_DIR}...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,        # apunta al adapter; Unsloth resuelve el base
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,            # para merge: 16-bit
    dtype=None,
)
print(f"[1/3] Cargado en {time.time()-t0:.1f}s")

print(f"\n[2/3] Merge adapter → {MERGED_DIR} (16-bit)...")
t0 = time.time()
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
)
print(f"[2/3] Merge OK en {time.time()-t0:.1f}s")

print(f"\n[3/3] Export GGUF Q4_K_M → {GGUF_DIR}/{GGUF_NAME}-Q4_K_M.gguf...")
t0 = time.time()
model.save_pretrained_gguf(
    f"{GGUF_DIR}/{GGUF_NAME}",
    tokenizer,
    quantization_method="q4_k_m",
)
print(f"[3/3] Export OK en {(time.time()-t0)/60:.1f} min")

print("\n" + "=" * 60)
print("Export terminado. Para descargar al laptop local desde RunPod:")
print(f"  scp -P <port> root@<pod-ip>:/workspace/Gemma-4/{GGUF_DIR}/{GGUF_NAME}*Q4_K_M.gguf ~/Escritorio/Gemma\\ 4/modelos/")
print("=" * 60)
