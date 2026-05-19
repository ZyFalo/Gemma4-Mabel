"""
§8 Entrenamiento real — Gemma 4 E4B (Mabel) — RunPod RTX 4090

Dataset: 7.870 ej (60% EN counselling + 40% ES sintético colombiano), 3 épocas.
Hiperparámetros docs/21 + dos protecciones contra regresión:
  - evaluation_strategy="epoch" sobre eval.jsonl (500 ej estratificados)
  - load_best_model_at_end=True con metric_for_best_model="eval_loss"
    → si la época 3 empeora respecto a la 2, automáticamente vuelve a la 2

Precisión: bf16 (4090 lo soporta nativo, evita el AltUp+fp32 issue del local).
Checkpoints: 1 por época en outputs/real_e4b/checkpoint-epoch-N (save_total_limit=2).
Adapter final: outputs/real_e4b/adapter (versión que minimizó eval_loss).

Ejecutar (después del prototipo §7 exitoso):
    cd /workspace/Gemma-4
    nohup python3 training/train_real_e4b.py > outputs/real_e4b/run.log 2>&1 &
    tail -f outputs/real_e4b/run.log
"""
import json
import os
import time
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ----- Configuración -----
MODEL_NAME = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LENGTH = 2048
TRAIN_PATH = "data/train.jsonl"
EVAL_PATH = "data/eval.jsonl"
OUTPUT_DIR = "outputs/real_e4b"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"
SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"[setup] OUTPUT_DIR  = {OUTPUT_DIR}")
print(f"[setup] TRAIN_PATH  = {TRAIN_PATH}")
print(f"[setup] EVAL_PATH   = {EVAL_PATH}")

# ----- 1. Cargar modelo base en 4-bit NF4 -----
print(f"\n[1/5] Cargando {MODEL_NAME} en 4-bit NF4...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # auto-detect bf16 en Ada Lovelace
)
print(f"[1/5] Modelo cargado en {time.time()-t0:.1f}s")

# ----- 2. LoRA r=32, α=64, 7 módulos (docs/21) -----
print("\n[2/5] Aplicando LoRA r=32, alpha=64, 7 módulos...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# ----- 3. Cargar y formatear datasets (train + eval) -----
def load_and_format(path: str, label: str) -> Dataset:
    raw = [json.loads(line) for line in open(path)]
    print(f"[3/5] {label}: {len(raw)} ejemplos cargados desde {path}")
    ds = Dataset.from_list(raw)
    ds = ds.map(
        lambda ex: {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False,
            )
        },
        remove_columns=["source", "lang", "messages"],
    )
    return ds

print("\n[3/5] Cargando datasets...")
train_ds = load_and_format(TRAIN_PATH, "TRAIN")
eval_ds = load_and_format(EVAL_PATH, "EVAL")

# ----- 4. SFTTrainer con eval + best-checkpoint protection -----
print("\n[4/5] Configurando SFTTrainer (con eval por época + best-model selection)...")
config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=False,
    bf16=True,
    optim="adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,

    # ----- Protección contra regresión -----
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    seed=SEED,
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=config,
)

# ----- 5. Entrenar + guardar adapter mejor-época -----
total_steps = len(train_ds) * 3 // 8  # 7870 * 3 / 8 ≈ 2.951 pasos
print(f"\n[5/5] Iniciando entrenamiento ({len(train_ds)} ej × 3 ep ÷ batch 8 ≈ {total_steps} pasos)...")
t0 = time.time()
stats = trainer.train()
elapsed_h = (time.time() - t0) / 3600
print(f"\n[5/5] Entrenamiento terminado en {elapsed_h:.2f} h")
print(f"  - Loss final (train): {stats.training_loss:.4f}")

print(f"\n[5/5] Guardando adapter LoRA (best-eval) en {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"[5/5] OK — adapter listo en {ADAPTER_DIR}")

print("\n" + "=" * 60)
print("Entrenamiento real terminado. Siguientes pasos:")
print("  1. python3 training/test_inference.py --model e4b   (§7.4/§10 sanity check)")
print("  2. python3 training/export_gguf.py                  (§9 export GGUF Q4_K_M)")
print("=" * 60)
