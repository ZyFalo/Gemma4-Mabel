"""
§7 Prototipo de entrenamiento — Gemma 4 E2B (Mabel)

Objetivo: validar que el pipeline completo (carga + LoRA + SFTTrainer + save)
funciona en la RTX 2060 6GB con un subset pequeño antes de comprometer
horas de GPU al entrenamiento real con E4B.

- Modelo: unsloth/gemma-4-E2B-it (4-bit NF4)
- Dataset: data/train_subset200.jsonl (200 ej estratificados)
- Épocas: 1 (real serán 3 con E4B)
- Hiperparámetros: idénticos a docs/21 salvo num_train_epochs

Ejecutar:
    cd "/home/zyfalo/Escritorio/Gemma 4"
    source .venv/bin/activate
    python3 training/train_prototype_e2b.py 2>&1 | tee outputs/prototype_e2b/run.log
"""
import json
import os
import time
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ----- Configuración -----
MODEL_NAME = "unsloth/gemma-3n-E2B-it"  # alias HF de Gemma 4 E2B; Unsloth lo redirige a unsloth/gemma-3n-e2b-it-unsloth-bnb-4bit
MAX_SEQ_LENGTH = 2048
DATA_PATH = "data/train_subset200.jsonl"
OUTPUT_DIR = "outputs/prototype_e2b"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"
SEED = 42

# ----- Setup -----
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f"[setup] OUTPUT_DIR = {OUTPUT_DIR}")
print(f"[setup] DATA_PATH  = {DATA_PATH}")

# ----- 1. Cargar modelo base en 4-bit -----
print(f"\n[1/5] Cargando {MODEL_NAME} en 4-bit NF4...")
t0 = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # auto: fp16 en Turing
)
print(f"[1/5] Modelo cargado en {time.time()-t0:.1f}s")

# ----- 2. Aplicar LoRA (parámetros docs/21) -----
print("\n[2/5] Aplicando LoRA r=32, alpha=64, 7 módulos, gradient_checkpointing=unsloth...")
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

# ----- 3. Cargar y formatear dataset -----
print(f"\n[3/5] Cargando dataset desde {DATA_PATH}...")
raw = [json.loads(line) for line in open(DATA_PATH)]
print(f"[3/5] {len(raw)} ejemplos cargados")

def format_example(ex):
    return {
        "text": tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

ds = Dataset.from_list(raw).map(
    format_example,
    remove_columns=["source", "lang", "messages"],
)
print(f"[3/5] Dataset formateado. Preview primer ejemplo (400 chars):")
print(ds[0]["text"][:400])
print("...")

# ----- 4. Trainer (SFTConfig en TRL 0.24) -----
print("\n[4/5] Configurando SFTTrainer...")
config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,            # prototipo: 1 ep (real serán 3)
    learning_rate=1e-4,
    fp16=True,
    bf16=False,                    # Turing no soporta bf16
    optim="adamw_8bit",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="no",            # prototipo: no checkpoints intermedios
    seed=SEED,
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=config,
)

# ----- 5. Entrenar y guardar adapter -----
print("\n[5/5] Iniciando entrenamiento...")
print(f"  - 200 ej × 1 ep ÷ batch_efectivo 8 = {200 // 8} pasos de optimización")
t0 = time.time()
stats = trainer.train()
elapsed_min = (time.time() - t0) / 60
print(f"\n[5/5] Entrenamiento completado en {elapsed_min:.1f} min")
print(f"  - Loss final: {stats.training_loss:.4f}")

print(f"\n[5/5] Guardando adapter LoRA en {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"[5/5] OK — adapter listo en {ADAPTER_DIR}")

print("\n" + "=" * 60)
print("Prototipo terminado. Siguiente paso:")
print("  python3 training/test_inference.py")
print("=" * 60)
