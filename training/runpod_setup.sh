#!/usr/bin/env bash
# Setup de RunPod Pod (PyTorch 2.4 / CUDA 12.4 template) para entrenar Mabel.
# Ejecutar UNA SOLA VEZ tras crear el pod, desde /workspace:
#
#   bash training/runpod_setup.sh
#
# Tras esto, el pod queda listo para correr train_prototype_e2b.py / train_real_e4b.py.

set -euo pipefail

echo "===== [1/4] Verificando GPU y CUDA ====="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0))"

echo ""
echo "===== [2/4] Instalando Unsloth + dependencias ====="
pip install --upgrade pip
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"
pip install --no-cache-dir timm                  # Gemma 3n vision tower obligatorio
pip install --no-cache-dir bitsandbytes trl peft accelerate datasets

echo ""
echo "===== [3/4] HuggingFace login (necesario para Gemma 4) ====="
echo "Pega tu HF token (necesita licencias Gemma 4 aceptadas en https://huggingface.co/google/gemma-4-E2B-it y https://huggingface.co/google/gemma-4-E4B-it):"
huggingface-cli login

echo ""
echo "===== [4/4] Sanity check: import Unsloth + carga E2B mini ====="
python3 -c "
from unsloth import FastLanguageModel
print('Unsloth: OK')
from unsloth.models.loader_utils import get_model_name
print('E2B resuelve a:', get_model_name('unsloth/gemma-4-E2B-it', load_in_4bit=True))
print('E4B resuelve a:', get_model_name('unsloth/gemma-4-E4B-it', load_in_4bit=True))
"

echo ""
echo "===== Setup completo ====="
echo "Siguiente paso: python3 training/train_prototype_e2b.py"
