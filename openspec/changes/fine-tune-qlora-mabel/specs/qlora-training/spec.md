## ADDED Requirements

### Requirement: Entorno de entrenamiento Python aislado
El sistema SHALL tener un entorno virtual Python con Unsloth y todas las dependencias necesarias instaladas y verificadas.

#### Scenario: Creación del venv e instalación
- **WHEN** se ejecuta la creación del venv y la instalación de Unsloth
- **THEN** el entorno contiene: unsloth, transformers, peft, bitsandbytes, trl, accelerate, datasets, sentencepiece, y torch con soporte CUDA, verificable con `python -c "import unsloth; import torch; print(torch.cuda.is_available())"`

#### Scenario: Verificación de CUDA y GPU
- **WHEN** se ejecuta la verificación del entorno
- **THEN** PyTorch detecta la RTX 2060 Mobile con 6 GB VRAM y CUDA disponible

### Requirement: Prototipo de validación con E2B
El sistema SHALL ejecutar un entrenamiento de prueba con Gemma 4 E2B sobre 200 ejemplos del dataset para validar que el pipeline funciona end-to-end antes de invertir en E4B.

#### Scenario: Entrenamiento prototipo exitoso
- **WHEN** se ejecuta el entrenamiento con E2B, 200 ejemplos, 1 época, mismos parámetros que E4B (r=32, lora_alpha=64, etc.)
- **THEN** el entrenamiento completa sin errores OOM, el training loss disminuye, y el modelo genera una respuesta coherente en español a un prompt de prueba

#### Scenario: Detección de OOM en prototipo
- **WHEN** el entrenamiento prototipo falla con error OOM (Out Of Memory)
- **THEN** se reduce context_length a 1024 y/o r a 16 y se reintenta, documentando el ajuste

### Requirement: Entrenamiento completo con E4B
El sistema SHALL ejecutar el fine-tuning de Gemma 4 E4B con QLoRA sobre el dataset completo usando los parámetros documentados en docs/21.

#### Scenario: Entrenamiento completo exitoso
- **WHEN** se ejecuta el entrenamiento con E4B, dataset completo (~21K), 3 épocas, parámetros de docs/21
- **THEN** el entrenamiento completa las 3 épocas, el training loss converge, los adapters LoRA se guardan en `outputs/`, y el proceso se documenta con capturas de TensorBoard

#### Scenario: Monitorización de temperatura y VRAM
- **WHEN** el entrenamiento está en progreso
- **THEN** se monitoriza `nvidia-smi` para VRAM y `sensors` para temperatura, pausando entre épocas si hay thermal throttling

#### Scenario: Guardado de checkpoints intermedios
- **WHEN** el entrenamiento completa cada época
- **THEN** se guarda un checkpoint del adapter en `outputs/checkpoint-epoch-N/` para poder retomar si hay interrupción
