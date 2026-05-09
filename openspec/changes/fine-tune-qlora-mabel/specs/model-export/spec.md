## ADDED Requirements

### Requirement: Merge de adapters LoRA con modelo base
El sistema SHALL fusionar los adapters LoRA entrenados con el modelo base E4B para producir un modelo completo con los nuevos pesos integrados.

#### Scenario: Merge exitoso
- **WHEN** se ejecuta el merge usando `model.save_pretrained_merged()` de Unsloth
- **THEN** se genera un directorio `outputs/merged/` con el modelo completo en formato safetensors, verificable cargándolo con transformers

### Requirement: Exportación a GGUF
El sistema SHALL exportar el modelo mergeado a formato GGUF con cuantización Q4_K_M para uso con llama-server.

#### Scenario: Exportación GGUF exitosa
- **WHEN** se ejecuta `model.save_pretrained_gguf()` con quantization_method="q4_k_m"
- **THEN** se genera el archivo `modelos/gemma-4-E4B-mabel-Q4_K_M.gguf` con un tamaño aproximado de ~4.7 GB

#### Scenario: Verificación de carga en llama-server
- **WHEN** se arranca llama-server con el GGUF fine-tuneado
- **THEN** el servidor carga el modelo sin errores, responde al endpoint `/health` con status OK, y el chat template se detecta correctamente

#### Scenario: Verificación de inferencia post-export
- **WHEN** se envía un prompt de prueba al modelo fine-tuneado vía `/v1/chat/completions`
- **THEN** el modelo responde en español, se presenta como Mabel, y la respuesta muestra diferencias cualitativas respecto al modelo base (tono, formato, contenido)

### Requirement: Compatibilidad con chat.py
El GGUF fine-tuneado SHALL funcionar como drop-in replacement en el flujo existente de chat.py sin modificación del código del cliente.

#### Scenario: Chat.py funciona con modelo fine-tuneado
- **WHEN** llama-server carga el GGUF fine-tuneado y el usuario ejecuta `python3 chat.py`
- **THEN** chat.py detecta el modelo automáticamente, muestra su nombre en el header, y la conversación funciona con streaming, thinking visible y todos los comandos
