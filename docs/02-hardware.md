# 02 — Hardware disponible y sus implicaciones

## Especificaciones del equipo de desarrollo

| Componente | Valor |
|---|---|
| **CPU** | Intel Core i7-10750H (6 núcleos físicos / 12 hilos) |
| **Frecuencia** | 2.6 GHz base, hasta 5.0 GHz turbo |
| **Instrucciones relevantes** | AVX2 (sí), AVX-512 (no) |
| **RAM total** | 31 GiB |
| **RAM libre habitual** | ~26 GiB |
| **GPU** | NVIDIA GeForce RTX 2060 Mobile |
| **VRAM** | 6 GB |
| **Compute capability** | 7.5 (Turing) |
| **Precisión nativa** | FP32, FP16. **Sin BF16 nativo.** |
| **Driver CUDA** | 580.65.06 |
| **Disco libre** | ~75 GB en /home |

## Implicaciones clave para el proyecto

### 1. Entrenamiento → VRAM (GPU)

Cualquier fine-tuning razonable se hace en GPU. QLoRA depende de `bitsandbytes`, que requiere CUDA para sus kernels 4-bit NF4. Intentar entrenar solo en CPU/RAM es 20–50× más lento y prácticamente inviable para modelos de miles de millones de parámetros.

**Consecuencia**: los únicos modelos de la familia Gemma 4 que se pueden entrenar en este equipo con QLoRA son:

| Modelo | VRAM necesaria (QLoRA + gradient checkpointing + Unsloth) | ¿Viable? |
|---|---|---|
| **E2B** (~2B params efectivos) | ~3–4 GB | ✅ Holgado |
| **E4B** (~4B params efectivos) | ~5–6 GB | ✅ Al límite, pero viable |
| **26B MoE** (A4B) | ~18–24 GB | ❌ No cabe |
| **31B Dense** | ~24–32 GB | ❌ No cabe |

**Decisión**: se entrenará **Gemma 4 E4B** con prototipo previo en **E2B** para validar pipeline.

### 2. Inferencia → RAM (CPU) también viable

Para *usar* el modelo entrenado, la RAM del sistema sí puede alojar modelos mucho más grandes que los que caben en VRAM para entrenamiento. Esto abre la posibilidad de usar modelos grandes como **comparador académico** en la evaluación.

| Modelo | Tamaño en RAM (Q4_K_M) | ¿Cabe en 26 GB libres? | Velocidad CPU esperada |
|---|---|---|---|
| **E2B fine-tuneado** | ~2 GB | ✅ | 15–25 tok/s |
| **E4B fine-tuneado** | ~3–4 GB | ✅ | 8–15 tok/s |
| **26B MoE base** | ~14–16 GB | ✅ | 6–12 tok/s |
| **31B Dense base** | ~18–20 GB | ⚠️ Apretado | 1.5–3 tok/s |

El **26B MoE** es especialmente atractivo en CPU porque, a pesar de tener 25.2B parámetros totales, solo activa ~3.8B por token gracias a la arquitectura *Mixture of Experts*. Eso lo hace más rápido en CPU que modelos densos de tamaño equivalente.

### 3. Ventana de contexto realista

Aunque Gemma 4 soporta nominalmente hasta **256K tokens** de contexto, el tamaño del KV cache escala linealmente con esa longitud y consume memoria enorme:

```
KV cache ≈ 2 × num_capas × num_kv_heads × head_dim × bytes_por_valor × num_tokens
```

Para E4B en FP16, cada token consume ~130 KB de KV cache. A **256K tokens** serían **~33 GB solo de KV cache**, lo cual excede la RAM total del equipo.

**Decisión técnica**: no perseguir los 256K de contexto. Se usa una ventana operativa de **8K–16K tokens** por sesión, combinada con un sistema de **memoria externa** (RAG + resumen jerárquico) para persistir información entre sesiones.

### 4. Consideraciones térmicas (laptop)

El equipo es un portátil. El entrenamiento sostenido (varias horas) y la inferencia pesada generan calor significativo, lo que puede llevar a *thermal throttling* y degradación de rendimiento tras períodos prolongados.

**Recomendaciones operativas**:
- Mantener el equipo enchufado durante entrenamiento.
- Base refrigerada si es posible.
- Pausas programadas entre épocas para evitar throttling continuo.
- Monitorización de temperatura con `sensors` o `nvtop`.

### 5. Ausencia de BF16 nativo

La RTX 2060 Mobile (Turing, compute 7.5) no soporta BF16 nativo — solo FP16. Los frameworks modernos por defecto asumen BF16 para modelos nuevos.

**Decisión técnica**: todas las configuraciones de entrenamiento deben especificar explícitamente `fp16=True` en lugar de `bf16=True`. Unsloth maneja esto correctamente cuando se configura.
