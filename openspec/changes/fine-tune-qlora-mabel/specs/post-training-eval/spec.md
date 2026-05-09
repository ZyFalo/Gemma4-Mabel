## ADDED Requirements

### Requirement: Re-ejecución de batería de 12 turnos post-fine-tuning
El sistema SHALL ejecutar la batería de evaluación (docs/15) sobre el modelo fine-tuneado con la misma configuración usada en las evaluaciones baseline, al menos 2 veces para medir varianza.

#### Scenario: Ejecución de batería sobre modelo fine-tuneado
- **WHEN** se ejecuta `python3 eval/run_battery.py E4B_finetuned_run1` con el GGUF fine-tuneado cargado en llama-server
- **THEN** se completan los 12 turnos sin errores, los resultados se guardan en `eval/results/E4B_finetuned_run1_*.md`, y se incluyen las respuestas textuales y reasoning de cada turno

#### Scenario: Segunda ejecución para medir varianza
- **WHEN** se ejecuta `python3 eval/run_battery.py E4B_finetuned_run2` inmediatamente después
- **THEN** se genera un segundo archivo de resultados, permitiendo comparar varianza entre runs del modelo fine-tuneado

### Requirement: Comparación scorecard pre/post fine-tuning
El sistema SHALL producir una tabla comparativa de los 15 criterios del scorecard entre el E4B baseline y el E4B fine-tuneado.

#### Scenario: Mejora en los 5 objetivos de fine-tuning
- **WHEN** se compara el scorecard del E4B fine-tuneado con el del E4B baseline
- **THEN** se observa mejora medible (aumento de al menos 1 punto Likert) en los 5 criterios objetivo: neutralidad de género, gradación de crisis, manejo del afterglow, pregunta por persona de confianza, y consistencia entre runs

#### Scenario: Sin regresión significativa en otros criterios
- **WHEN** se compara el scorecard completo
- **THEN** ningún criterio no-objetivo presenta una regresión mayor a 0.5 puntos Likert respecto al baseline

#### Scenario: Comparación con gold standard (26B MoE)
- **WHEN** se compara el scorecard del E4B fine-tuneado con el del 26B MoE baseline
- **THEN** la brecha se reduce visiblemente (E4B fine-tuneado se acerca al 4.40/5 del 26B MoE desde el 3.93/5 del E4B baseline)

### Requirement: Documentación de resultados post-fine-tuning
El sistema SHALL generar un documento `docs/22-resultados-post-finetuning.md` con el análisis completo de la evaluación post-fine-tuning.

#### Scenario: Documento generado y completo
- **WHEN** la evaluación post-fine-tuning termina
- **THEN** se crea `docs/22-resultados-post-finetuning.md` con: tabla comparativa pre/post, análisis de los turnos críticos (T5, T8, T9, T10), conclusión sobre si los 5 objetivos se cumplieron, y recomendaciones para iteración futura
