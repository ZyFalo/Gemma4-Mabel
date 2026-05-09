#!/usr/bin/env python3
"""
Script de traducción masiva de chunks de counselling del inglés al español colombiano.
Procesa batches de 25 ejemplos, guarda progreso, y puede retomarse si se interrumpe.

Uso:
    cd "/home/zyfalo/Escritorio/Gemma 4"
    .venv/bin/python data/scripts/run_translation.py --dataset mentalchat
    .venv/bin/python data/scripts/run_translation.py --dataset amod
    .venv/bin/python data/scripts/run_translation.py --status
"""

import json
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Intentar importar anthropic SDK
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

BASE_DIR = Path("/home/zyfalo/Escritorio/Gemma 4")
BATCH_SIZE = 25

TRANSLATION_SYSTEM = """Eres un traductor especializado en psicología clínica y consejería emocional. Traduce conversaciones de apoyo emocional del inglés al español colombiano.

Reglas:
1. Español natural colombiano/bogotano, NO formal. Debe sonar escrito originalmente en español.
2. "Tú" por defecto. "Usted" solo en contexto de autoridad o cercanía bogotana. NO "vos".
3. Neutralidad de género: doble marcado (agobiado/a, solo/a) cuando el género no es evidente.
4. Familismo: "support system" → "tu familia y las personas cercanas", NO "sistema de apoyo".
5. Estigma: "mental illness" → "dificultades emocionales", "therapy" → "hablar con alguien profesional".
6. Académico: "grades" → "notas", "GPA" → "promedio", "midterms/finals" → "parciales". UMB: notas 0-5.0, se aprueba con 3.0.
7. NO añadir ni quitar contenido.
8. Si una respuesta ante ideación suicida/autolesión NO evalúa riesgo: "needs_clinical_review": true.
9. Limpia artefactos ("Counselor (continued):", "Response:" etc.).

Glosario: overwhelmed→agobiado/a; I feel down→bajoneado/a; coping mechanism→la forma en que lidias con eso; burnout→estar quemado/a; to reach out→buscar ayuda; safe space→un lugar donde puedas hablar tranquilo/a; trigger→lo que te detona; support system→tu familia y personas cercanas; self-care→cuidarte, darte un respiro.

Responde SOLO con el array JSON traducido, sin explicaciones ni markdown."""


def translate_batch_anthropic(client, examples, batch_id):
    """Traduce un batch usando la API de Anthropic directamente."""
    items = []
    for i, ex in enumerate(examples):
        items.append({
            "idx": i,
            "input": ex.get("input") or ex.get("Context", ""),
            "output": ex.get("output") or ex.get("Response", ""),
        })

    user_msg = f"Traduce estos {len(items)} ejemplos al español colombiano. Responde SOLO con JSON:\n\n{json.dumps(items, ensure_ascii=False)}"

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            system=TRANSLATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        # Limpiar markdown si viene envuelto
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text), None
    except Exception as e:
        return None, str(e)


def process_dataset(dataset_name):
    if not HAS_ANTHROPIC:
        print("ERROR: pip install anthropic")
        print("Luego configura ANTHROPIC_API_KEY en el entorno.")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Configura ANTHROPIC_API_KEY")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    chunks_dir = BASE_DIR / f"data/chunks/{dataset_name}"
    output_dir = BASE_DIR / f"data/translated/{dataset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
    if not chunk_files:
        print(f"No se encontraron chunks en {chunks_dir}")
        return

    print(f"Dataset: {dataset_name}")
    print(f"Chunks: {len(chunk_files)}")

    total_translated = 0
    total_errors = 0

    for chunk_file in chunk_files:
        out_file = output_dir / chunk_file.name
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"  {chunk_file.name}: ya traducido ({len(existing)} ejemplos), saltando")
            total_translated += len(existing)
            continue

        with open(chunk_file) as f:
            examples = json.load(f)

        print(f"  {chunk_file.name}: traduciendo {len(examples)} ejemplos...", end="", flush=True)
        t0 = time.time()

        all_translated = []
        for batch_start in range(0, len(examples), BATCH_SIZE):
            batch = examples[batch_start:batch_start + BATCH_SIZE]
            batch_id = f"{chunk_file.stem}_b{batch_start}"

            result, error = translate_batch_anthropic(client, batch, batch_id)
            if error:
                print(f"\n    ERROR en batch {batch_id}: {error}")
                total_errors += 1
                # Guardar lo que tengamos y continuar
                break
            else:
                all_translated.extend(result)

            # Rate limiting suave
            time.sleep(0.5)

        if all_translated:
            with open(out_file, "w") as f:
                json.dump(all_translated, f, ensure_ascii=False, indent=2)
            total_translated += len(all_translated)
            elapsed = time.time() - t0
            print(f" OK ({len(all_translated)} traducidos, {elapsed:.1f}s)")
        else:
            print(f" FALLÓ (sin traducciones)")

    print(f"\nResumen: {total_translated} traducidos, {total_errors} errores")


def show_status():
    for dataset in ["mentalchat", "amod"]:
        chunks_dir = BASE_DIR / f"data/chunks/{dataset}"
        output_dir = BASE_DIR / f"data/translated/{dataset}"

        if not chunks_dir.exists():
            continue

        chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
        trans_files = sorted(output_dir.glob("chunk_*.json")) if output_dir.exists() else []

        total_src = sum(len(json.load(open(f))) for f in chunk_files)
        total_trans = sum(len(json.load(open(f))) for f in trans_files) if trans_files else 0

        print(f"\n{dataset.upper()}:")
        print(f"  Chunks: {len(trans_files)}/{len(chunk_files)}")
        print(f"  Ejemplos: {total_trans}/{total_src}")
        print(f"  Progreso: {total_trans/total_src*100:.1f}%" if total_src else "  Sin datos")


if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        dataset = sys.argv[idx + 1]
        process_dataset(dataset)
    else:
        print("Uso:")
        print("  --status              Ver progreso")
        print("  --dataset mentalchat  Traducir MentalChat16K")
        print("  --dataset amod        Traducir Amod")
