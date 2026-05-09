#!/usr/bin/env python3
"""
Traduce chunks de datasets de counselling del inglés al español colombiano.
Usa agentes Sonnet vía el Agent tool de Claude Code (debe ejecutarse desde Claude Code).

Para uso manual: este script genera los prompts formateados que se pasan a los agentes.
El proceso real de traducción se ejecuta desde Claude Code con Agent tool.

Uso:
    # Desde Claude Code, el script se usa como librería de utilidades
    python3 data/scripts/translate_chunks.py --preview 0  # Ver preview del chunk 0
    python3 data/scripts/translate_chunks.py --status      # Ver progreso
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path("/home/zyfalo/Escritorio/Gemma 4")
CHUNKS_DIR = BASE_DIR / "data/chunks/mentalchat"
TRANSLATED_DIR = BASE_DIR / "data/translated/mentalchat"
PROMPT_FILE = BASE_DIR / "data/prompts/traduccion_counselling.md"

SYSTEM_PROMPT = """Eres un traductor especializado en psicología clínica y consejería emocional. Traduce las siguientes conversaciones de apoyo emocional del inglés al español colombiano.

Reglas clave:
1. Español natural colombiano/bogotano, NO formal. Debe sonar como si se escribió originalmente en español.
2. Usa "tú" por defecto. Permite "usted" solo en contexto de autoridad o cercanía bogotana.
3. Neutralidad de género: usa doble marcado (agobiado/a, solo/a) cuando el género no es evidente.
4. Familismo: "support system" → "tu familia y las personas cercanas", NO "sistema de apoyo".
5. Reducir estigma: "mental illness" → "dificultades emocionales", "therapy" → "hablar con alguien profesional".
6. Adaptar referencias académicas: "grades" → "notas", "GPA" → "promedio", "midterms/finals" → "parciales". En la UMB las notas van de 0 a 5.0 y se aprueba con 3.0.
7. NO añadir ni quitar contenido.
8. Si una respuesta ante ideación suicida/autolesión NO evalúa riesgo, marca con "needs_clinical_review": true.

Glosario rápido:
- overwhelmed → agobiado/a, "que no doy más"
- I feel down → bajoneado/a, ando mal
- coping mechanism → la forma en que lidias con eso
- burnout → estar quemado/a
- to reach out → buscar ayuda
- safe space → un lugar donde puedas hablar tranquilo/a
- trigger → lo que te detona/activa
- support system → tu familia y personas cercanas

Responde SOLO con el JSON traducido, sin explicaciones."""


def load_chunk(chunk_idx):
    path = CHUNKS_DIR / f"chunk_{chunk_idx:03d}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def format_batch_for_translation(examples, batch_start=0, batch_size=25):
    """Formatea un sub-batch de ejemplos para enviarlo a un agente traductor."""
    batch = examples[batch_start:batch_start + batch_size]
    items = []
    for i, ex in enumerate(batch):
        items.append({
            "idx": batch_start + i,
            "input": ex.get("input") or ex.get("Context", ""),
            "output": ex.get("output") or ex.get("Response", ""),
        })
    return json.dumps(items, ensure_ascii=False, indent=2)


def get_status():
    """Muestra el estado de la traducción."""
    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = sorted(CHUNKS_DIR.glob("chunk_*.json"))
    translated = sorted(TRANSLATED_DIR.glob("chunk_*.json"))

    total_examples = 0
    translated_examples = 0

    for chunk_path in chunks:
        with open(chunk_path) as f:
            data = json.load(f)
            total_examples += len(data)

    for t_path in translated:
        with open(t_path) as f:
            data = json.load(f)
            translated_examples += len(data)

    print(f"Chunks totales:     {len(chunks)}")
    print(f"Chunks traducidos:  {len(translated)}")
    print(f"Chunks pendientes:  {len(chunks) - len(translated)}")
    print(f"Ejemplos totales:   {total_examples}")
    print(f"Ejemplos traducidos:{translated_examples}")
    print(f"Progreso:           {translated_examples/total_examples*100:.1f}%")

    # Listar chunks pendientes
    translated_names = {p.name for p in translated}
    pending = [p for p in chunks if p.name not in translated_names]
    if pending:
        print(f"\nPróximos chunks pendientes:")
        for p in pending[:5]:
            with open(p) as f:
                n = len(json.load(f))
            print(f"  {p.name} ({n} ejemplos)")


if __name__ == "__main__":
    if "--status" in sys.argv:
        get_status()
    elif "--preview" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--preview") + 1])
        chunk = load_chunk(idx)
        if chunk:
            print(f"Chunk {idx}: {len(chunk)} ejemplos")
            print(f"\nPrimeros 3 formateados para traducción:")
            print(format_batch_for_translation(chunk, 0, 3))
        else:
            print(f"Chunk {idx} no encontrado")
    else:
        print("Uso: --status | --preview N")
