#!/usr/bin/env python3
"""
Sincroniza docs/28-model-card-hf.md → https://huggingface.co/ZyFalo/mabel-gemma4-e4b
como README.md del repo del modelo.

Política Opción C (acordada 2026-05-20):
- docs/28-model-card-hf.md es la FUENTE en GitHub.
- Cada cambio al model card del modelo se edita allí, este script lo replica a HF.
- Bitácoras internas y ADRs NO van a HF; solo este archivo se sincroniza.

Uso:
    HF_TOKEN=hf_xxxx python scripts/sync_hf_readme.py
    HF_TOKEN=hf_xxxx python scripts/sync_hf_readme.py --message "tu mensaje de commit"

NUNCA hardcodear el token en este archivo: leer siempre de la variable de entorno.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ID = "ZyFalo/mabel-gemma4-e4b"
SOURCE_FILE = Path(__file__).resolve().parent.parent / "docs" / "28-model-card-hf.md"
DEFAULT_COMMIT_MSG = "docs(readme): sync from docs/28-model-card-hf.md"


def sha256_local(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_remote_readme(token: str) -> str | None:
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="model",
            filename="README.md",
            token=token,
            force_download=True,
        )
        return sha256_local(Path(local))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--message", "-m",
        default=DEFAULT_COMMIT_MSG,
        help=f"Mensaje de commit en HF (default: '{DEFAULT_COMMIT_MSG}')",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Sube aunque el contenido sea idéntico (útil para re-sellar metadatos)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: variable de entorno HF_TOKEN no definida.", file=sys.stderr)
        print("Usá: HF_TOKEN=hf_xxx python scripts/sync_hf_readme.py", file=sys.stderr)
        return 2

    if not SOURCE_FILE.exists():
        print(f"ERROR: no encontré {SOURCE_FILE}", file=sys.stderr)
        return 3

    local_hash = sha256_local(SOURCE_FILE)
    print(f"Fuente local : {SOURCE_FILE.name}  SHA256={local_hash[:16]}...")

    remote_hash = sha256_remote_readme(token)
    if remote_hash:
        print(f"Remoto en HF : README.md            SHA256={remote_hash[:16]}...")
    else:
        print("Remoto en HF : (no se pudo descargar — se asume primera subida)")

    if remote_hash == local_hash and not args.force:
        print("Sin cambios: el README en HF ya está idéntico. Nada que subir.")
        print("(Usá --force si querés forzar una nueva revisión igualmente.)")
        return 0

    from huggingface_hub import upload_file

    print(f"\nSubiendo a HF con mensaje: {args.message!r} ...")
    upload_file(
        path_or_fileobj=str(SOURCE_FILE),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=args.message,
        token=token,
    )
    print(f"OK. Repo: https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
