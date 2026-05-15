from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.config import ProjectPaths
from tfg_analysis.io import descargar_sequences_partido, listar_tracking_disponible


def _headers_from_env() -> dict[str, str]:
    token = os.getenv("BEPRO_DATA_TOKEN", "").strip() or os.getenv("BEPRO_AUTH_TOKEN", "").strip()
    if not token:
        raise SystemExit("Falta BEPRO_DATA_TOKEN o BEPRO_AUTH_TOKEN en el entorno.")
    auth = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return {"Authorization": auth}


def main():
    parser = argparse.ArgumentParser(description="Descarga sequences de BePro y las guarda como CSV local.")
    parser.add_argument("--match-id", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(root=ROOT).resolve()
    if args.match_id:
        match_ids = args.match_id
    else:
        partidos = listar_tracking_disponible(paths)
        match_ids = partidos["match_id"].dropna().astype(int).tolist()

    headers = _headers_from_env()
    for match_id in match_ids:
        print(f"Descargando sequences {match_id}...")
        df = descargar_sequences_partido(
            match_id=match_id,
            headers=headers,
            paths=paths,
            limit=args.limit,
            force=args.force,
        )
        print(f"OK {match_id}: {len(df)} sequences")


if __name__ == "__main__":
    main()
