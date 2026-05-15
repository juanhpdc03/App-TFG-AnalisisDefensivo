from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.app_data import generar_app_data_partido
from tfg_analysis.config import ProjectPaths
from tfg_analysis.io import listar_tracking_disponible
from tfg_analysis.models.global_clustering import global_model_path


def main():
    parser = argparse.ArgumentParser(description="Genera los archivos ligeros que lee la app web.")
    parser.add_argument("--team-id", type=int, default=12987)
    parser.add_argument("--match-id", type=int, action="append", default=None)
    parser.add_argument("--force-analysis", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(root=ROOT).resolve()
    if args.match_id:
        match_ids = args.match_id
    else:
        partidos = listar_tracking_disponible(paths)
        match_ids = partidos["match_id"].dropna().astype(int).tolist()

    if not match_ids:
        print("No hay partidos en tracking_partidos.")
        return

    if global_model_path(paths).exists() and not args.force_analysis:
        print(
            "AVISO: existe un modelo de clustering global. "
            "Usa --force-analysis para recalcular los partidos con ese modelo."
        )

    for match_id in match_ids:
        print(f"Generando datos web del partido {match_id}...")
        out = generar_app_data_partido(
            match_id=match_id,
            team_id=args.team_id,
            paths=paths,
            force_analysis=args.force_analysis,
        )
        print(f"OK {match_id}: {out}")


if __name__ == "__main__":
    main()
