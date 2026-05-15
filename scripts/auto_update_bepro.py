from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.app_data import generar_app_data_partido
from tfg_analysis.config import ProjectPaths
from tfg_analysis.io import (
    bepro_headers_from_env,
    descargar_paquete_partido_bepro,
    listar_partidos_bepro,
    partido_tiene_tracking,
)


def _split_ids(value: str | None) -> list[int]:
    if not value:
        return []
    ids: list[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids


def _processed_match_ids(paths: ProjectPaths) -> set[int]:
    app_root = paths.outputs_dir / "app_data"
    if not app_root.exists():
        return set()
    return {
        int(path.name)
        for path in app_root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "metadata.json").exists()
    }


def _discover_match_ids(args: argparse.Namespace, headers: dict[str, str]) -> list[int]:
    if args.match_id:
        return list(dict.fromkeys(int(match_id) for match_id in args.match_id))

    season_ids = args.season_id or _split_ids(os.getenv("BEPRO_SEASON_IDS") or os.getenv("BEPRO_SEASON_ID"))
    if not season_ids:
        # In the original notebook/export this id was also used to query BePro matches.
        season_ids = [int(args.team_id)]

    matches = listar_partidos_bepro(season_ids=season_ids, headers=headers, limit=args.limit)
    if matches.empty:
        return []

    matches_file = ROOT / "outputs" / "app_data" / "bepro_matches_detected.csv"
    matches_file.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(matches_file, index=False, encoding="utf-8-sig")
    return matches["match_id"].dropna().astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecta partidos nuevos en BePro, descarga datos y regenera la app."
    )
    parser.add_argument("--team-id", type=int, default=int(os.getenv("BEPRO_TEAM_ID", "12987")))
    parser.add_argument("--season-id", type=int, action="append", default=None)
    parser.add_argument("--match-id", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Procesa tambien partidos ya generados.")
    parser.add_argument("--force-analysis", action="store_true", help="Recalcula el analisis aunque exista cache.")
    parser.add_argument("--skip-tracking-check", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(root=ROOT).resolve()
    paths.ensure_dirs()
    headers = bepro_headers_from_env(required=True)

    discovered = _discover_match_ids(args, headers)
    processed = _processed_match_ids(paths)
    if args.match_id or args.force:
        candidates = discovered
    else:
        candidates = [match_id for match_id in discovered if match_id not in processed]

    if args.max_matches is not None:
        candidates = candidates[: int(args.max_matches)]

    if not candidates:
        print("No hay partidos nuevos pendientes de procesar.")
        return

    updated: list[int] = []
    failed: list[tuple[int, str]] = []
    for match_id in candidates:
        print(f"\n=== Partido {match_id} ===")
        try:
            if not args.skip_tracking_check and not partido_tiene_tracking(match_id, headers):
                print("Sin tracking disponible todavia. Se omite.")
                continue

            print("Descargando tracking, eventing y sequences...")
            descargar_paquete_partido_bepro(
                match_id=match_id,
                headers=headers,
                own_team_id=args.team_id,
                paths=paths,
                force=args.force,
            )

            print("Ejecutando pipeline y generando outputs/app_data...")
            out_dir = generar_app_data_partido(
                match_id=match_id,
                team_id=args.team_id,
                paths=paths,
                force_analysis=args.force_analysis,
            )
            print(f"OK {match_id}: {out_dir}")
            updated.append(match_id)
        except Exception as exc:
            failed.append((match_id, str(exc)))
            print(f"ERROR {match_id}: {exc}")

    if updated:
        print("\nPartidos actualizados: " + ", ".join(str(match_id) for match_id in updated))
    if failed:
        print("\nPartidos con error:")
        for match_id, message in failed:
            print(f"- {match_id}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
