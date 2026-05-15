from __future__ import annotations

from pathlib import Path

import pandas as pd

from tfg_analysis.config import ProjectPaths


def tracking_path(match_id: int, paths: ProjectPaths | None = None) -> Path:
    paths = (paths or ProjectPaths()).resolve()
    return paths.tracking_dir / f"partido_{int(match_id)}_tracking_poseedor.csv"


def cargar_tracking_partido(match_id: int, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Carga el CSV local de tracking de un partido."""
    path = tracking_path(match_id, paths)
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV de tracking: {path}")
    return pd.read_csv(path, low_memory=False)


def listar_tracking_disponible(paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Lista los partidos de tracking disponibles en la carpeta local."""
    paths = (paths or ProjectPaths()).resolve()
    rows = []
    for file in sorted(paths.tracking_dir.glob("partido_*_tracking_poseedor.csv")):
        parts = file.stem.split("_")
        match_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        rows.append(
            {
                "match_id": match_id,
                "path": str(file),
                "size_mb": round(file.stat().st_size / 1024 / 1024, 2),
                "modified": pd.Timestamp(file.stat().st_mtime, unit="s"),
            }
        )
    return pd.DataFrame(rows)

