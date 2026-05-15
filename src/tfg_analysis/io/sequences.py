from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from tfg_analysis.config import ProjectPaths
from tfg_analysis.io.eventing import _deserializar_eventing, _serializar_columnas_complejas


def sequences_path(match_id: int, paths: ProjectPaths | None = None) -> Path:
    paths = (paths or ProjectPaths()).resolve()
    return paths.sequences_dir / f"partido_{int(match_id)}_sequences.csv"


def descargar_sequences_partido(
    match_id: int,
    headers: dict,
    paths: ProjectPaths | None = None,
    limit: int = 1000,
    force: bool = False,
) -> pd.DataFrame:
    """Descarga las secuencias BePro de un partido y las guarda en CSV."""
    paths = (paths or ProjectPaths()).resolve()
    paths.sequences_dir.mkdir(parents=True, exist_ok=True)
    output_file = sequences_path(match_id, paths)

    if output_file.exists() and not force:
        return cargar_sequences_partido(match_id, paths)

    all_sequences = []
    offset = 0
    while True:
        url = (
            "https://ds.bepro.ai/data-api/data/sequences"
            f"?match_id={int(match_id)}&limit={int(limit)}&offset={int(offset)}"
        )
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        all_sequences.extend(data)
        if len(data) < limit:
            break
        offset += limit

    df_sequences = pd.DataFrame(all_sequences)
    _serializar_columnas_complejas(df_sequences).to_csv(output_file, index=False, encoding="utf-8-sig")
    return df_sequences


def descargar_sequences_partidos(
    match_ids: Iterable[int],
    headers: dict,
    paths: ProjectPaths | None = None,
    limit: int = 1000,
    force: bool = False,
) -> dict[int, pd.DataFrame]:
    return {
        int(match_id): descargar_sequences_partido(
            match_id=match_id,
            headers=headers,
            paths=paths,
            limit=limit,
            force=force,
        )
        for match_id in match_ids
    }


def cargar_sequences_partido(match_id: int, paths: ProjectPaths | None = None) -> pd.DataFrame:
    path = sequences_path(match_id, paths)
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV de sequences: {path}")
    return _deserializar_eventing(pd.read_csv(path, low_memory=False))
