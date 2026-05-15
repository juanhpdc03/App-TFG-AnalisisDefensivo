from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from tfg_analysis.config import ProjectPaths


def eventing_path(match_id: int, paths: ProjectPaths | None = None) -> Path:
    paths = (paths or ProjectPaths()).resolve()
    return paths.eventing_dir / f"partido_{int(match_id)}_eventing.csv"


def _serializar_columnas_complejas(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].map(lambda x: isinstance(x, (list, dict))).any():
            df_out[col] = df_out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    return df_out


def _deserializar_eventing(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    if "events" in df_out.columns:
        df_out["events"] = df_out["events"].apply(_loads_json_if_possible)
    return df_out


def _loads_json_if_possible(value):
    if not isinstance(value, str):
        return value
    value_strip = value.strip()
    if not value_strip or value_strip[0] not in "[{":
        return value
    try:
        return json.loads(value_strip)
    except json.JSONDecodeError:
        return value


def descargar_eventing_partido(
    match_id: int,
    headers: dict,
    paths: ProjectPaths | None = None,
    limit: int = 1000,
    force: bool = False,
) -> pd.DataFrame:
    """Descarga todos los eventos Bepro de un partido y los guarda en CSV."""
    paths = (paths or ProjectPaths()).resolve()
    paths.eventing_dir.mkdir(parents=True, exist_ok=True)
    output_file = eventing_path(match_id, paths)

    if output_file.exists() and not force:
        return cargar_eventing_partido(match_id, paths)

    all_events = []
    offset = 0

    while True:
        url = (
            "https://ds.bepro.ai/data-api/data/events"
            f"?match_id={int(match_id)}&limit={int(limit)}&offset={int(offset)}"
        )
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            break

        all_events.extend(data)

        if len(data) < limit:
            break

        offset += limit

    df_eventing = pd.DataFrame(all_events)
    _serializar_columnas_complejas(df_eventing).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )
    return df_eventing


def descargar_eventing_partidos(
    match_ids: Iterable[int],
    headers: dict,
    paths: ProjectPaths | None = None,
    limit: int = 1000,
    force: bool = False,
) -> dict[int, pd.DataFrame]:
    """Descarga el eventing de varios partidos."""
    return {
        int(match_id): descargar_eventing_partido(
            match_id=match_id,
            headers=headers,
            paths=paths,
            limit=limit,
            force=force,
        )
        for match_id in match_ids
    }


def cargar_eventing_partido(match_id: int, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Carga desde CSV local el eventing de un partido."""
    path = eventing_path(match_id, paths)
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV de eventing: {path}")
    return _deserializar_eventing(pd.read_csv(path, low_memory=False))

