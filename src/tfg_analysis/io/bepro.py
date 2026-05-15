from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from tfg_analysis.config import ProjectPaths
from tfg_analysis.io.eventing import descargar_eventing_partido
from tfg_analysis.io.sequences import descargar_sequences_partido
from tfg_analysis.io.tracking import cargar_tracking_partido, tracking_path


BEPRO_BASE_URL = "https://ds.bepro.ai/data-api"


def bepro_headers_from_env(required: bool = True) -> dict[str, str]:
    """Build BePro API headers from environment variables."""
    token = (
        os.getenv("BEPRO_API_KEY", "").strip()
        or os.getenv("BEPRO_DATA_TOKEN", "").strip()
        or os.getenv("BEPRO_AUTH_TOKEN", "").strip()
    )
    if required and not token:
        raise RuntimeError(
            "Falta BEPRO_API_KEY, BEPRO_DATA_TOKEN o BEPRO_AUTH_TOKEN en el entorno."
        )
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return headers


def _get_json(url: str, headers: dict[str, str], timeout: int = 60) -> dict:
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def listar_partidos_bepro(
    season_ids: Iterable[int | str],
    headers: dict[str, str],
    limit: int = 50,
) -> pd.DataFrame:
    """List matches returned by BePro meta/matches for the configured seasons."""
    ids = [str(item).strip() for item in season_ids if str(item).strip()]
    if not ids:
        raise ValueError("Debes indicar al menos un season_id para listar partidos en BePro.")

    rows: list[dict] = []
    url = f"{BEPRO_BASE_URL}/meta/matches?season_ids={','.join(ids)}&limit={int(limit)}"
    while url:
        payload = _get_json(url, headers=headers)
        for item in payload.get("data", []):
            match_id = item.get("id") or item.get("match_id")
            if match_id is None:
                continue
            home = item.get("home_team") or {}
            away = item.get("away_team") or {}
            rows.append(
                {
                    "match_id": int(match_id),
                    "match_date": item.get("match_date")
                    or item.get("date")
                    or item.get("start_time")
                    or item.get("match_start_time"),
                    "home_team": home.get("name") or home.get("team_name") or item.get("home_team_name"),
                    "away_team": away.get("name") or away.get("team_name") or item.get("away_team_name"),
                    "raw": json.dumps(item, ensure_ascii=False),
                }
            )
        url = payload.get("next")

    if not rows:
        return pd.DataFrame(columns=["match_id", "match_date", "home_team", "away_team", "raw"])
    return pd.DataFrame(rows).drop_duplicates("match_id").sort_values("match_id").reset_index(drop=True)


def partido_tiene_tracking(match_id: int, headers: dict[str, str]) -> bool:
    """Return True when BePro returns at least one tracking frame for the match."""
    url = (
        f"{BEPRO_BASE_URL}/data/tracking?match_id={int(match_id)}"
        "&period_order=0&start_match_time=0&end_match_time=15000"
    )
    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code in {400, 401, 403, 404}:
        return False
    response.raise_for_status()
    return bool(response.json().get("data", []))


def obtener_lineups_partido(match_id: int, headers: dict[str, str]) -> tuple[dict[int, str], dict[int, dict]]:
    url = f"{BEPRO_BASE_URL}/meta/lineups?match_ids={int(match_id)}"
    payload = _get_json(url, headers=headers)
    lineup = payload.get("data", [{}])[0].get("lineup", [])

    player_map: dict[int, str] = {}
    team_map: dict[int, dict] = {}
    for player in lineup:
        player_id = int(player["player_id"])
        player_map[player_id] = player.get("player_full_name") or player.get("player_name") or str(player_id)
        team_map[player_id] = {
            "team_id": player.get("team_id"),
            "team_name": player.get("team_full_name") or player.get("team_name"),
        }
    if not player_map:
        raise ValueError(f"No se han encontrado lineups para el partido {match_id}.")
    return player_map, team_map


def _descargar_tramo_tracking(
    match_id: int,
    period_order: int,
    start_ms: int,
    end_ms: int,
    headers: dict[str, str],
) -> dict:
    url = (
        f"{BEPRO_BASE_URL}/data/tracking?match_id={int(match_id)}"
        f"&period_order={int(period_order)}"
        f"&start_match_time={int(start_ms)}"
        f"&end_match_time={int(end_ms)}"
    )
    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code != 200:
        return {
            "period": int(period_order),
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "frames": [],
            "status_code": response.status_code,
        }
    return {
        "period": int(period_order),
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "frames": response.json().get("data", []),
        "status_code": 200,
    }


def _descargar_tramo_tracking_con_reintentos(
    match_id: int,
    period_order: int,
    start_ms: int,
    end_ms: int,
    headers: dict[str, str],
    max_retries: int = 5,
    sleep_base: int = 2,
) -> dict:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = _descargar_tramo_tracking(match_id, period_order, start_ms, end_ms, headers)
            if result.get("status_code") == 200:
                return result
            last_error = result.get("status_code")
        except Exception as exc:  # pragma: no cover - depends on remote API/network.
            last_error = str(exc)

        print(
            f"[REINTENTO] partido={match_id} periodo={period_order} "
            f"tramo={start_ms}-{end_ms} intento={attempt}/{max_retries} error={last_error}"
        )
        time.sleep(sleep_base * attempt)

    return {
        "period": int(period_order),
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "frames": [],
        "status_code": None,
    }


def _procesar_frames_tracking(
    frames: list[dict],
    team_map: dict[int, dict],
    player_map: dict[int, str],
    own_team_id: int,
    threshold_poseedor: float,
    match_id: int,
) -> list[dict]:
    rows: list[dict] = []

    for frame in frames:
        frame_idx = frame.get("frame_index")
        match_time = frame.get("match_time")
        period = frame.get("period_order")
        ball_state = frame.get("ball_state")

        balls = frame.get("balls") or []
        x_ball = balls[0].get("x") if balls else None
        y_ball = balls[0].get("y") if balls else None

        frame_rows: list[dict] = []
        for player in frame.get("players", []):
            player_id = int(player["player_id"])
            x_player = player.get("x")
            y_player = player.get("y")
            info = team_map.get(player_id, {})
            dist = (
                float(np.sqrt((x_player - x_ball) ** 2 + (y_player - y_ball) ** 2))
                if x_ball is not None and y_ball is not None and x_player is not None and y_player is not None
                else None
            )
            frame_rows.append(
                {
                    "match_id": int(match_id),
                    "frame": frame_idx,
                    "match_time": match_time,
                    "period": period,
                    "player_id": player_id,
                    "player_name": player_map.get(player_id, f"ID: {player_id}"),
                    "team_id": info.get("team_id"),
                    "team_name": info.get("team_name"),
                    "x": x_player,
                    "y": y_player,
                    "ball_x": x_ball,
                    "ball_y": y_ball,
                    "ball_state": ball_state,
                    "dist_ball": dist,
                }
            )

        valid = [row for row in frame_rows if row["dist_ball"] is not None]
        if valid:
            closest = min(valid, key=lambda row: row["dist_ball"])
            possessor_id = closest["player_id"] if closest["dist_ball"] <= threshold_poseedor else None
            possessor_team_id = closest["team_id"] if possessor_id is not None else None
        else:
            possessor_id = None
            possessor_team_id = None

        possession_type = (
            "propia" if possessor_team_id == own_team_id else ("rival" if possessor_team_id else None)
        )
        for row in frame_rows:
            row["poseedor"] = 1 if row["player_id"] == possessor_id else 0
            row["tipo_posesion"] = possession_type
            rows.append(row)

    return rows


def _descargar_periodo_completo(
    match_id: int,
    period_order: int,
    headers: dict[str, str],
    player_map: dict[int, str],
    team_map: dict[int, dict],
    own_team_id: int,
    start_ms: int,
    fixed_duration_ms: int = 2_700_000,
    chunk_ms: int = 30_000,
    max_workers: int = 4,
    threshold_poseedor: float = 0.02,
) -> tuple[pd.DataFrame, int | None]:
    starts = list(range(start_ms, start_ms + fixed_duration_ms, chunk_ms))
    tasks = [(match_id, period_order, start, start + chunk_ms, headers) for start in starts]
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_descargar_tramo_tracking_con_reintentos, *task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    rows: list[dict] = []
    last_match_time: int | None = None
    for result in sorted(results, key=lambda item: item["start_ms"]):
        frames = result.get("frames", [])
        if result.get("status_code") != 200 or not frames:
            continue
        last_match_time = max(last_match_time or 0, max(int(frame["match_time"]) for frame in frames))
        rows.extend(
            _procesar_frames_tracking(
                frames,
                team_map=team_map,
                player_map=player_map,
                own_team_id=own_team_id,
                threshold_poseedor=threshold_poseedor,
                match_id=match_id,
            )
        )

    empty_chunks = 0
    extra_start = start_ms + fixed_duration_ms
    while empty_chunks < 2:
        result = _descargar_tramo_tracking_con_reintentos(
            match_id,
            period_order,
            extra_start,
            extra_start + chunk_ms,
            headers,
        )
        frames = result.get("frames", [])
        if result.get("status_code") != 200 or not frames:
            empty_chunks += 1
            extra_start += chunk_ms
            continue
        empty_chunks = 0
        last_match_time = max(last_match_time or 0, max(int(frame["match_time"]) for frame in frames))
        rows.extend(
            _procesar_frames_tracking(
                frames,
                team_map=team_map,
                player_map=player_map,
                own_team_id=own_team_id,
                threshold_poseedor=threshold_poseedor,
                match_id=match_id,
            )
        )
        extra_start += chunk_ms

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["period", "frame", "player_id"])
            .sort_values(by=["period", "match_time", "player_id"])
            .reset_index(drop=True)
        )
    return df, last_match_time


def descargar_tracking_partido_bepro(
    match_id: int,
    headers: dict[str, str],
    own_team_id: int,
    paths: ProjectPaths | None = None,
    force: bool = False,
    chunk_ms: int = 30_000,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Download full BePro tracking and save it with the project naming convention."""
    paths = (paths or ProjectPaths()).resolve()
    paths.tracking_dir.mkdir(parents=True, exist_ok=True)
    out = tracking_path(match_id, paths)
    if out.exists() and not force:
        return cargar_tracking_partido(match_id, paths)

    player_map, team_map = obtener_lineups_partido(match_id, headers)
    first_half, last_first = _descargar_periodo_completo(
        match_id=match_id,
        period_order=0,
        headers=headers,
        player_map=player_map,
        team_map=team_map,
        own_team_id=own_team_id,
        start_ms=0,
        chunk_ms=chunk_ms,
        max_workers=max_workers,
    )
    extra_first = max(0, int(last_first or 2_700_000) - 2_700_000)
    second_half, _ = _descargar_periodo_completo(
        match_id=match_id,
        period_order=1,
        headers=headers,
        player_map=player_map,
        team_map=team_map,
        own_team_id=own_team_id,
        start_ms=2_700_000 + extra_first,
        chunk_ms=chunk_ms,
        max_workers=max_workers,
    )
    df = pd.concat([first_half, second_half], ignore_index=True)
    if df.empty:
        raise ValueError(f"La descarga de tracking del partido {match_id} no devolvio datos.")
    df = (
        df.drop_duplicates(subset=["period", "frame", "player_id"])
        .sort_values(by=["period", "match_time", "player_id"])
        .reset_index(drop=True)
    )
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def descargar_paquete_partido_bepro(
    match_id: int,
    headers: dict[str, str],
    own_team_id: int,
    paths: ProjectPaths | None = None,
    force: bool = False,
    sequence_limit: int = 1000,
) -> dict[str, Path]:
    """Download tracking, eventing and sequences for one match."""
    paths = (paths or ProjectPaths()).resolve()
    paths.ensure_dirs()
    descargar_tracking_partido_bepro(match_id, headers, own_team_id, paths=paths, force=force)
    descargar_eventing_partido(match_id, headers=headers, paths=paths, force=force)
    descargar_sequences_partido(match_id, headers=headers, paths=paths, limit=sequence_limit, force=force)
    return {
        "tracking": tracking_path(match_id, paths),
        "eventing": paths.eventing_dir / f"partido_{int(match_id)}_eventing.csv",
        "sequences": paths.sequences_dir / f"partido_{int(match_id)}_sequences.csv",
    }
