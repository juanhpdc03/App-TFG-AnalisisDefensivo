from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _ensure_events(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def extraer_event_types(events) -> list[str]:
    tipos = []
    for item in _ensure_events(events):
        if isinstance(item, dict):
            et = item.get("event_type")
            if et is not None:
                tipos.append(str(et).lower())
    return tipos


def extraer_prop_evento(events, tipo_objetivo: str) -> dict:
    for item in _ensure_events(events):
        if not isinstance(item, dict):
            continue
        et = str(item.get("event_type", "")).lower()
        if et == tipo_objetivo:
            prop = item.get("property", {})
            return prop if isinstance(prop, dict) else {}
    return {}


def es_tiro_a_puerta(prop: dict) -> bool:
    texto = " ".join(str(v).lower() for v in prop.values())
    claves_positivas = [
        "ontarget",
        "on_target",
        "on target",
        "goal",
        "gol",
        "save",
        "saved",
        "savedshot",
        "shotontarget",
    ]
    claves_negativas = [
        "offtarget",
        "off_target",
        "off target",
        "blocked",
        "woodwork",
        "post",
        "bar",
        "outside",
    ]
    if any(k in texto for k in claves_negativas):
        return False
    return any(k in texto for k in claves_positivas)


def extraer_xg(prop: dict) -> float:
    """Extrae xG de propiedades BePro si viene disponible."""
    if not isinstance(prop, dict):
        return np.nan
    claves = [
        "xg",
        "xG",
        "expected_goals",
        "expected_goal",
        "expectedGoal",
        "expectedGoals",
        "shot_xg",
        "shotXg",
    ]
    for key in claves:
        if key in prop:
            return pd.to_numeric(prop.get(key), errors="coerce")
    for key, value in prop.items():
        key_text = str(key).lower()
        if "xg" in key_text or "expected" in key_text:
            val = pd.to_numeric(value, errors="coerce")
            if pd.notna(val):
                return float(val)
    return np.nan


def extraer_tiros_oficiales(
    df_eventing: pd.DataFrame,
    team_id: int | None = None,
    rival_team_id: int | None = None,
) -> pd.DataFrame:
    """Extrae eventos oficiales `shot` de Bepro.

    Si se pasa `rival_team_id`, filtra los tiros de ese equipo. Si se pasa
    `team_id`, filtra los tiros del equipo contrario.
    """
    df = df_eventing.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "events" not in df.columns:
        return pd.DataFrame()

    df["event_types"] = df["events"].apply(extraer_event_types)

    rename = {
        "team_id": "team_id_event",
        "player_id": "player_id_event",
        "period_order": "period_event",
        "event_time": "match_time_event",
        "x": "x_event",
        "y": "y_event",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ["team_id_event", "player_id_event", "period_event", "match_time_event", "x_event", "y_event"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    shots = df[df["event_types"].apply(lambda lst: "shot" in lst)].copy()
    if shots.empty:
        return shots

    if rival_team_id is not None and "team_id_event" in shots.columns:
        shots = shots[shots["team_id_event"] == rival_team_id].copy()
    elif team_id is not None and "team_id_event" in shots.columns:
        shots = shots[shots["team_id_event"] != team_id].copy()

    saves = df[df["event_types"].apply(lambda lst: "save" in lst)].copy()

    shots["shot_property"] = shots["events"].apply(lambda ev: extraer_prop_evento(ev, "shot"))
    shots["xg_tiro"] = shots["shot_property"].apply(extraer_xg)
    shots["tipo_finalizacion_tiro"] = 1
    shots["tipo_finalizacion_tiro_puerta"] = shots["shot_property"].apply(es_tiro_a_puerta).astype(int)
    shots["es_gol"] = shots["shot_property"].apply(
        lambda prop: "goal" in " ".join(str(v).lower() for v in prop.values())
        or "gol" in " ".join(str(v).lower() for v in prop.values())
    )

    if not saves.empty:
        for idx_shot, shot in shots.iterrows():
            if int(shots.loc[idx_shot, "tipo_finalizacion_tiro_puerta"]) == 1:
                continue
            per = shot.get("period_event", np.nan)
            t = shot.get("match_time_event", np.nan)
            if pd.isna(per) or pd.isna(t):
                continue
            saves_cercanos = saves[
                (saves["period_event"] == per)
                & (np.abs(saves["match_time_event"] - t) <= 1500)
            ]
            if not saves_cercanos.empty:
                shots.loc[idx_shot, "tipo_finalizacion_tiro_puerta"] = 1

    keep = [
        c
        for c in [
            "player_id_event",
            "team_id_event",
            "period_event",
            "match_time_event",
            "x_event",
            "y_event",
            "player_name",
            "team_name",
            "shot_property",
            "xg_tiro",
            "tipo_finalizacion_tiro",
            "tipo_finalizacion_tiro_puerta",
            "es_gol",
        ]
        if c in shots.columns
    ]
    return shots[keep].sort_values([c for c in ["period_event", "match_time_event"] if c in keep]).reset_index(drop=True)
