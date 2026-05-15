from __future__ import annotations

import pandas as pd

from .shots import extraer_event_types


def extraer_setpiece_type(events):
    if isinstance(events, str):
        from .shots import _ensure_events

        events = _ensure_events(events)
    if isinstance(events, list):
        for item in events:
            if isinstance(item, dict) and str(item.get("event_type", "")).lower() == "setpiece":
                prop = item.get("property", {})
                if isinstance(prop, dict) and prop.get("type") is not None:
                    return str(prop.get("type")).lower()
    return None


def crear_eventos_clave(df_eventing_raw: pd.DataFrame) -> pd.DataFrame:
    """Replica la tabla `df_eventos_clave` del notebook."""
    df = df_eventing_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "events" not in df.columns:
        return pd.DataFrame()

    df["event_types"] = df["events"].apply(extraer_event_types)
    df["setpiece_type"] = df["events"].apply(extraer_setpiece_type)
    df["tipo_evento_raw"] = df["events"].astype(str)
    if "id" in df.columns:
        df["id_event"] = df["id"].astype(str)
    df = df.rename(
        columns={
            "player_id": "player_id_event",
            "team_id": "team_id_event",
            "x": "x_event",
            "y": "y_event",
            "to_x": "to_x_event",
            "to_y": "to_y_event",
            "period_order": "period_event",
            "event_time": "match_time_event",
        }
    )
    for col in [
        "player_id_event",
        "team_id_event",
        "x_event",
        "y_event",
        "to_x_event",
        "to_y_event",
        "period_event",
        "match_time_event",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["x_event", "y_event", "to_x_event", "to_y_event"]:
        if col in df.columns and df[col].max(skipna=True) > 1.5:
            df[col] = df[col] / 100.0

    eventos_validos = {
        "pass",
        "passreceived",
        "recovery",
        "interception",
        "ballrecovery",
        "clearance",
        "setpiece",
        "shot",
    }
    setpieces_validos = {"throwin", "freekick", "goalkick", "corner", "kickoff"}
    df = df[df["event_types"].apply(lambda lst: any(e in eventos_validos for e in lst))].copy()
    df = df[(~df["event_types"].apply(lambda lst: "setpiece" in lst)) | (df["setpiece_type"].isin(setpieces_validos))].copy()
    df = df.dropna(subset=["player_id_event", "x_event", "y_event", "period_event", "match_time_event"]).copy()
    df["player_id_event"] = df["player_id_event"].astype(int)
    df["period_event"] = df["period_event"].astype(int)
    keep = [
        c
        for c in [
            "player_id_event",
            "id_event",
            "team_id_event",
            "x_event",
            "y_event",
            "to_x_event",
            "to_y_event",
            "period_event",
            "match_time_event",
            "player_name",
            "team_name",
            "event_types",
            "setpiece_type",
            "tipo_evento_raw",
            "attack_direction",
        ]
        if c in df.columns
    ]
    return df[keep].reset_index(drop=True)
