from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M


def reorganizar_tracking_250ms(df_tracking: pd.DataFrame) -> pd.DataFrame:
    """Reorganiza el tracking en bloques de 250 ms por jugador y periodo."""
    df = df_tracking.copy()
    cols_num = [
        "match_id",
        "match_time",
        "period",
        "frame",
        "player_id",
        "team_id",
        "x",
        "y",
        "ball_x",
        "ball_y",
        "dist_ball",
        "poseedor",
    ]
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["player_id", "period", "match_time", "x", "y"]).copy()
    sort_cols = [c for c in ["match_id", "period", "player_id", "match_time", "frame"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["bloque_250ms"] = (df["match_time"] // 250).astype(int)
    df["tiempo_seg"] = df["match_time"] / 1000.0

    group_cols = [c for c in ["match_id", "period", "player_id", "bloque_250ms"] if c in df.columns]
    df_250ms = df.groupby(group_cols, as_index=False).first()
    sort_cols_250 = [c for c in ["match_id", "period", "player_id", "match_time", "frame"] if c in df_250ms.columns]
    return df_250ms.sort_values(sort_cols_250).reset_index(drop=True)


def calcular_velocidad_jugadores(df_250ms: pd.DataFrame) -> pd.DataFrame:
    """Calcula velocidades y direccion de movimiento de jugadores."""
    df = df_250ms.copy()
    df["x_m"] = df["x"] * FIELD_LENGTH_M
    df["y_m"] = df["y"] * FIELD_WIDTH_M

    sort_cols = [c for c in ["match_id", "player_id", "period", "match_time", "frame"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    group_cols = [c for c in ["match_id", "player_id", "period"] if c in df.columns]

    df["x_m_prev"] = df.groupby(group_cols)["x_m"].shift(1)
    df["y_m_prev"] = df.groupby(group_cols)["y_m"].shift(1)
    df["t_prev"] = df.groupby(group_cols)["match_time"].shift(1)

    df["dt_seg"] = (df["match_time"] - df["t_prev"]) / 1000.0
    df["dx_m"] = df["x_m"] - df["x_m_prev"]
    df["dy_m"] = df["y_m"] - df["y_m_prev"]
    df["dist_m"] = np.sqrt(df["dx_m"] ** 2 + df["dy_m"] ** 2)
    df["vel_m_s"] = np.where(df["dt_seg"] > 0, df["dist_m"] / df["dt_seg"], np.nan)
    df["dir_rad"] = np.arctan2(df["dy_m"], df["dx_m"])
    df["vel_m_s"] = df["vel_m_s"].clip(upper=12)
    df["vel_m_s_suav"] = df.groupby(group_cols)["vel_m_s"].transform(
        lambda s: s.rolling(window=3, min_periods=1, center=True).mean()
    )
    df["velocidad_m_s"] = df["vel_m_s_suav"]
    df["direccion_rad"] = df["dir_rad"]
    return df


def calcular_velocidad_balon(df_vel: pd.DataFrame) -> pd.DataFrame:
    """Calcula velocidad del balon y la incorpora al dataframe de jugadores."""
    df = df_vel.copy()
    ball_cols = [c for c in ["match_id", "period", "frame", "match_time", "tiempo_seg", "ball_x", "ball_y"] if c in df.columns]
    ball = (
        df[ball_cols]
        .drop_duplicates(subset=[c for c in ["match_id", "period", "frame"] if c in ball_cols])
        .sort_values([c for c in ["match_id", "period", "match_time", "frame"] if c in ball_cols])
        .reset_index(drop=True)
    )
    ball["ball_x_m"] = ball["ball_x"] * FIELD_LENGTH_M
    ball["ball_y_m"] = ball["ball_y"] * FIELD_WIDTH_M

    group_cols = [c for c in ["match_id", "period"] if c in ball.columns]
    ball["ball_x_m_prev"] = ball.groupby(group_cols)["ball_x_m"].shift(1)
    ball["ball_y_m_prev"] = ball.groupby(group_cols)["ball_y_m"].shift(1)
    ball["t_prev"] = ball.groupby(group_cols)["match_time"].shift(1)
    ball["dt_ball_seg"] = (ball["match_time"] - ball["t_prev"]) / 1000.0
    ball["dx_ball_m"] = ball["ball_x_m"] - ball["ball_x_m_prev"]
    ball["dy_ball_m"] = ball["ball_y_m"] - ball["ball_y_m_prev"]
    ball["dist_ball_m"] = np.sqrt(ball["dx_ball_m"] ** 2 + ball["dy_ball_m"] ** 2)
    ball["vel_ball_m_s"] = np.where(ball["dt_ball_seg"] > 0, ball["dist_ball_m"] / ball["dt_ball_seg"], np.nan)
    ball["dir_ball_rad"] = np.arctan2(ball["dy_ball_m"], ball["dx_ball_m"])

    merge_keys = [c for c in ["match_id", "period", "frame"] if c in df.columns and c in ball.columns]
    ball_features = merge_keys + ["ball_x_m", "ball_y_m", "vel_ball_m_s", "dir_ball_rad"]
    out = df.merge(ball[ball_features], on=merge_keys, how="left")
    out["dist_ball_m"] = np.sqrt((out["x_m"] - out["ball_x_m"]) ** 2 + (out["y_m"] - out["ball_y_m"]) ** 2)
    return out


def detectar_poseedor(
    df_tracking: pd.DataFrame,
    radio_candidato: float = 2.0,
    umbral_vel: float = 5.0,
    radio_recepcion_clara: float = 0.30,
) -> pd.DataFrame:
    """Recalcula poseedor con reglas fisicas y contexto `ball_state`."""
    df = df_tracking.copy()
    if "dist_ball_m" not in df.columns:
        df["dist_ball_m"] = np.sqrt((df["x_m"] - df["ball_x_m"]) ** 2 + (df["y_m"] - df["ball_y_m"]) ** 2)

    team_ids = df["team_id"].dropna().unique()
    home_team_id = team_ids[0] if len(team_ids) else None
    away_team_id = team_ids[1] if len(team_ids) > 1 else None

    def get_team_side(team_id):
        if pd.isna(team_id):
            return None
        if team_id == home_team_id:
            return "home"
        if team_id == away_team_id:
            return "away"
        return None

    df["team_side"] = df["team_id"].apply(get_team_side)
    vel_jugador = df.get("vel_m_s_suav", df.get("vel_m_s", 0)).fillna(0)
    vel_balon = df.get("vel_ball_m_s", 0).fillna(0)
    cond_dist = df["dist_ball_m"] <= radio_candidato
    cond_vel = (vel_jugador - vel_balon).abs() <= umbral_vel
    cond_candidato = cond_dist & cond_vel
    cond_recepcion_clara = df["dist_ball_m"] <= radio_recepcion_clara
    ball_state = df.get("ball_state", pd.Series("neutral", index=df.index)).fillna("none")
    cond_team = (df["team_side"] == ball_state) & (ball_state != "none")
    cond_neutral = ball_state == "neutral"
    df["poseedor"] = np.select(
        [cond_recepcion_clara, cond_candidato & cond_team, cond_candidato & cond_neutral],
        [1, 1, 2],
        default=0,
    ).astype(int)
    return df
