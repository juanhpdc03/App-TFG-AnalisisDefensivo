from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M


def emparejar_eventing_tracking(
    df_tracking: pd.DataFrame,
    df_eventos_clave: pd.DataFrame,
    max_time_diff_ms: int = 400,
) -> pd.DataFrame:
    """Empareja cada evento clave con el frame de tracking mas cercano."""
    tracking_cols = [
        "match_id",
        "period",
        "frame",
        "match_time",
        "player_id",
        "x",
        "y",
        "ball_x",
        "ball_y",
    ]
    df_tracking_base = df_tracking[[c for c in tracking_cols if c in df_tracking.columns]].copy()
    matches = []
    for _, ev in df_eventos_clave.iterrows():
        t_evt = ev["match_time_event"]
        per = ev["period_event"]
        pid = ev["player_id_event"]
        df_sub = df_tracking_base[
            (df_tracking_base["period"] == per) & (df_tracking_base["player_id"] == pid)
        ].copy()
        if df_sub.empty:
            continue
        df_sub["time_diff"] = np.abs(df_sub["match_time"] - t_evt)
        row_best = df_sub.loc[df_sub["time_diff"].idxmin()]
        if row_best["time_diff"] > max_time_diff_ms:
            continue
        matches.append(
            {
                "player_id": pid,
                "period": per,
                "match_time_event": t_evt,
                "frame_tracking": row_best["frame"],
                "match_time_tracking": row_best["match_time"],
                "time_diff": row_best["time_diff"],
                "x_tracking": row_best["x"],
                "y_tracking": row_best["y"],
                "ball_x_tracking": row_best["ball_x"],
                "ball_y_tracking": row_best["ball_y"],
                "x_event": ev["x_event"],
                "y_event": ev["y_event"],
                "event_types": ev["event_types"],
                "setpiece_type": ev["setpiece_type"],
            }
        )
    return pd.DataFrame(matches)


def detectar_candidatos_recalibracion(
    df_matches: pd.DataFrame,
    th_error: float = 1.5,
    th_dist_ball: float = 2.0,
) -> pd.DataFrame:
    """Detecta eventos donde tracking y eventing estan suficientemente desalineados."""
    if df_matches.empty:
        return pd.DataFrame()
    df = df_matches.copy()
    df["error_m"] = np.sqrt(
        ((df["x_event"] - df["x_tracking"]) * FIELD_LENGTH_M) ** 2
        + ((df["y_event"] - df["y_tracking"]) * FIELD_WIDTH_M) ** 2
    )
    df["dist_player_ball_m"] = np.sqrt(
        ((df["x_tracking"] - df["ball_x_tracking"]) * FIELD_LENGTH_M) ** 2
        + ((df["y_tracking"] - df["ball_y_tracking"]) * FIELD_WIDTH_M) ** 2
    )
    return df[(df["error_m"] > th_error) & (df["dist_player_ball_m"] > th_dist_ball)].copy()


def reasignar_eventos_referencia(
    df_candidatos_recal: pd.DataFrame,
    df_eventos_clave: pd.DataFrame,
    ventana_ms: int = 500,
) -> pd.DataFrame:
    """Busca el evento de referencia mas fiable para cada candidato."""
    rows = []
    for _, cand in df_candidatos_recal.iterrows():
        pid = cand["player_id"]
        per = cand["period"]
        t_cand = cand["match_time_event"]
        df_sub = df_eventos_clave[
            (df_eventos_clave["period_event"] == per)
            & (np.abs(df_eventos_clave["match_time_event"] - t_cand) <= ventana_ms)
        ].copy()
        if df_sub.empty:
            continue
        df_same_player = df_sub[df_sub["player_id_event"] == pid]
        if not df_same_player.empty:
            df_sub = df_same_player
        else:
            team_id = cand.get("team_id", None)
            if team_id is not None:
                df_same_team = df_sub[df_sub["team_id_event"] == team_id]
                if not df_same_team.empty:
                    df_sub = df_same_team
        df_sub["time_diff_evt"] = np.abs(df_sub["match_time_event"] - t_cand)
        evt = df_sub.loc[df_sub["time_diff_evt"].idxmin()]
        dx = evt["x_event"] - cand["x_tracking"]
        dy = evt["y_event"] - cand["y_tracking"]
        error_m = np.sqrt((dx * FIELD_LENGTH_M) ** 2 + (dy * FIELD_WIDTH_M) ** 2)
        rows.append(
            {
                "player_id": pid,
                "period": per,
                "match_time_cand": t_cand,
                "match_time_event_ref": evt["match_time_event"],
                "time_diff_evt": evt["time_diff_evt"],
                "event_types": evt["event_types"],
                "setpiece_type": evt["setpiece_type"],
                "x_tracking": cand["x_tracking"],
                "y_tracking": cand["y_tracking"],
                "x_event_ref": evt["x_event"],
                "y_event_ref": evt["y_event"],
                "error_m": error_m,
            }
        )
    return pd.DataFrame(rows)


def _pesos_pre_normal(n: int) -> list[float]:
    if n <= 0:
        return []
    z = np.linspace(0, 1, n + 1)[1:]
    return (z**0.6).tolist()


def _pesos_pre_setpiece(n: int) -> list[float]:
    base = [0.30, 0.50, 0.70, 0.85, 0.97]
    return base[-n:] if n > 0 else []


def _pesos_post(n: int) -> list[float]:
    base = [0.95, 0.85, 0.70, 0.50, 0.30, 0.15]
    return base[:n]


def _aplicar_throwin_temprano(df_tracking: pd.DataFrame, ev: pd.Series) -> tuple[pd.DataFrame, dict | None]:
    pid = ev["player_id"]
    per = ev["period"]
    t_evt = ev["match_time_event_ref"]
    x_evt = ev["x_event_ref"]
    y_evt = ev["y_event_ref"]
    traj = df_tracking[(df_tracking["player_id"] == pid) & (df_tracking["period"] == per)].sort_values(
        "match_time"
    )
    if traj.empty:
        return df_tracking, None
    frame_ball = (
        df_tracking[df_tracking["period"] == per][["frame", "match_time", "ball_x", "ball_y"]]
        .drop_duplicates()
        .sort_values("match_time")
        .reset_index(drop=True)
        .copy()
    )
    frame_ball["ball_valid"] = frame_ball["ball_x"].notna() & frame_ball["ball_y"].notna()
    frame_ball_pre = frame_ball[frame_ball["match_time"] <= t_evt].copy().reset_index(drop=True)
    if frame_ball_pre.empty:
        return df_tracking, None
    t_out = None
    for i in range(len(frame_ball_pre) - 1):
        if frame_ball_pre.loc[i, "ball_valid"] and not frame_ball_pre.loc[i + 1, "ball_valid"]:
            j = i + 1
            nan_count = 0
            while j < len(frame_ball_pre) and not frame_ball_pre.loc[j, "ball_valid"]:
                nan_count += 1
                j += 1
            if nan_count >= 3:
                t_out = frame_ball_pre.loc[i, "match_time"]
    if t_out is None:
        t_out = max(traj["match_time"].min(), t_evt - 4000)
    if (t_evt - t_out) < 2500:
        t_out = max(traj["match_time"].min(), t_evt - 4000)
    t_hold = t_out + 0.45 * (t_evt - t_out)
    t_post_end = t_evt + 3000
    mask_player = (
        (df_tracking["player_id"] == pid)
        & (df_tracking["period"] == per)
        & (df_tracking["match_time"] >= t_out)
        & (df_tracking["match_time"] <= t_post_end)
    )
    sub = df_tracking.loc[mask_player].copy().sort_values("match_time")
    for idx, row in sub.iterrows():
        t = row["match_time"]
        old_x = row["x"]
        old_y = row["y"]
        if t <= t_hold:
            alpha = np.clip((t - t_out) / max(1, (t_hold - t_out)), 0, 1)
            w = alpha**0.65
        elif t <= t_evt:
            w = 0.985
        else:
            beta = np.clip((t - t_evt) / max(1, (t_post_end - t_evt)), 0, 1)
            w = 0.985 * ((1 - beta) ** 0.55)
        df_tracking.loc[idx, "x"] = (1 - w) * old_x + w * x_evt
        df_tracking.loc[idx, "y"] = (1 - w) * old_y + w * y_evt
        if t <= t_evt:
            mask_frame = (df_tracking["period"] == per) & (df_tracking["frame"] == row["frame"])
            df_tracking.loc[mask_frame, "ball_x"] = x_evt
            df_tracking.loc[mask_frame, "ball_y"] = y_evt
    return df_tracking, {"t_out": t_out, "t_hold": t_hold, "t_evt": t_evt, "t_post_end": t_post_end}


def aplicar_recalibracion_suavizada(
    df_tracking: pd.DataFrame,
    df_eventos_recal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica la recalibracion suavizada de la trayectoria del notebook."""
    df_recal = df_tracking.sort_values(
        [c for c in ["match_id", "period", "match_time", "frame", "player_id"] if c in df_tracking.columns]
    ).reset_index(drop=True).copy()
    correcciones_log = []
    for _, ev in df_eventos_recal.iterrows():
        pid = ev["player_id"]
        per = ev["period"]
        t_evt = ev["match_time_event_ref"]
        x_evt = ev["x_event_ref"]
        y_evt = ev["y_event_ref"]
        setpiece_type = str(ev.get("setpiece_type", "")).lower()
        es_setpiece = setpiece_type in {"throwin", "goalkick", "freekick"}
        if setpiece_type == "throwin":
            df_recal, info = _aplicar_throwin_temprano(df_recal, ev)
            correcciones_log.append(
                {
                    "player_id": pid,
                    "period": per,
                    "match_time_event_ref": t_evt,
                    "error_m": ev.get("error_m", None),
                    "event_types": ev.get("event_types", None),
                    "setpiece_type": ev.get("setpiece_type", None),
                    "modo": "throwin_temprano",
                    "t_out": None if info is None else info["t_out"],
                    "t_hold": None if info is None else info["t_hold"],
                }
            )
            continue
        traj = df_recal[(df_recal["player_id"] == pid) & (df_recal["period"] == per)].sort_values("match_time").copy()
        if traj.empty:
            continue
        traj["abs_dt_event"] = np.abs(traj["match_time"] - t_evt)
        pos_nearest = traj["abs_dt_event"].values.argmin()
        row_anchor = traj.iloc[pos_nearest]
        dx = x_evt - row_anchor["x"]
        dy = y_evt - row_anchor["y"]
        err_m = np.sqrt((dx * FIELD_LENGTH_M) ** 2 + (dy * FIELD_WIDTH_M) ** 2)
        if err_m <= 0.25:
            continue
        k_prev_local = 4
        start_pos = max(0, pos_nearest - k_prev_local)
        end_pos = min(len(traj) - 1, pos_nearest + 4)
        traj_pre = traj.iloc[start_pos:pos_nearest].copy()
        traj_evt = traj.iloc[[pos_nearest]].copy()
        traj_post = traj.iloc[pos_nearest + 1 : end_pos + 1].copy()
        pesos_window = (
            (_pesos_pre_setpiece(len(traj_pre)) if es_setpiece else _pesos_pre_normal(len(traj_pre)))
            + [1.0]
            + _pesos_post(len(traj_post))
        )
        traj_win = pd.concat([traj_pre, traj_evt, traj_post], axis=0)
        for (idx_row, row_win), w in zip(traj_win.iterrows(), pesos_window):
            old_x = df_recal.loc[idx_row, "x"]
            old_y = df_recal.loc[idx_row, "y"]
            new_x = old_x + w * dx
            new_y = old_y + w * dy
            df_recal.loc[idx_row, "x"] = new_x
            df_recal.loc[idx_row, "y"] = new_y
            mask_frame = (df_recal["period"] == per) & (df_recal["frame"] == df_recal.loc[idx_row, "frame"])
            if not mask_frame.any():
                continue
            ball_x_old = df_recal.loc[mask_frame, "ball_x"].iloc[0]
            ball_y_old = df_recal.loc[mask_frame, "ball_y"].iloc[0]
            if pd.isna(ball_x_old) or pd.isna(ball_y_old):
                continue
            if es_setpiece:
                if w == 1.0:
                    df_recal.loc[mask_frame, "ball_x"] = 0.02 * ball_x_old + 0.98 * new_x
                    df_recal.loc[mask_frame, "ball_y"] = 0.02 * ball_y_old + 0.98 * new_y
                continue
            blend = 0.90 if w == 1.0 else (0.35 if df_recal.loc[idx_row, "match_time"] < t_evt else 0.20)
            df_recal.loc[mask_frame, "ball_x"] = (1 - blend) * ball_x_old + blend * new_x
            df_recal.loc[mask_frame, "ball_y"] = (1 - blend) * ball_y_old + blend * new_y
        correcciones_log.append(
            {
                "player_id": pid,
                "period": per,
                "match_time_event_ref": t_evt,
                "frame_anchor": row_anchor["frame"],
                "match_time_anchor": row_anchor["match_time"],
                "dx": dx,
                "dy": dy,
                "error_m": err_m,
                "event_types": ev.get("event_types", None),
                "setpiece_type": ev.get("setpiece_type", None),
                "modo": "general",
            }
        )
    df_recal["x"] = df_recal["x"].clip(0, 1)
    df_recal["y"] = df_recal["y"].clip(0, 1)
    df_recal["ball_x"] = df_recal["ball_x"].clip(0, 1)
    df_recal["ball_y"] = df_recal["ball_y"].clip(0, 1)
    return df_recal, pd.DataFrame(correcciones_log)


def recalibrar_tracking_con_eventing(
    df_tracking_vel_ball: pd.DataFrame,
    df_eventos_clave: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo de recalibracion usado antes de detectar poseedor."""
    matches = emparejar_eventing_tracking(df_tracking_vel_ball, df_eventos_clave)
    candidatos = detectar_candidatos_recalibracion(matches)
    candidatos_ref = reasignar_eventos_referencia(candidatos, df_eventos_clave)
    df_recal, log = aplicar_recalibracion_suavizada(df_tracking_vel_ball, candidatos_ref)
    df_recal["x_m"] = df_recal["x"] * FIELD_LENGTH_M
    df_recal["y_m"] = df_recal["y"] * FIELD_WIDTH_M
    df_recal["ball_x_m"] = df_recal["ball_x"] * FIELD_LENGTH_M
    df_recal["ball_y_m"] = df_recal["ball_y"] * FIELD_WIDTH_M
    df_recal = df_recal.sort_values(["period", "frame", "player_id"]).reset_index(drop=True)
    dx_ball = df_recal["x_m"].values - df_recal["ball_x_m"].values
    dy_ball = df_recal["y_m"].values - df_recal["ball_y_m"].values
    df_recal["dist_ball_m"] = np.sqrt(dx_ball**2 + dy_ball**2)
    return df_recal, log
