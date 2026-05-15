from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M


def asignar_tiros_a_secuencias(
    tiros: pd.DataFrame,
    secuencias: pd.DataFrame,
    max_dist_ms: int = 3500,
) -> pd.DataFrame:
    """Asigna tiros oficiales a la secuencia rival temporalmente mas cercana."""
    tiros_out = tiros.copy()
    if tiros_out.empty or secuencias.empty:
        tiros_out["secuencia_rival_id"] = np.nan
        return tiros_out

    tiros_out["secuencia_rival_id"] = np.nan
    tiros_out["distancia_seq_ms"] = np.nan
    tiros_out["tipo_asignacion_seq"] = None

    intervals = secuencias[
        ["secuencia_rival_id", "period", "start_time_ms", "end_time_ms", "duracion_seg"]
    ].copy()

    for idx, shot in tiros_out.iterrows():
        per = shot.get("period_event", np.nan)
        t = shot.get("match_time_event", np.nan)
        if pd.isna(per) or pd.isna(t):
            continue
        cand = intervals[intervals["period"] == int(per)].copy()
        if cand.empty:
            continue
        cand["distancia_seq_ms"] = np.where(
            t < cand["start_time_ms"],
            cand["start_time_ms"] - t,
            np.where(t > cand["end_time_ms"], t - cand["end_time_ms"], 0),
        )
        cand = cand[cand["distancia_seq_ms"] <= max_dist_ms].copy()
        if cand.empty:
            continue
        cand = cand.sort_values(["distancia_seq_ms", "duracion_seg"], ascending=[True, False])
        best = cand.iloc[0]
        tiros_out.loc[idx, "secuencia_rival_id"] = best["secuencia_rival_id"]
        tiros_out.loc[idx, "distancia_seq_ms"] = best["distancia_seq_ms"]
        tiros_out.loc[idx, "tipo_asignacion_seq"] = (
            "dentro_intervalo" if best["distancia_seq_ms"] == 0 else "tiro_cercano_a_intervalo"
        )

    return tiros_out


def crear_resumen_secuencias(
    df_final_con_secuencias: pd.DataFrame,
    secuencias: pd.DataFrame,
    clusters: pd.DataFrame,
    tiros_oficiales: pd.DataFrame,
) -> pd.DataFrame:
    """Crea una tabla resumen por secuencia con finalizacion oficial."""
    if secuencias.empty:
        return pd.DataFrame()

    tiros_asignados = asignar_tiros_a_secuencias(tiros_oficiales, secuencias)
    base = secuencias.merge(clusters, on="secuencia_rival_id", how="left")
    rows = []
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)

    for _, sec in base.iterrows():
        seq_id = int(sec["secuencia_rival_id"])
        gr = df[df["secuencia_rival_id"] == seq_id].sort_values("match_time")
        if gr.empty:
            continue
        x = gr["ball_x_m"].dropna().to_numpy()
        y = gr["ball_y_m"].dropna().to_numpy()
        if len(x) < 2:
            continue
        tiros_seq = tiros_asignados[tiros_asignados["secuencia_rival_id"] == seq_id].copy()
        tipo_tiro = int(not tiros_seq.empty)
        tipo_tiro_puerta = int(tiros_seq["tipo_finalizacion_tiro_puerta"].max()) if not tiros_seq.empty else 0
        es_gol = bool(tiros_seq["es_gol"].max()) if not tiros_seq.empty and "es_gol" in tiros_seq.columns else False
        match_time_tiro = tiros_seq["match_time_event"].min() if not tiros_seq.empty else np.nan
        distancia_tiro = tiros_seq["distancia_seq_ms"].min() if not tiros_seq.empty else np.nan
        tipo_asignacion = tiros_seq["tipo_asignacion_seq"].iloc[0] if not tiros_seq.empty else None
        xg_tiro = np.nan
        if not tiros_seq.empty and "xg_tiro" in tiros_seq.columns:
            xg_vals = pd.to_numeric(tiros_seq["xg_tiro"], errors="coerce").dropna()
            if not xg_vals.empty:
                xg_tiro = float(xg_vals.max())

        rows.append(
            {
                "secuencia_rival_id": seq_id,
                "cluster_trayectoria": sec.get("cluster_trayectoria", np.nan),
                "match_id": sec.get("match_id", gr["match_id"].iloc[0]),
                "period": sec.get("period", gr["period"].iloc[0]),
                "duracion_seg": sec.get("duracion_seg", np.nan),
                "recorrido_balon_m": sec.get("recorrido_balon_m", np.nan),
                "progresion_x": x[-1] - x[0],
                "amplitud_balon_m": np.nanmax(y) - np.nanmin(y),
                "num_eventos_rivales": sec.get("num_eventos_rivales", sec.get("num_frames", len(gr))),
                "vel_media_balon": sec.get("vel_media_balon", np.nan),
                "x_inicio": x[0],
                "x_fin": x[-1],
                "y_inicio": y[0],
                "y_fin": y[-1],
                "tipo_finalizacion_tiro": tipo_tiro,
                "tipo_finalizacion_tiro_puerta": tipo_tiro_puerta,
                "tipo_finalizacion_centro": int((x[-1] > FIELD_LENGTH_M * 0.75) and (y[-1] < 12 or y[-1] > FIELD_WIDTH_M - 12)),
                "tipo_finalizacion_perdida": int((x[-1] < FIELD_LENGTH_M * 0.5) and tipo_tiro == 0),
                "es_gol": es_gol,
                "xg_tiro": xg_tiro,
                "match_time_tiro_oficial": match_time_tiro,
                "distancia_tiro_seq_ms": distancia_tiro,
                "tipo_asignacion_tiro_seq": tipo_asignacion,
            }
        )
    return pd.DataFrame(rows)
