from __future__ import annotations

import numpy as np
import pandas as pd

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M


def _interp(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(n)
    if len(values) == 1:
        return np.repeat(values[0], n)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)


def crear_perfil_defensivo_dinamico(
    df_final_con_secuencias: pd.DataFrame,
    df_clusters: pd.DataFrame,
    team_id: int,
    n_points: int = 20,
) -> pd.DataFrame:
    """Resume la estructura defensiva del equipo propio en cada secuencia rival."""
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    df["x_norm"] = np.where(df["period"] == 0, df["x_m"], FIELD_LENGTH_M - df["x_m"])
    df["y_norm"] = np.where(df["period"] == 0, df["y_m"], FIELD_WIDTH_M - df["y_m"])
    df["ball_x_norm"] = np.where(df["period"] == 0, df["ball_x_m"], FIELD_LENGTH_M - df["ball_x_m"])
    df["ball_y_norm"] = np.where(df["period"] == 0, df["ball_y_m"], FIELD_WIDTH_M - df["ball_y_m"])

    own = df[df["team_id"] == team_id].copy()
    own["dist_balon"] = np.sqrt((own["x_norm"] - own["ball_x_norm"]) ** 2 + (own["y_norm"] - own["ball_y_norm"]) ** 2)

    rows = []
    for seq_id, gr in own.groupby("secuencia_rival_id"):
        frames = []
        for _, gf in gr.groupby("match_time"):
            frames.append(
                {
                    "altura": gf["x_norm"].mean(),
                    "anchura": gf["y_norm"].max() - gf["y_norm"].min(),
                    "jug5": (gf["dist_balon"] <= 5).sum(),
                    "jug10": (gf["dist_balon"] <= 10).sum(),
                    "dist": gf["dist_balon"].mean(),
                    "dist_cercano": gf["dist_balon"].min(),
                    "ballx": gf["ball_x_norm"].iloc[0],
                }
            )
        fdf = pd.DataFrame(frames)
        if len(fdf) < 2:
            continue
        row = {"secuencia_rival_id": seq_id}
        for var in ["altura", "anchura", "jug5", "jug10", "dist", "dist_cercano", "ballx"]:
            vals = _interp(fdf[var].to_numpy(dtype=float), n_points)
            row.update({f"{var}_{i}": vals[i] for i in range(n_points)})
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.merge(df_clusters[["secuencia_rival_id", "cluster_trayectoria"]], on="secuencia_rival_id", how="left")
    return out


def anadir_pitch_control(
    df_def_dinamico: pd.DataFrame,
    df_final_con_secuencias: pd.DataFrame,
    team_id: int,
    n_points: int = 20,
    reaction_time: float = 0.7,
    v_max: float = 5.5,
    n_grid_x: int = 30,
    n_grid_y: int = 20,
) -> pd.DataFrame:
    """Replica el pitch control defensivo dinamico del notebook."""
    df_pc = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    if df_pc.empty:
        return df_def_dinamico.copy()
    df_pc["secuencia_rival_id"] = df_pc["secuencia_rival_id"].astype(int)
    df_pc["x_norm"] = np.where(df_pc["period"] == 0, df_pc["x_m"], FIELD_LENGTH_M - df_pc["x_m"])
    df_pc["y_norm"] = np.where(df_pc["period"] == 0, df_pc["y_m"], FIELD_WIDTH_M - df_pc["y_m"])
    df_pc["velocidad_m_s"] = df_pc.get("velocidad_m_s", df_pc.get("vel_m_s_suav", 0)).fillna(0)
    df_pc["direccion_rad"] = df_pc.get("direccion_rad", df_pc.get("dir_rad", 0)).fillna(0)
    df_pc["vx_norm"] = df_pc["velocidad_m_s"] * np.cos(df_pc["direccion_rad"])
    df_pc["vy_norm"] = df_pc["velocidad_m_s"] * np.sin(df_pc["direccion_rad"])
    df_pc.loc[df_pc["period"] == 1, "vx_norm"] *= -1
    df_pc.loc[df_pc["period"] == 1, "vy_norm"] *= -1

    x_grid = np.linspace(0, FIELD_LENGTH_M, n_grid_x)
    y_grid = np.linspace(0, FIELD_WIDTH_M, n_grid_y)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    zona_peligrosa_mask = (
        (points[:, 0] >= FIELD_LENGTH_M * 0.60)
        & (points[:, 1] >= FIELD_WIDTH_M * 0.15)
        & (points[:, 1] <= FIELD_WIDTH_M * 0.85)
    )

    def arrival_times(players_df, pts):
        if players_df.empty:
            return np.full(len(pts), np.inf)
        x = players_df["x_norm"].values[:, None]
        y = players_df["y_norm"].values[:, None]
        vx = players_df["vx_norm"].values[:, None]
        vy = players_df["vy_norm"].values[:, None]
        x_react = x + vx * reaction_time
        y_react = y + vy * reaction_time
        dist = np.sqrt((pts[:, 0][None, :] - x_react) ** 2 + (pts[:, 1][None, :] - y_react) ** 2)
        return (reaction_time + dist / v_max).min(axis=0)

    rows = []
    for seq_id, gr_seq in df_pc.groupby("secuencia_rival_id"):
        frames = []
        for t_frame, gr_f in gr_seq.groupby("match_time"):
            own = gr_f[gr_f["team_id"] == team_id].copy()
            rival = gr_f[gr_f["team_id"] != team_id].copy()
            if len(own) < 5 or len(rival) < 5:
                continue
            t_own = arrival_times(own, points)
            t_rival = arrival_times(rival, points)
            p_own = 1 / (1 + np.exp(-8 * (t_rival - t_own)))
            p_rival = 1 - p_own
            frames.append(
                {
                    "match_time": t_frame,
                    "pct_campo_subiza": p_own.mean(),
                    "pct_campo_rival": p_rival.mean(),
                    "pct_zona_peligrosa_subiza": p_own[zona_peligrosa_mask].mean(),
                    "pct_zona_peligrosa_rival": p_rival[zona_peligrosa_mask].mean(),
                }
            )
        fdf = pd.DataFrame(frames).sort_values("match_time") if frames else pd.DataFrame()
        if len(fdf) < 3:
            continue
        row = {"secuencia_rival_id": seq_id}
        for col in ["pct_campo_subiza", "pct_campo_rival", "pct_zona_peligrosa_subiza", "pct_zona_peligrosa_rival"]:
            vals = _interp(fdf[col].to_numpy(dtype=float), n_points)
            row.update({f"{col}_{i}": vals[i] for i in range(n_points)})
        rows.append(row)

    pc = pd.DataFrame(rows)
    if pc.empty:
        return df_def_dinamico.copy()

    cols_campo_subiza = [f"pct_campo_subiza_{i}" for i in range(n_points)]
    cols_campo_rival = [f"pct_campo_rival_{i}" for i in range(n_points)]
    cols_zona_subiza = [f"pct_zona_peligrosa_subiza_{i}" for i in range(n_points)]
    cols_zona_rival = [f"pct_zona_peligrosa_rival_{i}" for i in range(n_points)]
    pc["control_campo_subiza_medio"] = pc[cols_campo_subiza].mean(axis=1)
    pc["control_campo_rival_medio"] = pc[cols_campo_rival].mean(axis=1)
    pc["control_zona_peligrosa_subiza_medio"] = pc[cols_zona_subiza].mean(axis=1)
    pc["control_zona_peligrosa_rival_medio"] = pc[cols_zona_rival].mean(axis=1)
    pc["pitch_control_rival_detras_linea"] = pc["control_zona_peligrosa_rival_medio"]
    pc["perdida_control_campo_subiza"] = pc[f"pct_campo_subiza_0"] - pc[f"pct_campo_subiza_{n_points - 1}"]
    pc["incremento_control_zona_peligrosa_rival"] = pc[f"pct_zona_peligrosa_rival_{n_points - 1}"] - pc[
        f"pct_zona_peligrosa_rival_0"
    ]
    cols_drop = [
        c
        for c in df_def_dinamico.columns
        if c.startswith("pct_campo_")
        or c.startswith("pct_zona_peligrosa_")
        or c
        in [
            "control_campo_subiza_medio",
            "control_campo_rival_medio",
            "control_zona_peligrosa_subiza_medio",
            "control_zona_peligrosa_rival_medio",
            "pitch_control_rival_detras_linea",
            "perdida_control_campo_subiza",
            "incremento_control_zona_peligrosa_rival",
        ]
    ]
    return df_def_dinamico.drop(columns=cols_drop, errors="ignore").merge(pc, on="secuencia_rival_id", how="left")


def anadir_pitch_control_simplificado(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    """Compatibilidad antigua. Usa `anadir_pitch_control` para fidelidad notebook."""
    return df_def_dinamico.copy()


def minmax_01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    rng = s.max() - s.min()
    if pd.isna(rng) or rng == 0:
        return pd.Series(0, index=s.index)
    return (s - s.min()) / rng


def normalizar_percentiles(s: pd.Series, p_low: int = 10, p_high: int = 90) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    low = s.quantile(p_low / 100)
    high = s.quantile(p_high / 100)
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0, index=s.index)
    return ((s - low) / (high - low)).clip(0, 1)


def calcular_ddi(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    """Calcula el IDD: índice de desorganización defensiva."""
    df = df_def_dinamico.copy()
    cols_anchura = [c for c in df.columns if c.startswith("anchura_")]
    cols_dist = [c for c in df.columns if c.startswith("dist_") and not c.startswith("dist_cercano_")]
    cols_dist_cercano = [c for c in df.columns if c.startswith("dist_cercano_")]
    cols_ballx = [c for c in df.columns if c.startswith("ballx_")]
    df["anchura_media"] = df[cols_anchura].mean(axis=1) if cols_anchura else 0
    df["anchura_maxima"] = df[cols_anchura].max(axis=1) if cols_anchura else 0
    df["dist_media_global"] = df[cols_dist].mean(axis=1) if cols_dist else 0
    df["dist_jugador_mas_cercano_balon"] = df[cols_dist_cercano].mean(axis=1) if cols_dist_cercano else df["dist_media_global"]
    df["retroceso_defensivo"] = df[cols_ballx].iloc[:, -1] - df[cols_ballx].iloc[:, 0] if len(cols_ballx) >= 2 else 0
    for col in ["perdida_control_campo_subiza", "incremento_control_zona_peligrosa_rival", "control_campo_rival_medio"]:
        if col not in df.columns:
            df[col] = 0
    df["sub_ddi_anchura"] = normalizar_percentiles(df["anchura_media"], 10, 90)
    df["sub_ddi_distancia_cercano_balon"] = normalizar_percentiles(df["dist_jugador_mas_cercano_balon"], 10, 90)
    df["sub_ddi_retroceso"] = normalizar_percentiles(df["retroceso_defensivo"].clip(lower=0), 10, 90)
    df["sub_ddi_pitch_control_rival"] = normalizar_percentiles(df["control_campo_rival_medio"], 5, 95)
    for col in [
        "sub_ddi_anchura",
        "sub_ddi_distancia_cercano_balon",
        "sub_ddi_retroceso",
        "sub_ddi_pitch_control_rival",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    pesos = {
        "sub_ddi_anchura": 0.10,
        "sub_ddi_distancia_cercano_balon": 0.10,
        "sub_ddi_retroceso": 0.30,
        "sub_ddi_pitch_control_rival": 0.50,
    }
    df["indice_desorganizacion"] = sum(df[col] * peso for col, peso in pesos.items())
    mapa = {
        "sub_ddi_anchura": "estructura_anchura",
        "sub_ddi_distancia_cercano_balon": "presion_distancia",
        "sub_ddi_retroceso": "estructura_retroceso",
        "sub_ddi_pitch_control_rival": "pitch_control_rival",
    }
    cols = list(pesos)
    orden = df[cols].apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
    df["tipo_desorganizacion_principal"] = orden.apply(lambda xs: mapa[xs[0]])
    df["tipo_desorganizacion_secundaria"] = orden.apply(lambda xs: mapa[xs[1]])
    return df
