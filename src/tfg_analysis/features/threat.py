from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M
from tfg_analysis.features.defensive import normalizar_percentiles


def _crear_xt_grid(n_x: int = 32, n_y: int = 24) -> np.ndarray:
    xT_grid = np.zeros((n_x, n_y))
    goal_x = FIELD_LENGTH_M
    goal_y = FIELD_WIDTH_M / 2
    for cx in range(n_x):
        for cy in range(n_y):
            x_center = (cx + 0.5) / n_x * FIELD_LENGTH_M
            y_center = (cy + 0.5) / n_y * FIELD_WIDTH_M
            profundidad = x_center / FIELD_LENGTH_M
            centralidad = 1 - abs(y_center - FIELD_WIDTH_M / 2) / (FIELD_WIDTH_M / 2)
            centralidad = np.clip(centralidad, 0, 1)
            dist_goal = np.sqrt((goal_x - x_center) ** 2 + (goal_y - y_center) ** 2)
            cercania = 1 - dist_goal / np.sqrt(FIELD_LENGTH_M**2 + (FIELD_WIDTH_M / 2) ** 2)
            cercania = np.clip(cercania, 0, 1)
            xT_grid[cx, cy] = (
                0.005
                + 0.035 * (profundidad**1.8)
                + 0.090 * (profundidad**3.2) * centralidad
                + 0.220 * (cercania**5.0) * (centralidad**1.5)
            )
    xT_grid = gaussian_filter(xT_grid, sigma=0.8)
    return xT_grid / xT_grid.max() * 0.32


def _asignar_celda_xt(x: float, y: float, n_x: int, n_y: int) -> tuple[int, int]:
    cell_x = int(np.floor(x / FIELD_LENGTH_M * n_x))
    cell_y = int(np.floor(y / FIELD_WIDTH_M * n_y))
    return max(0, min(n_x - 1, cell_x)), max(0, min(n_y - 1, cell_y))


def calcular_xt_secuencias(
    df_final_con_secuencias: pd.DataFrame,
    n_points: int = 20,
    n_x: int = 32,
    n_y: int = 24,
) -> pd.DataFrame:
    """xThreat geometrico estable y amenaza concedida dinamica del notebook."""
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    if "ball_x_norm" not in df.columns:
        df["ball_x_norm"] = np.where(df["period"] == 0, df["ball_x_m"], FIELD_LENGTH_M - df["ball_x_m"])
    if "ball_y_norm" not in df.columns:
        df["ball_y_norm"] = np.where(df["period"] == 0, df["ball_y_m"], FIELD_WIDTH_M - df["ball_y_m"])
    df = (
        df.dropna(subset=["ball_x_norm", "ball_y_norm"])
        .drop_duplicates(subset=["secuencia_rival_id", "period", "frame"])
        .sort_values(["secuencia_rival_id", "match_time"])
    )
    df = df[df["ball_x_norm"].between(0, FIELD_LENGTH_M) & df["ball_y_norm"].between(0, FIELD_WIDTH_M)].copy()
    xT_grid = _crear_xt_grid(n_x=n_x, n_y=n_y)

    rows = []
    for seq_id, gr in df.groupby("secuencia_rival_id"):
        gr = gr.sort_values("match_time").dropna(subset=["ball_x_norm", "ball_y_norm"]).copy()
        if len(gr) < 3:
            continue
        t = gr["match_time"].to_numpy(dtype=float)
        x = gr["ball_x_norm"].to_numpy(dtype=float)
        y = gr["ball_y_norm"].to_numpy(dtype=float)
        _, idx_unique = np.unique(t, return_index=True)
        t = t[idx_unique]
        x = x[idx_unique]
        y = y[idx_unique]
        if len(t) < 3 or t.max() == t.min():
            continue
        t_norm = (t - t.min()) / (t.max() - t.min())
        t_new = np.linspace(0, 1, n_points)
        x_new = np.interp(t_new, t_norm, x)
        y_new = np.interp(t_new, t_norm, y)
        xt_values = []
        for xi, yi in zip(x_new, y_new):
            cell_x, cell_y = _asignar_celda_xt(np.clip(xi, 0, FIELD_LENGTH_M), np.clip(yi, 0, FIELD_WIDTH_M), n_x, n_y)
            xt_values.append(float(xT_grid[cell_x, cell_y]))
        row = {"secuencia_rival_id": int(seq_id)}
        row.update({f"amenaza_concedida_{i}": xt_values[i] for i in range(n_points)})
        row["amenaza_concedida_inicio"] = xt_values[0]
        row["amenaza_concedida_final"] = xt_values[-1]
        row["incremento_amenaza_concedida"] = xt_values[-1] - xt_values[0]
        row["pico_amenaza_concedida"] = max(xt_values)
        row["exposicion_amenaza_concedida"] = sum(xt_values)
        row["area_amenaza_concedida"] = float(np.trapezoid(xt_values, dx=1 / (n_points - 1)))
        row["xT_inicio"] = row["amenaza_concedida_inicio"]
        row["xT_final"] = row["amenaza_concedida_final"]
        row["xT_added"] = row["incremento_amenaza_concedida"]
        row["xT_max"] = row["pico_amenaza_concedida"]
        row["xT_peak_added"] = row["pico_amenaza_concedida"] - row["amenaza_concedida_inicio"]
        rows.append(row)
    return pd.DataFrame(rows)


def score_finalizacion_avanzado(row) -> float:
    def _flag(name: str) -> int:
        value = pd.to_numeric(row.get(name, 0), errors="coerce")
        return int(value) if pd.notna(value) else 0

    tiro = _flag("tipo_finalizacion_tiro")
    tiro_puerta = _flag("tipo_finalizacion_tiro_puerta")
    centro = _flag("tipo_finalizacion_centro")
    perdida = _flag("tipo_finalizacion_perdida")
    es_gol = bool(row.get("es_gol", False))
    if es_gol:
        return 1.0

    if tiro == 1:
        xg = pd.to_numeric(row.get("xg_tiro", row.get("xG", row.get("xg", np.nan))), errors="coerce")
        if pd.notna(xg):
            return float(np.clip(0.30 + 0.70 * xg, 0.30, 1.0))
        return 0.75 if tiro_puerta == 1 else 0.60

    if centro == 1:
        return 0.30

    if perdida == 1:
        x_fin = pd.to_numeric(row.get("x_fin", np.nan), errors="coerce")
        if pd.notna(x_fin) and x_fin >= FIELD_LENGTH_M * 0.66:
            return 0.20
        return 0.10

    return 0.05


def calcular_ipar(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    """Calcula IPO con tres subindices normalizados y pesos tacticos."""
    df = df_def_dinamico.copy()
    if "pitch_control_rival_detras_linea" not in df.columns:
        df["pitch_control_rival_detras_linea"] = df.get("control_zona_peligrosa_rival_medio", 0)
    if "xT_max" not in df.columns:
        df["xT_max"] = df.get("pico_amenaza_concedida", 0)
    if "score_finalizacion" not in df.columns:
        df["score_finalizacion"] = 0
    df["sub_ipar_pc_zona_peligrosa"] = normalizar_percentiles(df["pitch_control_rival_detras_linea"], 10, 90)
    df["sub_ipar_xt_max"] = normalizar_percentiles(df["xT_max"], 15, 85)
    df["sub_ipar_finalizacion"] = normalizar_percentiles(df["score_finalizacion"], 10, 90)
    df["subindice_pitch_control"] = df["sub_ipar_pc_zona_peligrosa"]
    df["subindice_xt"] = df["sub_ipar_xt_max"]
    df["subindice_finalizacion"] = df["sub_ipar_finalizacion"]
    df["indice_peligrosidad_accion"] = (
        0.15 * df["sub_ipar_pc_zona_peligrosa"]
        + 0.35 * df["sub_ipar_xt_max"]
        + 0.50 * df["sub_ipar_finalizacion"]
    )
    return df


def categorizar_percentiles(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df[out_col] = []
        return df
    serie = pd.to_numeric(df[col], errors="coerce")
    if serie.dropna().empty:
        df[out_col] = "media"
        return df
    p50 = serie.quantile(0.50)
    p75 = serie.quantile(0.75)
    p90 = serie.quantile(0.90)
    p97 = serie.quantile(0.97)

    def categorizar(valor):
        if pd.isna(valor):
            return np.nan
        if valor >= p97:
            return "critica"
        if valor >= p90:
            return "muy_alta"
        if valor >= p75:
            return "alta"
        if valor >= p50:
            return "media"
        return "baja"

    orden = ["baja", "media", "alta", "muy_alta", "critica"]
    df[out_col] = pd.Categorical(serie.apply(categorizar), categories=orden, ordered=True)
    return df
