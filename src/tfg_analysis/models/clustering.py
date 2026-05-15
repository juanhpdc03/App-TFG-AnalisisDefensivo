from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M


def _interp_series(values: np.ndarray, n_points: int) -> np.ndarray:
    if len(values) == 0:
        return np.full(n_points, np.nan)
    if len(values) == 1:
        return np.repeat(values[0], n_points)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, values)


def preparar_trayectorias(
    df_final_con_secuencias: pd.DataFrame,
    n_points: int = 20,
) -> pd.DataFrame:
    """Representa cada secuencia como en el notebook base: x/y normalizados e intercalados."""
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    if "ball_x_m" not in df.columns:
        df["ball_x_m"] = df["ball_x"] * FIELD_LENGTH_M
    if "ball_y_m" not in df.columns:
        df["ball_y_m"] = df["ball_y"] * FIELD_WIDTH_M

    rows = []
    for seq_id, gr in df.groupby("secuencia_rival_id"):
        gr = (
            gr.drop_duplicates(subset=[c for c in ["period", "frame"] if c in gr.columns])
            .sort_values("match_time")
            .copy()
        )
        if len(gr) < 2:
            continue
        x = gr["ball_x_m"].to_numpy(dtype=float)
        y = gr["ball_y_m"].to_numpy(dtype=float)
        period = int(gr["period"].iloc[0]) if "period" in gr.columns else 0
        if period == 1:
            x = FIELD_LENGTH_M - x
            y = FIELD_WIDTH_M - y
        x_new = _interp_series(x, n_points)
        y_new = _interp_series(y, n_points)
        vector = np.column_stack([x_new, y_new]).ravel()
        rows.append({"secuencia_rival_id": seq_id, **{f"traj_{i}": v for i, v in enumerate(vector)}})
    return pd.DataFrame(rows)


def crear_features_tacticas(
    df_secuencias: pd.DataFrame,
    df_final_con_secuencias: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Features tacticas del notebook base para el clustering."""
    if df_secuencias.empty:
        return pd.DataFrame()
    df = df_secuencias.copy()
    if "progresion_x" not in df.columns:
        df["progresion_x"] = df["ball_x_fin_m"] - df["ball_x_ini_m"]
    df["verticalidad"] = (df["progresion_x"] / df["recorrido_balon_m"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["horizontalidad"] = (
        abs(df["ball_y_fin_m"] - df["ball_y_ini_m"]) / df["recorrido_balon_m"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    if "num_eventos_rivales" not in df.columns:
        df["num_eventos_rivales"] = df.get("num_frames", 0)
    if "num_pases_bepro" not in df.columns:
        df["num_pases_bepro"] = 0
    df["ritmo_pases"] = (df["num_pases_bepro"] / df["duracion_seg"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["duracion_log"] = np.log1p(df["duracion_seg"])
    df["eventos_log"] = np.log1p(df["num_eventos_rivales"])
    df["recorrido_log"] = np.log1p(df["recorrido_balon_m"])

    if df_final_con_secuencias is not None and not df_final_con_secuencias.empty:
        amp = (
            df_final_con_secuencias.dropna(subset=["secuencia_rival_id"])
            .assign(secuencia_rival_id=lambda d: d["secuencia_rival_id"].astype(int))
            .groupby("secuencia_rival_id")["ball_y_m"]
            .apply(lambda s: s.max() - s.min())
            .reset_index(name="amplitud_balon_m")
        )
        df = df.drop(columns=["amplitud_balon_m"], errors="ignore").merge(amp, on="secuencia_rival_id", how="left")
    elif "amplitud_balon_m" not in df.columns:
        df["amplitud_balon_m"] = 0
    df["amplitud_balon_m"] = df["amplitud_balon_m"].fillna(0)

    keep = [
        "secuencia_rival_id",
        "verticalidad",
        "horizontalidad",
        "ritmo_pases",
        "duracion_log",
        "eventos_log",
        "recorrido_log",
        "vel_media_balon",
        "amplitud_balon_m",
    ]
    return df[[c for c in keep if c in df.columns]]


def _evaluar_kmeans(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8, random_state: int = 42) -> pd.DataFrame:
    n = len(X_scaled)
    if n < 4:
        return pd.DataFrame()
    k_max = min(k_max, n - 1)
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_scaled)
        counts = np.bincount(labels, minlength=k)
        if len(np.unique(labels)) < 2:
            sil = np.nan
        else:
            sil = silhouette_score(X_scaled, labels)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(sil) if pd.notna(sil) else np.nan,
                "min_cluster_size": int(counts.min()),
            }
        )
    diagnostics = pd.DataFrame(rows)
    if len(diagnostics) >= 3:
        x = diagnostics["k"].to_numpy(dtype=float)
        y = diagnostics["inertia"].to_numpy(dtype=float)
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        denom = np.linalg.norm(p2 - p1) or 1.0
        distances = []
        for xi, yi in zip(x, y):
            p = np.array([xi, yi])
            distances.append(abs(np.cross(p2 - p1, p1 - p)) / denom)
        diagnostics["elbow_strength"] = distances
    else:
        diagnostics["elbow_strength"] = 0.0
    return diagnostics


def seleccionar_k_optimo(X_scaled: np.ndarray, random_state: int = 42, k_min: int = 2, k_max: int = 8) -> tuple[int, pd.DataFrame]:
    """Selecciona K de forma automatica combinando codo y silhouette.

    La silhouette evita clusters artificiales; el codo evita quedarse siempre con
    k pequeno cuando hay varias familias claras de trayectorias.
    """
    diagnostics = _evaluar_kmeans(X_scaled, k_min=k_min, k_max=k_max, random_state=random_state)
    if diagnostics.empty:
        return 1, diagnostics

    n = len(X_scaled)
    min_size = max(2, int(np.ceil(n * 0.04)))
    candidates = diagnostics[diagnostics["min_cluster_size"] >= min_size].copy()
    if candidates.empty:
        candidates = diagnostics.copy()

    for col in ["silhouette", "elbow_strength"]:
        values = candidates[col].fillna(candidates[col].min()).astype(float)
        span = values.max() - values.min()
        candidates[f"{col}_norm"] = 0.0 if span == 0 else (values - values.min()) / span

    candidates["auto_score"] = 0.65 * candidates["silhouette_norm"] + 0.35 * candidates["elbow_strength_norm"]
    chosen = candidates.sort_values(["auto_score", "silhouette", "k"], ascending=[False, False, True]).iloc[0]
    diagnostics = diagnostics.merge(candidates[["k", "auto_score"]], on="k", how="left")
    return int(chosen["k"]), diagnostics


def clusterizar_ataques(
    df_trayectorias: pd.DataFrame,
    df_features: pd.DataFrame | None = None,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna cluster de trayectoria a cada secuencia."""
    if df_trayectorias.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_model = df_trayectorias.copy()
    if df_features is not None and not df_features.empty:
        df_model = df_model.merge(df_features, on="secuencia_rival_id", how="left")

    ids = df_model["secuencia_rival_id"].astype(int)
    X = df_model.drop(columns=["secuencia_rival_id"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    if n_clusters is None:
        k, diagnostics = seleccionar_k_optimo(X_scaled, random_state=random_state)
    else:
        k = min(n_clusters, len(X))
        diagnostics = pd.DataFrame()
    if k <= 1:
        labels = np.zeros(len(X), dtype=int)
    else:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(X_scaled)

    asignacion = pd.DataFrame({"secuencia_rival_id": ids, "cluster_trayectoria": labels})
    asignacion["n_clusters_optimo"] = int(k)
    df_model = df_model.merge(asignacion, on="secuencia_rival_id", how="left")
    if not diagnostics.empty:
        df_model.attrs["cluster_diagnostics"] = diagnostics
    return asignacion, df_model


def _clasificar_zona_x(x: float) -> str:
    if x < FIELD_LENGTH_M * 0.25:
        return "zona_1"
    if x < FIELD_LENGTH_M * 0.50:
        return "zona_2"
    if x < FIELD_LENGTH_M * 0.75:
        return "zona_3"
    return "zona_4"


def _clasificar_carril_y(y: float) -> str:
    if y < FIELD_WIDTH_M * 0.20:
        return "carril_1"
    if y < FIELD_WIDTH_M * 0.40:
        return "carril_2"
    if y < FIELD_WIDTH_M * 0.60:
        return "carril_3"
    if y < FIELD_WIDTH_M * 0.80:
        return "carril_4"
    return "carril_5"


def crear_taxonomia_clusters(
    df_clusters: pd.DataFrame,
    df_final_con_secuencias: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Crea la taxonomia tactica neutra del notebook: zonas 1-4 y carriles 1-5."""
    if df_clusters.empty or "cluster_trayectoria" not in df_clusters.columns:
        return pd.DataFrame()

    df_plot = pd.DataFrame()
    if df_final_con_secuencias is not None and not df_final_con_secuencias.empty:
        df_plot = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
        df_plot["secuencia_rival_id"] = df_plot["secuencia_rival_id"].astype(int)
        if "ball_x_m" not in df_plot.columns:
            df_plot["ball_x_m"] = df_plot["ball_x"] * FIELD_LENGTH_M
        if "ball_y_m" not in df_plot.columns:
            df_plot["ball_y_m"] = df_plot["ball_y"] * FIELD_WIDTH_M
        df_plot["ball_x_norm"] = np.where(df_plot["period"] == 0, df_plot["ball_x_m"], FIELD_LENGTH_M - df_plot["ball_x_m"])
        df_plot["ball_y_norm"] = np.where(df_plot["period"] == 0, df_plot["ball_y_m"], FIELD_WIDTH_M - df_plot["ball_y_m"])
        df_plot = df_plot.merge(
            df_clusters[["secuencia_rival_id", "cluster_trayectoria"]],
            on="secuencia_rival_id",
            how="inner",
        )

    perfiles = []
    for cluster, gr in df_clusters.groupby("cluster_trayectoria"):
        if not df_plot.empty:
            gr_plot = df_plot[df_plot["cluster_trayectoria"] == cluster]
            x_dom = gr_plot["ball_x_norm"].median()
            y_dom = gr_plot["ball_y_norm"].median()
        else:
            x_dom = gr.get("x_fin", pd.Series([0])).median()
            y_dom = gr.get("y_fin", pd.Series([FIELD_WIDTH_M / 2])).median()
        zona_dom = _clasificar_zona_x(x_dom)
        carril_dom = _clasificar_carril_y(y_dom)
        default_zero = pd.Series(0, index=gr.index)
        prog = gr.get("progresion_x", default_zero).median()
        dur = gr.get("duracion_seg", default_zero).median()
        amp = gr.get("amplitud_balon_m", default_zero).median()
        tiro_pct = gr.get("tipo_finalizacion_tiro", default_zero).mean()
        centro_pct = gr.get("tipo_finalizacion_centro", default_zero).mean()
        perdida_pct = gr.get("tipo_finalizacion_perdida", default_zero).mean()
        if prog > 10:
            movimiento = "progresion_vertical"
        elif prog < -8:
            movimiento = "retroceso"
        else:
            movimiento = "circulacion"
        if amp > FIELD_WIDTH_M * 0.45:
            estructura = "cambio_orientacion"
        elif dur > 13:
            estructura = "posesion_larga"
        else:
            estructura = "ataque_corto"
        if tiro_pct >= 0.20:
            amenaza = "alta_amenaza"
        elif centro_pct >= 0.10:
            amenaza = "amenaza_centro"
        elif perdida_pct >= 0.65:
            amenaza = "baja_amenaza"
        else:
            amenaza = "amenaza_media"
        perfiles.append(
            {
                "cluster_trayectoria": cluster,
                "n_secuencias": len(gr),
                "x_dominante": x_dom,
                "y_dominante": y_dom,
                "zona_dominante": zona_dom,
                "carril_dominante": carril_dom,
                "progresion_mediana": prog,
                "duracion_mediana": dur,
                "amplitud_mediana": amp,
                "tiro_pct": tiro_pct,
                "centro_pct": centro_pct,
                "perdida_pct": perdida_pct,
                "estructura": estructura,
                "etiqueta_tactica": f"{zona_dom}_{carril_dom}_{movimiento}_{amenaza}",
                "etiqueta_detallada": f"{zona_dom}_{carril_dom}_{movimiento}_{estructura}_{amenaza}",
            }
        )
    return pd.DataFrame(perfiles)
