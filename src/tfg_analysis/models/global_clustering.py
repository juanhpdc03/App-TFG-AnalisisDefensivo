from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from tfg_analysis.config import ProjectPaths
from tfg_analysis.models.clustering import _evaluar_kmeans


GLOBAL_CLUSTER_DIRNAME = "global_clusters"
GLOBAL_MODEL_FILENAME = "global_cluster_model.pkl"


def global_cluster_dir(paths: ProjectPaths | None = None) -> Path:
    paths = (paths or ProjectPaths()).resolve()
    out = paths.outputs_dir / GLOBAL_CLUSTER_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def global_model_path(paths: ProjectPaths | None = None) -> Path:
    return global_cluster_dir(paths) / GLOBAL_MODEL_FILENAME


def _preparar_model_frame(
    df_trayectorias: pd.DataFrame,
    df_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_trayectorias.empty:
        return pd.DataFrame(), pd.DataFrame()

    join_cols = ["secuencia_rival_id"]
    if "match_id" in df_trayectorias.columns and df_features is not None and "match_id" in df_features.columns:
        join_cols = ["match_id", "secuencia_rival_id"]

    df_model = df_trayectorias.copy()
    if df_features is not None and not df_features.empty:
        # The feature table is the authoritative population of valid sequences.
        # Keep only trajectories that also have the tactical variables used by the
        # model so diagnostics, assignments and app counts refer to the same set.
        df_model = df_model.merge(df_features, on=join_cols, how="inner")

    id_cols = [c for c in ["match_id", "secuencia_rival_id"] if c in df_model.columns]
    ids = df_model[id_cols].copy()
    X = (
        df_model.drop(columns=id_cols, errors="ignore")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return ids, X


def evaluar_clustering_global(
    df_trayectorias: pd.DataFrame,
    df_features: pd.DataFrame | None = None,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids, X = _preparar_model_frame(df_trayectorias, df_features)
    if X.empty:
        return ids, pd.DataFrame()
    X_scaled = StandardScaler().fit_transform(X)
    diagnostics = _evaluar_kmeans(X_scaled, k_min=k_min, k_max=k_max, random_state=random_state)
    return ids, diagnostics


def ajustar_modelo_global(
    df_trayectorias: pd.DataFrame,
    df_features: pd.DataFrame | None,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[dict, pd.DataFrame]:
    ids, X = _preparar_model_frame(df_trayectorias, df_features)
    if X.empty:
        raise ValueError("No hay secuencias para ajustar el clustering global.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = min(int(n_clusters), len(X))
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    assignments = ids.copy()
    assignments["cluster_trayectoria"] = labels
    assignments["n_clusters_global"] = int(k)

    model = {
        "scope": "global",
        "n_clusters": int(k),
        "feature_columns": list(X.columns),
        "scaler": scaler,
        "kmeans": kmeans,
        "random_state": int(random_state),
    }
    return model, assignments


def guardar_modelo_global(model: dict, paths: ProjectPaths | None = None) -> Path:
    path = global_model_path(paths)
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def cargar_modelo_global(paths: ProjectPaths | None = None) -> dict | None:
    path = global_model_path(paths)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def clusterizar_con_modelo_global(
    df_trayectorias: pd.DataFrame,
    df_features: pd.DataFrame | None,
    model: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids, X = _preparar_model_frame(df_trayectorias, df_features)
    if X.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_columns = model["feature_columns"]
    X = X.reindex(columns=feature_columns, fill_value=0)
    X_scaled = model["scaler"].transform(X)
    labels = model["kmeans"].predict(X_scaled)

    asignacion = ids.copy()
    asignacion["secuencia_rival_id"] = asignacion["secuencia_rival_id"].astype(int)
    asignacion["cluster_trayectoria"] = labels
    asignacion["n_clusters_global"] = int(model["n_clusters"])
    asignacion["cluster_scope"] = "global"

    df_model = pd.concat([ids.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    df_model["cluster_trayectoria"] = labels
    return asignacion, df_model
