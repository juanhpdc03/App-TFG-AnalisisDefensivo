from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDEX_LEVELS = {
    "indice_desorganizacion": [
        ("Estable", "#2ecc71"),
        ("Vulnerable", "#f2c94c"),
        ("Inestable", "#f2994a"),
        ("Crítica", "#c8102e"),
    ],
    "indice_peligrosidad_accion": [
        ("Baja amenaza", "#2ecc71"),
        ("Amenaza moderada", "#f2c94c"),
        ("Alta amenaza", "#f2994a"),
        ("Amenaza crítica", "#c8102e"),
    ],
}

INDEX_NAMES = {
    "indice_desorganizacion": "IDD",
    "indice_peligrosidad_accion": "IPO",
}


def load_sequence_population(app_data_dir: str | Path) -> pd.DataFrame:
    root = Path(app_data_dir)
    rows = []
    for path in sorted(root.glob("*/secuencias_detalle.csv")):
        if not path.parent.name.isdigit():
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue
        df.insert(0, "match_id_app", int(path.parent.name))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quantile_thresholds(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"p25": np.nan, "p50": np.nan, "p75": np.nan, "n": 0}
    q = values.quantile([0.25, 0.50, 0.75])
    return {
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "n": int(values.shape[0]),
    }


def classify_index_value(metric: str, value, thresholds: dict[str, float]) -> dict[str, object]:
    levels = INDEX_LEVELS[metric]
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return {"label": "-", "color": "#6b7280", "value": np.nan, "level": None}
    p25 = thresholds.get("p25", np.nan)
    p50 = thresholds.get("p50", np.nan)
    p75 = thresholds.get("p75", np.nan)
    if pd.isna(p25) or pd.isna(p50) or pd.isna(p75):
        idx = 0
    elif numeric < p25:
        idx = 0
    elif numeric < p50:
        idx = 1
    elif numeric < p75:
        idx = 2
    else:
        idx = 3
    label, color = levels[idx]
    return {"label": label, "color": color, "value": float(numeric), "level": idx + 1}


def classify_series(metric: str, series: pd.Series, thresholds: dict[str, float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    labels = []
    for value in values:
        labels.append(classify_index_value(metric, value, thresholds)["label"])
    return pd.Series(labels, index=series.index)


def build_interpretability_reference(app_data_dir: str | Path) -> dict[str, object]:
    population = load_sequence_population(app_data_dir)
    thresholds = {}
    counts = {}
    for metric in INDEX_LEVELS:
        if population.empty or metric not in population.columns:
            thresholds[metric] = {"p25": np.nan, "p50": np.nan, "p75": np.nan, "n": 0}
            counts[metric] = {label: 0 for label, _ in INDEX_LEVELS[metric]}
            continue
        metric_thresholds = quantile_thresholds(population[metric])
        thresholds[metric] = metric_thresholds
        labels = classify_series(metric, population[metric], metric_thresholds)
        ordered = [label for label, _ in INDEX_LEVELS[metric]]
        counts[metric] = labels.value_counts().reindex(ordered, fill_value=0).astype(int).to_dict()
    return {"thresholds": thresholds, "counts": counts, "n_sequences": int(len(population))}


def xthreat_reference_text() -> str:
    return (
        "En la malla xThreat usada como referencia, valores cercanos a 0.00-0.05 indican amenaza baja; "
        "entre 0.05 y 0.10 amenaza moderada; a partir de 0.10 la accion ya entra en zona alta, "
        "y por encima de 0.20 se interpreta como amenaza muy peligrosa cerca del area."
    )
