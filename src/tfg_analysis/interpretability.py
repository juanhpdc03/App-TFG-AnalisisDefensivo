"""Interpretacion tactica de indices propios del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


IDD_METRIC = "indice_desorganizacion"
IPO_METRIC = "indice_peligrosidad_accion"

INDEX_NAMES = {
    IDD_METRIC: "IDD",
    IPO_METRIC: "IPO",
}

INDEX_LEVELS = {
    IDD_METRIC: [
        {
            "label": "Estable",
            "description": "La estructura mantiene orden relativo.",
            "color": "#38c172",
        },
        {
            "label": "Vulnerable",
            "description": "Aparecen desajustes, todavia sin ruptura clara.",
            "color": "#f4c542",
        },
        {
            "label": "Inestable",
            "description": "Hay perdida relevante de control defensivo.",
            "color": "#ff9f43",
        },
        {
            "label": "Crítica",
            "description": "La estructura queda muy expuesta.",
            "color": "#e4143a",
        },
    ],
    IPO_METRIC: [
        {
            "label": "Baja",
            "description": "La secuencia genera poco peligro potencial.",
            "color": "#38c172",
        },
        {
            "label": "Moderada",
            "description": "El rival progresa, pero sin alcanzar peligro alto.",
            "color": "#f4c542",
        },
        {
            "label": "Alta",
            "description": "La accion alcanza condiciones ofensivas relevantes.",
            "color": "#ff9f43",
        },
        {
            "label": "Crítica",
            "description": "La secuencia combina ventaja ofensiva y alto peligro.",
            "color": "#e4143a",
        },
    ],
}

IPO_MANUAL_THRESHOLDS = {
    "p25": 0.15,
    "p50": 0.35,
    "p75": 0.60,
    "method": "manual",
}


@dataclass(frozen=True)
class ThresholdReference:
    metric: str
    p25: float
    p50: float
    p75: float
    n: int
    method: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "p25": float(self.p25),
            "p50": float(self.p50),
            "p75": float(self.p75),
            "n": int(self.n),
            "method": self.method,
        }


def _clean_values(values: Iterable[float] | pd.Series | np.ndarray) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def metric_thresholds(metric: str, values: Iterable[float] | pd.Series | np.ndarray) -> ThresholdReference:
    """Devuelve umbrales de interpretacion para un indice.

    IDD se interpreta de forma relativa con percentiles de la muestra global.
    IPO/IPAR usa cortes manuales definidos por criterio futbolistico.
    """

    clean = _clean_values(values)
    n = int(clean.shape[0])
    if metric == IPO_METRIC:
        return ThresholdReference(
            metric=metric,
            p25=IPO_MANUAL_THRESHOLDS["p25"],
            p50=IPO_MANUAL_THRESHOLDS["p50"],
            p75=IPO_MANUAL_THRESHOLDS["p75"],
            n=n,
            method="manual",
        )

    if clean.empty:
        return ThresholdReference(metric=metric, p25=0.25, p50=0.50, p75=0.75, n=0, method="percentiles")

    p25, p50, p75 = np.nanpercentile(clean, [25, 50, 75])
    return ThresholdReference(metric=metric, p25=float(p25), p50=float(p50), p75=float(p75), n=n, method="percentiles")


def classify_index_value(
    metric: str,
    value: float | int | None,
    thresholds: ThresholdReference | dict[str, float | int | str],
) -> dict[str, object]:
    """Clasifica un valor exacto en uno de los cuatro niveles tacticos."""

    levels = INDEX_LEVELS.get(metric, INDEX_LEVELS[IDD_METRIC])
    if value is None or pd.isna(value):
        return {
            "label": "-",
            "level": 0,
            "color": "#6b7280",
            "description": "Sin dato disponible.",
            "value": None,
        }

    if isinstance(thresholds, ThresholdReference):
        reference = thresholds.as_dict()
    else:
        reference = thresholds

    val = float(value)
    p25 = float(reference.get("p25", 0.25))
    p50 = float(reference.get("p50", 0.50))
    p75 = float(reference.get("p75", 0.75))

    if val < p25:
        level = 0
    elif val < p50:
        level = 1
    elif val < p75:
        level = 2
    else:
        level = 3

    current = levels[level]
    return {
        "label": current["label"],
        "level": level + 1,
        "color": current["color"],
        "description": current["description"],
        "value": val,
    }


def load_sequence_population(app_data_dir: str | Path) -> pd.DataFrame:
    root = Path(app_data_dir)
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        for filename in ("secuencias_detalle.csv", "ranking_secuencias.csv"):
            path = folder / filename
            if path.exists():
                try:
                    rows.append(pd.read_csv(path))
                except (OSError, pd.errors.EmptyDataError):
                    pass
                break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _coerce_sequence_population(source: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if source is None:
        return pd.DataFrame()
    return load_sequence_population(source)


def build_interpretability_reference(sequences: pd.DataFrame | str | Path | None) -> dict[str, dict[str, object]]:
    """Construye referencias globales para IDD e IPO a partir de las secuencias."""

    sequences = _coerce_sequence_population(sequences)
    references: dict[str, dict[str, object]] = {}
    for metric in (IDD_METRIC, IPO_METRIC):
        values = sequences[metric] if metric in sequences.columns else pd.Series(dtype=float)
        thresholds = metric_thresholds(metric, values)
        counts = [0, 0, 0, 0]
        for raw_value in _clean_values(values):
            classified = classify_index_value(metric, raw_value, thresholds)
            counts[int(classified["level"]) - 1] += 1

        references[metric] = {
            "thresholds": thresholds.as_dict(),
            "counts": counts,
            "levels": INDEX_LEVELS[metric],
        }
    return references


def xthreat_reference_text() -> list[dict[str, str]]:
    return [
        {
            "label": "Baja",
            "range": "< 0.05",
            "color": "#38c172",
            "description": "Zonas de bajo valor esperado.",
        },
        {
            "label": "Moderada",
            "range": "0.05 - 0.10",
            "color": "#f4c542",
            "description": "Progresion con amenaza reconocible.",
        },
        {
            "label": "Alta",
            "range": "0.10 - 0.20",
            "color": "#ff9f43",
            "description": "Entrada en zonas de peligro relevante.",
        },
        {
            "label": "Muy peligrosa",
            "range": ">= 0.20",
            "color": "#e4143a",
            "description": "Cercania al area y alto potencial de ocasion.",
        },
    ]
