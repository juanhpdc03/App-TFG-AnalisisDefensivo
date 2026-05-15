from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M, ProjectPaths
from tfg_analysis.features.threat import _crear_xt_grid
from tfg_analysis.interpretability import (
    INDEX_LEVELS,
    INDEX_NAMES,
    build_interpretability_reference,
    classify_series,
    load_sequence_population,
)


def _save_reference_tables(population: pd.DataFrame, reference: dict, out_dir: Path) -> None:
    rows = []
    count_rows = []
    for metric, thresholds in reference["thresholds"].items():
        rows.append(
            {
                "metric": metric,
                "name": INDEX_NAMES.get(metric, metric),
                "n": thresholds["n"],
                "p25": thresholds["p25"],
                "p50": thresholds["p50"],
                "p75": thresholds["p75"],
            }
        )
        for label, color in INDEX_LEVELS[metric]:
            count_rows.append(
                {
                    "metric": metric,
                    "name": INDEX_NAMES.get(metric, metric),
                    "level": label,
                    "color": color,
                    "count": reference["counts"][metric].get(label, 0),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "index_quantiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(count_rows).to_csv(out_dir / "index_level_counts.csv", index=False, encoding="utf-8-sig")

    labelled = population.copy()
    for metric in INDEX_LEVELS:
        if metric in labelled.columns:
            labelled[f"{metric}_nivel"] = classify_series(metric, labelled[metric], reference["thresholds"][metric])
    labelled.to_csv(out_dir / "global_sequence_population_labelled.csv", index=False, encoding="utf-8-sig")


def _plot_histograms(population: pd.DataFrame, reference: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=160)
    configs = [
        ("indice_desorganizacion", "Distribucion global IDD", "Indice de desorganizacion defensiva"),
        ("indice_peligrosidad_accion", "Distribucion global IPO", "Indice de peligrosidad ofensiva rival"),
    ]
    level_colors = ["#2ecc71", "#f2c94c", "#f2994a", "#c8102e"]
    for ax, (metric, title, xlabel) in zip(axes, configs):
        values = pd.to_numeric(population[metric], errors="coerce").dropna()
        thresholds = reference["thresholds"][metric]
        bins = np.linspace(max(0, values.min() - 0.02), min(1, values.max() + 0.02), 24)
        ax.hist(values, bins=bins, color="#2d3a52", edgecolor="#ffffff", linewidth=0.5, alpha=0.92)
        for color, key, label in zip(level_colors[1:], ["p25", "p50", "p75"], ["P25", "P50", "P75"]):
            value = thresholds[key]
            ax.axvline(value, color=color, lw=2.2, ls="--", label=f"{label}: {value:.3f}")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Secuencias")
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Cuantiles interpretativos sobre las 273 secuencias analizadas", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "histogramas_indices.png", bbox_inches="tight")
    plt.close(fig)


def _plot_xt_reference(out_dir: Path) -> None:
    xt_grid = _crear_xt_grid()
    xt_plot = xt_grid.T
    fig, ax = plt.subplots(figsize=(10.4, 5.8), dpi=160)
    ax.set_facecolor("#0e1b2b")
    im = ax.imshow(
        xt_plot,
        extent=[0, FIELD_LENGTH_M, 0, FIELD_WIDTH_M],
        origin="lower",
        cmap="YlGnBu_r",
        vmin=0,
        vmax=max(float(np.nanmax(xt_plot)), 0.30),
        alpha=0.94,
    )

    line_color = "#f8fafc"
    ax.plot([0, FIELD_LENGTH_M, FIELD_LENGTH_M, 0, 0], [0, 0, FIELD_WIDTH_M, FIELD_WIDTH_M, 0], color=line_color, lw=2.0)
    ax.plot([FIELD_LENGTH_M / 2, FIELD_LENGTH_M / 2], [0, FIELD_WIDTH_M], color=line_color, lw=1.8)
    centre = plt.Circle((FIELD_LENGTH_M / 2, FIELD_WIDTH_M / 2), 9.15, fill=False, color=line_color, lw=1.8)
    ax.add_patch(centre)
    for x0 in [0, FIELD_LENGTH_M - 16.5]:
        ax.add_patch(plt.Rectangle((x0, FIELD_WIDTH_M / 2 - 20.16), 16.5, 40.32, fill=False, color=line_color, lw=1.8))
        ax.add_patch(plt.Rectangle((x0, FIELD_WIDTH_M / 2 - 9.16), 5.5, 18.32, fill=False, color=line_color, lw=1.8))
    y_grid = np.linspace(0, FIELD_WIDTH_M, xt_plot.shape[0])
    x_grid = np.linspace(0, FIELD_LENGTH_M, xt_plot.shape[1])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    for threshold, label, color in [(0.05, "moderada", "#f2c94c"), (0.10, "alta", "#f2994a"), (0.20, "muy peligrosa", "#c8102e")]:
        ax.contour(
            x_mesh,
            y_mesh,
            xt_plot,
            levels=[threshold],
            colors=[color],
            linewidths=1.8,
        )
        ax.text(FIELD_LENGTH_M - 18, 5 + threshold * 80, f"xT {label} > {threshold:.2f}", color=color, fontsize=9, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label("xT", rotation=90)
    ax.set_title("Referencia xThreat: que zonas empiezan a ser peligrosas", color="#ffffff", fontsize=15, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("#111827")
    fig.tight_layout()
    fig.savefig(out_dir / "xt_reference_grid.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    paths = ProjectPaths(root=ROOT).resolve()
    app_data_dir = paths.outputs_dir / "app_data"
    out_dir = app_data_dir / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    population = load_sequence_population(app_data_dir)
    if population.empty:
        raise SystemExit("No hay secuencias en outputs/app_data para construir interpretabilidad.")
    reference = build_interpretability_reference(app_data_dir)
    _save_reference_tables(population, reference, out_dir)
    _plot_histograms(population, reference, out_dir)
    _plot_xt_reference(out_dir)

    print(f"Secuencias globales: {reference['n_sequences']}")
    for metric, thresholds in reference["thresholds"].items():
        name = INDEX_NAMES.get(metric, metric)
        print(f"{name}: P25={thresholds['p25']:.3f}, P50={thresholds['p50']:.3f}, P75={thresholds['p75']:.3f}")
        for level, count in reference["counts"][metric].items():
            print(f"  {level}: {count}")
    print(f"Graficas guardadas en: {out_dir}")


if __name__ == "__main__":
    main()
