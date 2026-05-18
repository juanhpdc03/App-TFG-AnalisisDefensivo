from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M
from tfg_analysis.models.global_clustering import _preparar_model_frame


OUT_DIR = ROOT / "outputs" / "memoria_4_3"
GLOBAL_DIR = ROOT / "outputs" / "global_clusters"

PALETTE = {
    0: "#d0183f",  # Tipologia 1
    1: "#2f62b3",  # Tipologia 2
    2: "#f2c94c",  # Tipologia 3
    3: "#7b8493",  # Tipologia 4, gris en lugar de verde
}
PITCH_LINE = "#111827"


def _load_global_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tray = pd.read_csv(GLOBAL_DIR / "global_trayectorias.csv")
    features = pd.read_csv(GLOBAL_DIR / "global_features_tacticas.csv")
    assignments = pd.read_csv(GLOBAL_DIR / "global_cluster_assignments.csv")
    with (GLOBAL_DIR / "global_cluster_model.pkl").open("rb") as f:
        model = pickle.load(f)
    return tray, features, assignments, model


def _trajectory_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("traj_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )


def _points_from_row(row: pd.Series, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = row[cols].astype(float).to_numpy()
    xs = np.clip(values[0::2], 0, FIELD_LENGTH_M)
    ys = np.clip(values[1::2], 0, FIELD_WIDTH_M)
    return xs, ys


def _draw_pitch(ax, facecolor: str = "white"):
    ax.set_facecolor(facecolor)
    ax.set_xlim(0, FIELD_LENGTH_M)
    ax.set_ylim(0, FIELD_WIDTH_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PITCH_LINE)
        spine.set_linewidth(0.9)

    ax.plot([0, FIELD_LENGTH_M, FIELD_LENGTH_M, 0, 0], [0, 0, FIELD_WIDTH_M, FIELD_WIDTH_M, 0], color=PITCH_LINE, lw=1.05)
    ax.plot([FIELD_LENGTH_M / 2, FIELD_LENGTH_M / 2], [0, FIELD_WIDTH_M], color=PITCH_LINE, lw=1.05)
    ax.add_patch(Circle((FIELD_LENGTH_M / 2, FIELD_WIDTH_M / 2), 9.15, fill=False, ec=PITCH_LINE, lw=1.05))
    ax.scatter([FIELD_LENGTH_M / 2], [FIELD_WIDTH_M / 2], s=8, color=PITCH_LINE, zorder=5)

    box_y = (FIELD_WIDTH_M - 40.3) / 2
    goal_y = (FIELD_WIDTH_M - 18.3) / 2
    ax.add_patch(Rectangle((0, box_y), 16.5, 40.3, fill=False, ec=PITCH_LINE, lw=1.05))
    ax.add_patch(Rectangle((0, goal_y), 5.5, 18.3, fill=False, ec=PITCH_LINE, lw=1.05))
    ax.add_patch(Rectangle((FIELD_LENGTH_M - 16.5, box_y), 16.5, 40.3, fill=False, ec=PITCH_LINE, lw=1.05))
    ax.add_patch(Rectangle((FIELD_LENGTH_M - 5.5, goal_y), 5.5, 18.3, fill=False, ec=PITCH_LINE, lw=1.05))
    ax.scatter([11, FIELD_LENGTH_M - 11], [FIELD_WIDTH_M / 2, FIELD_WIDTH_M / 2], s=8, color=PITCH_LINE, zorder=5)


def _add_global_attack_arrow(fig):
    arrow = FancyArrowPatch(
        (0.23, 0.055),
        (0.77, 0.055),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=28,
        linewidth=3.2,
        color="#111827",
    )
    fig.add_artist(arrow)
    fig.text(
        0.50,
        0.095,
        "Sentido del ataque rival",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#111827",
    )


def plot_global_trajectories(tray: pd.DataFrame, assignments: pd.DataFrame):
    df = tray.merge(assignments[["match_id", "secuencia_rival_id", "cluster_trayectoria"]], on=["match_id", "secuencia_rival_id"], how="inner")
    cols = _trajectory_columns(df)
    counts = df["cluster_trayectoria"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), facecolor="white")

    for cluster, ax in enumerate(axes):
        part = df[df["cluster_trayectoria"].eq(cluster)]
        _draw_pitch(ax, facecolor="#f8fafc")
        color = PALETTE[cluster]
        for _, row in part.iterrows():
            xs, ys = _points_from_row(row, cols)
            ax.plot(xs, ys, color=color, alpha=0.34, lw=1.25, solid_capstyle="round", zorder=3)
            ax.scatter(xs[-1], ys[-1], color=color, s=8, alpha=0.75, zorder=4)
        ax.set_title(f"Tipologia {cluster + 1}\n{int(counts.get(cluster, 0))} secuencias", fontsize=13, fontweight="bold", color="#111827", pad=8)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.86, bottom=0.18, wspace=0.035)
    _add_global_attack_arrow(fig)
    out = OUT_DIR / "figura_4_3_clusters_trayectorias_globales.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_global_heatmaps(tray: pd.DataFrame, assignments: pd.DataFrame):
    df = tray.merge(assignments[["match_id", "secuencia_rival_id", "cluster_trayectoria"]], on=["match_id", "secuencia_rival_id"], how="inner")
    cols = _trajectory_columns(df)
    counts = df["cluster_trayectoria"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.55), facecolor="white")

    for cluster, ax in enumerate(axes):
        part = df[df["cluster_trayectoria"].eq(cluster)]
        all_x, all_y = [], []
        for _, row in part.iterrows():
            xs, ys = _points_from_row(row, cols)
            all_x.extend(xs)
            all_y.extend(ys)

        _draw_pitch(ax, facecolor="white")
        if all_x:
            heat, xedges, yedges = np.histogram2d(
                all_x,
                all_y,
                bins=(58, 38),
                range=[[0, FIELD_LENGTH_M], [0, FIELD_WIDTH_M]],
            )
            heat = gaussian_filter(heat.T, sigma=1.35)
            vmax = max(np.percentile(heat[heat > 0], 97) if np.any(heat > 0) else 1, 1)
            ax.imshow(
                heat,
                extent=[0, FIELD_LENGTH_M, 0, FIELD_WIDTH_M],
                origin="lower",
                cmap="YlOrRd",
                norm=PowerNorm(gamma=0.52, vmin=0, vmax=vmax),
                alpha=0.92,
                zorder=1,
            )
            _draw_pitch(ax, facecolor="none")
        ax.set_title(f"Mapa de calor | Tipologia {cluster + 1}\n{int(counts.get(cluster, 0))} secuencias", fontsize=13, fontweight="bold", color="#111827", pad=8)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.86, bottom=0.18, wspace=0.035)
    _add_global_attack_arrow(fig)
    out = OUT_DIR / "figura_4_3_clusters_mapas_calor_globales.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _model_matrix(tray: pd.DataFrame, features: pd.DataFrame, model: dict):
    ids, X = _preparar_model_frame(tray, features)
    X = X.reindex(columns=model["feature_columns"], fill_value=0)
    X_scaled = model["scaler"].transform(X)
    labels = model["kmeans"].predict(X_scaled)
    return ids, X_scaled, labels


def plot_silhouette(tray: pd.DataFrame, features: pd.DataFrame, model: dict):
    _, X_scaled, labels = _model_matrix(tray, features, model)
    sil_values = silhouette_samples(X_scaled, labels)
    sil_avg = silhouette_score(X_scaled, labels)
    counts = pd.Series(labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10.8, 6.1), facecolor="white")
    ax.set_facecolor("#f8fafc")
    y_lower = 10
    y_ticks = []
    y_labels = []

    for cluster in sorted(np.unique(labels)):
        values = np.sort(sil_values[labels == cluster])
        size = len(values)
        y_upper = y_lower + size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            values,
            facecolor=PALETTE[int(cluster)],
            edgecolor=PALETTE[int(cluster)],
            alpha=0.92,
        )
        y_ticks.append(y_lower + size / 2)
        y_labels.append(f"T{int(cluster) + 1}\n({int(counts.get(cluster, 0))})")
        y_lower = y_upper + 14

    ax.axvline(sil_avg, color="#c8102e", linestyle="--", lw=2.4, label=f"Media = {sil_avg:.2f}")
    ax.set_title("Analisis de silueta del clustering global", fontsize=18, fontweight="bold", color="#111827", pad=14)
    ax.set_xlabel("Coeficiente de silueta", fontsize=12.5, fontweight="bold", color="#111827")
    ax.set_ylabel("Secuencias agrupadas por tipologia", fontsize=12.5, fontweight="bold", color="#111827")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="bold", color="#111827")
    ax.grid(axis="x", color="#d8dee9", lw=1, alpha=0.75)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_xlim(min(-0.05, sil_values.min() - 0.02), max(0.55, sil_values.max() + 0.04))
    ax.set_ylim(0, y_lower)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "figura_4_3_silhouette_kmeans.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_pca(tray: pd.DataFrame, features: pd.DataFrame, model: dict):
    _, X_scaled, labels = _model_matrix(tray, features, model)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    ratios = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9.8, 7.2), facecolor="white")
    ax.set_facecolor("#f8fafc")
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=58,
            color=PALETTE[int(cluster)],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.65,
            label=f"Tipologia {int(cluster) + 1} ({int(mask.sum())})",
        )
        centroid = coords[mask].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            marker="X",
            s=150,
            color=PALETTE[int(cluster)],
            edgecolor="#111827",
            linewidth=1.2,
            zorder=5,
        )

    ax.axhline(0, color="#d8dee9", lw=1)
    ax.axvline(0, color="#d8dee9", lw=1)
    ax.grid(color="#d8dee9", lw=0.8, alpha=0.65)
    ax.set_title("PCA exploratorio de las tipologias ofensivas", fontsize=18, fontweight="bold", color="#111827", pad=14)
    ax.set_xlabel(f"Componente principal 1 ({ratios[0]:.1f}% varianza)", fontsize=12.5, fontweight="bold", color="#111827")
    ax.set_ylabel(f"Componente principal 2 ({ratios[1]:.1f}% varianza)", fontsize=12.5, fontweight="bold", color="#111827")
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "figura_4_3_pca_clusters_globales.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tray, features, assignments, model = _load_global_data()
    paths = [
        plot_global_trajectories(tray, assignments),
        plot_global_heatmaps(tray, assignments),
        plot_silhouette(tray, features, model),
        plot_pca(tray, features, model),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
