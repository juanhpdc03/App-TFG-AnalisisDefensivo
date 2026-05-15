from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.config import ProjectPaths
from tfg_analysis.io import listar_tracking_disponible
from tfg_analysis.models import ajustar_modelo_global, evaluar_clustering_global, guardar_modelo_global
from tfg_analysis.models.global_clustering import global_cluster_dir
from tfg_analysis.pipeline.cache import analizar_partido_cacheado


def _plot_diagnostics(diagnostics: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(diagnostics["k"], diagnostics["inertia"], marker="o", color="#c8102e", lw=2)
    axes[0].set_title("Metodo del codo")
    axes[0].set_xlabel("Numero de clusters (K)")
    axes[0].set_ylabel("Inercia")
    axes[0].grid(alpha=0.25)

    axes[1].plot(diagnostics["k"], diagnostics["silhouette"], marker="o", color="#15223b", lw=2)
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("Numero de clusters (K)")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _collect_global_dataset(match_ids: list[int], team_id: int, paths: ProjectPaths, force_analysis: bool):
    tray_rows = []
    feat_rows = []
    for match_id in match_ids:
        print(f"Preparando secuencias del partido {match_id}...")
        resultado, from_cache = analizar_partido_cacheado(
            match_id=match_id,
            team_id=team_id,
            paths=paths,
            force=force_analysis,
        )
        tray = resultado.trayectorias.copy()
        feat = resultado.features_tacticas.copy()
        if tray.empty:
            print(f"  Sin trayectorias validas: {match_id}")
            continue
        tray.insert(0, "match_id", int(match_id))
        if not feat.empty:
            feat.insert(0, "match_id", int(match_id))
        tray_rows.append(tray)
        feat_rows.append(feat)
        source = "cache" if from_cache else "analisis"
        print(f"  OK {len(tray)} secuencias ({source})")

    trayectorias = pd.concat(tray_rows, ignore_index=True) if tray_rows else pd.DataFrame()
    features = pd.concat(feat_rows, ignore_index=True) if feat_rows else pd.DataFrame()
    return trayectorias, features


def main():
    parser = argparse.ArgumentParser(
        description="Construye el dataset de clustering global y, si se indica K, ajusta el modelo global."
    )
    parser.add_argument("--team-id", type=int, default=12987)
    parser.add_argument("--match-id", type=int, action="append", default=None)
    parser.add_argument("--k", type=int, default=None, help="K final elegido para guardar el modelo global.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--force-analysis", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths(root=ROOT).resolve()
    if args.match_id:
        match_ids = args.match_id
    else:
        partidos = listar_tracking_disponible(paths)
        match_ids = partidos["match_id"].dropna().astype(int).tolist()

    if not match_ids:
        print("No hay partidos en tracking_partidos.")
        return

    out_dir = global_cluster_dir(paths)
    trayectorias, features = _collect_global_dataset(match_ids, args.team_id, paths, args.force_analysis)
    if trayectorias.empty:
        raise SystemExit("No se han encontrado secuencias para clusterizar.")

    trayectorias.to_csv(out_dir / "global_trayectorias.csv", index=False, encoding="utf-8-sig")
    features.to_csv(out_dir / "global_features_tacticas.csv", index=False, encoding="utf-8-sig")
    merged_preview = trayectorias[["match_id", "secuencia_rival_id"]].merge(
        features,
        on=["match_id", "secuencia_rival_id"],
        how="left",
    )
    merged_preview.to_csv(out_dir / "global_dataset_preview.csv", index=False, encoding="utf-8-sig")

    _, diagnostics = evaluar_clustering_global(
        trayectorias,
        features,
        k_min=args.k_min,
        k_max=args.k_max,
    )
    diagnostics.to_csv(out_dir / "global_cluster_diagnostics.csv", index=False, encoding="utf-8-sig")
    _plot_diagnostics(diagnostics, out_dir / "global_cluster_diagnostics.png")
    print(f"Diagnostico guardado en: {out_dir / 'global_cluster_diagnostics.png'}")
    print(diagnostics.to_string(index=False))

    if args.k is None:
        print("\nNo se ha guardado modelo global porque falta --k.")
        print(f"Cuando elijas K, ejecuta: .\\.venv\\Scripts\\python.exe scripts\\build_global_clusters.py --team-id {args.team_id} --k <K>")
        return

    model, assignments = ajustar_modelo_global(trayectorias, features, n_clusters=args.k)
    model_path = guardar_modelo_global(model, paths)
    assignments.to_csv(out_dir / "global_cluster_assignments.csv", index=False, encoding="utf-8-sig")
    print(f"Modelo global guardado en: {model_path}")
    print(f"Asignaciones guardadas en: {out_dir / 'global_cluster_assignments.csv'}")


if __name__ == "__main__":
    main()
