from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tfg_analysis.config import ProjectPaths


FIGURE_FILES = {
    "matriz_ddi_ipar": "matriz_ddi_ipar.png",
    "matriz_5x5": "matriz_5x5.png",
    "evolucion_temporal": "evolucion_temporal.png",
    "evolucion_temporal_tiros": "evolucion_temporal_tiros.png",
    "trayectorias_cluster": "trayectorias_cluster.png",
    "mapa_calor_notebook": "mapa_calor_clusters.png",
    "amenaza_media_cluster": "amenaza_media_cluster.png",
    "ranking_secuencias": "ranking_secuencias.png",
    "causas_danio": "causas_danio.png",
    "zona_cluster_danio": "zona_cluster_danio.png",
}

TABLE_FILES = {
    "kpis": "kpis.csv",
    "clusters_resumen": "clusters_resumen.csv",
    "riesgo_resumen": "riesgo_resumen.csv",
    "ranking_secuencias": "ranking_secuencias.csv",
    "secuencias_detalle": "secuencias_detalle.csv",
    "matriz_5x5": "matriz_5x5.csv",
    "causas_danio": "causas_danio.csv",
    "zona_cluster_danio": "zona_cluster_danio.csv",
    "danio_sin_desorden_simple": "danio_sin_desorden.csv",
    "caos_sin_castigo_simple": "caos_sin_castigo.csv",
    "trayectorias_ligeras": "trayectorias_ligeras.csv",
    "amenaza_media_cluster": "amenaza_media_cluster.csv",
}


def app_data_root(paths: ProjectPaths | None = None) -> Path:
    paths = (paths or ProjectPaths()).resolve()
    root = paths.outputs_dir / "app_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def partido_app_dir(match_id: int, paths: ProjectPaths | None = None) -> Path:
    out = app_data_root(paths) / str(int(match_id))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _initials(name: str) -> str:
    banned = {"cd", "cf", "ad", "a", "d", "de", "la", "el"}
    words = [w for w in str(name).replace(".", " ").split() if w and w.lower() not in banned]
    if not words:
        words = str(name).split()
    return "".join(w[0].upper() for w in words[:2]) or "R"


def _inferir_nombres_equipos(resultado, team_id: int) -> dict:
    propio = str(team_id)
    rival_id = resultado.resumen.get("rival_team_id")
    rival = str(rival_id) if rival_id is not None else "Rival"

    tracking = getattr(resultado, "tracking_con_secuencias", pd.DataFrame())
    if not tracking.empty and {"team_id", "team_name"}.issubset(tracking.columns):
        teams = tracking[["team_id", "team_name"]].dropna().drop_duplicates()
        mapping = dict(zip(teams["team_id"].astype(int), teams["team_name"].astype(str)))
        propio = mapping.get(int(team_id), propio)
        if rival_id is not None:
            rival = mapping.get(int(rival_id), rival)

    return {
        "team_name": propio,
        "rival_team_id": _json_safe(rival_id),
        "rival_name": rival,
        "rival_initials": _initials(rival),
    }


def generar_informe_markdown(resultado, dashboard: dict) -> str:
    resumen = resultado.resumen
    kpis = dashboard["tablas"]["kpis"].set_index("kpi")["valor"].to_dict()
    riesgo = dashboard["tablas"]["riesgo_resumen"]
    causas = dashboard["tablas"]["causas_danio"]
    zona = dashboard["tablas"]["zona_cluster_danio"]

    cluster_top = riesgo.iloc[0] if not riesgo.empty else {}
    causa_top = causas.iloc[0]["causa"] if not causas.empty else "no identificada"
    zona_top = zona.iloc[0]["zona_ataque"] if not zona.empty else "no identificada"
    ddi = float(kpis.get("IDD medio", 0) or 0)
    ipar = float(kpis.get("IPO medio", 0) or 0)

    return f"""# Informe del partido {resumen.get('match_id')}

El analisis detecta **{resumen.get('n_secuencias_rivales')} secuencias ofensivas rivales**. El rival registra **{resumen.get('tiros_rival')} tiros oficiales**, de los cuales **{resumen.get('tiros_puerta_rival')}** van a puerta. Dentro de las secuencias analizadas quedan asociados **{resumen.get('tiros_rival_asignados_secuencia')} tiros**.

## Lectura defensiva global

El **IDD medio** del partido es **{ddi:.2f}** y el **IPO medio** es **{ipar:.2f}**. La causa defensiva dominante es **{causa_top}** y la zona/carril que concentra mas dano eficiente es **{zona_top}**.

## Patron prioritario

El cluster de mayor prioridad analitica es el **cluster {cluster_top.get('cluster_trayectoria', '-')}**, con **{cluster_top.get('secuencias', '-')} secuencias**, IDD medio **{cluster_top.get('ddi_medio', 0):.2f}** e IPO medio **{cluster_top.get('ipar_medio', 0):.2f}**.

## Interpretacion

El foco del informe no es explicar todos los tiros aislados del rival, sino los tiros y acciones peligrosas que aparecen dentro de las secuencias ofensivas detectadas por la metodología. Esto permite estudiar patrones comparables de desorganización defensiva y relacionarlos con la peligrosidad real de la acción.
"""


def tabla_secuencias_detalle(resultado) -> pd.DataFrame:
    """Tabla ligera de trabajo: una fila por secuencia rival analizada.

    La app la usa como capa operativa para filtrar, ordenar y saltar a jugadas
    concretas sin recalcular el pipeline pesado.
    """
    df = getattr(resultado, "df_def_dinamico", pd.DataFrame()).copy()
    if df.empty:
        return pd.DataFrame()

    secuencias = getattr(resultado, "secuencias", pd.DataFrame()).copy()
    tracking_seq = getattr(resultado, "tracking_con_secuencias", pd.DataFrame()).copy()
    resumen = getattr(resultado, "resumen", {}) or {}
    team_id = resumen.get("team_id")
    rival_team_id = resumen.get("rival_team_id")
    cols_sec = [
        "secuencia_rival_id",
        "bepro_sequence_id",
        "origen_segmentacion",
        "period",
        "start_frame",
        "end_frame",
        "start_time_ms",
        "end_time_ms",
        "end_time_bepro_ms",
        "motivo_cierre",
        "start_time_seg",
        "end_time_seg",
        "duracion_seg",
        "ball_x_ini_m",
        "ball_y_ini_m",
        "ball_x_fin_m",
        "ball_y_fin_m",
        "ball_x_ini_norm_m",
        "ball_y_ini_norm_m",
        "ball_x_fin_norm_m",
        "ball_y_fin_norm_m",
        "progresion_x_norm_m",
        "x_max_norm_m",
        "recorrido_balon_m",
        "recorrido_bepro_m",
        "progresion_bepro_m",
        "num_pases_bepro",
        "direccion_progresion_bepro",
        "num_eventos",
        "num_eventos_rivales",
        "tiene_setpiece",
        "tipo_secuencia_ofensiva",
        "motivo_conservacion",
    ]
    if not secuencias.empty:
        df = df.merge(secuencias[[c for c in cols_sec if c in secuencias.columns]], on="secuencia_rival_id", how="left")

    if (
        not tracking_seq.empty
        and team_id is not None
        and rival_team_id is not None
        and {"secuencia_rival_id", "poseedor", "team_id"}.issubset(tracking_seq.columns)
    ):
        poss = tracking_seq.dropna(subset=["secuencia_rival_id"]).copy()
        poss["secuencia_rival_id"] = pd.to_numeric(poss["secuencia_rival_id"], errors="coerce")
        poss["poseedor"] = pd.to_numeric(poss["poseedor"], errors="coerce").fillna(0)
        poss["team_id"] = pd.to_numeric(poss["team_id"], errors="coerce")
        poss = poss[poss["poseedor"].eq(1)].copy()
        if not poss.empty:
            validacion = (
                poss.groupby("secuencia_rival_id")
                .agg(
                    frames_poseedor_total=("team_id", "size"),
                    frames_poseedor_rival=("team_id", lambda s: int(s.eq(int(rival_team_id)).sum())),
                    frames_poseedor_propio=("team_id", lambda s: int(s.eq(int(team_id)).sum())),
                )
                .reset_index()
            )
            validacion["pct_poseedor_rival"] = (
                validacion["frames_poseedor_rival"] / validacion["frames_poseedor_total"].replace(0, np.nan)
            ).fillna(0)
            validacion["validacion_rival_ok"] = (
                validacion["frames_poseedor_rival"].gt(0)
                & validacion["frames_poseedor_rival"].ge(validacion["frames_poseedor_propio"])
            )
            df = df.merge(validacion, on="secuencia_rival_id", how="left")

    if "validacion_rival_ok" not in df.columns:
        df["validacion_rival_ok"] = True
    df["validacion_rival_ok"] = df["validacion_rival_ok"].fillna(False).astype(bool)
    df["alerta_validacion"] = np.where(
        df["validacion_rival_ok"],
        "",
        "Revisar: el poseedor detectado no confirma dominio rival en la secuencia.",
    )

    if "cluster_trayectoria" in df.columns:
        df["tipologia"] = df["cluster_trayectoria"].apply(lambda v: f"T{int(v)}" if pd.notna(v) else "-")
    else:
        df["tipologia"] = "-"

    if {"period", "end_time_seg"}.issubset(df.columns):
        minuto = pd.to_numeric(df["end_time_seg"], errors="coerce") / 60
        if (df["period"].eq(1) & minuto.lt(40)).any():
            minuto = minuto.where(df["period"] != 1, minuto + 45)
        df["minuto_partido"] = minuto.round(1)
    elif "match_time_tiro_oficial" in df.columns:
        df["minuto_partido"] = (pd.to_numeric(df["match_time_tiro_oficial"], errors="coerce") / 60000).round(1)

    ddi = pd.to_numeric(df.get("indice_desorganizacion", 0), errors="coerce").fillna(0)
    ipar = pd.to_numeric(df.get("indice_peligrosidad_accion", 0), errors="coerce").fillna(0)
    xt = pd.to_numeric(df.get("xT_max", df.get("pico_amenaza_concedida", 0)), errors="coerce").fillna(0)
    df["score_critico"] = (0.45 * ipar + 0.35 * ddi + 0.20 * xt).round(3)

    cols = [
        "secuencia_rival_id",
        "bepro_sequence_id",
        "origen_segmentacion",
        "tipologia",
        "cluster_trayectoria",
        "period",
        "minuto_partido",
        "start_time_seg",
        "end_time_seg",
        "end_time_bepro_ms",
        "motivo_cierre",
        "duracion_seg",
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "score_critico",
        "xT_max",
        "xT_added",
        "pico_amenaza_concedida",
        "exposicion_amenaza_concedida",
        "control_campo_subiza_medio",
        "control_campo_rival_medio",
        "control_zona_peligrosa_subiza_medio",
        "control_zona_peligrosa_rival_medio",
        "pitch_control_rival_detras_linea",
        "perdida_control_campo_subiza",
        "incremento_control_zona_peligrosa_rival",
        "categoria_desorganizacion_auto",
        "categoria_peligrosidad_auto",
        "tipo_desorganizacion_principal",
        "tipo_desorganizacion_secundaria",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
        "es_gol",
        "xg_tiro",
        "score_finalizacion",
        "match_time_tiro_oficial",
        "distancia_tiro_seq_ms",
        "tipo_asignacion_tiro_seq",
        "x_fin",
        "y_fin",
        "ball_x_ini_m",
        "ball_y_ini_m",
        "ball_x_fin_m",
        "ball_y_fin_m",
        "ball_x_ini_norm_m",
        "ball_y_ini_norm_m",
        "ball_x_fin_norm_m",
        "ball_y_fin_norm_m",
        "progresion_x_norm_m",
        "x_max_norm_m",
        "recorrido_balon_m",
        "recorrido_bepro_m",
        "progresion_bepro_m",
        "num_pases_bepro",
        "direccion_progresion_bepro",
        "num_eventos",
        "num_eventos_rivales",
        "tiene_setpiece",
        "tipo_secuencia_ofensiva",
        "motivo_conservacion",
        "motivo_cierre",
        "frames_poseedor_rival",
        "frames_poseedor_propio",
        "pct_poseedor_rival",
        "validacion_rival_ok",
        "alerta_validacion",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    numeric_cols = [
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "xT_max",
        "xT_added",
        "pico_amenaza_concedida",
        "exposicion_amenaza_concedida",
        "xg_tiro",
        "score_finalizacion",
        "duracion_seg",
        "start_time_seg",
        "end_time_seg",
    ]
    round_map = {c: 3 for c in numeric_cols if c in out.columns}
    if round_map:
        out = out.round(round_map)
    return out.sort_values(["score_critico", "indice_peligrosidad_accion"], ascending=False)


def tabla_amenaza_media_cluster(resultado) -> pd.DataFrame:
    df = getattr(resultado, "df_def_dinamico", pd.DataFrame()).copy()
    cols = [c for c in df.columns if c.startswith("amenaza_concedida_") and str(c).split("_")[-1].isdigit()]
    if df.empty or "cluster_trayectoria" not in df.columns or not cols:
        return pd.DataFrame(columns=["cluster_trayectoria", "punto", "amenaza_media"])
    cols = sorted(cols, key=lambda c: int(str(c).split("_")[-1]))
    rows = []
    for cluster, gr in df.groupby("cluster_trayectoria"):
        serie = gr[cols].mean(axis=0).rolling(window=3, min_periods=1, center=True).mean()
        for point, value in enumerate(serie.values):
            rows.append(
                {
                    "cluster_trayectoria": cluster,
                    "punto": int(point),
                    "amenaza_media": value,
                }
            )
    return pd.DataFrame(rows)


def tabla_trayectorias_ligeras(resultado, n_points: int = 24) -> pd.DataFrame:
    tracking = getattr(resultado, "tracking_con_secuencias", pd.DataFrame()).copy()
    clusters = getattr(resultado, "df_clusters", pd.DataFrame()).copy()
    if tracking.empty or clusters.empty or "secuencia_rival_id" not in tracking.columns:
        return pd.DataFrame()
    df = tracking.dropna(subset=["secuencia_rival_id", "ball_x_m", "ball_y_m"]).copy()
    df["secuencia_rival_id"] = pd.to_numeric(df["secuencia_rival_id"], errors="coerce")
    df = df.dropna(subset=["secuencia_rival_id"])
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    cols_cluster = ["secuencia_rival_id", "cluster_trayectoria", "tipo_finalizacion_tiro"]
    df = df.merge(clusters[[c for c in cols_cluster if c in clusters.columns]], on="secuencia_rival_id", how="inner")

    rows = []
    for seq_id, gr in df.groupby("secuencia_rival_id"):
        gr = gr.sort_values("match_time").drop_duplicates("frame")
        if gr.empty:
            continue
        if len(gr) > n_points:
            idx = np.linspace(0, len(gr) - 1, n_points).round().astype(int)
            gr = gr.iloc[np.unique(idx)]
        for order, (_, row) in enumerate(gr.iterrows()):
            rows.append(
                {
                    "secuencia_rival_id": int(seq_id),
                    "tipologia": f"T{int(row['cluster_trayectoria'])}" if pd.notna(row.get("cluster_trayectoria")) else "-",
                    "cluster_trayectoria": row.get("cluster_trayectoria"),
                    "period": row.get("period"),
                    "match_time": row.get("match_time"),
                    "point_order": int(order),
                    "ball_x_m": row.get("ball_x_m"),
                    "ball_y_m": row.get("ball_y_m"),
                    "tipo_finalizacion_tiro": row.get("tipo_finalizacion_tiro", 0),
                }
            )
    return pd.DataFrame(rows)


def generar_app_data_partido(
    match_id: int,
    team_id: int = 12987,
    paths: ProjectPaths | None = None,
    force_analysis: bool = False,
) -> Path:
    import matplotlib.pyplot as plt

    from tfg_analysis.pipeline.cache import analizar_partido_cacheado
    from tfg_analysis.visualization import crear_dashboard_completo

    paths = (paths or ProjectPaths()).resolve()
    resultado, _ = analizar_partido_cacheado(match_id=match_id, team_id=team_id, paths=paths, force=force_analysis)
    dashboard = crear_dashboard_completo(resultado)
    dashboard["tablas"]["secuencias_detalle"] = tabla_secuencias_detalle(resultado)
    dashboard["tablas"]["trayectorias_ligeras"] = tabla_trayectorias_ligeras(resultado)
    dashboard["tablas"]["amenaza_media_cluster"] = tabla_amenaza_media_cluster(resultado)
    out = partido_app_dir(match_id, paths)

    metadata = {
        "match_id": int(match_id),
        "team_id": int(team_id),
        **_inferir_nombres_equipos(resultado, team_id),
        "resumen": {k: _json_safe(v) for k, v in resultado.resumen.items()},
        "figures": FIGURE_FILES,
        "tables": TABLE_FILES,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "informe.md").write_text(generar_informe_markdown(resultado, dashboard), encoding="utf-8")

    for key, filename in FIGURE_FILES.items():
        fig = dashboard["figuras"].get(key)
        if fig is None:
            continue
        fig.savefig(out / filename, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    for key, filename in TABLE_FILES.items():
        table = dashboard["tablas"].get(key)
        if isinstance(table, pd.DataFrame):
            table = table.replace([np.inf, -np.inf], np.nan)
            table.to_csv(out / filename, index=False, encoding="utf-8-sig")

    return out


def listar_app_data_disponible(paths: ProjectPaths | None = None) -> pd.DataFrame:
    root = app_data_root(paths)
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        meta = folder / "metadata.json"
        if not meta.exists():
            continue
        data = json.loads(meta.read_text(encoding="utf-8"))
        rows.append(
            {
                "match_id": int(data["match_id"]),
                "rival": data.get("rival_name", "Rival"),
                "path": str(folder),
                "modified": pd.Timestamp(meta.stat().st_mtime, unit="s"),
            }
        )
    return pd.DataFrame(rows)
