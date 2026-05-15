from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tfg_analysis.config import ProjectPaths
from tfg_analysis.eventing import crear_eventos_clave, extraer_tiros_oficiales
from tfg_analysis.features import (
    anadir_pitch_control,
    calcular_ddi,
    calcular_ipar,
    calcular_xt_secuencias,
    categorizar_percentiles,
    crear_perfil_defensivo_dinamico,
    crear_resumen_secuencias,
    score_finalizacion_avanzado,
)
from tfg_analysis.io import cargar_eventing_partido, cargar_sequences_partido, cargar_tracking_partido
from tfg_analysis.models import (
    cargar_modelo_global,
    clusterizar_ataques,
    clusterizar_con_modelo_global,
    crear_features_tacticas,
    crear_taxonomia_clusters,
    preparar_trayectorias,
)
from tfg_analysis.preprocessing import (
    calcular_velocidad_balon,
    calcular_velocidad_jugadores,
    detectar_poseedor,
    recalibrar_tracking_con_eventing,
    reorganizar_tracking_250ms,
)
from tfg_analysis.reports import crear_tabla_patrones, generar_informe_partido
from tfg_analysis.sequences import detectar_secuencias_rivales


@dataclass
class MatchAnalysisResult:
    """Resultado inicial del pipeline por partido.

    Esta clase ira creciendo conforme migremos clustering, IDD, IPO y reporting
    desde el notebook a funciones.
    """

    match_id: int
    tracking_raw: pd.DataFrame
    eventing_raw: pd.DataFrame
    eventos_clave: pd.DataFrame
    tracking_250ms: pd.DataFrame
    tracking_vel: pd.DataFrame
    tracking_vel_ball: pd.DataFrame
    tracking_recalibrado: pd.DataFrame
    recalibracion_log: pd.DataFrame
    secuencias: pd.DataFrame
    tracking_con_secuencias: pd.DataFrame
    trayectorias: pd.DataFrame
    features_tacticas: pd.DataFrame
    clusters: pd.DataFrame
    df_clusters: pd.DataFrame
    taxonomia_clusters: pd.DataFrame
    df_def_dinamico: pd.DataFrame
    tabla_patrones: pd.DataFrame
    informe_texto: str
    tiros_oficiales_rival: pd.DataFrame
    resumen: dict


def inferir_equipo_propio(df_tracking: pd.DataFrame, team_id: int | None = None) -> int:
    """Devuelve el equipo propio.

    De momento conserva la convencion del notebook: si se proporciona `team_id`,
    se usa ese valor. Si no, toma el equipo con mas filas de tracking.
    """
    if team_id is not None:
        return int(team_id)
    if "team_id" not in df_tracking.columns:
        raise ValueError("No se puede inferir team_id porque falta la columna team_id.")
    return int(df_tracking["team_id"].dropna().value_counts().idxmax())


def inferir_equipo_rival(df_tracking: pd.DataFrame, team_id: int) -> int:
    teams = [int(t) for t in df_tracking["team_id"].dropna().unique() if int(t) != int(team_id)]
    if not teams:
        raise ValueError("No se ha encontrado equipo rival en el tracking.")
    return teams[0]


def analizar_partido(
    match_id: int,
    team_id: int | None = None,
    paths: ProjectPaths | None = None,
) -> MatchAnalysisResult:
    """Ejecuta el primer tramo del analisis para un partido.

    Esta funcion ya es llamable desde una futura app. Las siguientes fases del
    notebook se iran migrando a este pipeline manteniendo la misma interfaz.
    """
    paths = (paths or ProjectPaths()).resolve()

    tracking_raw = cargar_tracking_partido(match_id, paths)
    eventing_raw = cargar_eventing_partido(match_id, paths)
    try:
        sequences_raw = cargar_sequences_partido(match_id, paths)
    except FileNotFoundError:
        sequences_raw = pd.DataFrame()

    equipo_propio = inferir_equipo_propio(tracking_raw, team_id)
    equipo_rival = inferir_equipo_rival(tracking_raw, equipo_propio)

    tracking_250ms = reorganizar_tracking_250ms(tracking_raw)
    tracking_vel = calcular_velocidad_jugadores(tracking_250ms)
    tracking_vel_ball = calcular_velocidad_balon(tracking_vel)
    eventos_clave = crear_eventos_clave(eventing_raw)
    tracking_recalibrado, recalibracion_log = recalibrar_tracking_con_eventing(tracking_vel_ball, eventos_clave)
    tracking_vel_ball = detectar_poseedor(tracking_recalibrado)
    tiros_oficiales_rival = extraer_tiros_oficiales(
        eventing_raw,
        rival_team_id=equipo_rival,
    )

    secuencias, tracking_con_secuencias = detectar_secuencias_rivales(
        tracking_vel_ball,
        team_id=equipo_propio,
        rival_team_id=equipo_rival,
        df_eventos_clave=eventos_clave,
        df_sequences_bepro=sequences_raw,
    )
    trayectorias = preparar_trayectorias(tracking_con_secuencias)
    features_tacticas = crear_features_tacticas(secuencias, tracking_con_secuencias)
    modelo_global = cargar_modelo_global(paths)
    if modelo_global is not None:
        clusters, clustering_model_df = clusterizar_con_modelo_global(trayectorias, features_tacticas, modelo_global)
        cluster_scope = "global"
        n_clusters_modelo = int(modelo_global.get("n_clusters", 0) or 0)
    else:
        clusters, clustering_model_df = clusterizar_ataques(trayectorias, features_tacticas)
        cluster_scope = "partido_auto"
        n_clusters_modelo = int(clusters["cluster_trayectoria"].nunique()) if not clusters.empty else 0
    df_clusters = crear_resumen_secuencias(
        tracking_con_secuencias,
        secuencias,
        clusters,
        tiros_oficiales_rival,
    )
    if not df_clusters.empty:
        taxonomia_clusters = crear_taxonomia_clusters(df_clusters, tracking_con_secuencias)
        if not taxonomia_clusters.empty:
            df_clusters = df_clusters.drop(columns=["etiqueta_tactica", "etiqueta_detallada"], errors="ignore").merge(
                taxonomia_clusters[["cluster_trayectoria", "etiqueta_tactica", "etiqueta_detallada"]],
                on="cluster_trayectoria",
                how="left",
            )
        xt_seq = calcular_xt_secuencias(tracking_con_secuencias)
        df_def_dinamico = crear_perfil_defensivo_dinamico(
            tracking_con_secuencias,
            df_clusters[["secuencia_rival_id", "cluster_trayectoria"]],
            team_id=equipo_propio,
        )
        df_def_dinamico = anadir_pitch_control(df_def_dinamico, tracking_con_secuencias, equipo_propio)
        df_def_dinamico = calcular_ddi(df_def_dinamico)
        cols_fin = [
            "secuencia_rival_id",
            "tipo_finalizacion_tiro",
            "tipo_finalizacion_tiro_puerta",
            "tipo_finalizacion_centro",
            "tipo_finalizacion_perdida",
            "es_gol",
            "xg_tiro",
            "match_time_tiro_oficial",
            "distancia_tiro_seq_ms",
            "tipo_asignacion_tiro_seq",
            "x_fin",
            "y_fin",
        ]
        df_def_dinamico = df_def_dinamico.merge(
            df_clusters[[c for c in cols_fin if c in df_clusters.columns]],
            on="secuencia_rival_id",
            how="left",
        )
        df_def_dinamico = df_def_dinamico.merge(xt_seq, on="secuencia_rival_id", how="left")
        df_def_dinamico["score_finalizacion"] = df_def_dinamico.apply(score_finalizacion_avanzado, axis=1)
        df_def_dinamico = calcular_ipar(df_def_dinamico)
        df_def_dinamico = categorizar_percentiles(
            df_def_dinamico,
            "indice_desorganizacion",
            "categoria_desorganizacion_auto",
        )
        df_def_dinamico = categorizar_percentiles(
            df_def_dinamico,
            "indice_peligrosidad_accion",
            "categoria_peligrosidad_auto",
        )
    else:
        taxonomia_clusters = pd.DataFrame()
        df_def_dinamico = pd.DataFrame()

    resumen = {
        "match_id": int(match_id),
        "team_id": equipo_propio,
        "rival_team_id": equipo_rival,
        "n_tracking_raw": int(len(tracking_raw)),
        "n_tracking_250ms": int(len(tracking_250ms)),
        "n_eventing_raw": int(len(eventing_raw)),
        "n_secuencias_detectadas": int(len(secuencias)),
        "n_secuencias_rivales": int(len(df_clusters)),
        "n_secuencias_clusterizadas": int(len(df_clusters)),
        "n_tipologias_detectadas": int(df_clusters["cluster_trayectoria"].nunique())
        if not df_clusters.empty and "cluster_trayectoria" in df_clusters.columns
        else 0,
        "cluster_scope": cluster_scope,
        "n_clusters_modelo": n_clusters_modelo,
        "tiros_rival": int(len(tiros_oficiales_rival)),
        "tiros_puerta_rival": int(tiros_oficiales_rival["tipo_finalizacion_tiro_puerta"].sum())
        if not tiros_oficiales_rival.empty
        else 0,
        "tiros_rival_asignados_secuencia": int(df_clusters["tipo_finalizacion_tiro"].sum())
        if not df_clusters.empty and "tipo_finalizacion_tiro" in df_clusters.columns
        else 0,
    }
    tabla_patrones = crear_tabla_patrones(df_def_dinamico)
    informe_texto = generar_informe_partido(resumen, tabla_patrones)

    return MatchAnalysisResult(
        match_id=int(match_id),
        tracking_raw=tracking_raw,
        eventing_raw=eventing_raw,
        eventos_clave=eventos_clave,
        tracking_250ms=tracking_250ms,
        tracking_vel=tracking_vel,
        tracking_vel_ball=tracking_vel_ball,
        tracking_recalibrado=tracking_recalibrado,
        recalibracion_log=recalibracion_log,
        secuencias=secuencias,
        tracking_con_secuencias=tracking_con_secuencias,
        trayectorias=trayectorias,
        features_tacticas=features_tacticas,
        clusters=clusters,
        df_clusters=df_clusters,
        taxonomia_clusters=taxonomia_clusters,
        df_def_dinamico=df_def_dinamico,
        tabla_patrones=tabla_patrones,
        informe_texto=informe_texto,
        tiros_oficiales_rival=tiros_oficiales_rival,
        resumen=resumen,
    )
