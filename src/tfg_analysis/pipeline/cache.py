from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

from tfg_analysis.config import ProjectPaths
from tfg_analysis.pipeline.match_analysis import analizar_partido


def cache_dir(paths: ProjectPaths | None = None) -> Path:
    paths = (paths or ProjectPaths()).resolve()
    out = paths.outputs_dir / "cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def analysis_cache_path(match_id: int, team_id: int, paths: ProjectPaths | None = None) -> Path:
    return cache_dir(paths) / f"analysis_match_{int(match_id)}_team_{int(team_id)}.pkl"


def compactar_resultado(resultado):
    """Reduce el resultado a lo necesario para dashboard y app."""
    return SimpleNamespace(
        match_id=resultado.match_id,
        eventos_clave=resultado.eventos_clave,
        tracking_250ms=resultado.tracking_250ms,
        recalibracion_log=resultado.recalibracion_log,
        secuencias=resultado.secuencias,
        tracking_con_secuencias=resultado.tracking_con_secuencias,
        trayectorias=resultado.trayectorias,
        features_tacticas=resultado.features_tacticas,
        clusters=resultado.clusters,
        df_clusters=resultado.df_clusters,
        taxonomia_clusters=resultado.taxonomia_clusters,
        df_def_dinamico=resultado.df_def_dinamico,
        tabla_patrones=resultado.tabla_patrones,
        informe_texto=resultado.informe_texto,
        tiros_oficiales_rival=resultado.tiros_oficiales_rival,
        resumen=resultado.resumen,
    )


def guardar_resultado_cache(resultado, team_id: int, paths: ProjectPaths | None = None):
    path = analysis_cache_path(resultado.match_id, team_id, paths)
    with path.open("wb") as f:
        pickle.dump(compactar_resultado(resultado), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def cargar_resultado_cache(match_id: int, team_id: int, paths: ProjectPaths | None = None):
    path = analysis_cache_path(match_id, team_id, paths)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def analizar_partido_cacheado(
    match_id: int,
    team_id: int,
    paths: ProjectPaths | None = None,
    force: bool = False,
):
    if not force:
        cached = cargar_resultado_cache(match_id, team_id, paths)
        if cached is not None:
            return cached, True
    resultado = analizar_partido(match_id=match_id, team_id=team_id, paths=paths)
    compacto = compactar_resultado(resultado)
    path = analysis_cache_path(match_id, team_id, paths)
    with path.open("wb") as f:
        pickle.dump(compacto, f, protocol=pickle.HIGHEST_PROTOCOL)
    return compacto, False
