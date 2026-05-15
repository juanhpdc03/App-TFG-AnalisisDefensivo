from __future__ import annotations


def resumen_texto_basico(resumen: dict) -> str:
    """Genera un resumen textual simple a partir del diccionario del pipeline."""
    return (
        f"Partido {resumen['match_id']}: equipo propio {resumen['team_id']} vs "
        f"rival {resumen['rival_team_id']}. "
        f"Tracking original: {resumen['n_tracking_raw']:,} filas; "
        f"tracking a 250 ms: {resumen['n_tracking_250ms']:,} filas. "
        f"Eventing: {resumen['n_eventing_raw']:,} eventos. "
        f"Tiros rivales oficiales: {resumen['tiros_rival']} "
        f"({resumen['tiros_puerta_rival']} a puerta)."
    )

