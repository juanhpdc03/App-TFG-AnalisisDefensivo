from __future__ import annotations

import pandas as pd


def crear_tabla_patrones(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    """Resume patrones tacticos por cluster combinando IDD, IPO y finalizacion."""
    if df_def_dinamico.empty:
        return pd.DataFrame()
    agg = {
        "secuencia_rival_id": "count",
        "indice_desorganizacion": "mean",
        "indice_peligrosidad_accion": "mean",
        "tipo_finalizacion_tiro": "sum",
        "tipo_finalizacion_tiro_puerta": "sum",
    }
    for col in ["tipo_desorganizacion_principal", "categoria_desorganizacion_auto", "categoria_peligrosidad_auto"]:
        if col in df_def_dinamico.columns:
            agg[col] = lambda s: s.mode().iloc[0] if not s.mode().empty else None
    tabla = (
        df_def_dinamico.groupby("cluster_trayectoria", as_index=False)
        .agg(agg)
        .rename(
            columns={
                "secuencia_rival_id": "n_secuencias",
                "indice_desorganizacion": "ddi_medio",
                "indice_peligrosidad_accion": "ipar_medio",
                "tipo_finalizacion_tiro": "tiros",
                "tipo_finalizacion_tiro_puerta": "tiros_puerta",
                "tipo_desorganizacion_principal": "causa_principal",
            }
        )
    )
    tabla["riesgo"] = pd.cut(
        (tabla["ddi_medio"] + tabla["ipar_medio"]) / 2,
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["bajo", "medio", "alto"],
    )
    return tabla.sort_values(["riesgo", "ipar_medio"], ascending=[False, False]).reset_index(drop=True)


def generar_informe_partido(resumen: dict, tabla_patrones: pd.DataFrame) -> str:
    """Informe textual determinista para usar antes de conectar IA generativa."""
    lineas = [
        f"Partido {resumen['match_id']}.",
        f"Se han detectado {resumen.get('n_secuencias_rivales', 0)} secuencias ofensivas rivales.",
        (
            f"El rival registra {resumen.get('tiros_rival', 0)} tiros oficiales, "
            f"{resumen.get('tiros_puerta_rival', 0)} de ellos a puerta."
        ),
    ]
    if resumen.get("tiros_rival_asignados_secuencia", 0) != resumen.get("tiros_rival", 0):
        lineas.append(
            f"{resumen.get('tiros_rival_asignados_secuencia', 0)} tiros quedan asociados "
            "a secuencias analizadas; el resto quedan fuera por criterios de continuidad."
        )
    if not tabla_patrones.empty:
        top = tabla_patrones.iloc[0]
        lineas.append(
            "El patron de mayor riesgo aparece en el cluster "
            f"{top['cluster_trayectoria']}, con IDD medio {top['ddi_medio']:.2f} "
            f"e IPO medio {top['ipar_medio']:.2f}."
        )
        if "causa_principal" in tabla_patrones.columns:
            lineas.append(f"La causa principal dominante es {top.get('causa_principal')}.")
    return "\n".join(lineas)

