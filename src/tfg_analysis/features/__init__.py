"""Features tacticas: xThreat, pitch control, IDD e IPO."""

from .attack_summary import asignar_tiros_a_secuencias, crear_resumen_secuencias
from .defensive import (
    anadir_pitch_control,
    anadir_pitch_control_simplificado,
    calcular_ddi,
    crear_perfil_defensivo_dinamico,
)
from .threat import (
    calcular_ipar,
    calcular_xt_secuencias,
    categorizar_percentiles,
    score_finalizacion_avanzado,
)

__all__ = [
    "asignar_tiros_a_secuencias",
    "crear_resumen_secuencias",
    "anadir_pitch_control",
    "anadir_pitch_control_simplificado",
    "calcular_ddi",
    "crear_perfil_defensivo_dinamico",
    "calcular_ipar",
    "calcular_xt_secuencias",
    "categorizar_percentiles",
    "score_finalizacion_avanzado",
]
