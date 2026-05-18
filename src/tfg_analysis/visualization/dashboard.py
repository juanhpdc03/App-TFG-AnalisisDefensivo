from __future__ import annotations

import math
from collections.abc import Mapping

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M

try:
    from mplsoccer import VerticalPitch
except ImportError:  # pragma: no cover - fallback para entornos ligeros
    VerticalPitch = None

ORDEN_CATEGORIAS = ["baja", "media", "alta", "muy_alta", "critica"]
APP_COLORS = ["#c8102e", "#2453a6", "#f2c94c", "#ead7a4", "#6b7280"]
DARK_FIG_BG = "#252d3c"
DARK_PANEL_BG = "#30384a"
TEXT_LIGHT = "#f4f6fb"
PITCH_BG = DARK_FIG_BG
PITCH_STRIPE_A = "#252d3c"
PITCH_STRIPE_B = "#252d3c"
PITCH_LINE = "#edf2fb"
PITCH_MARKER = "#f8fafc"
MOMENTUM_ALPHA = 0.88
MOMENTUM_IDD_WEIGHT = 0.40
MOMENTUM_IPO_WEIGHT = 0.60
MOMENTUM_ALERT_THRESHOLD = 0.20
MOMENTUM_STRESS_THRESHOLD = 0.32


def _patron_label(cluster) -> str:
    try:
        return f"Tipologia {int(cluster) + 1}"
    except (TypeError, ValueError):
        return "Tipologia"


def _empty_fig(title: str):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_FIG_BG)
    ax.set_axis_off()
    ax.set_title(title, color=TEXT_LIGHT)
    return fig


def _draw_pitch(ax):
    ax.set_facecolor(PITCH_BG)
    ax.set_xlim(0, FIELD_LENGTH_M)
    ax.set_ylim(0, FIELD_WIDTH_M)
    ax.set_aspect("equal")
    ax.plot([0, FIELD_LENGTH_M, FIELD_LENGTH_M, 0, 0], [0, 0, FIELD_WIDTH_M, FIELD_WIDTH_M, 0], color=PITCH_LINE, lw=1.2)
    ax.axvline(FIELD_LENGTH_M / 2, color=PITCH_LINE, lw=0.8)
    ax.add_patch(plt.Circle((FIELD_LENGTH_M / 2, FIELD_WIDTH_M / 2), 9.15, fill=False, color=PITCH_LINE, lw=0.8))
    ax.add_patch(plt.Rectangle((0, FIELD_WIDTH_M / 2 - 20.16), 16.5, 40.32, fill=False, color=PITCH_LINE, lw=0.8))
    ax.add_patch(
        plt.Rectangle((FIELD_LENGTH_M - 16.5, FIELD_WIDTH_M / 2 - 20.16), 16.5, 40.32, fill=False, color=PITCH_LINE, lw=0.8)
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_pitch_vertical(ax):
    ax.set_xlim(0, FIELD_WIDTH_M)
    ax.set_ylim(0, FIELD_LENGTH_M)
    ax.set_aspect("equal")
    ax.set_facecolor(PITCH_BG)
    ax.plot([0, FIELD_WIDTH_M, FIELD_WIDTH_M, 0, 0], [0, 0, FIELD_LENGTH_M, FIELD_LENGTH_M, 0], color=PITCH_LINE, lw=1)
    ax.axhline(FIELD_LENGTH_M / 2, color=PITCH_LINE, lw=0.8)
    ax.add_patch(plt.Circle((FIELD_WIDTH_M / 2, FIELD_LENGTH_M / 2), 9.15, fill=False, color=PITCH_LINE, lw=0.8))
    ax.add_patch(plt.Rectangle((FIELD_WIDTH_M / 2 - 20.16, 0), 40.32, 16.5, fill=False, color=PITCH_LINE, lw=0.8))
    ax.add_patch(plt.Rectangle((FIELD_WIDTH_M / 2 - 20.16, FIELD_LENGTH_M - 16.5), 40.32, 16.5, fill=False, color=PITCH_LINE, lw=0.8))
    ax.set_xticks([])
    ax.set_yticks([])


def _add_attack_direction_arrow(fig):
    arrow = patches.FancyArrowPatch(
        (0.935, 0.135),
        (0.935, 0.875),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=34,
        linewidth=5.2,
        color=TEXT_LIGHT,
        shrinkA=0,
        shrinkB=0,
        zorder=20,
    )
    fig.add_artist(arrow)
    fig.text(
        0.986,
        0.505,
        "SENTIDO DEL ATAQUE RIVAL",
        color=TEXT_LIGHT,
        fontsize=9.5,
        ha="center",
        va="center",
        rotation=90,
        weight="bold",
    )


def _normalizar_direccion_ataque(df: pd.DataFrame, df_clusters: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    finals = (
        out.sort_values("match_time")
        .groupby("secuencia_rival_id", as_index=False)
        .tail(1)[["secuencia_rival_id", "period", "ball_x_m"]]
    )
    if not df_clusters.empty:
        cols = ["secuencia_rival_id", "tipo_finalizacion_tiro"]
        finals = finals.merge(df_clusters[[c for c in cols if c in df_clusters.columns]], on="secuencia_rival_id", how="left")
    directions: dict[int, bool] = {}
    for period, gr in finals.groupby("period"):
        shots = gr[pd.to_numeric(gr.get("tipo_finalizacion_tiro", 0), errors="coerce").fillna(0).eq(1)]
        ref = shots["ball_x_m"] if len(shots) >= 1 else gr["ball_x_m"]
        directions[int(period)] = bool(pd.to_numeric(ref, errors="coerce").median() >= FIELD_LENGTH_M / 2)
    if not directions:
        directions = {0: True, 1: False}

    attack_high = out["period"].map(lambda p: directions.get(int(p), True))
    out["ball_x_norm"] = np.where(attack_high, out["ball_x_m"], FIELD_LENGTH_M - out["ball_x_m"])
    out["ball_y_norm"] = np.where(attack_high, out["ball_y_m"], FIELD_WIDTH_M - out["ball_y_m"])
    return out


def _minuto_partido(df: pd.DataFrame) -> pd.Series:
    minuto = df["end_time_seg"] / 60
    min_segunda = minuto[df["period"] == 1].min()
    if pd.notna(min_segunda) and min_segunda < 40:
        minuto = minuto.where(df["period"] != 1, minuto + 45)
    return minuto


def preparar_df_temporal(df_def_dinamico: pd.DataFrame, secuencias: pd.DataFrame) -> pd.DataFrame:
    if df_def_dinamico.empty or secuencias.empty:
        return pd.DataFrame()
    if "minuto_partido" in df_def_dinamico.columns:
        return df_def_dinamico.copy()
    cols = ["secuencia_rival_id", "period", "start_time_seg", "end_time_seg", "duracion_seg"]
    df = df_def_dinamico.merge(secuencias[[c for c in cols if c in secuencias.columns]], on="secuencia_rival_id", how="left")
    if {"period", "end_time_seg"}.issubset(df.columns):
        df["minuto_partido"] = _minuto_partido(df)
    return df


def preparar_momentum_defensivo(df_def_dinamico: pd.DataFrame, secuencias: pd.DataFrame, alpha: float = MOMENTUM_ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = preparar_df_temporal(df_def_dinamico, secuencias)
    if df.empty or "minuto_partido" not in df.columns:
        return df, pd.DataFrame(columns=["minuto", "senal", "momentum"])
    for col in ["minuto_partido", "duracion_seg", "indice_desorganizacion", "indice_peligrosidad_accion"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["minuto_partido"]).sort_values("minuto_partido").copy()
    df["duracion_seg"] = df.get("duracion_seg", pd.Series(6, index=df.index)).fillna(6).clip(lower=2)
    df["fin_min"] = df["minuto_partido"] + df["duracion_seg"] / 60
    df["senal_momentum"] = (
        MOMENTUM_IDD_WEIGHT * df.get("indice_desorganizacion", pd.Series(0, index=df.index)).fillna(0)
        + MOMENTUM_IPO_WEIGHT * df.get("indice_peligrosidad_accion", pd.Series(0, index=df.index)).fillna(0)
    ).clip(0, 1)
    max_minute = int(max(95, np.ceil(float(df["fin_min"].max()) + 2)))
    timeline = pd.DataFrame({"minuto": np.arange(0, max_minute + 1, 1)})
    df["minuto_modelo"] = np.floor(df["minuto_partido"]).astype(int).clip(0, max_minute)
    signal_by_minute = (
        df.groupby("minuto_modelo")["senal_momentum"]
        .apply(lambda values: 1 - float(np.prod(1 - pd.to_numeric(values, errors="coerce").fillna(0).clip(0, 1))))
        .clip(0, 1)
    )
    timeline["senal"] = timeline["minuto"].map(signal_by_minute).fillna(0.0)
    values = []
    acc = 0.0
    for signal in timeline["senal"]:
        acc = alpha * acc + (1 - alpha) * float(signal)
        values.append(acc)
    timeline["momentum"] = np.clip(values, 0, 1)
    return df, timeline


def tabla_kpis_partido(resumen: Mapping, df_def_dinamico: pd.DataFrame, tabla_patrones: pd.DataFrame | None = None) -> pd.DataFrame:
    zona = "-"
    carril = "-"
    causa = "-"
    cluster_riesgo = "-"
    if tabla_patrones is not None and not tabla_patrones.empty:
        top = tabla_patrones.iloc[0]
        cluster_riesgo = top.get("cluster_trayectoria", "-")
        causa = top.get("causa_principal", "-")
    if not df_def_dinamico.empty:
        if "y_fin" in df_def_dinamico.columns:
            zona_tab = tabla_zona_cluster_danio(df_def_dinamico)
            if not zona_tab.empty:
                zona = zona_tab.iloc[0]["zona_ataque"]
                cluster_riesgo = f"C{zona_tab.iloc[0]['cluster_trayectoria']}"
        if "tipo_desorganizacion_principal" in df_def_dinamico.columns:
            modo = df_def_dinamico["tipo_desorganizacion_principal"].mode()
            if not modo.empty:
                causa = modo.iloc[0]
        if "cluster_trayectoria" in df_def_dinamico.columns:
            carril = f"C{df_def_dinamico['cluster_trayectoria'].value_counts().idxmax()}"
    valores = {
        "Secuencias rivales": resumen.get("n_secuencias_rivales", len(df_def_dinamico)),
        "Tiros Bepro rival": resumen.get("tiros_rival", 0),
        "Tiros en secuencias": resumen.get("tiros_rival_asignados_secuencia", 0),
        "Tiros a puerta rival": resumen.get("tiros_puerta_rival", 0),
        "IDD medio": df_def_dinamico.get("indice_desorganizacion", pd.Series(dtype=float)).mean(),
        "IPO medio": df_def_dinamico.get("indice_peligrosidad_accion", pd.Series(dtype=float)).mean(),
        "Cluster mas repetido": carril,
        "Zona/carril mas danino": zona,
        "Foco defensivo": causa,
        "Cluster prioritario": cluster_riesgo,
    }
    return pd.DataFrame([{"kpi": k, "valor": v} for k, v in valores.items()])


def plot_kpis_partido(resumen: Mapping, df_def_dinamico: pd.DataFrame, tabla_patrones: pd.DataFrame | None = None):
    kpis = tabla_kpis_partido(resumen, df_def_dinamico, tabla_patrones)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_axis_off()
    labels = []
    for _, row in kpis.iterrows():
        val = row["valor"]
        if isinstance(val, float):
            val = f"{val:.2f}" if not math.isnan(val) else "-"
        labels.append(f"{row['kpi']}\n{val}")
    xs = np.linspace(0.05, 0.95, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.75"))
    ax.set_title("Resumen del partido", fontsize=14, pad=12)
    return fig


def plot_matriz_ddi_ipar(df_def_dinamico: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    if df_def_dinamico.empty:
        ax.set_title("Sin datos IDD/IPO")
        return fig
    scatter = ax.scatter(
        df_def_dinamico["indice_peligrosidad_accion"],
        df_def_dinamico["indice_desorganizacion"],
        c=df_def_dinamico.get("cluster_trayectoria", 0),
        cmap="tab10",
        s=70,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.85,
    )
    ax.axhline(df_def_dinamico["indice_desorganizacion"].median(), color="0.4", linestyle="--")
    ax.axvline(df_def_dinamico["indice_peligrosidad_accion"].median(), color="0.4", linestyle="--")
    ax.set_xlabel("IPO")
    ax.set_ylabel("IDD")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Matriz IDD vs IPO")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    return fig


def tabla_matriz_5x5(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    if df_def_dinamico.empty:
        return pd.DataFrame(0, index=ORDEN_CATEGORIAS, columns=ORDEN_CATEGORIAS)
    return pd.crosstab(
        df_def_dinamico["categoria_desorganizacion_auto"],
        df_def_dinamico["categoria_peligrosidad_auto"],
    ).reindex(index=ORDEN_CATEGORIAS, columns=ORDEN_CATEGORIAS, fill_value=0)


def plot_matriz_5x5_ddi_ipar(df_def_dinamico: pd.DataFrame):
    tabla = tabla_matriz_5x5(df_def_dinamico)
    fig, ax = plt.subplots(figsize=(7, 6))
    data = tabla.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="YlOrRd")
    total = data.sum()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pct = 0 if total == 0 else data[i, j] / total * 100
            ax.text(j, i, f"{int(data[i, j])}\n{pct:.1f}%", ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks(range(len(ORDEN_CATEGORIAS)), ORDEN_CATEGORIAS, rotation=35, ha="right")
    ax.set_yticks(range(len(ORDEN_CATEGORIAS)), ORDEN_CATEGORIAS)
    ax.set_xlabel("Peligrosidad IPO")
    ax.set_ylabel("Desorganizacion IDD")
    ax.set_title("Matriz tactica 5x5 IDD vs IPO")
    fig.colorbar(im, ax=ax, label="Secuencias")
    fig.tight_layout()
    return fig


def plot_evolucion_temporal(df_def_dinamico: pd.DataFrame, secuencias: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    df = preparar_df_temporal(df_def_dinamico, secuencias)
    if df.empty or "minuto_partido" not in df.columns:
        ax.set_title("Sin datos temporales")
        return fig
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    centers = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 95]
    df["tramo"] = pd.cut(df["minuto_partido"], bins=bins, labels=labels, include_lowest=True)
    line = df.groupby("tramo", observed=False).agg(
        ddi=("indice_desorganizacion", "mean"),
        ipar=("indice_peligrosidad_accion", "mean"),
    )
    ax.plot(centers[: len(line)], line["ddi"], marker="o", label="IDD medio", color="darkorange")
    ax.plot(centers[: len(line)], line["ipar"], marker="o", label="IPO medio", color="steelblue")
    ax.axvline(45, color="0.4", linestyle="--", lw=1)
    ax.text(45.5, 0.96, "Descanso", color="0.35", fontsize=9)
    ax.set_xlabel("Minuto de partido")
    ax.set_ylabel("Indice")
    ax.set_ylim(0, 1)
    ax.set_title("Evolucion temporal IDD/IPO")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def plot_evolucion_temporal_con_tiros(df_def_dinamico: pd.DataFrame, secuencias: pd.DataFrame):
    df, timeline = preparar_momentum_defensivo(df_def_dinamico, secuencias)
    if timeline.empty:
        return _empty_fig("Sin datos temporales")
    fig, ax = plt.subplots(figsize=(12, 5.8), facecolor=DARK_FIG_BG)
    ax.set_facecolor(DARK_PANEL_BG)
    ymax = max(0.46, min(1.0, float(timeline["momentum"].max()) + 0.08))
    ax.axhspan(0, MOMENTUM_ALERT_THRESHOLD, color="#2ea05a", alpha=0.14, lw=0)
    ax.axhspan(MOMENTUM_ALERT_THRESHOLD, MOMENTUM_STRESS_THRESHOLD, color="#f2c94c", alpha=0.15, lw=0)
    ax.axhspan(MOMENTUM_STRESS_THRESHOLD, 1, color="#c8102e", alpha=0.13, lw=0)
    ax.axhline(MOMENTUM_ALERT_THRESHOLD, color="#f2c94c", ls="--", lw=1.2, alpha=0.75)
    ax.axhline(MOMENTUM_STRESS_THRESHOLD, color="#c8102e", ls="--", lw=1.2, alpha=0.8)
    ax.fill_between(timeline["minuto"], timeline["momentum"], 0, color="#f4f6fb", alpha=0.08)
    ax.plot(timeline["minuto"], timeline["momentum"], color="#f4f6fb", lw=3.2, label="Momentum defensivo")
    if not df.empty:
        def _event_flag(column: str) -> pd.Series:
            return pd.to_numeric(df.get(column, pd.Series(0, index=df.index)), errors="coerce").fillna(0).gt(0)

        def _event_minutes(events: pd.DataFrame) -> np.ndarray:
            official = events.get("match_time_tiro_oficial", pd.Series(np.nan, index=events.index))
            return np.where(official.notna(), official / 60000, events["minuto_partido"])

        def _draw_ball_events(events: pd.DataFrame, ring_color: str, size: int, y_offset: float):
            if events.empty:
                return
            minutes = _event_minutes(events)
            y_vals = np.interp(minutes, timeline["minuto"], timeline["momentum"]) + y_offset
            ax.scatter(
                minutes,
                y_vals,
                s=size,
                marker="o",
                facecolors="#f8fafc",
                edgecolors=ring_color,
                linewidths=3.0,
                zorder=5,
            )
            ax.scatter(
                minutes,
                y_vals,
                s=size * 0.20,
                marker="p",
                facecolors="#111827",
                edgecolors="#111827",
                linewidths=0.4,
                zorder=6,
            )
            ax.scatter(
                minutes,
                y_vals,
                s=size * 0.055,
                marker="o",
                facecolors="#f8fafc",
                edgecolors="#f8fafc",
                linewidths=0.2,
                zorder=7,
            )

        goal_mask = _event_flag("es_gol")
        shot_on_target_mask = _event_flag("tipo_finalizacion_tiro_puerta") & ~goal_mask
        shot_mask = _event_flag("tipo_finalizacion_tiro") & ~shot_on_target_mask & ~goal_mask
        _draw_ball_events(df[shot_mask].copy(), "#2ea05a", 135, 0.018)
        _draw_ball_events(df[shot_on_target_mask].copy(), "#f2c94c", 160, 0.030)
        _draw_ball_events(df[goal_mask].copy(), "#c8102e", 190, 0.042)
    ax.text(1, MOMENTUM_ALERT_THRESHOLD / 2, "CONTROLADO", color="#8ee1ad", fontsize=10, weight="bold", va="center")
    ax.text(1, (MOMENTUM_ALERT_THRESHOLD + MOMENTUM_STRESS_THRESHOLD) / 2, "ALERTA", color="#f2c94c", fontsize=10, weight="bold", va="center")
    ax.text(1, min(ymax - 0.02, MOMENTUM_STRESS_THRESHOLD + 0.04), "ESTRES ALTO", color="#ff8a9b", fontsize=10, weight="bold", va="center")
    ax.axvline(45, color="white", linestyle=":", lw=1.2, alpha=0.45)
    ax.text(46, ymax * 0.95, "Descanso", color="#d8deea", fontsize=9)
    ax.set_xlim(0, int(timeline["minuto"].max()))
    ax.set_ylim(0, ymax)
    ax.set_title("Momentum defensivo acumulado (EWMA)", color=TEXT_LIGHT, fontsize=17, weight="bold", pad=14)
    ax.set_xlabel("Minuto de partido", color=TEXT_LIGHT)
    ax.set_ylabel("Presion acumulada", color=TEXT_LIGHT)
    ax.tick_params(colors="#d6deed")
    ax.grid(color="white", alpha=0.10)
    for spine in ax.spines.values():
        spine.set_color("#5f6b7d")
    legend_handles = [
        Line2D([0], [0], color="#f4f6fb", lw=3.2, label="Momentum defensivo"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#f8fafc", markeredgecolor="#2ea05a", markeredgewidth=3, markersize=11, label="Tiro"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#f8fafc", markeredgecolor="#f2c94c", markeredgewidth=3, markersize=12, label="Tiro a puerta"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#f8fafc", markeredgecolor="#c8102e", markeredgewidth=3, markersize=13, label="Gol"),
    ]
    legend = ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    legend.get_frame().set_facecolor("#1b2433")
    legend.get_frame().set_edgecolor("#5f6b7d")
    for text in legend.get_texts():
        text.set_color(TEXT_LIGHT)
    fig.tight_layout()
    return fig


def plot_trayectorias_por_cluster(df_final_con_secuencias: pd.DataFrame, df_clusters: pd.DataFrame, max_cols: int | None = None):
    if df_final_con_secuencias.empty or df_clusters.empty:
        return _empty_fig("Sin trayectorias")
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    df = df.merge(df_clusters[["secuencia_rival_id", "cluster_trayectoria"]], on="secuencia_rival_id", how="inner")
    df = _normalizar_direccion_ataque(df, df_clusters)
    clusters = sorted(df["cluster_trayectoria"].dropna().unique())
    ncols = max(1, len(clusters)) if max_cols is None else min(max_cols, max(1, len(clusters)))
    clusters = clusters[:ncols]

    if VerticalPitch is not None:
        fig, axes = plt.subplots(1, ncols, figsize=(2.55 * ncols + 1.15, 4.35), squeeze=False, facecolor=DARK_FIG_BG)
        pitch = VerticalPitch(
            pitch_type="custom",
            pitch_length=FIELD_LENGTH_M,
            pitch_width=FIELD_WIDTH_M,
            pitch_color=PITCH_BG,
            line_color=PITCH_LINE,
            linewidth=1.15,
        )
        colors = APP_COLORS
        for ax in axes.ravel():
            ax.set_axis_off()
        for pos, cluster in enumerate(clusters):
            ax = axes.ravel()[pos]
            ax.set_axis_on()
            pitch.draw(ax=ax)
            ax.set_facecolor(PITCH_BG)
            for j in range(10):
                y0 = j * FIELD_LENGTH_M / 10
                ax.add_patch(
                    patches.Rectangle(
                        (0, y0),
                        FIELD_WIDTH_M,
                        FIELD_LENGTH_M / 10,
                        facecolor=PITCH_STRIPE_A if j % 2 == 0 else PITCH_STRIPE_B,
                        edgecolor="none",
                        zorder=0,
                    )
                )
            pitch.draw(ax=ax)
            gr_c = df[df["cluster_trayectoria"] == cluster]
            color = colors[int(cluster) % len(colors)]
            for _, gr_seq in gr_c.groupby("secuencia_rival_id"):
                gr_seq = gr_seq.sort_values("match_time").drop_duplicates("frame")
                pitch.plot(
                    gr_seq["ball_x_norm"],
                    gr_seq["ball_y_norm"],
                    ax=ax,
                    color=color,
                    alpha=0.74,
                    lw=2.15,
                    zorder=2,
                )
            media = gr_c.sort_values("match_time").groupby("secuencia_rival_id").head(1)
            pitch.scatter(
                media["ball_x_norm"],
                media["ball_y_norm"],
                ax=ax,
                s=18,
                color=PITCH_MARKER,
                edgecolor="#0b1020",
                linewidth=0.55,
                alpha=0.92,
                zorder=3,
            )
            pitch.draw(ax=ax)
            ax.set_title(f"{_patron_label(cluster)} ({gr_c['secuencia_rival_id'].nunique()} sec.)", fontsize=8.5, color=TEXT_LIGHT)
        fig.subplots_adjust(left=0.01, right=0.885, top=0.90, bottom=0.05, wspace=0.05)
        _add_attack_direction_arrow(fig)
        return fig

    fig, axes = plt.subplots(1, ncols, figsize=(2.35 * ncols + 1.15, 4.35), squeeze=False, facecolor=DARK_FIG_BG)
    colors = APP_COLORS
    for ax in axes.ravel():
        ax.set_axis_off()
    for pos, cluster in enumerate(clusters):
        ax = axes.ravel()[pos]
        ax.set_axis_on()
        _draw_pitch_vertical(ax)
        gr_c = df[df["cluster_trayectoria"] == cluster]
        for _, gr_seq in gr_c.groupby("secuencia_rival_id"):
            gr_seq = gr_seq.sort_values("match_time").drop_duplicates("frame")
            ax.plot(gr_seq["ball_y_norm"], gr_seq["ball_x_norm"], color=colors[int(cluster) % len(colors)], alpha=0.72, lw=2.05)
        media = (
            gr_c.sort_values("match_time")
            .groupby("secuencia_rival_id")
            .head(1)
        )
        ax.scatter(media["ball_y_norm"], media["ball_x_norm"], s=18, color=PITCH_MARKER, edgecolors="#0b1020", linewidths=0.55, alpha=0.92)
        ax.set_title(f"{_patron_label(cluster)} ({gr_c['secuencia_rival_id'].nunique()} sec.)", fontsize=8.5, color=TEXT_LIGHT)
    fig.subplots_adjust(left=0.01, right=0.885, top=0.90, bottom=0.05, wspace=0.05)
    _add_attack_direction_arrow(fig)
    return fig


def plot_heatmap_territorial_clusters(df_final_con_secuencias: pd.DataFrame, df_clusters: pd.DataFrame, bins: tuple[int, int] = (24, 16)):
    if df_final_con_secuencias.empty or df_clusters.empty:
        return _empty_fig("Sin mapa territorial")
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    df = df.merge(df_clusters[["secuencia_rival_id", "cluster_trayectoria"]], on="secuencia_rival_id", how="inner")
    df = _normalizar_direccion_ataque(df, df_clusters)
    clusters = sorted(df["cluster_trayectoria"].dropna().unique())
    ncols = min(3, max(1, len(clusters)))
    nrows = math.ceil(len(clusters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for pos, cluster in enumerate(clusters):
        ax = axes.ravel()[pos]
        ax.set_axis_on()
        _draw_pitch(ax)
        gr = df[df["cluster_trayectoria"] == cluster]
        ax.hist2d(gr["ball_x_norm"], gr["ball_y_norm"], bins=bins, range=[[0, FIELD_LENGTH_M], [0, FIELD_WIDTH_M]], cmap="YlOrRd", alpha=0.82)
        ax.set_title(f"Mapa de calor | {_patron_label(cluster)}")
    fig.tight_layout()
    return fig


def plot_mapa_calor_clusters_notebook(df_final_con_secuencias: pd.DataFrame, df_clusters: pd.DataFrame):
    """Mapa de calor vertical por cluster, en estilo similar al notebook original."""
    if df_final_con_secuencias.empty or df_clusters.empty:
        return _empty_fig("Sin mapas de calor")
    df = df_final_con_secuencias.dropna(subset=["secuencia_rival_id"]).copy()
    df["secuencia_rival_id"] = df["secuencia_rival_id"].astype(int)
    df = df.merge(df_clusters[["secuencia_rival_id", "cluster_trayectoria"]], on="secuencia_rival_id", how="inner")
    df = _normalizar_direccion_ataque(df, df_clusters)
    clusters = sorted(df["cluster_trayectoria"].dropna().unique())

    if VerticalPitch is not None:
        fig, axes = plt.subplots(1, len(clusters), figsize=(2.55 * len(clusters) + 0.35, 4.35), squeeze=False, facecolor=DARK_FIG_BG)
        pitch = VerticalPitch(
            pitch_type="custom",
            pitch_length=FIELD_LENGTH_M,
            pitch_width=FIELD_WIDTH_M,
            pitch_color=PITCH_BG,
            line_color=PITCH_LINE,
            linewidth=1.15,
        )
        for ax, cluster in zip(axes.ravel(), clusters):
            pitch.draw(ax=ax)
            ax.set_facecolor(PITCH_BG)

            for j in range(10):
                y0 = j * FIELD_LENGTH_M / 10
                rect = patches.Rectangle(
                    (0, y0),
                    FIELD_WIDTH_M,
                    FIELD_LENGTH_M / 10,
                    facecolor=PITCH_STRIPE_A if j % 2 == 0 else PITCH_STRIPE_B,
                    edgecolor="none",
                    zorder=0,
                )
                ax.add_patch(rect)

            pitch.draw(ax=ax)
            gr = df[df["cluster_trayectoria"] == cluster].dropna(subset=["ball_x_norm", "ball_y_norm"])
            gr = gr[
                gr["ball_x_norm"].between(0, FIELD_LENGTH_M)
                & gr["ball_y_norm"].between(0, FIELD_WIDTH_M)
            ]
            if len(gr) > 1800:
                gr = gr.sample(1800, random_state=42)
            n_collections_before = len(ax.collections)
            if len(gr) >= 3:
                try:
                    pitch.kdeplot(
                        gr["ball_x_norm"],
                        gr["ball_y_norm"],
                        ax=ax,
                        fill=True,
                        levels=45,
                        thresh=0.50,
                        alpha=0.95,
                        cmap="YlOrRd",
                        cut=0,
                        zorder=2,
                    )
                    clip_rect = patches.Rectangle((0, 0), FIELD_WIDTH_M, FIELD_LENGTH_M, transform=ax.transData)
                    for coll in ax.collections[n_collections_before:]:
                        coll.set_clip_path(clip_rect)
                except Exception:
                    pitch.scatter(
                        gr["ball_x_norm"],
                        gr["ball_y_norm"],
                        ax=ax,
                        s=10,
                        c="#d90429",
                        alpha=0.6,
                        zorder=2,
                    )
            pitch.draw(ax=ax)
            ax.set_title(f"Mapa de calor | {_patron_label(cluster)}", fontsize=8.5, color=TEXT_LIGHT)
        fig.subplots_adjust(left=0.01, right=0.93, top=0.92, bottom=0.05, wspace=0.03)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.945, 0.17, 0.014, 0.66])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Densidad", rotation=90, fontsize=8, color=TEXT_LIGHT)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Baja", "Alta"])
        cbar.ax.tick_params(labelsize=7, colors=TEXT_LIGHT)
        cbar.outline.set_edgecolor((1, 1, 1, 0.35))
        return fig

    fig, axes = plt.subplots(1, len(clusters), figsize=(2.35 * len(clusters) + 0.35, 4.35), squeeze=False, facecolor=DARK_FIG_BG)
    for ax, cluster in zip(axes.ravel(), clusters):
        _draw_pitch_vertical(ax)
        gr = df[df["cluster_trayectoria"] == cluster].dropna(subset=["ball_x_norm", "ball_y_norm"])
        if not gr.empty:
            heat, xedges, yedges = np.histogram2d(
                gr["ball_y_norm"],
                gr["ball_x_norm"],
                bins=[72, 112],
                range=[[0, FIELD_WIDTH_M], [0, FIELD_LENGTH_M]],
            )
            heat = gaussian_filter(heat, sigma=3.0)
            masked = np.ma.masked_where(heat.T <= np.percentile(heat[heat > 0], 35) if (heat > 0).any() else heat.T <= 0, heat.T)
            ax.imshow(
                masked,
                extent=[0, FIELD_WIDTH_M, 0, FIELD_LENGTH_M],
                origin="lower",
                cmap="YlOrRd",
                alpha=0.88,
                aspect="auto",
                interpolation="bilinear",
            )
        ax.set_title(f"Mapa de calor | {_patron_label(cluster)}", fontsize=8.5, color=TEXT_LIGHT)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.92, bottom=0.05, wspace=0.03)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.945, 0.17, 0.014, 0.66])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Densidad", rotation=90, fontsize=8, color=TEXT_LIGHT)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Baja", "Alta"])
    cbar.ax.tick_params(labelsize=7, colors=TEXT_LIGHT)
    return fig


def plot_amenaza_media_por_cluster(df_def_dinamico: pd.DataFrame):
    cols = [c for c in df_def_dinamico.columns if c.startswith("amenaza_concedida_") and c.split("_")[-1].isdigit()]
    if df_def_dinamico.empty or not cols:
        return _empty_fig("Sin amenaza concedida")
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_FIG_BG)
    ax.set_facecolor(DARK_PANEL_BG)
    x_max = len(cols) - 1
    inicio_fin = x_max * 0.33
    progresion_fin = x_max * 0.67
    for cluster, gr in df_def_dinamico.groupby("cluster_trayectoria"):
        serie = gr[cols].mean(axis=0).rolling(window=3, min_periods=1, center=True).mean()
        color = APP_COLORS[int(cluster) % len(APP_COLORS)] if pd.notna(cluster) else APP_COLORS[0]
        ax.plot(range(len(cols)), serie.values, marker="o", lw=2.4, color=color, label=_patron_label(cluster))
    third_entry_x = progresion_fin
    y_min, y_max = ax.get_ylim()
    if "tipo_finalizacion_tiro" in df_def_dinamico.columns and pd.to_numeric(df_def_dinamico["tipo_finalizacion_tiro"], errors="coerce").fillna(0).sum() > 0:
        shot_x = x_max * 0.88
    else:
        shot_x = x_max * 0.88
    gradient_width = 260
    red = np.array(mcolors.to_rgba("#c8102e"))
    gradient = np.ones((2, gradient_width, 4))
    for i in range(gradient_width):
        frac = i / max(gradient_width - 1, 1)
        alpha = 0.10 + 0.24 * frac
        gradient[:, i, :3] = red[:3]
        gradient[:, i, 3] = alpha
    ax.imshow(
        gradient,
        extent=[third_entry_x, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )
    ax.axvspan(shot_x, x_max, color="#c8102e", alpha=0.16, zorder=0)
    ax.axvline(inicio_fin, color="#ffffff", ls="--", lw=1.4, alpha=0.80, zorder=1)
    ax.axvline(progresion_fin, color="#ffffff", ls="--", lw=1.4, alpha=0.80, zorder=1)
    ax.axvline(third_entry_x, color="#ff6b7f", ls="-", lw=4.2, alpha=0.82, zorder=2)
    ax.axvline(third_entry_x, color="#ffffff", ls="--", lw=1.1, alpha=0.90, zorder=3)
    ax.text(
        third_entry_x + 0.22,
        y_max - (y_max - y_min) * 0.08,
        "ENTRADA ULTIMO TERCIO",
        fontsize=9,
        color="#ffffff",
        weight="bold",
        va="top",
        bbox=dict(facecolor="#a90f28", edgecolor="#ffffff", linewidth=0.8, alpha=0.94, pad=2.5),
    )
    ax.axvline(shot_x, color="#c8102e", ls="-", lw=5.4, alpha=1.0, zorder=2)
    ax.axvline(shot_x, color="#ffffff", ls="--", lw=1.1, alpha=0.90, zorder=3)
    ax.text(
        shot_x + 0.18,
        y_max - (y_max - y_min) * 0.20,
        "ZONA MEDIA DE TIRO",
        fontsize=9,
        color="#ffffff",
        weight="bold",
        va="top",
        bbox=dict(facecolor="#c8102e", edgecolor="#ffffff", linewidth=0.8, alpha=0.98, pad=2.5),
    )
    phase_y = y_min + (y_max - y_min) * 0.13
    phase_box = dict(facecolor=DARK_PANEL_BG, edgecolor="none", alpha=0.70, pad=1.2)
    ax.text(inicio_fin / 2, phase_y, "INICIO", fontsize=9, weight="bold", color="#ffffff", ha="center", va="bottom", bbox=phase_box)
    ax.text((inicio_fin + progresion_fin) / 2, phase_y, "PROGRESION", fontsize=9, weight="bold", color="#ffffff", ha="center", va="bottom", bbox=phase_box)
    ax.text((progresion_fin + x_max) / 2, phase_y, "FINALIZACION", fontsize=9, weight="bold", color="#ffffff", ha="center", va="bottom", bbox=phase_box)
    ax.set_title("Evolucion media de la amenaza concedida por tipologia", color=TEXT_LIGHT)
    ax.set_xlabel("Momento normalizado de la secuencia", color=TEXT_LIGHT)
    ax.set_ylabel("Amenaza concedida xT", color=TEXT_LIGHT)
    ax.tick_params(colors=TEXT_LIGHT)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.45))
    ax.grid(alpha=0.18, color="white")
    legend = ax.legend(facecolor=DARK_PANEL_BG, edgecolor=(1, 1, 1, 0.28))
    for text in legend.get_texts():
        text.set_color(TEXT_LIGHT)
    fig.tight_layout()
    return fig


def tabla_resumen_clusters(df_clusters: pd.DataFrame, taxonomia_clusters: pd.DataFrame | None = None) -> pd.DataFrame:
    if df_clusters.empty:
        return pd.DataFrame()
    tabla = (
        df_clusters.groupby("cluster_trayectoria", as_index=False)
        .agg(
            secuencias=("secuencia_rival_id", "count"),
            tiros=("tipo_finalizacion_tiro", "sum"),
            tiros_puerta=("tipo_finalizacion_tiro_puerta", "sum"),
            duracion_media=("duracion_seg", "mean"),
            ipar_proxy=("tipo_finalizacion_tiro", "mean"),
        )
    )
    tabla["porcentaje"] = (tabla["secuencias"] / tabla["secuencias"].sum() * 100).round(1)
    tabla["duracion_media"] = tabla["duracion_media"].round(1)
    tabla["tiro_pct"] = (tabla["ipar_proxy"] * 100).round(1)
    tabla = tabla.drop(columns=["ipar_proxy"])
    if taxonomia_clusters is not None and not taxonomia_clusters.empty:
        cols = ["cluster_trayectoria", "zona_dominante", "carril_dominante", "etiqueta_tactica"]
        tabla = tabla.merge(taxonomia_clusters[[c for c in cols if c in taxonomia_clusters.columns]], on="cluster_trayectoria", how="left")
    return tabla


def tabla_resumen_riesgo(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    if df_def_dinamico.empty:
        return pd.DataFrame()
    return (
        df_def_dinamico.groupby("cluster_trayectoria", as_index=False)
        .agg(
            secuencias=("secuencia_rival_id", "count"),
            ddi_medio=("indice_desorganizacion", "mean"),
            ipar_medio=("indice_peligrosidad_accion", "mean"),
            tiros=("tipo_finalizacion_tiro", "sum"),
            tiros_puerta=("tipo_finalizacion_tiro_puerta", "sum"),
        )
        .round({"ddi_medio": 3, "ipar_medio": 3})
        .sort_values("ipar_medio", ascending=False)
    )


def tabla_extremos_simple(df_def_dinamico: pd.DataFrame, key: str, n: int = 8) -> pd.DataFrame:
    extremos = tabla_perfiles_extremos(df_def_dinamico).get(key, pd.DataFrame())
    cols = [
        "secuencia_rival_id",
        "cluster_trayectoria",
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "tipo_desorganizacion_principal",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
    ]
    if extremos.empty:
        return pd.DataFrame(columns=cols)
    return extremos[[c for c in cols if c in extremos.columns]].head(n).round(3)


def tabla_ranking_secuencias(df_def_dinamico: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    cols = [
        "secuencia_rival_id",
        "cluster_trayectoria",
        "indice_peligrosidad_accion",
        "indice_desorganizacion",
        "categoria_peligrosidad_auto",
        "categoria_desorganizacion_auto",
        "tipo_desorganizacion_principal",
        "score_finalizacion",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
    ]
    if df_def_dinamico.empty:
        return pd.DataFrame(columns=cols)
    return df_def_dinamico[[c for c in cols if c in df_def_dinamico.columns]].sort_values(
        ["indice_peligrosidad_accion", "indice_desorganizacion"], ascending=False
    ).head(n)


def plot_ranking_secuencias(df_def_dinamico: pd.DataFrame, n: int = 12):
    ranking = tabla_ranking_secuencias(df_def_dinamico, n=n)
    if ranking.empty:
        return _empty_fig("Sin ranking de secuencias")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ranking["secuencia_rival_id"].astype(int).astype(str)
    ax.barh(labels, ranking["indice_peligrosidad_accion"], color="steelblue", label="IPO")
    ax.barh(labels, ranking["indice_desorganizacion"], color="darkorange", alpha=0.65, label="IDD")
    ax.invert_yaxis()
    ax.set_xlabel("Indice")
    ax.set_title("Ranking de secuencias mas peligrosas")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    return fig


def tabla_perfiles_extremos(df_def_dinamico: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df_def_dinamico.empty:
        return {"danio_sin_desorden": pd.DataFrame(), "caos_sin_castigo": pd.DataFrame()}
    danio_sin_desorden = df_def_dinamico[
        (df_def_dinamico["categoria_desorganizacion_auto"] == "baja")
        & (df_def_dinamico["categoria_peligrosidad_auto"].isin(["muy_alta", "critica"]))
    ].copy()
    caos_sin_castigo = df_def_dinamico[
        (df_def_dinamico["categoria_desorganizacion_auto"].isin(["muy_alta", "critica"]))
        & (df_def_dinamico["categoria_peligrosidad_auto"] == "baja")
    ].copy()
    return {"danio_sin_desorden": danio_sin_desorden, "caos_sin_castigo": caos_sin_castigo}


def tabla_causas_danio(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    extremos = tabla_perfiles_extremos(df_def_dinamico)["danio_sin_desorden"]
    if extremos.empty or "tipo_desorganizacion_principal" not in extremos.columns:
        return pd.DataFrame(columns=["causa", "frecuencia", "porcentaje"])
    causas = extremos["tipo_desorganizacion_principal"].value_counts()
    return pd.DataFrame(
        {
            "causa": causas.index,
            "frecuencia": causas.values,
            "porcentaje": (causas / causas.sum() * 100).round(1).values,
        }
    )


def plot_causas_danio(df_def_dinamico: pd.DataFrame):
    causas = tabla_causas_danio(df_def_dinamico)
    if causas.empty:
        return _empty_fig("Sin perfil de dano alto sin desorden")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(causas["causa"], causas["frecuencia"], color="firebrick", alpha=0.82)
    ax.set_title("Causas principales del dano real")
    ax.set_ylabel("Secuencias")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    return fig


def _zona_horizontal(y: float) -> str:
    if pd.isna(y):
        return "desconocida"
    if y < FIELD_WIDTH_M * 0.20:
        return "banda_izquierda"
    if y < FIELD_WIDTH_M * 0.40:
        return "halfspace_izquierdo"
    if y < FIELD_WIDTH_M * 0.60:
        return "carril_central"
    if y < FIELD_WIDTH_M * 0.80:
        return "halfspace_derecho"
    return "banda_derecha"


def tabla_zona_cluster_danio(df_def_dinamico: pd.DataFrame) -> pd.DataFrame:
    extremos = tabla_perfiles_extremos(df_def_dinamico)["danio_sin_desorden"].copy()
    if extremos.empty:
        return pd.DataFrame()
    extremos["zona_ataque"] = extremos["y_fin"].apply(_zona_horizontal)
    return (
        extremos.groupby(["zona_ataque", "cluster_trayectoria"], observed=False)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )


def plot_zona_cluster_danio(df_def_dinamico: pd.DataFrame):
    tabla = tabla_zona_cluster_danio(df_def_dinamico)
    if tabla.empty:
        return _empty_fig("Sin dano territorial")
    tabla["label"] = tabla["zona_ataque"].astype(str) + " | " + tabla["cluster_trayectoria"].apply(_patron_label)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(tabla["label"], tabla["n"], color="teal", alpha=0.82)
    ax.invert_yaxis()
    ax.set_title("Zona x patron del dano real")
    ax.set_xlabel("Secuencias")
    ax.grid(axis="x", alpha=0.25)
    return fig


def crear_dashboard_completo(resultado) -> dict[str, object]:
    """Devuelve figuras y tablas principales para renderizar en la futura app."""
    figs = {
        "kpis": plot_kpis_partido(resultado.resumen, resultado.df_def_dinamico, resultado.tabla_patrones),
        "matriz_ddi_ipar": plot_matriz_ddi_ipar(resultado.df_def_dinamico),
        "matriz_5x5": plot_matriz_5x5_ddi_ipar(resultado.df_def_dinamico),
        "evolucion_temporal": plot_evolucion_temporal(resultado.df_def_dinamico, resultado.secuencias),
        "evolucion_temporal_tiros": plot_evolucion_temporal_con_tiros(resultado.df_def_dinamico, resultado.secuencias),
        "trayectorias_cluster": plot_trayectorias_por_cluster(resultado.tracking_con_secuencias, resultado.df_clusters),
        "heatmap_clusters": plot_heatmap_territorial_clusters(resultado.tracking_con_secuencias, resultado.df_clusters),
        "mapa_calor_notebook": plot_mapa_calor_clusters_notebook(resultado.tracking_con_secuencias, resultado.df_clusters),
        "amenaza_media_cluster": plot_amenaza_media_por_cluster(resultado.df_def_dinamico),
        "ranking_secuencias": plot_ranking_secuencias(resultado.df_def_dinamico),
        "causas_danio": plot_causas_danio(resultado.df_def_dinamico),
        "zona_cluster_danio": plot_zona_cluster_danio(resultado.df_def_dinamico),
    }
    tablas = {
        "kpis": tabla_kpis_partido(resultado.resumen, resultado.df_def_dinamico, resultado.tabla_patrones),
        "patrones": resultado.tabla_patrones,
        "clusters_resumen": tabla_resumen_clusters(resultado.df_clusters, resultado.taxonomia_clusters),
        "riesgo_resumen": tabla_resumen_riesgo(resultado.df_def_dinamico),
        "ranking_secuencias": tabla_ranking_secuencias(resultado.df_def_dinamico),
        "matriz_5x5": tabla_matriz_5x5(resultado.df_def_dinamico),
        "perfiles_extremos": tabla_perfiles_extremos(resultado.df_def_dinamico),
        "causas_danio": tabla_causas_danio(resultado.df_def_dinamico),
        "zona_cluster_danio": tabla_zona_cluster_danio(resultado.df_def_dinamico),
        "danio_sin_desorden_simple": tabla_extremos_simple(resultado.df_def_dinamico, "danio_sin_desorden"),
        "caos_sin_castigo_simple": tabla_extremos_simple(resultado.df_def_dinamico, "caos_sin_castigo"),
    }
    return {"figuras": figs, "tablas": tablas}
