from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _time_to_ms(series: pd.Series, col_name: str = "") -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values
    name = str(col_name).lower()
    if "sec" in name and "ms" not in name:
        return values * 1000
    return values


def _event_has(event_types, names: set[str]) -> bool:
    if isinstance(event_types, list):
        return any(str(e).lower() in names for e in event_types)
    return any(name in str(event_types).lower() for name in names)


def _parse_event_ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _normalizar_sequences_bepro(df_sequences: pd.DataFrame, rival_team_id: int) -> pd.DataFrame:
    if df_sequences is None or df_sequences.empty:
        return pd.DataFrame()
    seq = df_sequences.copy()
    seq.columns = [str(c).strip() for c in seq.columns]

    id_col = _first_existing_col(seq, ["sequence_id", "sequenceId", "id"])
    team_col = _first_existing_col(seq, ["team_id", "teamId", "possession_team_id", "possessionTeamId"])
    period_col = _first_existing_col(seq, ["period_order", "period", "periodOrder"])
    start_col = _first_existing_col(
        seq,
        ["start_time_ms", "startTimeMs", "start_time", "startTime", "start_event_time", "startEventTime"],
    )
    end_col = _first_existing_col(
        seq,
        ["end_time_ms", "endTimeMs", "end_time", "endTime", "end_event_time", "endEventTime"],
    )
    duration_col = _first_existing_col(seq, ["duration", "duration_ms", "durationMs", "duration_sec", "durationSec"])
    event_ids_col = _first_existing_col(seq, ["event_ids", "eventIds", "events"])
    total_distance_col = _first_existing_col(seq, ["total_distance", "totalDistance"])
    num_passes_col = _first_existing_col(seq, ["num_of_passes", "numOfPasses", "passes"])
    progress_distance_col = _first_existing_col(seq, ["progress_distance", "progressDistance"])
    progress_direction_col = _first_existing_col(seq, ["progress_direction", "progressDirection"])
    ball_status_col = _first_existing_col(seq, ["ball_status", "ballStatus"])
    if team_col is None or period_col is None or start_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["bepro_sequence_id"] = seq[id_col].astype(str) if id_col is not None else np.arange(1, len(seq) + 1).astype(str)
    out["team_id_sequence"] = pd.to_numeric(seq[team_col], errors="coerce")
    out["period"] = pd.to_numeric(seq[period_col], errors="coerce")
    out["start_time_ms"] = _time_to_ms(seq[start_col], start_col)
    if end_col is not None:
        out["end_time_ms"] = _time_to_ms(seq[end_col], end_col)
    elif duration_col is not None:
        duration = _time_to_ms(seq[duration_col], duration_col)
        out["end_time_ms"] = out["start_time_ms"] + duration
    else:
        return pd.DataFrame()
    out["event_ids"] = seq[event_ids_col].apply(_parse_event_ids) if event_ids_col is not None else [[] for _ in range(len(seq))]
    out["bepro_total_distance_m"] = (
        pd.to_numeric(seq[total_distance_col], errors="coerce") if total_distance_col is not None else np.nan
    )
    out["bepro_num_passes"] = pd.to_numeric(seq[num_passes_col], errors="coerce") if num_passes_col is not None else np.nan
    out["bepro_progress_distance_m"] = (
        pd.to_numeric(seq[progress_distance_col], errors="coerce") if progress_distance_col is not None else np.nan
    )
    out["bepro_progress_direction"] = (
        seq[progress_direction_col].astype(str).str.lower() if progress_direction_col is not None else ""
    )
    out["bepro_ball_status"] = seq[ball_status_col].astype(str).str.lower() if ball_status_col is not None else ""

    out = out.dropna(subset=["team_id_sequence", "period", "start_time_ms", "end_time_ms"]).copy()
    out = out[out["team_id_sequence"].astype(int).eq(int(rival_team_id))].copy()
    out = out[out["end_time_ms"].gt(out["start_time_ms"])].copy()
    out["period"] = out["period"].astype(int)
    return out.sort_values(["period", "start_time_ms"]).reset_index(drop=True)


def detectar_secuencias_rivales(
    df_tracking: pd.DataFrame,
    team_id: int,
    rival_team_id: int,
    df_eventos_clave: pd.DataFrame | None = None,
    df_sequences_bepro: pd.DataFrame | None = None,
    max_gap_ms: int = 4000,
    max_transito_ms: int = 3500,
    vel_balon_transito_min: float = 0.3,
    min_eventos_rivales: int = 1,
    min_recorrido_balon_m: float = 10.0,
    min_duracion_activa_seg: float = 1.0,
    min_duracion_elaborada_seg: float = 6.0,
    min_eventos_elaborada: int = 3,
    min_recorrido_corto_peligroso_m: float = 7.5,
    min_vel_media_balon: float = 5.5,
    min_duracion_activa_seg_corta: float = 1.0,
    max_distancia_saque_corto_m: float = 15.0,
    min_eventos_post_setpiece: int = 2,
    min_duracion_activa_post_setpiece: float = 4.0,
    umbral_movimiento_saque_m: float = 0.5,
    max_duracion_secuencia_seg: float = 45.0,
    min_progresion_elaborada_m: float = 5.0,
    min_progresion_directa_m: float = 8.0,
    min_recorrido_directo_m: float = 15.0,
    min_xmax_directo_m: float = FIELD_LENGTH_M * 0.62,
    max_duracion_transicion_seg: float = 10.0,
    min_recorrido_transicion_m: float = 10.0,
    ventana_recuperacion_ms: int = 3000,
    ventana_cierre_terminal_ms: int = 3500,
    buffer_cierre_terminal_ms: int = 500,
    min_xmax_accion_ligera_m: float = FIELD_LENGTH_M * 0.57,
    min_progresion_accion_ligera_m: float = 14.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta secuencias ofensivas rivales desde tracking con poseedor.

    Devuelve:
    - `df_secuencias_validas`: una fila por secuencia rival.
    - `df_final_con_secuencias`: tracking con columna `secuencia_rival_id`.
    """
    df = df_tracking.copy()

    if "ball_x_m" not in df.columns:
        df["ball_x_m"] = df["ball_x"] * FIELD_LENGTH_M
    if "ball_y_m" not in df.columns:
        df["ball_y_m"] = df["ball_y"] * FIELD_WIDTH_M

    sort_cols = [c for c in ["match_id", "period", "match_time", "frame", "player_id"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    poseedores = (
        df[df.get("poseedor", 0) == 1]
        .groupby(["match_id", "period", "frame"], as_index=False)
        .agg(team_poseedor=("team_id", "first"))
    )

    frame_agg = {
        "match_time": ("match_time", "first"),
        "ball_x_m": ("ball_x_m", "first"),
        "ball_y_m": ("ball_y_m", "first"),
    }
    if "vel_ball_m_s" in df.columns:
        frame_agg["vel_ball_m_s"] = ("vel_ball_m_s", "first")

    df_frames = df.groupby(["match_id", "period", "frame"], as_index=False).agg(**frame_agg)
    if "vel_ball_m_s" not in df_frames.columns:
        df_frames["vel_ball_m_s"] = 0.0

    df_frames = df_frames.merge(poseedores, on=["match_id", "period", "frame"], how="left")
    df_frames = df_frames.sort_values(["match_id", "period", "match_time"]).reset_index(drop=True)

    attack_high_by_period: dict[int, bool] = {}
    if df_eventos_clave is not None and not df_eventos_clave.empty and "attack_direction" in df_eventos_clave.columns:
        ev_dir = df_eventos_clave.copy()
        ev_dir["team_id_event"] = pd.to_numeric(ev_dir.get("team_id_event"), errors="coerce")
        ev_dir["period_event"] = pd.to_numeric(ev_dir.get("period_event"), errors="coerce")
        ev_dir = ev_dir[ev_dir["team_id_event"].eq(int(rival_team_id))].dropna(subset=["period_event", "attack_direction"])
        for period, gr_dir in ev_dir.groupby("period_event"):
            mode = gr_dir["attack_direction"].astype(str).str.lower().mode()
            if not mode.empty:
                attack_high_by_period[int(period)] = mode.iloc[0] == "right"
    if not attack_high_by_period:
        attack_high_by_period = {0: True, 1: False}

    attack_high = df_frames["period"].map(lambda p: attack_high_by_period.get(int(p), True))
    df_frames["ball_x_norm"] = np.where(attack_high, df_frames["ball_x_m"], FIELD_LENGTH_M - df_frames["ball_x_m"])
    df_frames["ball_y_norm"] = np.where(attack_high, df_frames["ball_y_m"], FIELD_WIDTH_M - df_frames["ball_y_m"])

    if df_eventos_clave is not None and not df_eventos_clave.empty:
        eventos_base = df_eventos_clave.copy()
        eventos_base["team_id_event"] = pd.to_numeric(eventos_base["team_id_event"], errors="coerce")
        eventos_base["match_time_event"] = pd.to_numeric(eventos_base["match_time_event"], errors="coerce")
        eventos_base["period_event"] = pd.to_numeric(eventos_base["period_event"], errors="coerce")
    else:
        eventos_base = pd.DataFrame()

    def _x_norm_event(value, period) -> float:
        x = pd.to_numeric(value, errors="coerce")
        if pd.isna(x):
            return float("nan")
        attack_high_period = attack_high_by_period.get(int(period), True)
        x_m = float(x) * FIELD_LENGTH_M
        return x_m if attack_high_period else FIELD_LENGTH_M - x_m

    def _event_x_max_norm(eventos_sec: pd.DataFrame, period) -> float:
        if eventos_sec.empty:
            return float("nan")
        vals: list[float] = []
        for col in ["x_event", "to_x_event"]:
            if col in eventos_sec.columns:
                vals.extend([_x_norm_event(v, period) for v in eventos_sec[col].dropna()])
        vals = [v for v in vals if pd.notna(v)]
        return max(vals) if vals else float("nan")

    def _extender_hasta_cierre_terminal(period, end_time: float, max_end_time: float | None = None) -> tuple[float, str]:
        if eventos_base.empty:
            return end_time, "cierre_bepro"
        limite = end_time + ventana_cierre_terminal_ms
        if max_end_time is not None and pd.notna(max_end_time):
            limite = min(limite, float(max_end_time) - 1)
        if limite <= end_time:
            return end_time, "cierre_bepro"

        post = eventos_base[
            (eventos_base["period_event"] == int(period))
            & (eventos_base["match_time_event"] > end_time)
            & (eventos_base["match_time_event"] <= limite)
        ].sort_values("match_time_event")
        if post.empty:
            return end_time, "cierre_bepro"

        terminales = {
            "recovery",
            "interception",
            "ballrecovery",
            "pass",
            "clearance",
            "shot",
        }
        for _, ev in post.iterrows():
            types = ev.get("event_types", [])
            team_ev = pd.to_numeric(ev.get("team_id_event"), errors="coerce")
            es_terminal_propio = pd.notna(team_ev) and int(team_ev) != int(rival_team_id)
            es_tiro_rival = pd.notna(team_ev) and int(team_ev) == int(rival_team_id) and _event_has(types, {"shot"})
            if (es_terminal_propio or es_tiro_rival) and _event_has(types, terminales):
                cierre = min(float(ev["match_time_event"]) + buffer_cierre_terminal_ms, limite)
                motivo = "cierre_recuperacion_propia" if es_terminal_propio else "cierre_tiro_rival"
                return cierre, motivo
        return end_time, "cierre_bepro"

    seq_bepro = _normalizar_sequences_bepro(df_sequences_bepro, rival_team_id)
    if not seq_bepro.empty:
        secuencias_bepro = []
        for _, seq in seq_bepro.iterrows():
            seq_event_ids = set(seq.get("event_ids", []) or [])
            eventos_sec = pd.DataFrame()
            if not eventos_base.empty:
                if seq_event_ids and "id_event" in eventos_base.columns:
                    eventos_sec = eventos_base[
                        eventos_base["id_event"].astype(str).isin(seq_event_ids)
                        & (eventos_base["team_id_event"] == rival_team_id)
                    ].copy()
                else:
                    eventos_sec = eventos_base[
                        (eventos_base["period_event"] == seq["period"])
                        & (eventos_base["match_time_event"] >= seq["start_time_ms"])
                        & (eventos_base["match_time_event"] <= seq["end_time_ms"])
                        & (eventos_base["team_id_event"] == rival_team_id)
                    ].copy()
                eventos_sec = eventos_sec.sort_values("match_time_event").copy()

            start_time = float(seq["start_time_ms"])
            end_time = float(seq["end_time_ms"])
            end_time_bepro = end_time
            contiene_tiro = not eventos_sec.empty and eventos_sec["event_types"].apply(lambda x: _event_has(x, {"shot"})).any()
            texto_eventos_sec = eventos_sec.get("tipo_evento_raw", pd.Series(dtype=str)).astype(str).str.lower()
            contiene_gol = bool(
                contiene_tiro
                and texto_eventos_sec.str.contains("goal|gol").any()
            )
            contiene_tiro_puerta = bool(contiene_tiro and texto_eventos_sec.str.contains("shotontarget|shot_on_target|goal|gol").any())
            if contiene_tiro:
                chain_ids = set(seq_event_ids)
                chain_start = start_time
                while True:
                    prevs = seq_bepro[
                        (seq_bepro["period"] == int(seq["period"]))
                        & (seq_bepro["end_time_ms"] <= chain_start)
                        & ((chain_start - seq_bepro["end_time_ms"]) <= 5000)
                    ].copy()
                    if prevs.empty:
                        break
                    prev = prevs.sort_values("end_time_ms", ascending=False).iloc[0]
                    chain_ids.update(prev.get("event_ids", []) or [])
                    chain_start = float(prev["start_time_ms"])
                    if start_time - chain_start >= 15000:
                        break
                if chain_start < start_time:
                    start_time = chain_start
                    if not eventos_base.empty and chain_ids and "id_event" in eventos_base.columns:
                        eventos_sec = eventos_base[
                            eventos_base["id_event"].astype(str).isin(chain_ids)
                            & (eventos_base["team_id_event"] == rival_team_id)
                        ].sort_values("match_time_event").copy()

            cierre_terminal = "cierre_bepro"
            if not contiene_tiro:
                siguientes = seq_bepro[
                    (seq_bepro["period"] == int(seq["period"]))
                    & (seq_bepro["start_time_ms"] > end_time)
                ].sort_values("start_time_ms")
                max_end_time = float(siguientes.iloc[0]["start_time_ms"]) if not siguientes.empty else None
                end_time_ext, cierre_terminal = _extender_hasta_cierre_terminal(seq["period"], end_time, max_end_time)
                if end_time_ext > end_time:
                    end_time = end_time_ext
                    if not eventos_base.empty:
                        eventos_sec = eventos_base[
                            (eventos_base["period_event"] == int(seq["period"]))
                            & (eventos_base["match_time_event"] >= start_time)
                            & (eventos_base["match_time_event"] <= end_time)
                            & (eventos_base["team_id_event"] == rival_team_id)
                        ].sort_values("match_time_event").copy()

            gr = df_frames[
                (df_frames["period"] == seq["period"])
                & (df_frames["match_time"] >= start_time)
                & (df_frames["match_time"] <= end_time)
            ].copy()
            gr_ball = gr.drop_duplicates("frame").dropna(subset=["ball_x_m", "ball_y_m"]).sort_values("match_time")
            if len(gr_ball) < 2:
                continue
            duracion_seg = (end_time - start_time) / 1000
            recorrido = float(
                np.sqrt(
                    (gr_ball["ball_x_m"].iloc[-1] - gr_ball["ball_x_m"].iloc[0]) ** 2
                    + (gr_ball["ball_y_m"].iloc[-1] - gr_ball["ball_y_m"].iloc[0]) ** 2
                )
            )
            progresion_x = float(gr_ball["ball_x_norm"].iloc[-1] - gr_ball["ball_x_norm"].iloc[0])
            x_max_norm = float(gr_ball["ball_x_norm"].max())
            vel_media = float(gr_ball["vel_ball_m_s"].replace([np.inf, -np.inf], np.nan).mean())
            duracion_bepro_original_seg = (end_time_bepro - float(seq["start_time_ms"])) / 1000
            total_distance_bepro = pd.to_numeric(seq.get("bepro_total_distance_m", np.nan), errors="coerce")
            progress_distance_bepro = pd.to_numeric(seq.get("bepro_progress_distance_m", np.nan), errors="coerce")
            num_passes_bepro = pd.to_numeric(seq.get("bepro_num_passes", np.nan), errors="coerce")
            progress_direction = str(seq.get("bepro_progress_direction", "")).lower()
            distancia_accion = float(total_distance_bepro) if pd.notna(total_distance_bepro) else recorrido
            progresion_accion = float(progress_distance_bepro) if pd.notna(progress_distance_bepro) else max(0.0, progresion_x)
            if contiene_tiro:
                distancia_accion = max(float(distancia_accion), float(recorrido))
                progresion_accion = max(float(progresion_accion), float(max(0.0, progresion_x)))
            num_pases = int(num_passes_bepro) if pd.notna(num_passes_bepro) else 0
            first_event = eventos_sec.iloc[0] if not eventos_sec.empty else pd.Series(dtype=object)
            first_types = first_event.get("event_types", [])
            tiene_setpiece = not eventos_sec.empty and eventos_sec["setpiece_type"].notna().any()
            contiene_tiro = not eventos_sec.empty and eventos_sec["event_types"].apply(lambda x: _event_has(x, {"shot"})).any()
            texto_eventos_sec = eventos_sec.get("tipo_evento_raw", pd.Series(dtype=str)).astype(str).str.lower()
            contiene_gol = bool(
                contiene_tiro
                and texto_eventos_sec.str.contains("goal|gol").any()
            )
            contiene_tiro_puerta = bool(contiene_tiro and texto_eventos_sec.str.contains("shotontarget|shot_on_target|goal|gol").any())
            x_max_eventos_norm = _event_x_max_norm(eventos_sec, seq["period"])
            x_max_ataque_norm = max(
                [v for v in [x_max_norm, x_max_eventos_norm] if pd.notna(v)] or [0.0]
            )

            tipo = "descartada"
            motivo = "ruido_sin_continuidad_ni_progresion"
            conservar = False
            num_eventos_rivales = len(eventos_sec)
            first_recovery = _event_has(first_types, {"recovery", "interception", "ballrecovery"})
            first_pass = _event_has(first_types, {"pass"})
            pass_dist = 0.0
            pass_prog = 0.0
            if first_pass and {"x_event", "y_event", "to_x_event", "to_y_event"}.issubset(eventos_sec.columns):
                fx = pd.to_numeric(first_event.get("x_event"), errors="coerce")
                fy = pd.to_numeric(first_event.get("y_event"), errors="coerce")
                tx = pd.to_numeric(first_event.get("to_x_event"), errors="coerce")
                ty = pd.to_numeric(first_event.get("to_y_event"), errors="coerce")
                if pd.notna(fx) and pd.notna(fy) and pd.notna(tx) and pd.notna(ty):
                    pass_dist = float(np.sqrt(((tx - fx) * FIELD_LENGTH_M) ** 2 + ((ty - fy) * FIELD_WIDTH_M) ** 2))
                    attack_high_period = attack_high_by_period.get(int(seq["period"]), True)
                    x_from = fx * FIELD_LENGTH_M if attack_high_period else FIELD_LENGTH_M - fx * FIELD_LENGTH_M
                    x_to = tx * FIELD_LENGTH_M if attack_high_period else FIELD_LENGTH_M - tx * FIELD_LENGTH_M
                    pass_prog = float(x_to - x_from)

            if duracion_seg < min_duracion_activa_seg_corta:
                motivo = "microaccion_menos_1s"
            elif duracion_seg > max_duracion_secuencia_seg:
                motivo = "secuencia_demasiado_larga"
            elif tiene_setpiece:
                t_sp = eventos_sec[eventos_sec["setpiece_type"].notna()]["match_time_event"].min()
                eventos_post = eventos_sec[eventos_sec["match_time_event"] >= t_sp].copy()
                duracion_post = (end_time - float(t_sp)) / 1000 if pd.notna(t_sp) else 0
                continuidad_post = (
                    len(eventos_post) >= min_eventos_post_setpiece
                    and duracion_post >= min_duracion_activa_post_setpiece
                    and (
                        distancia_accion >= min_recorrido_directo_m
                        or progresion_accion >= min_progresion_directa_m
                        or num_pases >= 2
                    )
                )
                reinicio_inocuo = progress_direction == "backward" and progresion_accion < min_progresion_directa_m
                if continuidad_post and not reinicio_inocuo:
                    conservar, tipo, motivo = True, "balon_parado_con_continuidad", "setpiece_con_continuidad_bepro"
                else:
                    motivo = "setpiece_sin_continuidad_posterior"
            elif first_recovery:
                es_ataque_rapido = (
                    duracion_seg <= max_duracion_transicion_seg
                    and (
                        max(0.0, progresion_x) >= min_progresion_directa_m
                        or x_max_norm >= min_xmax_directo_m
                        or (progress_direction == "forward" and progresion_accion >= min_progresion_directa_m)
                    )
                )
                accion_ligera_sin_finalizacion = (
                    not contiene_tiro
                    and duracion_bepro_original_seg <= 3.0
                    and num_eventos_rivales <= 2
                    and num_pases <= 2
                )
                amenaza_ligera = (
                    x_max_ataque_norm >= min_xmax_accion_ligera_m
                    or (
                        max(0.0, progresion_x) >= min_progresion_accion_ligera_m
                        and x_max_ataque_norm >= FIELD_LENGTH_M * 0.50
                    )
                )
                if accion_ligera_sin_finalizacion and not amenaza_ligera:
                    motivo = "recuperacion_ligera_sin_amenaza_terminal"
                elif duracion_seg >= 2.0 and num_eventos_rivales >= 2 and distancia_accion >= min_recorrido_transicion_m and es_ataque_rapido:
                    conservar, tipo, motivo = True, "transicion_rapida", "origen_recuperacion_rival"
                elif (
                    num_eventos_rivales >= min_eventos_elaborada
                    and duracion_seg >= min_duracion_elaborada_seg
                    and distancia_accion >= min_recorrido_balon_m
                    and (num_pases >= 2 or num_eventos_rivales >= 4)
                ):
                    conservar, tipo, motivo = True, "ataque_elaborado", "recuperacion_con_posesion_elaborada"
                else:
                    motivo = "recuperacion_sin_continuidad_ofensiva"
            elif contiene_gol and duracion_seg >= 1.0 and num_eventos_rivales >= 2:
                tipo_final = "transicion_rapida" if duracion_seg <= max_duracion_transicion_seg else "ataque_elaborado"
                conservar, tipo, motivo = True, tipo_final, "finalizacion_gol_con_continuidad"
            elif contiene_tiro_puerta and duracion_seg >= 2.0 and num_eventos_rivales >= 2:
                tipo_final = "transicion_rapida" if duracion_seg <= max_duracion_transicion_seg else "ataque_elaborado"
                conservar, tipo, motivo = True, tipo_final, "tiro_a_puerta_con_continuidad"
            else:
                pase_largo_progresivo = pass_dist >= min_recorrido_directo_m and pass_prog >= min_progresion_directa_m
                secuencia_directa_bepro = (
                    first_pass
                    and
                    num_pases <= 2
                    and distancia_accion >= min_recorrido_directo_m
                    and progresion_accion >= min_progresion_directa_m
                    and progress_direction == "forward"
                )
                if (pase_largo_progresivo or secuencia_directa_bepro) and (num_eventos_rivales >= 2 or duracion_seg >= 2.0):
                    conservar, tipo, motivo = True, "juego_directo", "primer_pase_largo_progresivo"
                elif num_eventos_rivales >= min_eventos_elaborada and duracion_seg >= min_duracion_elaborada_seg:
                    if distancia_accion >= min_recorrido_balon_m and (num_pases >= 2 or num_eventos_rivales >= 4):
                        conservar, tipo, motivo = True, "ataque_elaborado", "posesion_bepro_elaborada"
                    else:
                        motivo = "posesion_larga_sin_recorrido_ofensivo"
                else:
                    motivo = "sin_desarrollo_ofensivo_evaluable"

            if conservar:
                secuencias_bepro.append(
                    {
                        "match_id": gr_ball["match_id"].iloc[0],
                        "period": int(seq["period"]),
                        "bepro_sequence_id": seq["bepro_sequence_id"],
                        "start_frame": gr_ball["frame"].iloc[0],
                        "end_frame": gr_ball["frame"].iloc[-1],
                        "start_time_ms": start_time,
                        "end_time_ms": end_time,
                        "end_time_bepro_ms": end_time_bepro,
                        "motivo_cierre": cierre_terminal,
                        "start_time_seg": start_time / 1000,
                        "end_time_seg": end_time / 1000,
                        "duracion_seg": duracion_seg,
                        "ball_x_ini_m": gr_ball["ball_x_m"].iloc[0],
                        "ball_y_ini_m": gr_ball["ball_y_m"].iloc[0],
                        "ball_x_fin_m": gr_ball["ball_x_m"].iloc[-1],
                        "ball_y_fin_m": gr_ball["ball_y_m"].iloc[-1],
                        "ball_x_ini_norm_m": gr_ball["ball_x_norm"].iloc[0],
                        "ball_y_ini_norm_m": gr_ball["ball_y_norm"].iloc[0],
                        "ball_x_fin_norm_m": gr_ball["ball_x_norm"].iloc[-1],
                        "ball_y_fin_norm_m": gr_ball["ball_y_norm"].iloc[-1],
                        "progresion_x_norm_m": progresion_x,
                        "x_max_norm_m": x_max_norm,
                        "recorrido_balon_m": recorrido,
                        "recorrido_bepro_m": distancia_accion,
                        "progresion_bepro_m": progresion_accion,
                        "num_pases_bepro": num_pases,
                        "direccion_progresion_bepro": progress_direction,
                        "vel_media_balon": vel_media,
                        "num_frames": len(gr_ball),
                        "num_eventos": len(eventos_sec),
                        "num_eventos_rivales": num_eventos_rivales,
                        "tiene_setpiece": tiene_setpiece,
                        "tipo_secuencia_ofensiva": tipo,
                        "motivo_conservacion": motivo,
                        "origen_segmentacion": "bepro_sequences",
                    }
                )

        df_secuencias_validas = pd.DataFrame(secuencias_bepro)
        if not df_secuencias_validas.empty:
            df_secuencias_validas = df_secuencias_validas.reset_index(drop=True)
            df_secuencias_validas["secuencia_rival_id"] = np.arange(1, len(df_secuencias_validas) + 1)
            df_final = df.copy()
            df_final["secuencia_rival_id"] = np.nan
            for _, sec in df_secuencias_validas.iterrows():
                mask = (
                    (df_final["match_id"] == sec["match_id"])
                    & (df_final["period"] == sec["period"])
                    & (df_final["match_time"] >= sec["start_time_ms"])
                    & (df_final["match_time"] <= sec["end_time_ms"])
                )
                df_final.loc[mask, "secuencia_rival_id"] = sec["secuencia_rival_id"]
            return df_secuencias_validas, df_final

    df_frames["team_posesion_extendida"] = np.nan
    df_frames["motivo_posesion_extendida"] = None

    for (_, _), idxs in df_frames.groupby(["match_id", "period"]).groups.items():
        ultimo_equipo = np.nan
        ultimo_tiempo = np.nan
        for idx in list(idxs):
            team_actual = df_frames.loc[idx, "team_poseedor"]
            t_actual = df_frames.loc[idx, "match_time"]
            vel_balon = df_frames.loc[idx, "vel_ball_m_s"]

            if pd.notna(team_actual):
                ultimo_equipo = team_actual
                ultimo_tiempo = t_actual
                df_frames.loc[idx, "team_posesion_extendida"] = team_actual
                df_frames.loc[idx, "motivo_posesion_extendida"] = "poseedor_claro"
            elif (
                pd.notna(ultimo_equipo)
                and pd.notna(ultimo_tiempo)
                and (t_actual - ultimo_tiempo <= max_transito_ms)
                and pd.notna(vel_balon)
                and vel_balon >= vel_balon_transito_min
            ):
                df_frames.loc[idx, "team_posesion_extendida"] = ultimo_equipo
                df_frames.loc[idx, "motivo_posesion_extendida"] = "balon_en_transito"
            else:
                df_frames.loc[idx, "motivo_posesion_extendida"] = "sin_control"

    df_frames["es_posesion_rival"] = df_frames["team_posesion_extendida"] == rival_team_id
    df_frames["prev_es_rival"] = df_frames.groupby(["match_id", "period"])["es_posesion_rival"].shift(1)
    df_frames["prev_time"] = df_frames.groupby(["match_id", "period"])["match_time"].shift(1)
    df_frames["gap_ms"] = df_frames["match_time"] - df_frames["prev_time"]
    df_frames["nuevo_bloque"] = (
        (df_frames["es_posesion_rival"] != df_frames["prev_es_rival"])
        | (df_frames["gap_ms"] > max_gap_ms)
    ).astype(int)
    df_frames["bloque_estado"] = df_frames.groupby(["match_id", "period"])["nuevo_bloque"].cumsum()

    secuencias = []
    rival_frames = df_frames[df_frames["es_posesion_rival"]].copy()
    for (match_id, period, bloque), gr in rival_frames.groupby(["match_id", "period", "bloque_estado"]):
        gr = gr.sort_values("match_time")
        start_time = gr["match_time"].min()
        end_time = gr["match_time"].max()
        duracion_seg = (end_time - start_time) / 1000
        recorrido = np.sqrt(
            (gr["ball_x_m"].iloc[-1] - gr["ball_x_m"].iloc[0]) ** 2
            + (gr["ball_y_m"].iloc[-1] - gr["ball_y_m"].iloc[0]) ** 2
        )
        progresion_x = gr["ball_x_norm"].iloc[-1] - gr["ball_x_norm"].iloc[0]
        x_max_norm = gr["ball_x_norm"].max()
        vel_media = gr["vel_ball_m_s"].replace([np.inf, -np.inf], np.nan).mean()
        secuencias.append(
            {
                "match_id": match_id,
                "period": period,
                "bloque_estado": bloque,
                "start_frame": gr["frame"].iloc[0],
                "end_frame": gr["frame"].iloc[-1],
                "start_time_ms": start_time,
                "end_time_ms": end_time,
                "start_time_seg": start_time / 1000,
                "end_time_seg": end_time / 1000,
                "duracion_seg": duracion_seg,
                "ball_x_ini_m": gr["ball_x_m"].iloc[0],
                "ball_y_ini_m": gr["ball_y_m"].iloc[0],
                "ball_x_fin_m": gr["ball_x_m"].iloc[-1],
                "ball_y_fin_m": gr["ball_y_m"].iloc[-1],
                "ball_x_ini_norm_m": gr["ball_x_norm"].iloc[0],
                "ball_y_ini_norm_m": gr["ball_y_norm"].iloc[0],
                "ball_x_fin_norm_m": gr["ball_x_norm"].iloc[-1],
                "ball_y_fin_norm_m": gr["ball_y_norm"].iloc[-1],
                "progresion_x_norm_m": progresion_x,
                "x_max_norm_m": x_max_norm,
                "recorrido_balon_m": recorrido,
                "vel_media_balon": vel_media,
                "num_frames": len(gr),
                "pct_balon_transito": (gr["motivo_posesion_extendida"] == "balon_en_transito").mean(),
            }
        )

    df_secuencias = pd.DataFrame(secuencias)
    if df_secuencias.empty:
        df["secuencia_rival_id"] = np.nan
        return df_secuencias, df

    if df_eventos_clave is not None and not df_eventos_clave.empty:
        eventos = df_eventos_clave.copy()
        eventos["team_id_event"] = pd.to_numeric(eventos["team_id_event"], errors="coerce")
        eventos["match_time_event"] = pd.to_numeric(eventos["match_time_event"], errors="coerce")
        eventos["period_event"] = pd.to_numeric(eventos["period_event"], errors="coerce")
    else:
        eventos = pd.DataFrame()

    eventos_info = []
    for _, sec in df_secuencias.iterrows():
        if eventos.empty:
            eventos_info.append({"num_eventos": 0, "num_eventos_rivales": 0, "tiene_setpiece": False})
            continue
        eventos_sec = eventos[
            (eventos["period_event"] == sec["period"])
            & (eventos["match_time_event"] >= sec["start_time_ms"])
            & (eventos["match_time_event"] <= sec["end_time_ms"])
        ].copy()
        eventos_rivales = eventos_sec[eventos_sec["team_id_event"] == rival_team_id].copy()
        tiene_setpiece = "setpiece_type" in eventos_rivales.columns and eventos_rivales["setpiece_type"].notna().any()
        eventos_info.append(
            {
                "num_eventos": len(eventos_sec),
                "num_eventos_rivales": len(eventos_rivales),
                "tiene_setpiece": tiene_setpiece,
            }
        )
    df_secuencias = pd.concat([df_secuencias.reset_index(drop=True), pd.DataFrame(eventos_info)], axis=1)

    def _hay_recuperacion_temprana(eventos_sec: pd.DataFrame, sec) -> bool:
        if eventos_sec.empty or "event_types" not in eventos_sec.columns:
            return False
        ventana_fin = sec["start_time_ms"] + ventana_recuperacion_ms
        tempranos = eventos_sec[eventos_sec["match_time_event"] <= ventana_fin].copy()
        return tempranos["event_types"].apply(lambda lst: "recovery" in lst if isinstance(lst, list) else "recovery" in str(lst)).any()

    def _metricas_tramo(frames_tramo: pd.DataFrame, period) -> dict:
        tramo = (
            frames_tramo[["frame", "match_time", "ball_x_m", "ball_y_m", "vel_ball_m_s"]]
            .drop_duplicates("frame")
            .dropna(subset=["ball_x_m", "ball_y_m"])
            .sort_values("match_time")
            .reset_index(drop=True)
        )
        if len(tramo) < 2:
            return {"duracion": 0.0, "recorrido": 0.0, "progresion": 0.0, "vel_media": 0.0}
        attack_high_period = attack_high_by_period.get(int(period), True)
        x_norm = tramo["ball_x_m"] if attack_high_period else FIELD_LENGTH_M - tramo["ball_x_m"]
        return {
            "duracion": float((tramo["match_time"].max() - tramo["match_time"].min()) / 1000),
            "recorrido": float(
                np.sqrt(
                    (tramo["ball_x_m"].iloc[-1] - tramo["ball_x_m"].iloc[0]) ** 2
                    + (tramo["ball_y_m"].iloc[-1] - tramo["ball_y_m"].iloc[0]) ** 2
                )
            ),
            "progresion": float(x_norm.iloc[-1] - x_norm.iloc[0]),
            "vel_media": float(pd.to_numeric(tramo["vel_ball_m_s"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        }

    def _cumple_elaboracion(sec) -> bool:
        return (
            sec["num_eventos_rivales"] >= min_eventos_elaborada
            and sec["duracion_seg"] >= min_duracion_elaborada_seg
            and sec["recorrido_balon_m"] >= min_recorrido_balon_m
        )

    def _cumple_directo(sec) -> bool:
        return (
            sec["num_eventos_rivales"] >= min_eventos_rivales
            and sec["duracion_seg"] >= min_duracion_activa_seg
            and sec["recorrido_balon_m"] >= min_recorrido_directo_m
            and sec["vel_media_balon"] >= min_vel_media_balon
            and sec["progresion_x_norm_m"] >= min_progresion_directa_m
        )

    def _cumple_transicion(sec, eventos_sec: pd.DataFrame) -> bool:
        return (
            _hay_recuperacion_temprana(eventos_sec, sec)
            and sec["num_eventos_rivales"] >= min_eventos_rivales
            and sec["duracion_seg"] >= min_duracion_activa_seg
            and sec["duracion_seg"] <= max_duracion_transicion_seg
            and sec["recorrido_balon_m"] >= min_recorrido_transicion_m
            and sec["vel_media_balon"] >= min_vel_media_balon
            and sec["progresion_x_norm_m"] >= min_progresion_directa_m
        )

    def evaluar_secuencia(sec):
        eventos_sec = pd.DataFrame()
        if not eventos.empty:
            eventos_sec = eventos[
                (eventos["period_event"] == sec["period"])
                & (eventos["match_time_event"] >= sec["start_time_ms"])
                & (eventos["match_time_event"] <= sec["end_time_ms"])
                & (eventos["team_id_event"] == rival_team_id)
            ].copy()

        tiene_setpiece = "setpiece_type" in eventos_sec.columns and eventos_sec["setpiece_type"].notna().any()
        if sec["duracion_seg"] < min_duracion_activa_seg_corta:
            return False, "descartada", "microaccion_menos_1s"
        if sec["duracion_seg"] > max_duracion_secuencia_seg:
            return False, "descartada", "secuencia_demasiado_larga"

        if tiene_setpiece:
            eventos_sp = eventos_sec[eventos_sec["setpiece_type"].notna()].copy()
            if eventos_sp.empty:
                return False, "descartada", "setpiece_sin_evento_claro"
            t_sp = eventos_sp["match_time_event"].min()
            frames_post = df_frames[
                (df_frames["period"] == sec["period"])
                & (df_frames["match_time"] >= t_sp)
                & (df_frames["match_time"] <= sec["end_time_ms"])
            ].copy()
            if frames_post.empty:
                return False, "descartada", "setpiece_sin_juego_posterior"
            ball_post = (
                frames_post[["frame", "match_time", "ball_x_m", "ball_y_m", "vel_ball_m_s"]]
                .drop_duplicates("frame")
                .dropna(subset=["ball_x_m", "ball_y_m"])
                .sort_values("match_time")
                .reset_index(drop=True)
            )
            if len(ball_post) < 2:
                return False, "descartada", "setpiece_sin_movimiento_balon"
            ball_post["dx"] = ball_post["ball_x_m"].diff()
            ball_post["dy"] = ball_post["ball_y_m"].diff()
            ball_post["mov_m"] = np.sqrt(ball_post["dx"] ** 2 + ball_post["dy"] ** 2)
            primer_mov = ball_post[ball_post["mov_m"] > umbral_movimiento_saque_m].copy()
            if primer_mov.empty:
                return False, "descartada", "setpiece_balon_parado_sin_saque_real"
            idx_mov = primer_mov.index[0]
            t_inicio_activo = ball_post.loc[idx_mov, "match_time"]
            distancia_saque_m = np.sqrt(
                (ball_post.loc[idx_mov, "ball_x_m"] - ball_post.loc[0, "ball_x_m"]) ** 2
                + (ball_post.loc[idx_mov, "ball_y_m"] - ball_post.loc[0, "ball_y_m"]) ** 2
            )
            duracion_activa_post = (ball_post["match_time"].max() - t_inicio_activo) / 1000
            eventos_post = eventos_sec[eventos_sec["match_time_event"] >= t_inicio_activo].copy()
            num_eventos_post = len(eventos_post)
            frames_activos = frames_post[frames_post["match_time"] >= t_inicio_activo].copy()
            post = _metricas_tramo(frames_activos, sec["period"])
            elaboracion_post = (
                num_eventos_post >= min_eventos_post_setpiece
                and duracion_activa_post >= min_duracion_activa_post_setpiece
                and post["recorrido"] >= min_recorrido_balon_m
            )
            directo_post = (
                num_eventos_post >= min_eventos_post_setpiece
                and duracion_activa_post >= min_duracion_activa_seg
                and post["recorrido"] >= min_recorrido_directo_m
                and post["vel_media"] >= min_vel_media_balon
                and post["progresion"] >= min_progresion_directa_m
            )
            if duracion_activa_post < min_duracion_activa_post_setpiece or num_eventos_post < min_eventos_post_setpiece:
                return False, "descartada", "setpiece_sin_continuidad_posterior"
            if elaboracion_post:
                motivo = "setpiece_corto_con_elaboracion" if distancia_saque_m <= max_distancia_saque_corto_m else "setpiece_largo_con_elaboracion"
                return True, "balon_parado_con_continuidad", motivo
            if directo_post:
                return True, "balon_parado_con_continuidad", "setpiece_con_accion_directa_posterior"
            return False, "descartada", "setpiece_sin_continuidad"

        if _cumple_transicion(sec, eventos_sec):
            return True, "transicion_rapida", "recuperacion_y_amenaza"

        if _cumple_elaboracion(sec):
            return True, "ataque_elaborado", "elaboracion_juego_fluido"

        if _cumple_directo(sec):
            return True, "juego_directo", "desplazamiento_vertical_claro"

        return False, "descartada", "ruido_sin_continuidad_ni_progresion"

    evaluaciones = df_secuencias.apply(evaluar_secuencia, axis=1)
    df_secuencias["conservar"] = evaluaciones.apply(lambda x: x[0])
    df_secuencias["tipo_secuencia_ofensiva"] = evaluaciones.apply(lambda x: x[1])
    df_secuencias["motivo_conservacion"] = evaluaciones.apply(lambda x: x[2])
    df_secuencias_validas = df_secuencias[df_secuencias["conservar"]].copy().reset_index(drop=True)
    df_secuencias_validas["origen_segmentacion"] = "tracking_poseedor"
    df_secuencias_validas["secuencia_rival_id"] = np.arange(1, len(df_secuencias_validas) + 1)

    df_final = df.copy()
    df_final["secuencia_rival_id"] = np.nan
    for _, sec in df_secuencias_validas.iterrows():
        mask = (
            (df_final["match_id"] == sec["match_id"])
            & (df_final["period"] == sec["period"])
            & (df_final["match_time"] >= sec["start_time_ms"])
            & (df_final["match_time"] <= sec["end_time_ms"])
        )
        df_final.loc[mask, "secuencia_rival_id"] = sec["secuencia_rival_id"]

    return df_secuencias_validas, df_final
