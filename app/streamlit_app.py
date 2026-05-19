from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots

try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tfg_analysis.app_data import listar_app_data_disponible
from tfg_analysis.config import FIELD_LENGTH_M, FIELD_WIDTH_M, ProjectPaths
from tfg_analysis.features.threat import _asignar_celda_xt, _crear_xt_grid
from tfg_analysis.index_interpretation import (
    build_interpretability_reference,
    classify_index_value,
    xthreat_reference_text,
)
from tfg_analysis.io import listar_tracking_disponible


TEAM_ID_DEFAULT = 12987
MATCH_PRESENTATION = {
    199542: {"home_team_id": 11016, "away_team_id": 12987},
    207673: {"home_team_id": 11459, "away_team_id": 12987},
    209425: {"home_team_id": 7608, "away_team_id": 12987},
    212200: {"home_team_id": 13020, "away_team_id": 12987},
}
BEPRO_SCHEDULE_IDS = {
    199542: 520395,
    207673: 536197,
    209425: 540944,
    212200: 550027,
}
BEPRO_MATCH_URL_TEMPLATE_DEFAULT = (
    "https://space.bepro.ai/center/matches/{bepro_match_id}/videos/video-type/fullMatchVideo"
)
BEPRO_API_BASE = "https://d.bepro11.com/api"
BEPRO_LIBRARY_URL_TEMPLATE_DEFAULT = "https://space.bepro.ai/center/library/me/root"
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
OLLAMA_MODEL_DEFAULT = "llama3.1"
APP_USERS = {
    "entrenador": "demo2026",
    "juan": "demo2026",
    "admin": "demo2026",
}
USER_ROLES = ("guest", "analista", "admin")
USERS_COLLECTION = "usuarios"
ANONYMIZE_TEAM_IDENTITIES = True
ANON_TEAM_NAME = "Tu Equipo"
ANON_RIVAL_PREFIX = "Equipo Rival"
ANON_RIVAL_IDS: list[int] = []
for _match_id in sorted(MATCH_PRESENTATION):
    for _side in ("home_team_id", "away_team_id"):
        _team_id = MATCH_PRESENTATION[_match_id].get(_side)
        if _team_id is not None and int(_team_id) != TEAM_ID_DEFAULT and int(_team_id) not in ANON_RIVAL_IDS:
            ANON_RIVAL_IDS.append(int(_team_id))
ANON_RIVAL_ORDER = {team_id: idx + 1 for idx, team_id in enumerate(ANON_RIVAL_IDS)}
REGISTERED_TEAMS = {
    TEAM_ID_DEFAULT: {
        "name": ANON_TEAM_NAME,
        "short_name": ANON_TEAM_NAME,
        "location": "Entorno privado",
        "role": "Equipo registrado",
        "description": (
            "Entorno activo de la plataforma con tracking, eventing, tipologías ofensivas "
            "rivales, estructura defensiva, secuencias críticas, momentum e informe final."
        ),
        "status": "Activo",
        "access_code": "demo2026",
    }
}


def _clean_app_text(value):
    if value is None:
        return value
    if not isinstance(value, str):
        return value
    text = value
    for _ in range(3):
        if not any(mark in text for mark in ("Ã", "Â", "\x8d", "\x81")):
            break
        repaired = None
        for encoding in ("latin1", "cp1252"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if not repaired or repaired == text:
            break
        text = repaired
    if any(mark in text for mark in ("Ã", "Â", "ï¿½")):
        for encoding in ("latin1", "cp1252"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if repaired and repaired != text:
                text = repaired
                break
    return text.replace("Â", "").replace("�", "")


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].map(_clean_app_text)
    return out
APP_COLOR_SEQUENCE = ["#c8102e", "#2453a6", "#f2c94c", "#ead7a4", "#6b7280"]
APP_COLOR_MAP = {
    "T0": "#c8102e",
    "T1": "#2453a6",
    "T2": "#f2c94c",
    "T3": "#ead7a4",
    "T4": "#6b7280",
    "Tipologia 1": "#c8102e",
    "Tipologia 2": "#2453a6",
    "Tipologia 3": "#f2c94c",
    "Tipologia 4": "#ead7a4",
    "Tipologia 5": "#6b7280",
    "Tipología 1": "#c8102e",
    "Tipología 2": "#2453a6",
    "Tipología 3": "#f2c94c",
    "Tipología 4": "#ead7a4",
    "Tipología 5": "#6b7280",
    "Todas": "#c8102e",
}
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
}


st.set_page_config(
    page_title="Plataforma de análisis defensivo",
    page_icon="TD",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _paths() -> ProjectPaths:
    return ProjectPaths(root=ROOT).resolve()


@st.cache_data(show_spinner=False)
def _interpretability_reference() -> dict:
    app_data_dir = _paths().outputs_dir / "app_data"
    frames: list[pd.DataFrame] = []
    for match_dir in sorted(app_data_dir.iterdir() if app_data_dir.exists() else []):
        if not match_dir.is_dir() or not match_dir.name.isdigit():
            continue
        for filename in ("secuencias_detalle.csv", "ranking_secuencias.csv"):
            path = match_dir / filename
            if path.exists():
                try:
                    frames.append(pd.read_csv(path))
                except (OSError, EmptyDataError):
                    pass
                break
    sequences = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    reference = build_interpretability_reference(sequences)
    reference["thresholds"] = {
        metric: data.get("thresholds", {}) for metric, data in reference.items() if isinstance(data, dict)
    }
    reference["counts"] = {
        metric: {
            str(level.get("label", "")): int(count)
            for level, count in zip(data.get("levels", []), data.get("counts", []))
        }
        for metric, data in reference.items()
        if isinstance(data, dict) and "levels" in data
    }
    reference["n_sequences"] = int(len(sequences))
    return reference


def inject_style():
    st.markdown(
        """
        <style>
        :root {
            --osasuna-red: #c8102e;
            --osasuna-red-dark: #8f0d23;
            --osasuna-navy: #15223b;
            --osasuna-navy-dark: #0a1020;
            --osasuna-blue: #2453a6;
            --page-dark: #202532;
            --panel-dark: #283040;
            --panel-dark-soft: #30384a;
            --text-main: #f4f6fb;
            --text-muted: #aab3c4;
            --soft-gray: #f7f4f5;
            --ink: #272b36;
            --line: rgba(200,16,46,0.28);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(200,16,46,0.12), transparent 34rem),
                linear-gradient(180deg, #2a3040 0%, #202532 58%, #1e2430 100%);
            color: var(--text-main);
        }
        div[data-testid="stToolbar"],
        div[data-testid="stElementToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stHeaderActionElements"],
        div[data-testid="stHeadingActionElements"],
        div[data-testid="stHeadingWithActionElements"] a,
        div[data-testid="stHeadingWithActionElements"] button,
        .block-container a[href^="#"],
        [data-testid="stMarkdownContainer"] a[href^="#"],
        [data-testid="stMarkdownContainer"] h1 a,
        [data-testid="stMarkdownContainer"] h2 a,
        [data-testid="stMarkdownContainer"] h3 a,
        [data-testid="stMarkdownContainer"] h4 a,
        [data-testid="stMarkdownContainer"] h5 a,
        [data-testid="stMarkdownContainer"] h6 a,
        a.anchor-link,
        .anchor-link,
        .heading-anchor,
        div[data-testid="StyledFullScreenButton"],
        button[title*="Fullscreen" i],
        button[title*="full screen" i],
        button[title*="expand" i],
        button[aria-label*="Fullscreen" i],
        button[aria-label*="full screen" i],
        button[aria-label*="expand" i],
        .modebar,
        .modebar-container,
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        header[data-testid="stHeader"] {
            display: block !important;
            visibility: visible !important;
            height: 0 !important;
            min-height: 0 !important;
            background: transparent !important;
            pointer-events: none !important;
        }
        header[data-testid="stHeader"] button,
        div[data-testid="stSidebarCollapsedControl"],
        div[data-testid="stSidebarCollapseButton"],
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        section[data-testid="stSidebar"] button[aria-label*="close" i],
        section[data-testid="stSidebar"] button[aria-label*="collapse" i],
        section[data-testid="stSidebar"] button[title*="close" i],
        section[data-testid="stSidebar"] button[title*="collapse" i] {
            opacity: 0 !important;
            pointer-events: none !important;
            visibility: hidden !important;
            overflow: hidden !important;
        }
        div[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            position: absolute !important;
            top: 30px !important;
            left: 72px !important;
            z-index: 100000 !important;
            width: 42px !important;
            height: 34px !important;
            display: grid !important;
            place-items: center !important;
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stSidebarCollapseButton"] {
            display: none !important;
            visibility: hidden !important;
            position: absolute !important;
            top: 30px !important;
            left: 334px !important;
            z-index: 100000 !important;
            width: 42px !important;
            height: 34px !important;
            display: none !important;
            place-items: center !important;
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stSidebarCollapsedControl"]::after,
        div[data-testid="stSidebarCollapseButton"]::after {
            position: absolute !important;
            inset: 0 !important;
            display: none !important;
            place-items: center !important;
            color: #ffffff !important;
            font-size: 27px !important;
            line-height: 1 !important;
            font-weight: 900 !important;
            text-shadow: 0 2px 8px rgba(0,0,0,0.55) !important;
            z-index: 2 !important;
            pointer-events: none !important;
        }
        div[data-testid="stSidebarCollapsedControl"]::after {
            content: none !important;
        }
        div[data-testid="stSidebarCollapseButton"]::after {
            content: none !important;
        }
        div[data-testid="stSidebarCollapsedControl"] button,
        div[data-testid="stSidebarCollapseButton"] button {
            position: absolute !important;
            inset: 0 !important;
            width: 100% !important;
            height: 100% !important;
            display: grid !important;
            place-items: center !important;
            background: transparent !important;
            border: 0 !important;
            color: transparent !important;
            font-size: 0 !important;
            line-height: 0 !important;
            overflow: hidden !important;
            z-index: 3 !important;
        }
        div[data-testid="stSidebarCollapsedControl"] button *,
        div[data-testid="stSidebarCollapseButton"] button * {
            opacity: 0 !important;
            color: transparent !important;
            font-size: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
        }
        div[data-testid="stSidebarCollapsedControl"] button::after,
        div[data-testid="stSidebarCollapseButton"] button::after {
            content: "" !important;
        }
        .block-container {
            background: transparent !important;
            border-left: 1px solid rgba(200,16,46,0.36);
            border-right: 1px solid rgba(255,255,255,0.06);
            box-shadow: inset 4px 0 0 rgba(200,16,46,0.95);
            min-height: 100vh;
            padding-left: 3rem;
            padding-right: 4rem;
            padding-bottom: 28rem !important;
            max-width: 1480px;
        }
        section.main > div,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stVerticalBlock"] {
            padding-bottom: 1.25rem;
        }
        div[data-testid="stAppViewContainer"],
        section[data-testid="stMain"],
        section.main,
        section.main > div,
        .main,
        .main .block-container,
        .main .block-container > div,
        div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlock"] > div,
        div[data-testid="stElementContainer"],
        div[data-testid="stMarkdown"],
        div[data-testid="stMarkdownContainer"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"] > div,
        div[data-testid="stBlock"] {
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border-top: 0 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 0 !important;
            outline: 0 !important;
        }
        div[data-testid="stRadio"],
        div[data-testid="stRadio"] *,
        div[data-testid="stRadio"] div,
        div[data-testid="stRadio"] section {
            border-top-color: transparent !important;
            outline: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stSegmentedControl"],
        div[data-testid="stSegmentedControl"] *,
        div[data-testid="stSegmentedControl"] > div,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] {
            outline: 0 !important;
            box-shadow: none !important;
        }
        div[data-testid="stSegmentedControl"] {
            border: 0 !important;
            border-top: 0 !important;
            background: transparent !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stSegmentedControl"]),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stRadio"]),
        div[data-testid="stElementContainer"]:has(div[data-testid="stSegmentedControl"]),
        div[data-testid="stElementContainer"]:has(div[data-testid="stRadio"]) {
            border: 0 !important;
            border-top: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stSegmentedControl"]) > div,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stRadio"]) > div {
            border: 0 !important;
            border-top: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        div[data-testid="stSegmentedControl"]::before,
        div[data-testid="stSegmentedControl"]::after {
            display: none !important;
            content: none !important;
        }
        div[data-testid="stExpander"] details > summary {
            list-style: none !important;
        }
        div[data-testid="stExpander"] details > summary::-webkit-details-marker {
            display: none !important;
        }
        div[data-testid="stExpander"] details > summary svg,
        div[data-testid="stExpander"] details > summary [data-testid="stIconMaterial"],
        div[data-testid="stExpander"] details > summary [data-testid="stExpanderToggleIcon"],
        div[data-testid="stExpander"] details > summary > div:last-child {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
            color: transparent !important;
            fill: transparent !important;
        }
        .bottom-safe-space {
            height: 26rem;
        }
        .graph-section-gap {
            height: 22px;
        }
        div[data-testid="stCaptionContainer"] {
            margin-bottom: 2px !important;
        }
        .app-shell {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 4px 0 14px 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 16px;
        }
        .page-title-lockup {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
            font-size: 1.42rem;
            font-weight: 900;
            line-height: 1.12;
        }
        .page-title-lockup:before {
            content: "";
            width: 7px;
            height: 34px;
            border-radius: 999px;
            background: linear-gradient(180deg, var(--osasuna-red), var(--osasuna-blue));
            box-shadow: 0 0 20px rgba(200,16,46,0.35);
        }
        .portal-hero {
            min-height: 0;
            border-radius: 8px;
            padding: 0;
            margin: 0;
            background: transparent;
            border: 0;
            display: none;
            box-shadow: none;
        }
        .portal-hero h1 {
            max-width: 850px;
            margin: 0 0 10px 0;
            color: var(--osasuna-navy-dark) !important;
            font-size: 1.85rem;
            line-height: 1.06;
        }
        .portal-hero p {
            max-width: 780px;
            margin: 0;
            color: #465570 !important;
            font-size: 1.02rem;
        }
        .login-wrap {
            min-height: 78vh;
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(330px, 0.7fr);
            gap: 28px;
            align-items: center;
        }
        .login-visual {
            min-height: 500px;
            border-radius: 8px;
            padding: 30px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            background:
                linear-gradient(180deg, rgba(5,10,22,0.62), rgba(5,10,22,0.93)),
                url("https://images.unsplash.com/photo-1574629810360-7efbbe195018?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center 58%;
            box-shadow: 0 20px 46px rgba(21,34,59,0.24);
            overflow: hidden;
        }
        .login-brand {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #ffffff !important;
            font-size: .78rem;
            font-weight: 950;
            letter-spacing: .12em;
            text-transform: uppercase;
            opacity: .92;
        }
        .login-brand span {
            color: #d7deec !important;
            font-size: .72rem;
            letter-spacing: .05em;
        }
        .login-kicker {
            display: inline-flex;
            color: #f2c94c !important;
            font-weight: 900;
            margin-bottom: 10px;
            font-size: .92rem;
        }
        .login-visual h1 {
            color: #ffffff !important;
            font-size: clamp(2.0rem, 3.2vw, 3.15rem);
            line-height: 1.04;
            margin: 0 0 10px 0;
            max-width: 720px;
        }
        .login-visual p {
            color: #eef3ff !important;
            max-width: 610px;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.52;
        }
        .login-access-heading {
            margin: 0 0 16px 0;
        }
        .login-access-heading span {
            display: block;
            color: #aab3c4 !important;
            font-size: .82rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 4px;
        }
        .login-access-heading strong {
            display: block;
            color: #ffffff !important;
            font-size: 1.55rem;
            line-height: 1.1;
        }
        .login-access-note {
            margin-top: 14px;
            color: #aab3c4 !important;
            font-size: .92rem;
            line-height: 1.35;
        }
        div[data-testid="stForm"] {
            border: 1px solid rgba(255,255,255,.08) !important;
            background: rgba(15,23,42,.30) !important;
            border-radius: 8px !important;
            padding: 18px 18px 16px 18px !important;
            box-shadow: 0 20px 44px rgba(0,0,0,.16) !important;
        }
        .login-panel {
            background: #ffffff;
            border: 1px solid rgba(200,16,46,0.26);
            border-top: 5px solid #c8102e;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 16px 38px rgba(21,34,59,0.12);
        }
        .login-panel h2 {
            margin: 0 0 6px 0;
        }
        .story-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 10px 0 18px 0;
        }
        .story-card {
            background: #ffffff;
            border: 1px solid rgba(21,34,59,0.12);
            border-top: 4px solid #c8102e;
            border-radius: 8px;
            padding: 16px;
            min-height: 132px;
            box-shadow: 0 8px 20px rgba(21,34,59,0.07);
        }
        .story-card h3 {
            margin: 0 0 8px 0;
            font-size: 1rem;
        }
        .story-card p {
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .section-band {
            margin: 22px 0 14px 0;
            padding-top: 14px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }
        .section-band h3 {
            margin-bottom: 4px;
        }
        .table-note {
            color: var(--text-muted);
            font-size: 0.86rem;
            margin: -4px 0 10px 0;
        }
        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4 {
            color: var(--text-main) !important;
            text-shadow: none;
            text-transform: uppercase;
        }
        .app-page-heading {
            color: #f8fafc !important;
            font-size: 1.72rem !important;
            line-height: 1.12 !important;
            font-weight: 950 !important;
            text-transform: uppercase !important;
            letter-spacing: 0 !important;
            margin: 14px 0 18px 0 !important;
        }
        .app-page-heading:before {
            content: "";
            display: inline-block;
            width: 7px;
            height: 32px;
            margin-right: 11px;
            border-radius: 999px;
            background: var(--osasuna-red);
            vertical-align: -7px;
        }
        .app-subsection-heading {
            color: #f4f6fb !important;
            font-size: 1.08rem !important;
            line-height: 1.2 !important;
            font-weight: 900 !important;
            text-transform: uppercase !important;
            letter-spacing: 0 !important;
            margin: 24px 0 7px 0 !important;
        }
        .app-subsection-heading:before {
            content: "";
            display: inline-block;
            width: 5px;
            height: 20px;
            margin-right: 8px;
            border-radius: 999px;
            background: rgba(200,16,46,0.92);
            vertical-align: -4px;
        }
        .app-selected-heading {
            color: #ffffff !important;
            font-size: 1rem !important;
            line-height: 1.22 !important;
            font-weight: 900 !important;
            text-transform: uppercase !important;
            letter-spacing: 0 !important;
            margin: 10px 0 8px 0 !important;
        }
        .block-container p,
        .block-container span,
        .block-container label,
        .block-container div,
        .block-container div[data-testid="stCaptionContainer"],
        .block-container div[data-testid="stMarkdownContainer"] > p {
            color: #ffffff !important;
            text-shadow: none;
        }
        .block-container div[data-testid="stCaptionContainer"],
        .block-container div[data-testid="stCaptionContainer"] * {
            color: #dbe3f3 !important;
        }
        .block-container hr {
            border-color: rgba(255,255,255,0.08);
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(42,48,64,0.98) 0%, rgba(32,37,50,0.98) 100%);
            border-right: 4px solid var(--osasuna-red);
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] *,
        section[data-testid="stSidebar"] input {
            color: #4b5563 !important;
        }
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] button *,
        section[data-testid="stSidebar"] div[data-testid="stButton"] * {
            color: #111827 !important;
        }
        div[data-baseweb="select"] > div {
            background: #eef2f7 !important;
            border: 1px solid rgba(255,255,255,0.20) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="select"] input {
            color: #4b5563 !important;
        }
        div[data-baseweb="popover"] *,
        ul[role="listbox"] *,
        li[role="option"] *,
        div[role="option"] * {
            color: #374151 !important;
            text-shadow: none !important;
        }
        div[data-testid="stExpander"] {
            border: 0 !important;
        }
        div[data-testid="stExpander"] details > summary {
            background:
                linear-gradient(135deg, rgba(48,56,74,0.98), rgba(37,43,56,0.98)) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            border-left: 5px solid var(--osasuna-red) !important;
            border-top: 1px solid rgba(200,16,46,0.72) !important;
            border-radius: 8px !important;
            color: #dbe3f3 !important;
            min-height: 54px !important;
            padding: 14px 16px 14px 54px !important;
            box-shadow: 0 12px 26px rgba(0,0,0,0.18) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        div[data-testid="stExpander"] details > summary::before {
            content: "+" !important;
            position: absolute !important;
            left: 18px !important;
            top: 50% !important;
            transform: translateY(-50%) !important;
            width: 24px !important;
            height: 24px !important;
            border-radius: 999px !important;
            background: var(--osasuna-red) !important;
            color: #ffffff !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 18px !important;
            font-weight: 950 !important;
            line-height: 1 !important;
            box-shadow: 0 8px 18px rgba(200,16,46,0.28) !important;
        }
        div[data-testid="stExpander"] details > summary::after {
            content: none !important;
            display: none !important;
        }
        div[data-testid="stExpander"] details[open] > summary::after {
            content: none !important;
            display: none !important;
        }
        div[data-testid="stExpander"] details > summary:hover {
            border-color: rgba(200,16,46,0.70) !important;
            background:
                linear-gradient(135deg, rgba(53,62,82,0.98), rgba(40,47,63,0.98)) !important;
        }
        div[data-testid="stExpander"] details > summary div,
        div[data-testid="stExpander"] details > summary *,
        div[data-testid="stExpander"] details > summary p,
        div[data-testid="stExpander"] details > summary span {
            color: #dbe3f3 !important;
            text-shadow: none !important;
            font-weight: 800 !important;
        }
        div[data-testid="stExpander"] details > summary svg,
        div[data-testid="stExpander"] details > summary svg * {
            display: none !important;
            color: transparent !important;
            fill: transparent !important;
            background: transparent !important;
        }
        div[data-testid="stExpander"] details > summary > div:last-child {
            display: none !important;
            background: transparent !important;
        }
        div[data-testid="stExpander"] details > summary span[data-testid="stIconMaterial"],
        div[data-testid="stExpander"] details > summary span[class*="material"] {
            display: none !important;
            width: 0 !important;
            min-width: 0 !important;
            color: transparent !important;
            font-size: 0 !important;
            line-height: 0 !important;
            overflow: hidden !important;
            background: transparent !important;
        }
        div[data-testid="stExpander"] details > summary span[data-testid="stIconMaterial"]::after,
        div[data-testid="stExpander"] details > summary span[class*="material"]::after {
            content: none !important;
            display: none !important;
        }
        div[data-testid="stExpander"] details[open] > summary span[data-testid="stIconMaterial"]::after,
        div[data-testid="stExpander"] details[open] > summary span[class*="material"]::after {
            content: none !important;
            display: none !important;
        }
        div[data-testid="stExpander"] details[open] {
            background: rgba(15,23,42,0.16) !important;
            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 8px !important;
            padding-bottom: 10px !important;
        }
        section[data-testid="stSidebar"] button {
            background: #ffffff !important;
            border: 1px solid rgba(200,16,46,0.38) !important;
            font-weight: 800 !important;
        }
        button[kind="primary"],
        button[data-testid="baseButton-primary"],
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stButton"] button[data-testid="baseButton-primary"],
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"],
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stFormSubmitButton"] button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"],
        div[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] {
            background: var(--osasuna-red) !important;
            background-color: var(--osasuna-red) !important;
            border-color: var(--osasuna-red) !important;
            color: #ffffff !important;
            box-shadow: 0 12px 22px rgba(200,16,46,0.24) !important;
        }
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stDownloadButton"] button:focus,
        button[kind="primary"]:focus,
        button[data-testid="baseButton-primary"]:focus,
        button[data-testid="stBaseButton-primary"]:focus,
        div[data-testid="stButton"] button[kind="primary"]:hover,
        div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover,
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:focus,
        div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"]:hover,
        div[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:hover {
            background: var(--osasuna-red) !important;
            background-color: var(--osasuna-red) !important;
            border-color: var(--osasuna-red) !important;
            color: #ffffff !important;
        }
        button[kind="primary"] *,
        button[data-testid="baseButton-primary"] *,
        button[data-testid="stBaseButton-primary"] *,
        div[data-testid="stDownloadButton"] button *,
        div[data-testid="stDownloadButton"] button p,
        div[data-testid="stButton"] button[kind="primary"] p,
        div[data-testid="stButton"] button[data-testid="baseButton-primary"] p,
        div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] p,
        div[data-testid="stFormSubmitButton"] button *,
        div[data-testid="stFormSubmitButton"] button[kind="primary"] p,
        div[data-testid="stFormSubmitButton"] button[data-testid="baseButton-primary"] p,
        div[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] p {
            color: #ffffff !important;
            font-weight: 900 !important;
        }
        div[data-testid="stDownloadButton"] {
            width: 100% !important;
        }
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stDownloadButton"] button[kind="secondary"],
        div[data-testid="stDownloadButton"] button[data-testid*="baseButton"] {
            width: 100% !important;
            min-height: 46px !important;
            background: var(--osasuna-red) !important;
            background-color: var(--osasuna-red) !important;
            border: 1px solid var(--osasuna-red) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            font-weight: 900 !important;
            box-shadow: 0 12px 22px rgba(200,16,46,0.24) !important;
        }
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stDownloadButton"] button:focus {
            background: var(--osasuna-red-dark) !important;
            background-color: var(--osasuna-red-dark) !important;
            border-color: var(--osasuna-red-dark) !important;
            color: #ffffff !important;
        }
        div[data-testid="stDownloadButton"] button *,
        div[data-testid="stDownloadButton"] button p {
            color: #ffffff !important;
            font-weight: 900 !important;
        }
        .sidebar-club {
            text-align: center;
            padding: 12px 6px 18px 6px;
            border-bottom: 1px solid rgba(255,255,255,0.18);
            margin-bottom: 12px;
        }
        .sidebar-crest {
            width: 86px;
            height: 86px;
            margin: 0 auto 10px auto;
            border-radius: 50%;
            background: #ffffff;
            color: var(--osasuna-red) !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 1.15rem;
            border: 4px solid var(--osasuna-red);
            overflow: hidden;
            box-sizing: border-box;
            padding: 9px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        }
        .sidebar-crest img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            object-position: center;
            display: block;
        }
        .sidebar-club h2 {
            margin: 0;
            font-size: 1.35rem;
            letter-spacing: 0;
        }
        .sidebar-club p {
            margin: 4px 0 0 0;
            color: #cdd7ea !important;
            font-size: 0.85rem;
        }
        .sidebar-nav-title {
            color: #ffffff !important;
            font-size: 1.02rem;
            font-weight: 950;
            margin: 8px 0 14px 0;
            text-transform: uppercase;
        }
        .sidebar-account {
            background: rgba(36,83,166,0.26);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 8px;
            padding: 11px 12px;
            margin: 10px 0 12px;
        }
        .sidebar-account strong {
            display: block;
            color: #ffffff !important;
            font-size: 0.92rem;
        }
        .portal-landing {
            max-width: 1160px;
            margin: 0 auto;
            padding: 10px 0 30px;
        }
        .portal-welcome {
            border: 1px solid rgba(255,255,255,0.16);
            border-left: 7px solid var(--osasuna-red);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(10,16,32,0.98), rgba(32,37,50,0.96) 54%, rgba(94,15,34,0.82));
            padding: 26px 28px;
            margin: 4px 0 20px;
            box-shadow: 0 20px 44px rgba(0,0,0,0.22);
        }
        .portal-welcome h1 {
            color: #ffffff !important;
            font-size: 2.25rem;
            line-height: 1.08;
            margin: 0 0 12px 0;
            text-transform: none;
            letter-spacing: 0;
        }
        .portal-welcome p {
            color: #dbe3f3 !important;
            max-width: 930px;
            margin: 0;
            font-size: 1rem;
            line-height: 1.48;
            font-weight: 650;
        }
        .portal-section-title {
            color: #f8fafc !important;
            font-size: 1.38rem;
            font-weight: 950;
            margin: 24px 0 12px;
            text-transform: uppercase;
        }
        .portal-section-title:before {
            content: "";
            display: inline-block;
            width: 6px;
            height: 25px;
            border-radius: 999px;
            margin-right: 10px;
            vertical-align: -5px;
            background: var(--osasuna-red);
        }
        .portal-module-grid,
        .portal-steps-grid,
        .team-card-grid,
        .club-stat-grid {
            display: grid;
            gap: 12px;
        }
        .portal-module-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .portal-steps-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .portal-module-card,
        .portal-step-card,
        .team-card,
        .club-home-card,
        .match-picker-card {
            background: rgba(32,37,50,0.78);
            border: 1px solid rgba(255,255,255,0.11);
            border-top: 4px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        }
        .portal-module-card strong,
        .portal-step-card strong,
        .team-card strong,
        .club-home-card strong,
        .match-picker-card strong {
            display: block;
            color: #ffffff !important;
            font-size: 1rem;
            line-height: 1.25;
            margin-bottom: 7px;
        }
        .portal-module-card span,
        .portal-step-card span,
        .team-card span,
        .club-home-card span,
        .match-picker-card span {
            display: block;
            color: #cdd7ea !important;
            font-size: 0.90rem;
            line-height: 1.42;
            font-weight: 650;
        }
        .portal-module-card ul,
        .portal-step-card ul {
            margin: 10px 0 0 18px;
            padding: 0;
        }
        .portal-module-card li,
        .portal-step-card li {
            color: #f4f6fb !important;
            font-size: 0.86rem;
            line-height: 1.45;
            margin: 4px 0;
            font-weight: 650;
        }
        .portal-selector {
            margin-top: 18px;
            background: rgba(15,23,42,0.28);
            border: 1px solid rgba(255,255,255,0.10);
            border-left: 6px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 16px;
        }
        .team-card-grid {
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            margin: 10px 0 18px;
        }
        .team-card {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 14px;
            align-items: center;
            border-top-color: var(--osasuna-red);
        }
        .registered-team-card-marker {
            display: none;
        }
        div[data-testid="stHorizontalBlock"]:has(.registered-team-card-marker) {
            background: rgba(32,37,50,0.78);
            border: 1px solid rgba(255,255,255,0.11);
            border-top: 4px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
            margin: 10px 0 18px;
            align-items: center;
        }
        div[data-testid="stHorizontalBlock"]:has(.registered-team-card-marker) .team-card {
            background: transparent;
            border: 0;
            box-shadow: none;
            padding: 0;
            margin: 0;
        }
        div[data-testid="stHorizontalBlock"]:has(.registered-team-card-marker) div[data-testid="stButton"] button {
            background: var(--osasuna-red) !important;
            background-color: var(--osasuna-red) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            border-radius: 8px !important;
            min-height: 48px !important;
            font-size: 0.95rem !important;
            font-weight: 950 !important;
            box-shadow: 0 12px 22px rgba(200,16,46,0.24) !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.registered-team-card-marker) div[data-testid="stButton"] button p {
            color: #ffffff !important;
            font-weight: 950 !important;
        }
        .team-card .team-card-logo .sidebar-crest {
            width: 72px;
            height: 72px;
            margin: 0;
            padding: 8px;
        }
        .team-card strong {
            font-size: 1.18rem !important;
        }
        .team-pill {
            display: inline-flex;
            width: max-content;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(46,160,90,0.18);
            color: #9be7b9 !important;
            padding: 4px 9px;
            font-size: 0.72rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-top: 8px;
        }
        .club-home-card {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 18px;
            align-items: center;
            margin-bottom: 18px;
            border-left: 7px solid var(--osasuna-red);
            border-top-color: rgba(255,255,255,0.16);
            background:
                linear-gradient(135deg, rgba(10,16,32,0.96), rgba(36,83,166,0.34) 48%, rgba(200,16,46,0.24));
        }
        .club-home-card .sidebar-crest {
            width: 92px;
            height: 92px;
            margin: 0;
            padding: 14px;
        }
        .club-home-card h1 {
            color: #ffffff !important;
            font-size: clamp(2.6rem, 4.2vw, 4.0rem) !important;
            line-height: 1.08;
            margin: 0;
            font-weight: 950;
            text-transform: none;
        }
        .club-name-large {
            color: #ffffff !important;
            font-size: clamp(2.05rem, 3.25vw, 3.05rem) !important;
            line-height: 1.06 !important;
            font-weight: 950 !important;
            text-transform: none !important;
        }
        .club-stat-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 12px 0 18px;
        }
        .club-stat {
            background: rgba(15,23,42,0.30);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 8px;
            padding: 12px;
        }
        .club-stat b {
            display: block;
            color: #ffffff !important;
            font-size: 1.22rem;
        }
        .club-stat span {
            display: block;
            color: #aab3c4 !important;
            font-size: 0.80rem;
            font-weight: 800;
            margin-top: 4px;
            text-transform: uppercase;
        }
        .app-breadcrumb {
            color: #cbd4e6 !important;
            font-size: 0.86rem;
            font-weight: 800;
            margin: -6px 0 12px;
        }
        .app-breadcrumb b {
            color: #ffffff !important;
        }
        @media (max-width: 1100px) {
            .portal-module-grid,
            .portal-steps-grid,
            .club-stat-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 720px) {
            .portal-module-grid,
            .portal-steps-grid,
            .club-stat-grid,
            .team-card,
            .club-home-card {
                grid-template-columns: 1fr;
            }
        }
        .hero {
            border: 1px solid rgba(255,255,255,0.3);
            border-left: 8px solid var(--osasuna-red);
            background:
                linear-gradient(135deg, #0a1020 0%, #15223b 42%, #2453a6 67%, #c8102e 100%);
            color: white;
            padding: 22px 28px;
            border-radius: 8px;
            margin-top: 0.5rem;
            margin-bottom: 18px;
            box-shadow: 0 18px 42px rgba(21, 34, 59, 0.24);
        }
        .hero h1 {
            margin: 0 0 8px 0;
            font-size: 1.75rem;
            color: #ffffff !important;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: #eef3ff !important;
            font-size: 1rem;
        }
        .hero,
        .hero * {
            text-shadow:
                -1px -1px 0 rgba(7, 10, 20, 0.62),
                 1px -1px 0 rgba(7, 10, 20, 0.62),
                -1px  1px 0 rgba(7, 10, 20, 0.62),
                 1px  1px 0 rgba(7, 10, 20, 0.62),
                 0 8px 22px rgba(0, 0, 0, 0.36) !important;
        }
        .team-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 58px;
            width: 58px;
            height: 58px;
            padding: 7px;
            margin-right: 12px;
            border-radius: 50%;
            background: #ffffff;
            color: #c8102e;
            font-weight: 900;
            border: 3px solid #c8102e;
            vertical-align: middle;
            overflow: hidden;
            box-shadow: 0 0 0 3px rgba(200,16,46,0.12), 0 8px 18px rgba(21,34,59,0.12);
        }
        .team-badge img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .match-card {
            background:
                linear-gradient(135deg, rgba(10,16,32,0.98) 0%, rgba(21,34,59,0.98) 52%, rgba(117,9,28,0.96) 100%);
            border: 1px solid rgba(200,16,46,0.46);
            border-left: 7px solid var(--osasuna-red);
            border-top: 1px solid rgba(255,255,255,0.10);
            border-radius: 8px;
            padding: 18px 22px;
            margin: 0 0 18px 0;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.26);
        }
        .match-card,
        .match-card * {
            text-shadow: none !important;
        }
        .match-grid {
            display: grid;
            grid-template-columns: minmax(0, 0.92fr) auto minmax(0, 0.92fr);
            align-items: center;
            gap: 10px;
            max-width: 980px;
            margin: 0 auto;
        }
        .match-team {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
        }
        .match-team.away {
            justify-content: flex-start;
            text-align: right;
            flex-direction: row-reverse;
        }
        .team-name {
            font-size: 1.2rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: none;
        }
        .team-label {
            font-size: 0.78rem;
            color: #cdd7ea !important;
            text-transform: uppercase;
            letter-spacing: 0;
            font-weight: 700;
        }
        .score-pill {
            min-width: 98px;
            text-align: center;
            border-radius: 8px;
            background: linear-gradient(135deg, #061026, #c8102e);
            color: #ffffff;
            padding: 10px 14px;
            font-weight: 900;
            font-size: 1.15rem;
            box-shadow: 0 12px 24px rgba(200,16,46,0.30);
        }
        .analysis-card {
            background:
                linear-gradient(180deg, rgba(48,56,74,0.98) 0%, rgba(39,47,64,0.98) 100%);
            border: 1px solid rgba(255,255,255,0.12);
            border-top: 5px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 14px 15px;
            min-height: 92px;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
        }
        .analysis-card,
        .analysis-card * {
            text-shadow: none !important;
        }
        .analysis-card .label {
            color: #aab3c4 !important;
            font-size: 0.82rem;
            line-height: 1.2;
            margin-bottom: 8px;
            font-weight: 700;
            text-shadow: none;
        }
        .analysis-card .value {
            color: #ffffff !important;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1;
        }
        .analysis-card.compact {
            min-height: 74px;
        }
        .analysis-card.compact .value {
            font-size: 1.05rem;
            line-height: 1.25;
        }
        .metric-badge {
            display: inline-grid;
            grid-template-columns: 12px minmax(0, 1fr);
            align-items: center;
            gap: 7px;
            min-width: 0;
            max-width: 100%;
            color: #ffffff !important;
            font-weight: 900;
            line-height: 1.12;
            white-space: normal;
            word-break: normal;
            overflow-wrap: normal;
            hyphens: none;
        }
        .metric-badge .metric-dot {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            flex: 0 0 auto;
            box-shadow: 0 0 0 4px rgba(255,255,255,0.08);
        }
        .metric-badge .metric-label {
            display: block;
            min-width: 0;
            white-space: normal;
            word-break: normal;
            overflow-wrap: normal;
            hyphens: none;
        }
        .metric-badge .metric-number {
            display: block;
            grid-column: 2;
            margin-top: 2px;
            color: #cbd4e6 !important;
            font-size: 0.86em;
            font-style: normal;
            font-weight: 850;
        }
        .metric-badge.compact {
            grid-template-columns: 10px minmax(0, 1fr);
            gap: 6px;
            font-size: 0.90rem;
        }
        .metric-badge.compact .metric-dot {
            width: 9px;
            height: 9px;
        }
        .xt-reference-box {
            display: grid;
            grid-template-columns: minmax(0, 0.78fr) minmax(320px, 0.50fr);
            gap: 14px;
            align-items: center;
            margin: 10px 0 18px 0;
            background: rgba(37,43,56,0.82);
            border: 1px solid rgba(255,255,255,0.10);
            border-left: 5px solid #2453a6;
            border-radius: 8px;
            padding: 13px;
        }
        .xt-reference-box p {
            margin: 0;
            color: #dbe3f3 !important;
            font-weight: 700;
            line-height: 1.42;
        }
        .xt-level-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin-top: 13px;
        }
        .xt-level-item {
            background: rgba(15,23,42,0.28);
            border: 1px solid rgba(255,255,255,0.11);
            border-radius: 8px;
            padding: 9px 10px;
            min-height: 76px;
        }
        .xt-level-item b {
            display: flex;
            align-items: center;
            gap: 7px;
            color: #ffffff !important;
            font-size: 0.88rem;
            line-height: 1.18;
        }
        .xt-level-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex: 0 0 auto;
        }
        .xt-level-item span {
            display: block;
            color: #cdd7ea !important;
            font-size: 0.78rem;
            line-height: 1.25;
            margin-top: 5px;
            font-weight: 750;
        }
        .xt-reference-box img {
            display: block;
            width: min(96%, 540px);
            max-height: 315px;
            object-fit: contain;
            margin: 0 auto;
            border-radius: 7px;
            border: 1px solid rgba(255,255,255,0.10);
            background: #111827;
            padding: 6px;
        }
        @media (max-width: 900px) {
            .xt-reference-box {
                grid-template-columns: 1fr;
            }
            .xt-level-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        .summary-section-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 14px;
            margin-top: 16px;
        }
        .summary-section {
            background: rgba(32,37,50,0.40);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 14px;
        }
        .summary-section.wide {
            grid-column: 1 / -1;
        }
        .summary-section-head {
            display: block;
            margin-bottom: 12px;
        }
        .summary-section-title {
            display: flex;
            align-items: center;
            gap: 9px;
            color: #f4f6fb !important;
            font-size: 0.92rem;
            font-weight: 900;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .summary-section-title:before {
            content: "";
            width: 5px;
            height: 22px;
            border-radius: 999px;
            background: var(--osasuna-red);
        }
        .info-tip {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            margin-left: 0;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.36);
            background: rgba(15,23,42,0.72);
            color: #ffffff !important;
            font-size: 0.72rem;
            line-height: 1;
            font-weight: 950;
            cursor: pointer;
            text-transform: none;
            z-index: 20;
        }
        .info-tip-panel {
            display: none;
            position: absolute;
            left: 0;
            top: calc(100% + 9px);
            width: min(620px, 74vw);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(200,16,46,0.54);
            border-left: 5px solid var(--osasuna-red);
            background: #151a25;
            color: #e8edf7 !important;
            box-shadow: 0 18px 38px rgba(0,0,0,0.38);
            font-size: 0.88rem;
            font-weight: 650;
            line-height: 1.42;
            letter-spacing: 0;
            text-transform: none;
            white-space: normal;
            z-index: 9999;
        }
        .info-tip-panel * {
            color: inherit !important;
            text-transform: none !important;
            letter-spacing: 0 !important;
        }
        .info-tip-title {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #ffffff !important;
            font-size: 0.94rem;
            font-weight: 950;
            line-height: 1.25;
            margin: 0 0 9px 0;
        }
        .info-tip-title:before {
            content: "i";
            display: inline-grid;
            place-items: center;
            width: 20px;
            height: 20px;
            border-radius: 5px;
            background: var(--osasuna-red);
            color: #ffffff !important;
            font-size: 0.72rem;
            font-weight: 950;
            flex: 0 0 auto;
        }
        .info-tip-grid {
            display: grid;
            gap: 8px;
        }
        .info-tip-row {
            display: grid;
            grid-template-columns: minmax(104px, 0.34fr) minmax(0, 1fr);
            gap: 10px;
            align-items: start;
            padding: 9px 10px;
            border-radius: 7px;
            background: linear-gradient(135deg, rgba(48,56,74,0.82), rgba(32,37,50,0.90));
            border: 1px solid rgba(255,255,255,0.10);
        }
        .info-tip-row.index-row {
            grid-template-columns: minmax(86px, 0.22fr) minmax(0, 1fr);
            background: linear-gradient(135deg, rgba(200,16,46,0.22), rgba(32,37,50,0.95));
            border-color: rgba(200,16,46,0.42);
            border-left: 5px solid var(--osasuna-red);
            margin-top: 5px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.18);
        }
        .info-tip-row.child-row {
            margin-left: 28px;
            grid-template-columns: minmax(122px, 0.34fr) minmax(0, 1fr);
            background: rgba(48,56,74,0.48);
            border-color: rgba(255,255,255,0.08);
            border-left: 3px solid rgba(200,16,46,0.78);
            box-shadow: inset 12px 0 0 rgba(200,16,46,0.08);
        }
        .info-tip-row.glossary-row {
            margin-top: 3px;
            background: rgba(15,23,42,0.70);
            border-color: rgba(255,255,255,0.14);
        }
        .info-tip-row b {
            color: #f8fafc !important;
            font-size: 0.77rem;
            line-height: 1.25;
            text-transform: uppercase !important;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .info-tip-row b:before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--osasuna-red);
            box-shadow: 0 0 0 3px rgba(200,16,46,0.16);
            flex: 0 0 auto;
        }
        .info-tip-row.child-row b:before {
            width: 5px;
            height: 5px;
            box-shadow: none;
            opacity: 0.82;
        }
        .info-tip-row.index-row b {
            font-size: 0.82rem;
        }
        .info-tip-row.index-row span {
            font-size: 0.82rem;
            font-weight: 800;
        }
        .info-tip-row span {
            color: #dbe3f3 !important;
            font-size: 0.80rem;
            line-height: 1.32;
            font-weight: 650;
        }
        .info-tip:hover .info-tip-panel,
        .info-tip:focus .info-tip-panel,
        .info-tip:focus-within .info-tip-panel {
            display: block;
        }
        .section-info-heading {
            display: inline-grid;
            grid-template-columns: max-content max-content;
            align-items: center;
            gap: 6px;
            margin: 10px 0 12px 0;
            width: max-content;
            max-width: 100%;
        }
        .section-info-heading h3,
        .section-info-heading h4 {
            margin: 0 !important;
            color: #f4f6fb !important;
            display: block !important;
            width: auto !important;
            max-width: max-content !important;
            flex: 0 0 auto !important;
        }
        .section-intro {
            background: rgba(37,43,56,0.98);
            border: 1px solid rgba(255,255,255,0.88);
            border-left: 8px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 6px 0 18px 0;
            box-shadow: 0 0 0 1px rgba(200,16,46,0.58), 0 14px 28px rgba(0,0,0,0.22);
        }
        .section-intro strong {
            display: block;
            color: #ffffff !important;
            font-size: 1rem;
            margin-bottom: 5px;
            text-transform: uppercase;
        }
        .section-intro span {
            display: block;
            color: #dbe3f3 !important;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .section-intro-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .section-intro-point {
            display: flex;
            gap: 8px;
            align-items: flex-start;
            padding: 9px 10px;
            border-radius: 7px;
            background: rgba(15,23,42,0.28);
            border: 1px solid rgba(255,255,255,0.08);
            color: #dbe3f3 !important;
            font-size: 0.88rem;
            line-height: 1.35;
            font-weight: 650;
        }
        .section-intro-point:before {
            content: "";
            width: 7px;
            height: 7px;
            margin-top: 0.42em;
            border-radius: 50%;
            background: var(--osasuna-red);
            box-shadow: 0 0 0 3px rgba(200,16,46,0.14);
            flex: 0 0 auto;
        }
        .summary-section-note {
            color: #cbd4e6 !important;
            font-size: 0.91rem;
            line-height: 1.35;
            margin: 12px 0 0 0;
            padding: 11px 13px;
            background: rgba(15,23,42,0.34);
            border: 1px solid rgba(255,255,255,0.08);
            border-left: 4px solid rgba(200,16,46,0.92);
            border-radius: 8px;
        }
        .summary-card-grid {
            display: grid;
            grid-template-columns: repeat(var(--summary-cols, 2), minmax(0, 1fr));
            gap: 10px;
        }
        .summary-card-grid .analysis-card {
            min-height: 84px;
        }
        .summary-card-grid .analysis-card .value {
            font-size: 1.08rem;
            line-height: 1.24;
        }
        .summary-combo-card {
            background:
                linear-gradient(180deg, rgba(48,56,74,0.98) 0%, rgba(39,47,64,0.98) 100%);
            border: 1px solid rgba(255,255,255,0.12);
            border-top: 5px solid var(--osasuna-red);
            border-radius: 8px;
            min-height: 84px;
            overflow: hidden;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
        }
        .summary-combo-card.triple {
            grid-column: 1 / -1;
        }
        .summary-combo-card.triple .summary-combo-grid,
        .summary-combo-card.cols-3 .summary-combo-grid {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
            grid-auto-flow: column !important;
            grid-auto-rows: auto !important;
        }
        .summary-combo-card.triple .summary-combo-item,
        .summary-combo-card.cols-3 .summary-combo-item {
            grid-row: 1 !important;
            min-width: 0 !important;
        }
        .summary-combo-grid {
            display: grid;
            grid-template-columns: repeat(var(--combo-cols, 2), minmax(0, 1fr));
            height: 100%;
        }
        .summary-combo-item {
            padding: 14px 15px;
            min-width: 0;
            overflow: hidden;
        }
        .summary-combo-item + .summary-combo-item {
            border-left: 1px solid rgba(255,255,255,0.12);
        }
        .summary-combo-item .label {
            color: #aab3c4 !important;
            font-size: 0.76rem;
            line-height: 1.2;
            margin-bottom: 8px;
            font-weight: 800;
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
        }
        .summary-combo-item .value {
            color: #ffffff !important;
            font-size: 1.04rem;
            line-height: 1.16;
            font-weight: 900;
            word-break: normal;
            overflow-wrap: normal;
            hyphens: none;
        }
        @media (max-width: 1100px) {
            .summary-card-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 720px) {
            .summary-card-grid {
                grid-template-columns: 1fr;
            }
        }
        .coach-note {
            background: rgba(37,43,56,0.98);
            border-left: 5px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 14px 16px;
            color: #dbe3f3;
            margin: 0.5rem 0 1rem 0;
        }
        .sequence-card {
            background: rgba(48,56,74,0.96);
            border: 1px solid rgba(255,255,255,0.12);
            border-top: 4px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 8px 20px rgba(21,34,59,0.07);
            min-height: 160px;
        }
        .sequence-card h4 {
            margin: 0 0 8px 0;
            color: #ffffff !important;
        }
        .sequence-card p {
            margin: 4px 0;
            color: #aab3c4 !important;
        }
        .sequence-kpi-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0;
        }
        .sequence-kpi {
            background: rgba(15,23,42,0.34);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 8px;
            padding: 10px 12px;
        }
        .sequence-kpi span {
            display: block;
            color: #aab3c4 !important;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .sequence-kpi strong {
            display: block;
            color: #ffffff !important;
            font-size: 1.08rem;
            line-height: 1.12;
            word-break: normal;
            overflow-wrap: normal;
            hyphens: none;
        }
        .sequence-read-box {
            background: rgba(15,23,42,0.30);
            border-left: 4px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 10px 12px;
            margin: 10px 0;
            color: #e8edf7 !important;
            font-weight: 700;
        }
        .sequence-tag-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .sequence-tag {
            background: rgba(15,23,42,0.26);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 8px;
            padding: 8px 10px;
            color: #dbe3f3 !important;
            font-weight: 800;
            line-height: 1.25;
        }
        .sequence-tag span {
            display: block;
            color: #aab3c4 !important;
            font-size: 0.76rem;
            margin-bottom: 4px;
            text-transform: uppercase;
        }
        .sequence-tag strong {
            color: #ffffff !important;
        }
        @media (max-width: 900px) {
            .sequence-kpi-row,
            .sequence-tag-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        .sequence-card .stButton > button {
            min-height: 46px;
            font-weight: 800;
        }
        .sequence-action-row {
            margin-top: 12px;
            margin-bottom: 12px;
            padding-left: 14px;
            position: relative;
            z-index: 3;
        }
        .sequence-action-row .stButton > button {
            background: #eef3ff !important;
            color: #202532 !important;
            border: 1px solid rgba(255,255,255,0.22) !important;
            font-weight: 900 !important;
        }
        .sequence-action-row .stButton > button * {
            color: #202532 !important;
        }
        .sequence-bar-label {
            min-height: 38px;
            display: flex;
            align-items: center;
            color: #e8edf7 !important;
            font-weight: 900;
            white-space: nowrap;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button {
            min-height: 38px;
            border-radius: 6px !important;
            font-weight: 850 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(48,56,74,0.96);
            border: 1px solid rgba(255,255,255,0.12);
            border-top: 4px solid #c8102e;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(200,16,46,0.38);
            border-top: 4px solid var(--osasuna-red);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(21, 34, 59, 0.08);
        }
        div[data-testid="stDataFrame"] [role="grid"] {
            background: #ffffff;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {
            text-shadow: none !important;
        }
        .static-table-wrap {
            background: linear-gradient(180deg, rgba(32,37,50,0.98), rgba(25,31,43,0.98));
            border: 1px solid rgba(255,255,255,0.12);
            border-top: 4px solid var(--osasuna-red);
            border-radius: 8px;
            overflow: auto;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.24);
            margin: 10px 0 1rem 0;
        }
        .static-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        .static-table th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #101722;
            color: #f4f6fb;
            font-weight: 850;
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            text-transform: uppercase;
            letter-spacing: 0.02em;
            font-size: 0.76rem;
            white-space: nowrap;
        }
        .static-table td {
            color: #eef3ff;
            padding: 12px 14px;
            border-top: 1px solid rgba(255,255,255,0.07);
            white-space: nowrap;
            background: rgba(37,43,56,0.76);
            font-weight: 650;
        }
        .static-table tr:nth-child(even) td {
            background: rgba(48,56,74,0.58);
        }
        .static-table tr:hover td {
            background: rgba(200,16,46,0.12);
        }
        .static-table th + th,
        .static-table td + td {
            border-left: 1px solid rgba(255,255,255,0.06);
        }
        .static-table th:first-child,
        .static-table td:first-child {
            border-left: 3px solid rgba(200,16,46,0.70);
            color: #ffffff !important;
            font-weight: 900;
        }
        .static-table-wrap,
        .static-table-wrap * {
            text-shadow: none !important;
        }
        .figure-box {
            background: rgba(37,43,56,0.98);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 8px;
            overflow: hidden;
            margin-bottom: 1rem;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
        }
        .figure-box img {
            display: block;
            width: 100%;
            max-width: 100%;
            height: auto;
            margin: 0 auto;
            image-rendering: auto;
        }
        .figure-caption {
            color: #aab3c4;
            font-size: 0.82rem;
            margin-top: 6px;
            text-align: center;
        }
        .report-shell {
            background: #ffffff;
            border: 1px solid rgba(200,16,46,0.20);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 14px 34px rgba(21,34,59,0.10);
            margin: 8px 0 18px 0;
        }
        .report-head {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
            align-items: center;
            gap: 18px;
            padding: 22px 24px;
            background:
                linear-gradient(90deg, #8f0d23 0%, #c8102e 48%, #15223b 100%);
            color: #ffffff;
        }
        .report-head * {
            color: #ffffff !important;
            text-shadow: none !important;
        }
        .report-team {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
        }
        .report-team.away {
            justify-content: flex-end;
            text-align: right;
        }
        .report-title {
            text-align: center;
            min-width: 180px;
        }
        .report-title h2 {
            margin: 0;
            font-size: 1.15rem;
            letter-spacing: 0;
        }
        .report-title p,
        .report-team-label {
            margin: 4px 0 0 0;
            font-size: 0.78rem;
            opacity: 0.86;
            font-weight: 700;
        }
        .report-team-name {
            font-weight: 900;
            font-size: 1.08rem;
            line-height: 1.1;
        }
        .report-body {
            padding: 20px 22px 22px 22px;
        }
        .report-kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 18px;
        }
        .report-kpi {
            border: 1px solid #e5e7eb;
            border-top: 4px solid #c8102e;
            border-radius: 8px;
            padding: 10px 11px;
            background: #fffdfd;
            min-height: 82px;
        }
        .report-kpi span {
            display: block;
            color: #6b7280 !important;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .report-kpi strong {
            display: block;
            color: #111827 !important;
            font-size: 1.38rem;
            margin-top: 8px;
            line-height: 1;
        }
        .report-section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 18px 0 10px 0;
            color: #15223b;
            font-weight: 900;
            font-size: 1.05rem;
        }
        .report-section-title:before {
            content: "";
            width: 7px;
            height: 22px;
            background: #c8102e;
            border-radius: 2px;
            display: inline-block;
        }
        .report-callouts {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 6px 0 16px 0;
        }
        .report-callout {
            background: #f7f4f5;
            border-left: 4px solid #c8102e;
            border-radius: 8px;
            padding: 12px;
        }
        .report-callout b {
            display: block;
            color: #15223b !important;
            margin-bottom: 4px;
        }
        .report-callout span {
            color: #374151 !important;
            font-size: 0.9rem;
        }
        .report-table-wrap {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow-x: auto;
            background: #ffffff;
            margin-bottom: 14px;
        }
        .report-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.84rem;
            table-layout: auto;
        }
        .report-table th {
            background: #15223b;
            color: #ffffff !important;
            padding: 9px 10px;
            text-align: left;
            white-space: nowrap;
        }
        .report-table td {
            padding: 9px 10px;
            border-top: 1px solid #edf0f4;
            color: #111827 !important;
            vertical-align: top;
        }
        .report-table tr:nth-child(even) td {
            background: #fbf8f9;
        }
        .report-compact-text {
            color: #374151 !important;
            font-size: 0.94rem;
            line-height: 1.45;
            margin: 0 0 12px 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 2px solid rgba(200,16,46,0.54);
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(37,43,56,0.98);
            color: #dbe3f3;
            border: 1px solid rgba(255,255,255,0.06);
            border-bottom: 0;
            border-radius: 6px 6px 0 0;
            padding: 10px 16px;
        }
        .stTabs [data-baseweb="tab"]:nth-of-type(4) {
            border: 2px solid rgba(255,255,255,0.88) !important;
            border-bottom: 0 !important;
            box-shadow: 0 0 0 2px rgba(200,16,46,0.78), 0 0 18px rgba(200,16,46,0.34);
        }
        .stTabs [data-baseweb="tab"] p,
        .stTabs [data-baseweb="tab"] span {
            color: #dbe3f3 !important;
            text-shadow: none !important;
            font-weight: 750;
        }
        .stTabs [aria-selected="true"] {
            background: #c8102e !important;
            color: white !important;
        }
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] span {
            color: #ffffff !important;
        }
        div[data-testid="stSegmentedControl"] {
            margin: 12px 0 24px 0;
            border-bottom: 2px solid rgba(200,16,46,0.54);
            border-top: 0 !important;
            border-left: 0 !important;
            border-right: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] {
            gap: 6px;
            flex-wrap: wrap;
            background: transparent !important;
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] > *,
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] [role="radio"],
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"] {
            background: rgba(37,43,56,0.98) !important;
            background-color: rgba(37,43,56,0.98) !important;
            color: #dbe3f3 !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-bottom: 0 !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 10px 16px !important;
            font-weight: 800 !important;
            min-height: 46px !important;
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] > * *,
        div[data-testid="stSegmentedControl"] button *,
        div[data-testid="stSegmentedControl"] label *,
        div[data-testid="stSegmentedControl"] [role="radio"] *,
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"] * {
            background: transparent !important;
            background-color: transparent !important;
            color: #dbe3f3 !important;
            text-shadow: none !important;
            font-weight: 800 !important;
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] p,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] span {
            color: #dbe3f3 !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] > *:has([aria-checked="true"]),
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] > *:has([data-checked="true"]),
        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[data-checked="true"],
        div[data-testid="stSegmentedControl"] label[aria-checked="true"],
        div[data-testid="stSegmentedControl"] label[data-checked="true"],
        div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"],
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"][aria-checked="true"],
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"][data-checked="true"] {
            background: #c8102e !important;
            background-color: #c8102e !important;
            color: #ffffff !important;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] > *:has([aria-checked="true"]) *,
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] > *:has([data-checked="true"]) *,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] *,
        div[data-testid="stSegmentedControl"] button[data-checked="true"] *,
        div[data-testid="stSegmentedControl"] label[aria-checked="true"] *,
        div[data-testid="stSegmentedControl"] label[data-checked="true"] *,
        div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] *,
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"][aria-checked="true"] *,
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"][data-checked="true"] * {
            color: #ffffff !important;
        }
        div[data-testid="stRadio"] {
            margin: 12px 0 24px 0;
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            width: fit-content !important;
            max-width: 100%;
        }
        div[data-testid="stRadio"] [data-testid="stWidgetLabel"],
        div[data-testid="stRadio"] > label {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            border: 0 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: inline-flex;
            flex-wrap: nowrap;
            gap: 0;
            align-items: stretch;
            width: fit-content;
            max-width: 100%;
            border: 0 !important;
            border-bottom: 2px solid rgba(200,16,46,0.70) !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        div[data-testid="stRadio"] > div,
        div[data-testid="stRadio"] > div > div,
        div[data-testid="stRadio"] div[role="radiogroup"] > div {
            border: 0 !important;
            outline: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
        }
        div[data-testid="stRadio"] label {
            min-height: 38px !important;
            padding: 9px 18px !important;
            margin: 0 !important;
            background: rgba(37,43,56,0.98) !important;
            background-color: rgba(37,43,56,0.98) !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-bottom: 0 !important;
            border-radius: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[data-testid="stRadio"] label:first-of-type {
            border-radius: 6px 0 0 0 !important;
        }
        div[data-testid="stRadio"] label:last-of-type {
            border-radius: 0 6px 0 0 !important;
        }
        div[data-testid="stRadio"] label > div:first-child {
            display: none !important;
        }
        div[data-testid="stRadio"] label *,
        div[data-testid="stRadio"] p,
        div[data-testid="stRadio"] span {
            color: #dbe3f3 !important;
            font-weight: 800 !important;
            background: transparent !important;
            background-color: transparent !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }
        div[data-testid="stRadio"] label:has(input:checked),
        div[data-testid="stRadio"] label:has([aria-checked="true"]),
        div[data-testid="stRadio"] label[data-checked="true"] {
            background: #c8102e !important;
            background-color: #c8102e !important;
            border-color: rgba(200,16,46,0.95) !important;
            box-shadow: none !important;
        }
        div[data-testid="stRadio"] label:has(input:checked) *,
        div[data-testid="stRadio"] label:has([aria-checked="true"]) *,
        div[data-testid="stRadio"] label[data-checked="true"] * {
            color: #ffffff !important;
        }
        .critical-focus-banner {
            background: rgba(37,43,56,0.98);
            border: 2px solid rgba(255,255,255,0.88);
            border-left: 8px solid var(--osasuna-red);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 2px 0 16px 0;
            box-shadow: 0 0 0 2px rgba(200,16,46,0.70), 0 14px 28px rgba(0,0,0,0.22);
        }
        .critical-focus-banner strong {
            display: block;
            color: #ffffff !important;
            font-size: 1rem;
            margin-bottom: 3px;
        }
        .critical-focus-banner span {
            color: #dbe3f3 !important;
            font-size: 0.9rem;
        }
        .bar-legend {
            margin: 8px 12px 22px 12px;
        }
        .bar-legend-title {
            color: #ffffff;
            font-size: 0.72rem;
            margin-bottom: 7px;
            text-align: center;
        }
        .bar-legend-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            column-gap: 34px;
            row-gap: 8px;
            max-width: 100%;
            margin: 0 auto;
            overflow: hidden;
        }
        .bar-legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
            max-width: 100%;
            white-space: nowrap;
            color: #ffffff;
            font-size: 0.63rem;
            line-height: 1.15;
            overflow: hidden;
        }
        .bar-legend-swatch {
            width: 11px;
            height: 11px;
            display: inline-block;
            flex: 0 0 11px;
        }
        .bar-legend-label {
            color: #ffffff !important;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .tipology-assignment {
            display: grid;
            gap: 10px;
            background: rgba(40,48,64,0.78);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            padding: 14px 16px;
            margin: 10px 0 20px 0;
        }
        .tipology-row {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
            font-weight: 750;
        }
        .tipology-swatch {
            width: 14px;
            height: 14px;
            flex: 0 0 14px;
            border-radius: 3px;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.22);
        }
        @media (max-width: 900px) {
            .login-wrap,
            .story-grid {
                grid-template-columns: 1fr;
            }
            .bar-legend-grid {
                grid-template-columns: 1fr;
            }
            .portal-hero h1,
            .login-visual h1 {
                font-size: 1.65rem;
            }
            .match-grid {
                grid-template-columns: 1fr;
            }
            .report-head,
            .report-kpi-grid,
            .report-callouts {
                grid-template-columns: 1fr;
            }
            .report-team.away {
                justify-content: flex-start;
                text-align: left;
            }
            .match-team.away {
                justify-content: flex-start;
                text-align: left;
            }
            .score-pill {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_sidebar_toggle():
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          const existing = doc.getElementById("tfg-sidebar-toggle");
          if (existing) existing.remove();
          const btn = doc.createElement("button");
          btn.id = "tfg-sidebar-toggle";
          btn.type = "button";
          btn.title = "Mostrar/ocultar barra lateral";
          btn.style.cssText = [
            "position:absolute",
            "left:10px",
            "top:104px",
            "z-index:2147483647",
            "width:38px",
            "height:38px",
            "border-radius:999px",
            "border:2px solid #c8102e",
            "background:#eef2f7",
            "color:#202532",
            "font:900 22px Arial,sans-serif",
            "line-height:30px",
            "cursor:pointer",
            "box-shadow:0 10px 24px rgba(0,0,0,0.28)"
          ].join(";");

          function sidebarOpen() {
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            return !!sidebar && sidebar.getBoundingClientRect().width > 80;
          }

          function refresh() {
            btn.textContent = sidebarOpen() ? "â€¹" : "â€º";
          }

          function findNativeToggle() {
            const selectors = [
              '[data-testid="stSidebarCollapseButton"] button',
              'button[data-testid="stSidebarCollapseButton"]',
              '[data-testid="stSidebarCollapsedControl"] button',
              'button[data-testid="stSidebarCollapsedControl"]',
              'button[aria-label*="sidebar" i]',
              'button[title*="sidebar" i]',
              'button[aria-label*="menu" i]'
            ];
            for (const selector of selectors) {
              const el = doc.querySelector(selector);
              if (el) return el;
            }
            return null;
          }

          btn.addEventListener("click", () => {
            const native = findNativeToggle();
            if (native) native.click();
            setTimeout(refresh, 260);
          });

          doc.body.appendChild(btn);
          refresh();
          setInterval(refresh, 700);
        })();
        </script>
        """,
        height=0,
        width=0,
    )
def inject_sidebar_toggle_v2():
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          const existing = doc.getElementById("tfg-sidebar-toggle");
          if (existing) existing.remove();
          const oldStyle = doc.getElementById("tfg-sidebar-toggle-style");
          if (oldStyle) oldStyle.remove();

          const style = doc.createElement("style");
          style.id = "tfg-sidebar-toggle-style";
          style.textContent = `
            #tfg-sidebar-toggle {
              position: absolute !important;
              top: 22px !important;
              left: 16px !important;
              z-index: 2147483647 !important;
              width: 46px !important;
              height: 42px !important;
              border: 0 !important;
              background: transparent !important;
              display: flex !important;
              align-items: center !important;
              justify-content: center !important;
              gap: 3px !important;
              cursor: pointer !important;
              padding: 0 !important;
              filter: drop-shadow(0 2px 7px rgba(0,0,0,0.62)) !important;
              appearance: none !important;
            }
            body.tfg-sidebar-forced-closed section[data-testid="stSidebar"] {
              width: 0 !important;
              min-width: 0 !important;
              max-width: 0 !important;
              flex-basis: 0 !important;
              transform: translateX(-110%) !important;
              opacity: 0 !important;
              overflow: hidden !important;
              border-right: 0 !important;
              padding: 0 !important;
              pointer-events: none !important;
            }
            body.tfg-sidebar-forced-closed section[data-testid="stSidebar"] * {
              visibility: hidden !important;
              pointer-events: none !important;
            }
          `;
          doc.head.appendChild(style);

          const btn = doc.createElement("button");
          btn.id = "tfg-sidebar-toggle";
          btn.type = "button";
          btn.title = "Mostrar/ocultar barra lateral";

          function paintArrows(open) {
            btn.innerHTML = "";
            for (let i = 0; i < 2; i++) {
              const arrow = doc.createElement("span");
              arrow.style.cssText = [
                "display:block",
                "width:16px",
                "height:16px",
                "border-top:5px solid rgba(255,255,255,.96)",
                "border-left:5px solid rgba(255,255,255,.96)",
                "transform:" + (open ? "rotate(-45deg)" : "rotate(135deg)"),
                "border-radius:1px"
              ].join(";");
              btn.appendChild(arrow);
            }
          }

          function sidebarOpen() {
            if (doc.body.classList.contains("tfg-sidebar-forced-closed")) return false;
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            if (!sidebar) return false;
            const rect = sidebar.getBoundingClientRect();
            return rect.width > 160 && rect.right > 180;
          }

          function refresh() {
            const open = sidebarOpen();
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            const pageX = window.parent.scrollX || doc.documentElement.scrollLeft || 0;
            paintArrows(open);
            if (open && sidebar) {
              const rect = sidebar.getBoundingClientRect();
              btn.style.setProperty("left", Math.max(18, pageX + rect.right + 14) + "px", "important");
            } else {
              btn.style.setProperty("left", (pageX + 16) + "px", "important");
            }
            btn.style.setProperty("top", "22px", "important");
            btn.style.setProperty("position", "absolute", "important");
            hideNativeCloseButtons();
          }

          function findNativeToggle() {
            const selectors = [
              '[data-testid="stSidebarCollapseButton"] button',
              'button[data-testid="stSidebarCollapseButton"]',
              '[data-testid="stSidebarCollapsedControl"] button',
              'button[data-testid="stSidebarCollapsedControl"]',
              'button[aria-label*="sidebar" i]',
              'button[title*="sidebar" i]',
              'button[aria-label*="menu" i]'
            ];
            for (const selector of selectors) {
              const el = doc.querySelector(selector);
              if (el) return el;
            }
            return null;
          }

          btn.addEventListener("click", () => {
            if (sidebarOpen()) {
              doc.body.classList.add("tfg-sidebar-forced-closed");
            } else {
              doc.body.classList.remove("tfg-sidebar-forced-closed");
            }
            setTimeout(refresh, 60);
            setTimeout(refresh, 260);
            setTimeout(refresh, 650);
          });

          function hideNativeCloseButtons() {
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            const buttonRoots = sidebar ? [sidebar, doc] : [doc];
            buttonRoots.forEach((root) => root.querySelectorAll('button, [role="button"]').forEach((button) => {
              if (button.id === "tfg-sidebar-toggle") return;
              const rect = button.getBoundingClientRect();
              const text = (button.textContent || "").trim();
              const aria = (button.getAttribute("aria-label") || "").toLowerCase();
              const title = (button.getAttribute("title") || "").toLowerCase();
              const inSidebarZone = rect.left < 420 && rect.top < 160;
              const smallTopButton = inSidebarZone && rect.width <= 82 && rect.height <= 82;
              const looksClose = text === "×" || text === "x" || text === "✕" || text === "✖" || aria.includes("close") || aria.includes("collapse") || title.includes("close") || title.includes("collapse");
              if (smallTopButton || (inSidebarZone && looksClose)) {
                button.style.setProperty("display", "none", "important");
                button.style.setProperty("visibility", "hidden", "important");
                button.style.setProperty("pointer-events", "none", "important");
              }
            }));
            doc.querySelectorAll('button, [role="button"], svg, span, div').forEach((el) => {
              if (el.id === "tfg-sidebar-toggle" || el.closest("#tfg-sidebar-toggle")) return;
              const rect = el.getBoundingClientRect();
              if (rect.left > 420 || rect.top > 160 || rect.width > 90 || rect.height > 90 || rect.width < 8 || rect.height < 8) return;
              const text = (el.textContent || "").trim().toLowerCase();
              const aria = (el.getAttribute("aria-label") || "").toLowerCase();
              const title = (el.getAttribute("title") || "").toLowerCase();
              const isCloseMark = text === "x" || text === "×" || text === "✕" || text === "✖" || aria.includes("close") || aria.includes("collapse") || title.includes("close") || title.includes("collapse");
              const hasCloseSvg = el.tagName.toLowerCase() === "svg" && rect.top < 120;
              if (isCloseMark || hasCloseSvg) {
                const target = el.closest('button, [role="button"]') || el;
                target.style.setProperty("display", "none", "important");
                target.style.setProperty("visibility", "hidden", "important");
                target.style.setProperty("pointer-events", "none", "important");
              }
            });
          }

          function cleanMenuAndExpanders() {
            doc.querySelectorAll('[data-testid="stRadio"]').forEach((radio) => {
              let el = radio;
              for (let i = 0; i < 5 && el; i++) {
                el.style.setProperty("border-top", "0", "important");
                el.style.setProperty("box-shadow", "none", "important");
                el.style.setProperty("outline", "0", "important");
                el = el.parentElement;
              }
            });
            doc.querySelectorAll('[data-testid="stSegmentedControl"]').forEach((menu) => {
              let el = menu;
              for (let i = 0; i < 14 && el; i++) {
                el.style.setProperty("border-top", "0", "important");
                el.style.setProperty("border-left", "0", "important");
                el.style.setProperty("border-right", "0", "important");
                if (i > 0) el.style.setProperty("border", "0", "important");
                el.style.setProperty("box-shadow", "none", "important");
                el.style.setProperty("outline", "0", "important");
                if (i > 0) {
                  el.style.setProperty("background", "transparent", "important");
                  el.style.setProperty("background-color", "transparent", "important");
                }
                el = el.parentElement;
              }
            });
            doc.querySelectorAll('[data-testid="stExpander"] summary').forEach((summary) => {
              summary.style.setProperty("list-style", "none", "important");
            });
            doc.querySelectorAll('[data-testid="stExpander"] summary svg, [data-testid="stExpander"] summary [data-testid="stIconMaterial"], [data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"], [data-testid="stExpander"] summary > div:last-child').forEach((el) => {
              el.style.setProperty("display", "none", "important");
              el.style.setProperty("background", "transparent", "important");
              el.style.setProperty("width", "0", "important");
              el.style.setProperty("min-width", "0", "important");
              el.style.setProperty("padding", "0", "important");
            });
          }

          doc.body.appendChild(btn);
          refresh();
          window.addEventListener("resize", refresh);
          setInterval(refresh, 700);
          setInterval(hideNativeCloseButtons, 500);
          setInterval(cleanMenuAndExpanders, 500);
          cleanMenuAndExpanders();
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def remove_sidebar_toggle_artifact():
    components.html(
        """
        <script>
        (() => {
          const btn = window.parent.document.getElementById("tfg-sidebar-toggle");
          if (btn) btn.remove();
        })();
        </script>
        """,
        height=0,
        width=0,
    )


@st.cache_data(show_spinner=False)
def _available_app_data() -> pd.DataFrame:
    return listar_app_data_disponible(_paths())


@st.cache_data(show_spinner=False)
def _available_tracking() -> pd.DataFrame:
    return listar_tracking_disponible(_paths())


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, mtime: float) -> dict:
    _ = mtime
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    _ = mtime
    try:
        df = pd.read_csv(path_str)
    except EmptyDataError:
        return pd.DataFrame()
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


@st.cache_data(show_spinner=False)
def _read_text_cached(path_str: str, mtime: float) -> str:
    _ = mtime
    return Path(path_str).read_text(encoding="utf-8")


def _load_metadata(match_id: int) -> dict:
    path = _paths().outputs_dir / "app_data" / str(int(match_id)) / "metadata.json"
    return _read_json_cached(str(path), _path_mtime(path))


def _load_table(match_id: int, filename: str) -> pd.DataFrame:
    path = _paths().outputs_dir / "app_data" / str(int(match_id)) / filename
    if not path.exists():
        return pd.DataFrame()
    return _clean_text_columns(_read_csv_cached(str(path), _path_mtime(path)))


def _load_bepro_video_index() -> pd.DataFrame:
    path = _paths().outputs_dir / "app_data" / "bepro_videos.csv"
    if not path.exists():
        return pd.DataFrame()
    return _clean_text_columns(_read_csv_cached(str(path), _path_mtime(path)))


def _load_text(match_id: int, filename: str) -> str:
    path = _paths().outputs_dir / "app_data" / str(int(match_id)) / filename
    if not path.exists():
        return ""
    return _read_text_cached(str(path), _path_mtime(path))


@st.cache_data(show_spinner=False)
def _match_date_label(match_id: int) -> str:
    path = ROOT / "eventing_partidos" / f"partido_{int(match_id)}_eventing.csv"
    if not path.exists():
        return "Fecha no disponible"
    try:
        sample = pd.read_csv(path, usecols=["match_start_time"], nrows=1)
    except (OSError, ValueError, EmptyDataError):
        return "Fecha no disponible"
    if sample.empty or "match_start_time" not in sample.columns:
        return "Fecha no disponible"
    value = sample["match_start_time"].iloc[0]
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return "Fecha no disponible"
    return dt.strftime("%d/%m/%Y")


@st.cache_data(show_spinner=False)
def _infer_score(match_id: int, local_team_id: int | None, rival_team_id: int | None) -> str:
    path = ROOT / "eventing_partidos" / f"partido_{int(match_id)}_eventing.csv"
    if not path.exists():
        return "vs"

    try:
        events = pd.read_csv(path, usecols=["team_id", "events"], low_memory=False)
    except (OSError, ValueError):
        return "vs"

    def _clean_team(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

    local_id = _clean_team(local_team_id) or TEAM_ID_DEFAULT
    rival_id = _clean_team(rival_team_id)
    score = {local_id: 0}
    if rival_id is not None:
        score[rival_id] = 0
    known_ids = {tid for tid in [local_id, rival_id] if tid is not None}

    for _, row in events.iterrows():
        team_id = _clean_team(row.get("team_id"))
        if team_id is None or (known_ids and team_id not in known_ids):
            continue
        try:
            parsed_events = json.loads(row.get("events") or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        for event in parsed_events:
            event_type = str(event.get("event_type", "")).lower()
            prop = event.get("property") or {}
            outcome = str(prop.get("outcome", "")).lower()
            if event_type == "shot" and outcome in {"goal", "shotgoal"}:
                score[team_id] = score.get(team_id, 0) + 1
            elif event_type == "owngoal":
                other_ids = [tid for tid in known_ids if tid != team_id]
                if other_ids:
                    score[other_ids[0]] = score.get(other_ids[0], 0) + 1

    local_score = score.get(local_id, 0)
    rival_score = score.get(rival_id, 0) if rival_id is not None else 0
    return f"{local_score} - {rival_score}"


def _score_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "vs"}:
        return ""
    return text.replace(" vs ", " - ")


def _asset(match_id: int, filename: str) -> Path:
    return _paths().outputs_dir / "app_data" / str(int(match_id)) / filename


def _anonymous_rival_number(team_id) -> int:
    try:
        clean_id = int(team_id)
    except (TypeError, ValueError):
        return 1
    return ANON_RIVAL_ORDER.get(clean_id, max(ANON_RIVAL_ORDER.values(), default=0) + 1)


def _anonymous_team_name(team_id=None, fallback: str = "Equipo") -> str:
    if not ANONYMIZE_TEAM_IDENTITIES:
        return fallback
    try:
        clean_id = int(team_id)
    except (TypeError, ValueError):
        clean_id = None
    if clean_id == TEAM_ID_DEFAULT:
        return ANON_TEAM_NAME
    if clean_id is not None:
        return f"{ANON_RIVAL_PREFIX} {_anonymous_rival_number(clean_id)}"
    fallback_text = str(_clean_app_text(fallback) or fallback)
    generic_labels = {
        "",
        "-",
        "equipo",
        "equipo local",
        "equipo visitante",
        "equipo rival",
        "rival",
        ANON_TEAM_NAME.lower(),
    }
    if fallback_text.strip().lower() in generic_labels:
        return fallback_text or "Equipo"
    return ANON_RIVAL_PREFIX


def _anonymous_badge_path(team_id=None) -> Path | None:
    if not ANONYMIZE_TEAM_IDENTITIES:
        return None
    try:
        clean_id = int(team_id)
    except (TypeError, ValueError):
        clean_id = None
    if clean_id == TEAM_ID_DEFAULT:
        filename = "team.png"
    else:
        filename = f"rival_{_anonymous_rival_number(clean_id)}.png"
    path = ROOT / "assets" / "anonymous_badges" / filename
    if not path.exists() and clean_id != TEAM_ID_DEFAULT:
        path = ROOT / "assets" / "anonymous_badges" / "rival_1.png"
    return path if path.exists() else None


def _team_display_name(team: dict | None, fallback: str = "Equipo") -> str:
    if not team:
        return fallback
    return _anonymous_team_name(team.get("team_id"), str(team.get("name", fallback)))


def _team_logo_html(team_id, fallback: str, css_class: str = "team-badge") -> str:
    anonymous = _anonymous_badge_path(team_id)
    if anonymous is not None:
        data = base64.b64encode(anonymous.read_bytes()).decode("ascii")
        return f'<span class="{css_class}"><img src="data:image/png;base64,{data}" alt="escudo anonimo"></span>'
    candidates = []
    if team_id is not None:
        candidates.extend(
            [
                ROOT / "assets" / "team_logos" / f"{team_id}.png",
                ROOT / "assets" / "team_logos" / f"{team_id}.jpg",
                ROOT / "assets" / "team_logos" / f"{team_id}.webp",
            ]
        )
    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".png":
                mime = "image/png"
            elif path.suffix.lower() == ".webp":
                mime = "image/webp"
            else:
                mime = "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'<span class="{css_class}"><img src="data:{mime};base64,{data}" alt="escudo"></span>'
    return f'<span class="{css_class}">{fallback}</span>'


def _initials(name: str, fallback: str = "EQ") -> str:
    initials = "".join(part[0] for part in str(name or "").split()[:3]).upper()
    return initials or fallback


def _team_name_for_id(meta: dict, team_id, fallback: str = "Equipo") -> str:
    try:
        clean_id = int(team_id)
    except (TypeError, ValueError):
        return fallback
    if ANONYMIZE_TEAM_IDENTITIES:
        return _anonymous_team_name(clean_id, fallback)
    if clean_id == int(meta.get("team_id", TEAM_ID_DEFAULT)):
        return str(meta.get("team_name", fallback))
    if clean_id == int(meta.get("rival_team_id", -1)):
        return str(meta.get("rival_name", fallback))
    return fallback


def _team_badge_for_id(meta: dict, team_id, fallback_name: str) -> str:
    return _team_logo_html(team_id, _initials(fallback_name))


def _cluster_color(cluster) -> str:
    try:
        return APP_COLOR_MAP.get(f"T{int(float(cluster))}", APP_COLOR_SEQUENCE[int(float(cluster)) % len(APP_COLOR_SEQUENCE)])
    except (TypeError, ValueError, OverflowError):
        return APP_COLOR_SEQUENCE[0]


def _presentation_teams(match_id: int, meta: dict) -> dict:
    override = MATCH_PRESENTATION.get(int(match_id), {})
    home_id = override.get("home_team_id", meta.get("team_id", TEAM_ID_DEFAULT))
    away_id = override.get("away_team_id", meta.get("rival_team_id"))
    home_name = _team_name_for_id(meta, home_id, "Equipo local")
    away_name = _team_name_for_id(meta, away_id, "Equipo visitante")
    return {
        "home_id": home_id,
        "away_id": away_id,
        "home_name": home_name,
        "away_name": away_name,
        "home_logo": _team_badge_for_id(meta, home_id, home_name),
        "away_logo": _team_badge_for_id(meta, away_id, away_name),
    }


def _registered_teams() -> list[dict]:
    app_data = _available_app_data()
    teams: dict[int, dict] = {
        int(team_id): {"team_id": int(team_id), "matches": 0, **data}
        for team_id, data in REGISTERED_TEAMS.items()
    }
    if not app_data.empty and "match_id" in app_data.columns:
        for match_id in app_data["match_id"].dropna().astype(int).tolist():
            try:
                meta = _load_metadata(int(match_id))
            except Exception:
                continue
            team_id = int(meta.get("team_id", TEAM_ID_DEFAULT))
            base = teams.get(team_id, {})
            teams[team_id] = {
                "team_id": team_id,
                "name": _anonymous_team_name(team_id, str(meta.get("team_name", base.get("name", f"Equipo {team_id}")))),
                "short_name": _anonymous_team_name(team_id, str(base.get("short_name", meta.get("team_name", f"Equipo {team_id}")))),
                "location": str(base.get("location", "Sede no especificada")),
                "role": str(base.get("role", "Equipo registrado")),
                "description": str(
                    base.get(
                        "description",
                        "Equipo con datos de tracking y eventing registrados en la plataforma.",
                    )
                ),
                "status": str(base.get("status", "Activo")),
                "access_code": str(base.get("access_code", "")),
                "matches": int(base.get("matches", 0)) + 1,
            }
    for team in teams.values():
        team["name"] = _anonymous_team_name(team.get("team_id"), str(team.get("name", "Equipo")))
        team["short_name"] = _anonymous_team_name(team.get("team_id"), str(team.get("short_name", team.get("name", "Equipo"))))
    return sorted(teams.values(), key=lambda item: str(item.get("name", "")).lower())


def _team_by_id(team_id: int | None) -> dict:
    teams = _registered_teams()
    if not teams:
        return {"team_id": TEAM_ID_DEFAULT, **REGISTERED_TEAMS[TEAM_ID_DEFAULT], "matches": 0}
    try:
        clean_id = int(team_id)
    except (TypeError, ValueError):
        clean_id = int(teams[0]["team_id"])
    for team in teams:
        if int(team.get("team_id", -1)) == clean_id:
            return team
    return teams[0]


def _team_matches(team_id: int) -> pd.DataFrame:
    app_data = _available_app_data()
    if app_data.empty or "match_id" not in app_data.columns:
        return pd.DataFrame(columns=["match_id", "label", "rival", "fecha", "score", "secuencias", "modified"])
    rows = []
    for _, row in app_data.iterrows():
        match_id = int(row.get("match_id"))
        try:
            meta = _load_metadata(match_id)
        except Exception:
            continue
        if int(meta.get("team_id", TEAM_ID_DEFAULT)) != int(team_id):
            continue
        teams = _presentation_teams(match_id, meta)
        score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(
            match_id,
            teams["home_id"],
            teams["away_id"],
        )
        fecha = _match_date_label(match_id)
        rival = _team_name_for_id(meta, meta.get("rival_team_id"), str(row.get("rival", "Rival")))
        secuencias = meta.get("resumen", {}).get("n_secuencias_rivales", "-")
        rows.append(
            {
                "match_id": match_id,
                "label": f"{fecha} | {rival} | Partido {match_id}",
                "rival": rival,
                "fecha": fecha,
                "score": score or "vs",
                "secuencias": secuencias,
                "modified": row.get("modified"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["modified", "match_id"], ascending=[False, False]).reset_index(drop=True)


def _match_label_map(matches: pd.DataFrame) -> dict[str, int]:
    if matches.empty:
        return {}
    labels: dict[str, int] = {}
    for _, row in matches.iterrows():
        label = str(row.get("label", row.get("match_id")))
        labels[label] = int(row.get("match_id"))
    return labels


def _set_app_view(view: str, team_id: int | None = None, match_id: int | None = None):
    st.session_state["app_view"] = view
    if team_id is not None:
        st.session_state["selected_team_id"] = int(team_id)
    if match_id is not None:
        st.session_state["selected_match_id"] = int(match_id)
    st.rerun()


def _logout():
    st.session_state.clear()
    st.rerun()


def _secret_value(*names: str) -> str:
    for name in names:
        try:
            value = st.secrets[name]
        except (KeyError, FileNotFoundError):
            value = None
        if value:
            return str(value).strip()
    try:
        firebase = st.secrets.get("firebase", {})
    except (AttributeError, FileNotFoundError):
        firebase = {}
    for name in names:
        value = firebase.get(name) if hasattr(firebase, "get") else None
        if value:
            return str(value).strip()
    return ""


def _firebase_config() -> dict:
    return {
        "api_key": _secret_value("FIREBASE_API_KEY", "api_key"),
        "project_id": _secret_value("FIREBASE_PROJECT_ID", "project_id"),
    }


def _firebase_enabled() -> bool:
    cfg = _firebase_config()
    return bool(cfg["api_key"] and cfg["project_id"])


def _auth_url(action: str) -> str:
    return f"https://identitytoolkit.googleapis.com/v1/accounts:{action}?key={_firebase_config()['api_key']}"


def _firestore_url(path: str) -> str:
    project_id = _firebase_config()["project_id"]
    clean_path = path.strip("/")
    return f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{clean_path}"


def _firebase_error(response: requests.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message", "")
    except ValueError:
        message = response.text
    friendly = {
        "EMAIL_EXISTS": "Ese correo ya esta registrado.",
        "EMAIL_NOT_FOUND": "No existe ninguna cuenta con ese correo.",
        "INVALID_LOGIN_CREDENTIALS": "Usuario o contrasena incorrectos.",
        "INVALID_PASSWORD": "Usuario o contrasena incorrectos.",
        "WEAK_PASSWORD : Password should be at least 6 characters": "La contrasena debe tener al menos 6 caracteres.",
    }
    return friendly.get(str(message), str(message) or "Firebase no ha aceptado la operacion.")


def _firestore_value(value):
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if value is None:
        return {"nullValue": None}
    return {"stringValue": str(value)}


def _firestore_fields(data: dict) -> dict:
    return {key: _firestore_value(value) for key, value in data.items()}


def _parse_firestore_value(value: dict):
    if "stringValue" in value:
        return value.get("stringValue", "")
    if "integerValue" in value:
        try:
            return int(value.get("integerValue", 0))
        except (TypeError, ValueError):
            return value.get("integerValue")
    if "doubleValue" in value:
        return float(value.get("doubleValue", 0))
    if "booleanValue" in value:
        return bool(value.get("booleanValue"))
    if "timestampValue" in value:
        return value.get("timestampValue")
    return None


def _parse_firestore_doc(doc: dict) -> dict:
    fields = doc.get("fields", {})
    out = {key: _parse_firestore_value(value) for key, value in fields.items()}
    name = str(doc.get("name", ""))
    out["_doc_id"] = name.rsplit("/", 1)[-1] if name else ""
    return out


def _firebase_auth(action: str, email: str, password: str) -> dict:
    response = requests.post(
        _auth_url(action),
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(_firebase_error(response))
    return response.json()


def _firestore_headers(id_token: str) -> dict:
    return {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}


def _save_user_profile(uid: str, id_token: str, profile: dict):
    response = requests.patch(
        _firestore_url(f"{USERS_COLLECTION}/{uid}"),
        headers=_firestore_headers(id_token),
        json={"fields": _firestore_fields(profile)},
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(_firebase_error(response))


def _load_user_profile(uid: str, id_token: str, email: str) -> dict:
    response = requests.get(
        _firestore_url(f"{USERS_COLLECTION}/{uid}"),
        headers=_firestore_headers(id_token),
        timeout=15,
    )
    if response.ok:
        profile = _parse_firestore_doc(response.json())
        if profile.get("email"):
            return profile
    if response.status_code not in (403, 404):
        raise RuntimeError(_firebase_error(response))

    query = {
        "structuredQuery": {
            "from": [{"collectionId": USERS_COLLECTION}],
            "where": {
                "fieldFilter": {
                    "field": {"fieldPath": "email"},
                    "op": "EQUAL",
                    "value": {"stringValue": email},
                }
            },
            "limit": 1,
        }
    }
    response = requests.post(
        _firestore_url(":runQuery"),
        headers=_firestore_headers(id_token),
        json=query,
        timeout=15,
    )
    if response.ok:
        for row in response.json():
            if row.get("document"):
                profile = _parse_firestore_doc(row["document"])
                if profile.get("_doc_id") != uid:
                    try:
                        _save_user_profile(uid, id_token, {k: v for k, v in profile.items() if not k.startswith("_")})
                    except RuntimeError:
                        pass
                return profile
    raise RuntimeError("La cuenta existe en Authentication, pero no tiene perfil en Firestore.")


def _set_authenticated_session(auth_data: dict, profile: dict):
    role = str(profile.get("rol", "guest")).strip().lower()
    if role not in USER_ROLES:
        role = "guest"
    st.session_state["logged_in"] = True
    st.session_state["access_mode"] = "account"
    st.session_state["login_user"] = profile.get("email") or auth_data.get("email", "")
    st.session_state["firebase_uid"] = auth_data.get("localId", "")
    st.session_state["firebase_id_token"] = auth_data.get("idToken", "")
    st.session_state["user_profile"] = {**profile, "rol": role}
    st.session_state["app_view"] = "portal"


def _register_guest(email: str, password: str):
    auth_data = _firebase_auth("signUp", email, password)
    profile = {"email": email, "rol": "guest", "equipo": ""}
    _save_user_profile(str(auth_data["localId"]), str(auth_data["idToken"]), profile)
    _set_authenticated_session(auth_data, profile)


def _login_firebase(email: str, password: str):
    auth_data = _firebase_auth("signInWithPassword", email, password)
    profile = _load_user_profile(str(auth_data["localId"]), str(auth_data["idToken"]), email)
    _set_authenticated_session(auth_data, profile)


def _login_local(user: str, password: str) -> bool:
    if APP_USERS.get(user.strip().lower()) != password:
        return False
    role = "admin" if user.strip().lower() in {"admin", "juan"} else "analista"
    st.session_state["logged_in"] = True
    st.session_state["access_mode"] = "account"
    st.session_state["login_user"] = user.strip()
    st.session_state["user_profile"] = {"email": user.strip(), "rol": role, "equipo": ANON_TEAM_NAME}
    st.session_state["app_view"] = "portal"
    return True


def _current_user_profile() -> dict:
    return dict(st.session_state.get("user_profile") or {"rol": "guest"})


def _current_user_role() -> str:
    role = str(_current_user_profile().get("rol", "guest")).strip().lower()
    return role if role in USER_ROLES else "guest"


def _is_admin() -> bool:
    return _current_user_role() == "admin"


def _is_guest() -> bool:
    return _current_user_role() == "guest"


def _user_allowed_team_ids() -> set[int] | None:
    if _is_admin():
        return None
    if _is_guest():
        return set()
    profile = _current_user_profile()
    assigned = str(profile.get("equipo", "")).strip().lower()
    if not assigned:
        return set()
    allowed: set[int] = set()
    for team in _registered_teams():
        names = {
            str(team.get("team_id", "")).lower(),
            str(team.get("name", "")).lower(),
            str(team.get("short_name", "")).lower(),
            "cd subiza" if int(team.get("team_id", 0)) == TEAM_ID_DEFAULT else "",
            "subiza" if int(team.get("team_id", 0)) == TEAM_ID_DEFAULT else "",
        }
        if assigned in names:
            allowed.add(int(team["team_id"]))
    return allowed


def _can_access_team(team_id: int) -> bool:
    allowed = _user_allowed_team_ids()
    return allowed is None or int(team_id) in allowed


def _list_firestore_users() -> list[dict]:
    id_token = str(st.session_state.get("firebase_id_token", ""))
    if not (_firebase_enabled() and id_token):
        return []
    response = requests.get(
        _firestore_url(USERS_COLLECTION),
        headers=_firestore_headers(id_token),
        timeout=15,
    )
    if response.status_code == 404:
        return []
    if not response.ok:
        raise RuntimeError(_firebase_error(response))
    users = [_parse_firestore_doc(doc) for doc in response.json().get("documents", [])]
    return sorted(users, key=lambda item: str(item.get("email", "")).lower())


def _update_firestore_user(doc_id: str, email: str, role: str, team: str):
    id_token = str(st.session_state.get("firebase_id_token", ""))
    if not (_firebase_enabled() and id_token):
        raise RuntimeError("Conecta Firebase para editar usuarios desde la app.")
    payload = {"email": email, "rol": role, "equipo": team if role == "analista" else ""}
    response = requests.patch(
        _firestore_url(f"{USERS_COLLECTION}/{doc_id}"),
        headers=_firestore_headers(id_token),
        json={"fields": _firestore_fields(payload)},
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(_firebase_error(response))


def _current_user_label() -> tuple[str, str]:
    if st.session_state.get("logged_in"):
        profile = _current_user_profile()
        role = str(profile.get("rol", "guest")).capitalize()
        team = str(profile.get("equipo", "")).strip()
        label = role if not team else f"{role} | {team}"
        return str(st.session_state.get("login_user", "usuario")), label
    return "Invitado", "Modo de consulta"


def _show_image(match_id: int, filename: str, caption: str | None = None, max_width: int | None = None):
    path = _asset(match_id, filename)
    if path.exists():
        if max_width:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            caption_html = f'<div class="figure-caption">{caption}</div>' if caption else ""
            st.markdown(
                f"""
                <div class="figure-box">
                    <img src="data:{mime};base64,{encoded}" style="max-width:{int(max_width)}px;" />
                    {caption_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"No se ha generado {filename}.")


def _global_app_asset(filename: str) -> Path:
    return ROOT / "assets" / filename


def _image_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_xt_reference_panel():
    image_uri = _image_uri(_global_app_asset("xt_grid_reference.png"))
    image_html = f'<img src="{image_uri}" alt="Referencia xThreat" />' if image_uri else ""
    xt_items = xthreat_reference_text()
    if isinstance(xt_items, str):
        xt_items = [
            {"label": "Baja", "range": "< 0.05", "color": "#38c172", "description": "Zonas de bajo valor esperado."},
            {"label": "Moderada", "range": "0.05 - 0.10", "color": "#f4c542", "description": "Progresión con amenaza reconocible."},
            {"label": "Alta", "range": "0.10 - 0.20", "color": "#ff9f43", "description": "Entrada en zonas de peligro relevante."},
            {"label": "Muy peligrosa", "range": ">= 0.20", "color": "#e4143a", "description": "Cercanía al área y alto potencial de ocasión."},
        ]
    levels_html = "".join(
        f"""
        <div class="xt-level-item">
            <b><span class="xt-level-dot" style="background:{html.escape(str(item.get('color', '#6b7280')))};"></span>{html.escape(str(item.get('label', '-')))}</b>
            <span>{html.escape(str(item.get('range', '-')))}</span>
            <span>{html.escape(str(item.get('description', '')))}</span>
        </div>
        """
        for item in xt_items
        if isinstance(item, dict)
    )
    st.markdown(
        f"""
        <div class="xt-reference-box">
            <div>
                <p>Referencia xThreat: la amenaza crece cuanto más cerca se sitúa la acción de zonas de remate. Estos niveles ayudan a leer la escala de la malla sin convertir el xT en un índice propio.</p>
                <div class="xt-level-grid">{levels_html}</div>
            </div>
            {image_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_table(df: pd.DataFrame, height: int | None = 300):
    if df.empty:
        st.info("No hay datos para esta tabla.")
        return
    row_height = 35
    header_height = 38
    dynamic_height = header_height + row_height * (len(df) + 1)
    if height is None:
        height = min(440, max(110, dynamic_height))
    else:
        height = max(110, dynamic_height)
    table = df.fillna("").copy()
    html_table = table.to_html(index=False, escape=True, classes="static-table", border=0)
    st.markdown(
        f'<div class="static-table-wrap" style="max-height:{int(height)}px;">{html_table}</div>',
        unsafe_allow_html=True,
    )


def _show_static_table(df: pd.DataFrame):
    if df.empty:
        st.info("No hay datos para esta tabla.")
        return
    html = df.fillna("").to_html(index=False, escape=True, classes="static-table", border=0)
    st.markdown(f'<div class="static-table-wrap">{html}</div>', unsafe_allow_html=True)


def _render_tipology_assignment(clusters: pd.DataFrame):
    if clusters.empty:
        return
    rows = []
    display = clusters.copy()
    if "tipologia" not in display.columns and "cluster_trayectoria" in display.columns:
        display["tipologia"] = display["cluster_trayectoria"].map(_as_tipologia)
    if "patron_tactico" not in display.columns:
        display["patron_tactico"] = display.get("tipologia", pd.Series(dtype=str))
    for _, row in display.sort_values("cluster_trayectoria" if "cluster_trayectoria" in display.columns else "tipologia").iterrows():
        tip = html.escape(str(row.get("tipologia", "-")))
        name = html.escape(str(row.get("patron_tactico", "Tipología sin nombre")))
        color = _cluster_color(row.get("cluster_trayectoria", 0))
        rows.append(
            f'<div class="tipology-row"><span class="tipology-swatch" style="background:{color};"></span><span><strong>{tip}:</strong> {name}</span></div>'
        )
    st.markdown(f'<div class="tipology-assignment">{"".join(rows)}</div>', unsafe_allow_html=True)


def _legend_color(label: str, idx: int) -> str:
    text = str(label)
    if text.startswith("T") and text[1:].isdigit():
        return APP_COLOR_MAP.get(text, APP_COLOR_SEQUENCE[int(text[1:]) % len(APP_COLOR_SEQUENCE)])
    return APP_COLOR_MAP.get(text, APP_COLOR_SEQUENCE[idx % len(APP_COLOR_SEQUENCE)])


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    clean = str(color).lstrip("#")
    if len(clean) != 6:
        return (200, 16, 46)
    return tuple(int(clean[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(v))) for v in rgb))


def _blend_hex(color: str, target: str, weight: float) -> str:
    base = _hex_to_rgb(color)
    dest = _hex_to_rgb(target)
    return _rgb_to_hex(tuple(base[i] * (1 - weight) + dest[i] * weight for i in range(3)))


def _tipology_gradient(tipologia: str) -> list[str]:
    base = APP_COLOR_MAP.get(str(tipologia), APP_COLOR_SEQUENCE[0])
    return [_blend_hex(base, "#ffffff", 0.62), _blend_hex(base, "#ffffff", 0.22), _blend_hex(base, "#111827", 0.22)]


def _category_color_map(labels: list[str]) -> dict[str, str]:
    return {str(label): _legend_color(str(label), idx) for idx, label in enumerate(labels)}


def _tipology_color_map(labels: list[str]) -> dict[str, str]:
    return {str(label): _legend_color(str(label), idx) for idx, label in enumerate(labels)}


def _apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor="#30384a",
        paper_bgcolor="#30384a",
        font=dict(color="#e8edf7"),
        title=dict(font=dict(size=17, color="#f4f6fb"), x=0.5, xanchor="center"),
        margin=dict(l=10, r=10, t=54, b=35),
        modebar=dict(remove=["toImage", "zoom", "pan", "select", "lasso", "zoomIn", "zoomOut", "autoScale", "resetScale"]),
    )
    fig.update_xaxes(
        color="#cbd4e6",
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(255,255,255,0.12)",
    )
    fig.update_yaxes(
        color="#cbd4e6",
        gridcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.12)",
        linecolor="rgba(255,255,255,0.12)",
    )
    return fig


def _add_plotly_red_frame(fig, x0: float = -0.065, y0: float = -0.165, x1: float = 1.02, y1: float = 1.13):
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        line=dict(color="rgba(200,16,46,0.95)", width=1.4),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )
    return fig


def _render_bar_legend(labels: list[str], title: str = "Tipología", color_map: dict[str, str] | None = None):
    if not labels:
        return
    items = []
    color_map = color_map or _category_color_map(labels)
    for idx, label in enumerate(labels):
        label_text = str(label)
        color = color_map.get(label_text, _legend_color(label_text, idx))
        safe_label = html.escape(label_text)
        items.append(
            f'<div class="bar-legend-item" title="{safe_label}"><span class="bar-legend-swatch" style="background:{color};"></span><span class="bar-legend-label">{safe_label}</span></div>'
        )
    legend_html = f'<div class="bar-legend"><div class="bar-legend-title">{html.escape(str(title))}</div><div class="bar-legend-grid">{"".join(items)}</div></div>'
    st.markdown(legend_html, unsafe_allow_html=True)


def _clean_plotly_events_frame():
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          function clean() {
            const frames = Array.from(doc.querySelectorAll("iframe"));
            frames.forEach((frame) => {
              try {
                const inner = frame.contentDocument || frame.contentWindow.document;
                if (!inner) return;
                const hasPlot = !!inner.querySelector(".js-plotly-plot, .plotly, .plot-container");
                if (!hasPlot) return;
                if (!inner.getElementById("tfg-plotly-clean-style")) {
                  const style = inner.createElement("style");
                  style.id = "tfg-plotly-clean-style";
                  style.textContent = `
                    html, body, #root, .container, .stPlotlyChart, .js-plotly-plot, .plot-container, .plotly {
                      margin: 0 !important;
                      padding: 0 !important;
                      background: #30384a !important;
                      background-color: #30384a !important;
                      box-shadow: none !important;
                      overflow: hidden !important;
                    }
                    .modebar { display: none !important; }
                  `;
                  inner.head.appendChild(style);
                }
                frame.style.setProperty("border", "0", "important");
                frame.style.setProperty("box-sizing", "border-box", "important");
                frame.style.setProperty("width", "100%", "important");
                frame.style.setProperty("max-width", "100%", "important");
                frame.style.setProperty("background", "#30384a", "important");
                frame.style.setProperty("box-shadow", "none", "important");
                frame.style.setProperty("display", "block", "important");
                frame.style.setProperty("border-radius", "0", "important");
                if (frame.parentElement) {
                  frame.parentElement.style.setProperty("border", "1px solid rgba(200,16,46,0.98)", "important");
                  frame.parentElement.style.setProperty("box-sizing", "border-box", "important");
                  frame.parentElement.style.setProperty("overflow", "hidden", "important");
                  frame.parentElement.style.setProperty("padding", "0", "important");
                  frame.parentElement.style.setProperty("margin-right", "12px", "important");
                  frame.parentElement.style.setProperty("width", "calc(100% - 12px)", "important");
                  frame.parentElement.style.setProperty("background", "#30384a", "important");
                  frame.parentElement.style.setProperty("box-shadow", "none", "important");
                }
                inner.documentElement.style.setProperty("margin", "0", "important");
                inner.documentElement.style.setProperty("padding", "0", "important");
                inner.documentElement.style.setProperty("background", "#30384a", "important");
                inner.body.style.setProperty("margin", "0", "important");
                inner.body.style.setProperty("padding", "0", "important");
                inner.body.style.setProperty("background", "#30384a", "important");
                inner.body.style.setProperty("overflow", "hidden", "important");
                inner.querySelectorAll("#root, .container, .stPlotlyChart, .js-plotly-plot").forEach((el) => {
                  el.style.setProperty("background", "#30384a", "important");
                  el.style.setProperty("margin", "0", "important");
                  el.style.setProperty("padding", "0", "important");
                });
                inner.querySelectorAll(".modebar").forEach((modebar) => {
                  modebar.style.setProperty("display", "none", "important");
                });
              } catch (err) {}
            });
          }
          clean();
          setTimeout(clean, 120);
          setTimeout(clean, 450);
          setTimeout(clean, 900);
          setTimeout(clean, 1600);
          setTimeout(clean, 2600);
          const interval = setInterval(clean, 500);
          setTimeout(() => clearInterval(interval), 10000);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def inject_selectbox_dropdown_fix():
    components.html(
        """
        <script>
        (() => {
          const doc = window.parent.document;
          function removeLargeGrayPanels() {
            const grayBackgrounds = new Set([
              "rgb(48, 56, 74)",
              "rgb(40, 48, 64)",
              "rgba(48, 56, 74, 0.96)",
              "rgba(48, 56, 74, 0.98)",
              "rgba(40, 48, 64, 0.78)"
            ]);
            doc.querySelectorAll("div, section, main").forEach((el) => {
              const rect = el.getBoundingClientRect();
              if (rect.width < 900 || rect.height < 360) return;
              if (el.closest(".analysis-card, .match-card, .sequence-card, .critical-focus-banner, .report-shell")) return;
              const bg = window.getComputedStyle(el).backgroundColor;
              if (!grayBackgrounds.has(bg)) return;
              el.style.setProperty("background", "transparent", "important");
              el.style.setProperty("background-color", "transparent", "important");
              el.style.setProperty("box-shadow", "none", "important");
            });
          }
          function nearestSelect(popover) {
            const popRect = popover.getBoundingClientRect();
            const selects = Array.from(doc.querySelectorAll('[data-testid="stSelectbox"] [data-baseweb="select"]'));
            let best = null;
            let bestScore = Infinity;
            selects.forEach((select) => {
              const rect = select.getBoundingClientRect();
              const horizontal = Math.abs(rect.left - popRect.left) + Math.abs(rect.width - popRect.width) * 0.3;
              const vertical = Math.abs(rect.bottom - popRect.top);
              const score = horizontal + vertical * 0.25;
              if (score < bestScore) {
                bestScore = score;
                best = rect;
              }
            });
            return best;
          }

          function fixPopovers() {
            doc.querySelectorAll('[data-baseweb="popover"]').forEach((popover) => {
              if (!popover.querySelector('[role="listbox"], [role="option"]')) return;
              const rect = nearestSelect(popover);
              if (!rect) return;
              const available = Math.max(180, window.innerHeight - rect.bottom - 22);
              popover.style.setProperty("transform", `translate3d(${rect.left}px, ${rect.bottom + 6}px, 0px)`, "important");
              popover.style.setProperty("width", `${rect.width}px`, "important");
              popover.style.setProperty("max-height", `${available}px`, "important");
              popover.style.setProperty("overflow-y", "auto", "important");
              popover.style.setProperty("z-index", "2147483646", "important");
              popover.querySelectorAll('[role="listbox"]').forEach((listbox) => {
                listbox.style.setProperty("max-height", `${available - 10}px`, "important");
                listbox.style.setProperty("overflow-y", "auto", "important");
              });
            });
          }
          function hideGraphControls() {
            const selectors = [
              '.modebar',
              '.modebar-container',
              '[data-testid="stElementToolbar"]',
              '[data-testid="StyledFullScreenButton"]',
              'button[title*="Fullscreen" i]',
              'button[title*="full screen" i]',
              'button[title*="expand" i]',
              'button[aria-label*="Fullscreen" i]',
              'button[aria-label*="full screen" i]',
              'button[aria-label*="expand" i]'
            ];
            doc.querySelectorAll(selectors.join(',')).forEach((el) => {
              el.style.setProperty("display", "none", "important");
              el.style.setProperty("visibility", "hidden", "important");
              el.style.setProperty("pointer-events", "none", "important");
              el.style.setProperty("width", "0", "important");
              el.style.setProperty("height", "0", "important");
            });
          }
          window.addEventListener("click", () => {
            removeLargeGrayPanels();
            hideGraphControls();
            setTimeout(fixPopovers, 20);
            setTimeout(fixPopovers, 120);
            setTimeout(fixPopovers, 260);
          }, true);
          removeLargeGrayPanels();
          hideGraphControls();
          setTimeout(removeLargeGrayPanels, 100);
          setTimeout(removeLargeGrayPanels, 500);
          setTimeout(removeLargeGrayPanels, 1200);
          setTimeout(hideGraphControls, 100);
          setTimeout(hideGraphControls, 500);
          setTimeout(hideGraphControls, 1200);
          setInterval(fixPopovers, 300);
          setInterval(removeLargeGrayPanels, 700);
          setInterval(hideGraphControls, 500);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def _plot_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    labels: dict | None = None,
    height: int = 340,
    color_map: dict[str, str] | None = None,
    red_gradient: bool = False,
    gradient_scale: list[str] | None = None,
):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este grafico.")
        return
    plot_df = df.copy()
    legend_labels = plot_df[color].dropna().astype(str).drop_duplicates().tolist() if color in plot_df.columns and not red_gradient else []
    category_colors = color_map or _category_color_map(legend_labels)
    if red_gradient:
        fig = px.bar(
            plot_df,
            x=x,
            y=y,
            color=y,
            text=y,
            labels=labels or {},
            height=height,
            title=title,
            color_continuous_scale=gradient_scale or ["#f7b5bd", "#dc1438", "#8b1025"],
        )
    else:
        fig = px.bar(
            plot_df,
            x=x,
            y=y,
            color=color if color in plot_df.columns else None,
            text=y,
            labels=labels or {},
            height=height,
            title=title,
            color_discrete_sequence=APP_COLOR_SEQUENCE,
            color_discrete_map=category_colors or APP_COLOR_MAP,
        )
    numeric_text = pd.to_numeric(df[y], errors="coerce").dropna()
    if not numeric_text.empty and np.allclose(numeric_text, numeric_text.round()):
        text_template = "%{text:.0f}"
    else:
        text_template = "%{text:.2f}"
    fig.update_traces(texttemplate=text_template, textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=54, b=35),
        showlegend=False,
        bargap=0.28,
        coloraxis_showscale=False,
    )
    _apply_plotly_theme(fig)
    fig.update_xaxes(showgrid=False, showticklabels=red_gradient, title=(labels or {}).get(x, "Tipología"))
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _plot_selectable_sequence_bar(
    df: pd.DataFrame,
    y: str,
    title: str,
    color: str,
    selected_id: int | None,
    key: str,
    labels: dict | None = None,
    height: int = 390,
    color_map: dict[str, str] | None = None,
) -> int | None:
    if df.empty or "secuencia_rival_id" not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este grafico.")
        return selected_id
    plot_df = df.copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y, "secuencia_rival_id"])
    if plot_df.empty:
        st.info("No hay datos suficientes para este grafico.")
        return selected_id
    plot_df["secuencia"] = plot_df["secuencia_rival_id"].astype(int).astype(str)
    plot_df["_xpos"] = np.arange(len(plot_df), dtype=int)
    legend_labels = plot_df[color].dropna().astype(str).drop_duplicates().tolist() if color in plot_df.columns else []
    category_colors = color_map or _category_color_map(legend_labels)
    if color in plot_df.columns:
        bar_colors = [
            category_colors.get(str(value), APP_COLOR_SEQUENCE[idx % len(APP_COLOR_SEQUENCE)])
            for idx, value in enumerate(plot_df[color].astype(str).tolist())
        ]
    else:
        bar_colors = [APP_COLOR_SEQUENCE[idx % len(APP_COLOR_SEQUENCE)] for idx in range(len(plot_df))]
    selected_seq = str(int(selected_id)) if selected_id is not None else None
    marker_line_colors = [
        "#c8102e" if selected_seq is not None and seq_id == selected_seq else "rgba(255,255,255,0.10)"
        for seq_id in plot_df["secuencia"].tolist()
    ]
    marker_line_widths = [2.4 if selected_seq is not None and seq_id == selected_seq else 0.8 for seq_id in plot_df["secuencia"].tolist()]
    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_df["_xpos"],
                y=plot_df[y],
                width=0.58,
                customdata=np.stack(
                    [
                        plot_df["secuencia_rival_id"].astype(int),
                        plot_df["secuencia"].astype(str),
                        plot_df[y].astype(float).round(3),
                    ],
                    axis=-1,
                ),
                marker=dict(
                    color=bar_colors,
                    opacity=0.95,
                    line=dict(color=marker_line_colors, width=marker_line_widths),
                ),
                hovertemplate="Secuencia %{customdata[1]}<br>%{y:.3f}<extra></extra>",
                text=[f"{v:.2f}" for v in plot_df[y].astype(float).tolist()],
                textposition="outside",
                textfont=dict(color="#f4f6fb", size=14, family="Arial Black"),
                cliponaxis=False,
            )
        ]
    )
    _apply_plotly_theme(fig)
    fig.update_layout(
        margin=dict(l=78, r=22, t=64, b=76),
        bargap=0.36,
        clickmode="event+select",
        showlegend=False,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22, color="#f4f6fb")),
        height=height,
        dragmode=False,
        paper_bgcolor="#30384a",
        plot_bgcolor="#30384a",
    )
    max_y = pd.to_numeric(plot_df[y], errors="coerce").max()
    y_range_top = max(float(max_y) * 1.16, 0.1) if pd.notna(max_y) else None
    fig.update_xaxes(
        showgrid=False,
        tickmode="array",
        tickvals=plot_df["_xpos"].tolist(),
        ticktext=plot_df["secuencia"].tolist(),
        title=dict(text="Secuencia", standoff=16, font=dict(size=16, color="#f4f6fb")),
        tickfont=dict(size=14, color="#dbe3f3"),
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        showgrid=True,
        title=dict(text=(labels or {}).get(y, y), standoff=14, font=dict(size=16, color="#f4f6fb")),
        tickformat=".2f",
        tickfont=dict(size=14, color="#dbe3f3"),
        range=[0, y_range_top] if y_range_top else None,
        zeroline=False,
        fixedrange=True,
    )
    frame_anchor = "critical-sequence-chart-" + re.sub(r"[^a-zA-Z0-9_-]+", "-", str(key))
    st.markdown(
        f"""
        <style>
        div[data-testid="stElementContainer"]:has(#{frame_anchor}) {{
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }}
        div[data-testid="stElementContainer"]:has(#{frame_anchor}) + div[data-testid="stElementContainer"] {{
            border: 1.5px solid #c8102e !important;
            border-radius: 0 !important;
            padding: 14px !important;
            background: #30384a !important;
            box-sizing: border-box !important;
            margin-top: 12px !important;
            margin-bottom: 18px !important;
            box-shadow: none !important;
        }}
        div[data-testid="stElementContainer"]:has(#{frame_anchor}) + div[data-testid="stElementContainer"] [data-testid="stPlotlyChart"] {{
            border: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            background: #30384a !important;
            box-shadow: none !important;
        }}
        </style>
        <span id="{frame_anchor}" class="critical-sequence-chart-anchor"></span>
        """,
        unsafe_allow_html=True,
    )
    points = []
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=key,
        on_select="rerun",
        selection_mode="points",
    )
    event_sources = [event, st.session_state.get(key)]
    for source in event_sources:
        if not source:
            continue
        try:
            points = source.selection.points
        except AttributeError:
            if isinstance(source, dict):
                points = source.get("selection", {}).get("points", [])
        if points:
            break
    if points:
        point = points[0]
        customdata = point.get("customdata") or point.get("custom_data")
        try:
            selected_id = int(customdata[0] if isinstance(customdata, (list, tuple)) else customdata)
        except (TypeError, ValueError, IndexError):
            for field in ("point_number", "pointIndex", "point_index"):
                try:
                    idx = int(point.get(field))
                    selected_id = int(plot_df.iloc[idx]["secuencia_rival_id"])
                    break
                except (TypeError, ValueError, IndexError):
                    pass
    return selected_id


def _clickable_sequence_bars(
    df: pd.DataFrame,
    y: str,
    title: str,
    selected_id: int | None,
    key_prefix: str,
    label: str,
) -> int | None:
    if df.empty or "secuencia_rival_id" not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este grafico.")
        return selected_id
    plot_df = df.copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce").fillna(0)
    max_value = max(float(plot_df[y].max()), 0.001)
    st.markdown(f"#### {html.escape(title)}")
    for idx, row in plot_df.iterrows():
        seq_id = int(row["secuencia_rival_id"])
        value = float(row[y])
        ratio = max(value / max_value, 0.06)
        tip = _display_text(row.get("tipologia", "-"))
        left, bar, rest = st.columns([0.9, max(0.45, ratio * 6.0), max(0.15, (1 - ratio) * 6.0)])
        with left:
            st.markdown(f'<div class="sequence-bar-label">Secuencia {seq_id}</div>', unsafe_allow_html=True)
        with bar:
            button_type = "primary" if selected_id == seq_id else "secondary"
            if st.button(
                f"{label}: {_format_metric(value)} · {tip}",
                key=f"{key_prefix}_{seq_id}",
                type=button_type,
                use_container_width=True,
            ):
                selected_id = seq_id
    return selected_id


def _plot_seq_scatter(seq: pd.DataFrame, title: str, height: int = 560):
    needed = {"indice_desorganizacion", "indice_peligrosidad_accion"}
    if seq.empty or not needed.issubset(seq.columns):
        st.info("No hay datos suficientes para este grafico.")
        return
    plot_df = seq.copy()
    for col in ["tipologia", "score_critico", "xT_max", "secuencia_rival_id", "minuto_partido", "tipo_secuencia_label"]:
        if col not in plot_df.columns:
            plot_df[col] = None
    plot_df["score_critico_plot"] = pd.to_numeric(plot_df["score_critico"], errors="coerce").fillna(0.05).clip(lower=0.05)
    plot_df["indice_desorganizacion"] = pd.to_numeric(plot_df["indice_desorganizacion"], errors="coerce")
    plot_df["indice_peligrosidad_accion"] = pd.to_numeric(plot_df["indice_peligrosidad_accion"], errors="coerce")
    plot_df = plot_df.dropna(subset=["indice_desorganizacion", "indice_peligrosidad_accion"])
    plot_df["lectura"] = np.select(
        [
            plot_df["indice_desorganizacion"].ge(0.66) & plot_df["indice_peligrosidad_accion"].ge(0.66),
            plot_df["indice_desorganizacion"].lt(0.66) & plot_df["indice_peligrosidad_accion"].ge(0.66),
            plot_df["indice_desorganizacion"].ge(0.66) & plot_df["indice_peligrosidad_accion"].lt(0.66),
        ],
        ["prioridad defensiva alta", "peligro eficiente", "ruptura sin castigo"],
        default="secuencia controlada",
    )
    tipology_colors = _category_color_map(plot_df["tipologia"].dropna().astype(str).drop_duplicates().tolist())
    fig = px.scatter(
        plot_df,
        x="indice_desorganizacion",
        y="indice_peligrosidad_accion",
        color="tipologia",
        size="score_critico_plot",
        hover_data={
            "secuencia_rival_id": True,
            "minuto_partido": ":.1f",
            "xT_max": ":.3f",
            "score_critico": ":.3f",
            "tipologia": True,
            "lectura": True,
            "tipo_secuencia_label": True,
            "score_critico_plot": False,
        },
        labels={
            "indice_desorganizacion": "IDD",
            "indice_peligrosidad_accion": "IPO",
            "tipologia": "Tipología de ataque rival",
        },
        height=height,
        title=title,
        size_max=18,
        color_discrete_map=tipology_colors or APP_COLOR_MAP,
    )
    fig.add_shape(type="rect", x0=0, x1=0.66, y0=0, y1=0.66, fillcolor="rgba(255,255,255,0.035)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0.66, x1=1.02, y0=0, y1=0.66, fillcolor="rgba(255,255,255,0.060)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=0.66, y0=0.66, y1=1.02, fillcolor="rgba(255,255,255,0.060)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0.66, x1=1.02, y0=0.66, y1=1.02, fillcolor="rgba(255,255,255,0.085)", line_width=0, layer="below")
    fig.add_hline(y=0.66, line_dash="dash", line_color="rgba(232,237,247,0.70)")
    fig.add_vline(x=0.66, line_dash="dash", line_color="rgba(232,237,247,0.70)")
    neutral_label = dict(size=11, color="#4b5563")
    neutral_box = "rgba(255,255,255,0.92)"
    fig.add_annotation(x=0.84, y=0.94, text="Alta prioridad", showarrow=False, font=neutral_label, bgcolor=neutral_box)
    fig.add_annotation(x=0.20, y=0.94, text="Peligro eficiente", showarrow=False, font=neutral_label, bgcolor=neutral_box)
    fig.add_annotation(x=0.84, y=0.08, text="Ruptura sin castigo", showarrow=False, font=neutral_label, bgcolor=neutral_box)
    fig.add_annotation(x=0.22, y=0.08, text="Controlada", showarrow=False, font=neutral_label, bgcolor=neutral_box)
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=1.4, color="#202633")))
    _apply_plotly_theme(fig)
    fig.update_layout(
        showlegend=False,
        autosize=True,
    )
    fig.update_xaxes(range=[0, 1.02], showgrid=True, constrain="domain")
    fig.update_yaxes(range=[0, 1.02], showgrid=True, scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _lane_label(value) -> str:
    text = str(value).lower()
    mapping = {
        "carril_1": "Banda izquierda",
        "carril_2": "Half-space izquierdo",
        "carril_3": "Carril central",
        "carril_4": "Half-space derecho",
        "carril_5": "Banda derecha",
        "carril_central": "Carril central",
        "banda_izquierda": "Banda izquierda",
        "banda_derecha": "Banda derecha",
    }
    return mapping.get(text, text.replace("_", " ").capitalize() if text and text != "nan" else "Zona no identificada")


def _zone_label(value) -> str:
    text = str(value).lower()
    mapping = {
        "zona_1": "Zona 1 - Inicio de campo propio",
        "zona_2": "Zona 2 - Zona media",
        "zona_3": "Zona 3 - Ultimo tercio",
        "zona_4": "Zona 4 - Zona de remate",
        "izquierda": "Banda izquierda",
        "derecha": "Banda derecha",
        "centro": "Carril central",
    }
    return mapping.get(text, text.replace("_", " ").capitalize() if text and text != "nan" else "Zona no identificada")


def _zone_code_label(value) -> str:
    text = str(value).lower()
    mapping = {
        "zona_1": "Zona 1",
        "zona_2": "Zona 2",
        "zona_3": "Zona 3",
        "zona_4": "Zona 4",
    }
    return mapping.get(text, _zone_label(value))


def _zone_lane_suffix(source: pd.Series | dict) -> str:
    zone_raw = source.get("zona_dominante", "") if hasattr(source, "get") else ""
    lane_raw = source.get("carril_dominante", "") if hasattr(source, "get") else ""
    zone = _zone_label(zone_raw)
    lane = _lane_label(lane_raw)
    if zone != "Zona no identificada" and lane != "Zona no identificada":
        return f"{zone} ({lane})"
    if zone != "Zona no identificada":
        return zone
    if lane != "Zona no identificada":
        return lane
    return "Zona no identificada"


def _sentence_label(value) -> str:
    text = str(value).strip()
    return text[:1].upper() + text[1:] if text else text


def _lower_first(value: str) -> str:
    text = str(value)
    return text[:1].lower() + text[1:] if text else text


def _cause_label(value) -> str:
    text = str(value).lower()
    mapping = {
        "estructura_retroceso": "problemas de repliegue",
        "estructura_anchura": "bloque demasiado abierto",
        "presion_distancia": "poca presión al balón",
        "pitch_control_rival": "dominio rival en zonas peligrosas",
        "presion_distancia": "poca presión sobre poseedor",
        "sin etiqueta": "sin causa dominante",
    }
    return mapping.get(text, text.replace("_", " ") if text and text != "nan" else "-")


def _sequence_type_label(value) -> str:
    text = str(value).lower()
    mapping = {
        "ataque_elaborado": "Ataque posicional",
        "juego_directo": "Juego directo vertical",
        "transicion_rapida": "Transición tras robo",
        "balon_parado_con_continuidad": "Balón parado con continuidad",
        "descartada": "Secuencia descartada",
    }
    return mapping.get(text, text.replace("_", " ").capitalize() if text and text != "nan" else "-")


def _pattern_name(cluster_row: pd.Series, seq_cluster: pd.DataFrame | None = None) -> str:
    lane = _lane_label(cluster_row.get("carril_dominante", ""))
    lane_lower = lane.lower()
    dominant_type = None
    if seq_cluster is not None and not seq_cluster.empty and "tipo_secuencia_ofensiva" in seq_cluster.columns:
        mode = seq_cluster["tipo_secuencia_ofensiva"].dropna().astype(str).mode()
        dominant_type = mode.iloc[0] if not mode.empty else None
    tiro_pct = pd.to_numeric(cluster_row.get("tiro_pct", 0), errors="coerce")
    ipar = pd.to_numeric(seq_cluster.get("indice_peligrosidad_accion", pd.Series(dtype=float)).mean(), errors="coerce") if seq_cluster is not None and not seq_cluster.empty else np.nan
    if dominant_type == "transicion_rapida":
        base = "Transiciones tras robo"
    elif dominant_type == "juego_directo":
        base = "Juego directo vertical"
    elif dominant_type == "balon_parado_con_continuidad":
        base = "Balón parado con segunda jugada"
    elif dominant_type == "ataque_elaborado":
        if "central" in lane_lower:
            base = "Circulación posicional interior"
        elif "banda" in lane_lower:
            base = f"Ataques por {_lower_first(lane)}"
        else:
            base = "Circulación posicional en U"
    elif pd.notna(tiro_pct) and tiro_pct >= 25:
        base = "Tipología de finalización frecuente"
    elif pd.notna(ipar) and ipar >= 0.55:
        base = "Tipología de alta amenaza"
    else:
        base = "Ataques de progresión controlada"
    suffix = _zone_lane_suffix(cluster_row)
    if suffix != "Zona no identificada":
        return _sentence_label(f"{base} - {suffix}")
    return _sentence_label(base)


def _apply_tactical_labels(seq: pd.DataFrame, clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seq = seq.copy()
    clusters = clusters.copy()
    label_map = {}
    if not clusters.empty and "cluster_trayectoria" in clusters.columns:
        raw_label_map = {}
        for _, row in clusters.iterrows():
            cid = row.get("cluster_trayectoria")
            if pd.isna(cid):
                continue
            seq_cluster = seq[pd.to_numeric(seq.get("cluster_trayectoria"), errors="coerce").eq(float(cid))] if "cluster_trayectoria" in seq.columns else pd.DataFrame()
            raw_label_map[int(float(cid))] = _pattern_name(row, seq_cluster)
        label_counts = pd.Series(list(raw_label_map.values()), dtype="object").value_counts().to_dict()
        for cid, label in raw_label_map.items():
            if label_counts.get(label, 0) > 1:
                cluster_row = clusters[pd.to_numeric(clusters["cluster_trayectoria"], errors="coerce").eq(float(cid))].head(1)
                suffix = _zone_lane_suffix(cluster_row.iloc[0]) if not cluster_row.empty else f"Sector {cid}"
                if suffix != "Zona no identificada" and suffix not in label:
                    label_map[cid] = f"{label} - {suffix}"
                else:
                    label_map[cid] = f"{label} - Sector {cid + 1}"
            else:
                label_map[cid] = label
            APP_COLOR_MAP[str(label_map[cid])] = _cluster_color(cid)
        clusters["patron_tactico"] = clusters["cluster_trayectoria"].map(lambda v: label_map.get(int(float(v)), "Tipología sin etiquetar") if pd.notna(v) else "Tipología sin etiquetar")
        clusters["tipologia"] = clusters["cluster_trayectoria"].map(_as_tipologia)
        for _, row in clusters.dropna(subset=["cluster_trayectoria"]).iterrows():
            APP_COLOR_MAP[str(row["patron_tactico"])] = _cluster_color(row["cluster_trayectoria"])
        clusters["zona_dominante"] = clusters.get("zona_dominante", pd.Series(dtype=str)).apply(_zone_label)
        clusters["carril_dominante"] = clusters.get("carril_dominante", pd.Series(dtype=str)).apply(_lane_label)
    if not seq.empty:
        if "cluster_trayectoria" in seq.columns:
            seq["patron_tactico"] = seq["cluster_trayectoria"].map(lambda v: label_map.get(int(float(v)), "Tipología sin etiquetar") if pd.notna(v) else "Tipología sin etiquetar")
            seq["tipologia_id"] = seq["cluster_trayectoria"].map(_as_tipologia)
            seq["tipologia"] = seq["patron_tactico"]
        if "tipo_desorganizacion_principal" in seq.columns:
            seq["causa_tactica"] = seq["tipo_desorganizacion_principal"].apply(_cause_label)
        if "tipo_secuencia_ofensiva" in seq.columns:
            seq["tipo_secuencia_label"] = seq["tipo_secuencia_ofensiva"].apply(_sequence_type_label)
    return _clean_text_columns(seq), _clean_text_columns(clusters), label_map


def _prepare_app_data(match_id: int, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    tables = meta["tables"]
    seq = _load_sequence_detail(match_id, meta)
    clusters = _load_table(match_id, tables["clusters_resumen"])
    zona = _load_table(match_id, tables.get("zona_cluster_danio", "zona_cluster_danio.csv"))
    seq, clusters, label_map = _apply_tactical_labels(seq, clusters)
    if not zona.empty:
        if "zona_ataque" in zona.columns:
            zona["zona_ataque"] = zona["zona_ataque"].apply(_lane_label)
        if "cluster_trayectoria" in zona.columns:
            zona["patron_tactico"] = zona["cluster_trayectoria"].map(lambda v: label_map.get(int(float(v)), "Tipología sin etiquetar") if pd.notna(v) else "Tipología sin etiquetar")
    return _clean_text_columns(seq), _clean_text_columns(clusters), _clean_text_columns(zona), label_map


def _add_pitch_shapes(fig: go.Figure):
    line = dict(color="#203040", width=1.2)
    faint = dict(color="rgba(32,48,64,0.55)", width=1)
    fig.update_layout(
        shapes=[
            dict(type="rect", x0=0, y0=0, x1=FIELD_WIDTH_M, y1=FIELD_LENGTH_M, line=line),
            dict(type="line", x0=0, y0=FIELD_LENGTH_M / 2, x1=FIELD_WIDTH_M, y1=FIELD_LENGTH_M / 2, line=faint),
            dict(type="circle", x0=FIELD_WIDTH_M / 2 - 9.15, y0=FIELD_LENGTH_M / 2 - 9.15, x1=FIELD_WIDTH_M / 2 + 9.15, y1=FIELD_LENGTH_M / 2 + 9.15, line=faint),
            dict(type="rect", x0=FIELD_WIDTH_M / 2 - 20.16, y0=0, x1=FIELD_WIDTH_M / 2 + 20.16, y1=16.5, line=faint),
            dict(type="rect", x0=FIELD_WIDTH_M / 2 - 20.16, y0=FIELD_LENGTH_M - 16.5, x1=FIELD_WIDTH_M / 2 + 20.16, y1=FIELD_LENGTH_M, line=faint),
            dict(type="rect", x0=FIELD_WIDTH_M * 0.18, y0=FIELD_LENGTH_M - 30, x1=FIELD_WIDTH_M * 0.82, y1=FIELD_LENGTH_M, line=dict(color="rgba(200,16,46,0.35)", width=1, dash="dot"), fillcolor="rgba(200,16,46,0.045)"),
        ]
    )


def _sequence_field_heatmap(match_id: int, meta: dict, seq: pd.DataFrame, title: str, height: int = 430):
    if seq.empty:
        st.info("No hay secuencias para dibujar este mapa.")
        return
    plot_df = seq.copy()
    x_col = "x_fin" if "x_fin" in plot_df.columns else "ball_x_fin_m"
    y_col = "y_fin" if "y_fin" in plot_df.columns else "ball_y_fin_m"
    if x_col not in plot_df.columns or y_col not in plot_df.columns:
        st.info("No hay coordenadas suficientes para este mapa.")
        return
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        st.info("No hay coordenadas validas para este mapa.")
        return

    for col in ["tipologia", "secuencia_rival_id", "indice_peligrosidad_accion", "indice_desorganizacion"]:
        if col not in plot_df.columns:
            plot_df[col] = None
    period = pd.to_numeric(plot_df.get("period", 0), errors="coerce").fillna(0).astype(int)
    attack_high_by_period = {}
    for per, gr in plot_df.assign(_period=period).groupby("_period"):
        shots = gr[pd.to_numeric(gr.get("tipo_finalizacion_tiro", 0), errors="coerce").fillna(0).eq(1)]
        ref = shots[x_col] if len(shots) >= 1 else gr[x_col]
        attack_high_by_period[int(per)] = bool(pd.to_numeric(ref, errors="coerce").median() >= FIELD_LENGTH_M / 2)
    attack_high = period.map(lambda p: attack_high_by_period.get(int(p), True))
    plot_df["_x_norm"] = plot_df[x_col].where(attack_high, FIELD_LENGTH_M - plot_df[x_col])
    plot_df["_y_norm"] = plot_df[y_col].where(attack_high, FIELD_WIDTH_M - plot_df[y_col])
    if {"ball_x_ini_m", "ball_y_ini_m", "ball_x_fin_m", "ball_y_fin_m"}.issubset(plot_df.columns):
        plot_df["_x_ini_norm"] = plot_df["ball_x_ini_m"].where(attack_high, FIELD_LENGTH_M - plot_df["ball_x_ini_m"])
        plot_df["_y_ini_norm"] = plot_df["ball_y_ini_m"].where(attack_high, FIELD_WIDTH_M - plot_df["ball_y_ini_m"])
        plot_df["_x_fin_norm"] = plot_df["ball_x_fin_m"].where(attack_high, FIELD_LENGTH_M - plot_df["ball_x_fin_m"])
        plot_df["_y_fin_norm"] = plot_df["ball_y_fin_m"].where(attack_high, FIELD_WIDTH_M - plot_df["ball_y_fin_m"])

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2dContour(
            x=plot_df["_y_norm"],
            y=plot_df["_x_norm"],
            colorscale=[[0, "#f7f0df"], [0.35, "#f2c94c"], [0.7, "#2453a6"], [1, "#c8102e"]],
            contours=dict(coloring="heatmap"),
            ncontours=18,
            showscale=True,
            colorbar=dict(title="Densidad", len=0.82),
            opacity=0.62,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["_y_norm"],
            y=plot_df["_x_norm"],
            mode="markers",
            marker=dict(size=8, color="#0a1020", opacity=0.58, line=dict(color="white", width=0.7)),
            customdata=plot_df[["secuencia_rival_id", "tipologia", "indice_desorganizacion", "indice_peligrosidad_accion"]],
            hovertemplate="Secuencia %{customdata[0]}<br>%{customdata[1]}<br>IDD %{customdata[2]:.3f}<br>IPO %{customdata[3]:.3f}<extra></extra>",
            name="Final de secuencia",
        )
    )
    path_df = pd.DataFrame()
    table_name = meta.get("tables", {}).get("trayectorias_ligeras", "trayectorias_ligeras.csv")
    if "secuencia_rival_id" in plot_df.columns:
        path_df = _load_table(match_id, table_name)
        if not path_df.empty:
            ids = set(pd.to_numeric(plot_df["secuencia_rival_id"], errors="coerce").dropna().astype(int))
            path_df["secuencia_rival_id"] = pd.to_numeric(path_df["secuencia_rival_id"], errors="coerce")
            path_df = path_df[path_df["secuencia_rival_id"].isin(ids)].copy()
            path_period = pd.to_numeric(path_df.get("period", 0), errors="coerce").fillna(0).astype(int)
            path_attack_high = path_period.map(lambda p: attack_high_by_period.get(int(p), True))
            path_df["_x_norm"] = pd.to_numeric(path_df["ball_x_m"], errors="coerce").where(path_attack_high, FIELD_LENGTH_M - pd.to_numeric(path_df["ball_x_m"], errors="coerce"))
            path_df["_y_norm"] = pd.to_numeric(path_df["ball_y_m"], errors="coerce").where(path_attack_high, FIELD_WIDTH_M - pd.to_numeric(path_df["ball_y_m"], errors="coerce"))
            path_df = path_df.dropna(subset=["_x_norm", "_y_norm"]).sort_values(["secuencia_rival_id", "point_order"])

    if not path_df.empty:
        for _, gr_path in path_df.groupby("secuencia_rival_id"):
            fig.add_trace(
                go.Scatter(
                    x=gr_path["_y_norm"],
                    y=gr_path["_x_norm"],
                    mode="lines",
                    line=dict(color="rgba(200,16,46,0.34)", width=1.25),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    elif {"_x_ini_norm", "_y_ini_norm", "_x_fin_norm", "_y_fin_norm"}.issubset(plot_df.columns):
        for _, row in plot_df.head(80).iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["_y_ini_norm"], row["_y_fin_norm"]],
                    y=[row["_x_ini_norm"], row["_x_fin_norm"]],
                    mode="lines",
                    line=dict(color="rgba(200,16,46,0.36)", width=1.4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    _add_pitch_shapes(fig)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=17, color="#f4f6fb")),
        height=height,
        plot_bgcolor="#5f7668",
        paper_bgcolor="#202633",
        font=dict(color="#e8edf7"),
        margin=dict(l=10, r=10, t=54, b=25),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, FIELD_WIDTH_M], showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[0, FIELD_LENGTH_M], showgrid=False, zeroline=False, visible=False)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _add_plotly_pitch(fig: go.Figure, row: int = 1, col: int = 1):
    line = dict(color="#111827", width=1.4)
    faint = dict(color="rgba(17,24,39,0.78)", width=1.0)
    fig.add_shape(type="rect", x0=0, y0=0, x1=FIELD_WIDTH_M, y1=FIELD_LENGTH_M, line=line, row=row, col=col)
    fig.add_shape(type="line", x0=0, y0=FIELD_LENGTH_M / 2, x1=FIELD_WIDTH_M, y1=FIELD_LENGTH_M / 2, line=faint, row=row, col=col)
    fig.add_shape(
        type="circle",
        x0=FIELD_WIDTH_M / 2 - 9.15,
        y0=FIELD_LENGTH_M / 2 - 9.15,
        x1=FIELD_WIDTH_M / 2 + 9.15,
        y1=FIELD_LENGTH_M / 2 + 9.15,
        line=faint,
        row=row,
        col=col,
    )
    fig.add_shape(type="rect", x0=FIELD_WIDTH_M / 2 - 20.16, y0=0, x1=FIELD_WIDTH_M / 2 + 20.16, y1=16.5, line=faint, row=row, col=col)
    fig.add_shape(
        type="rect",
        x0=FIELD_WIDTH_M / 2 - 20.16,
        y0=FIELD_LENGTH_M - 16.5,
        x1=FIELD_WIDTH_M / 2 + 20.16,
        y1=FIELD_LENGTH_M,
        line=faint,
        row=row,
        col=col,
    )
    for j in range(10):
        y0 = j * FIELD_LENGTH_M / 10
        fig.add_shape(
            type="rect",
            x0=0,
            y0=y0,
            x1=FIELD_WIDTH_M,
            y1=y0 + FIELD_LENGTH_M / 10,
            line=dict(width=0),
            fillcolor="rgba(255,255,255,0.035)" if j % 2 == 0 else "rgba(0,0,0,0.025)",
            layer="below",
            row=row,
            col=col,
        )


def _path_table_for_plot(match_id: int, meta: dict, seq: pd.DataFrame) -> pd.DataFrame:
    table_name = meta.get("tables", {}).get("trayectorias_ligeras", "trayectorias_ligeras.csv")
    path_df = _load_table(match_id, table_name)
    if path_df.empty:
        return pd.DataFrame()
    for col in ["secuencia_rival_id", "cluster_trayectoria", "ball_x_m", "ball_y_m", "point_order", "match_time"]:
        if col in path_df.columns:
            path_df[col] = pd.to_numeric(path_df[col], errors="coerce")
    cols = ["secuencia_rival_id", "tipologia", "patron_tactico", "tipologia_id"]
    if not seq.empty and "secuencia_rival_id" in seq.columns:
        lookup = seq[[c for c in cols if c in seq.columns]].drop_duplicates("secuencia_rival_id")
        path_df = path_df.merge(lookup, on="secuencia_rival_id", how="left")
    if "patron_tactico" not in path_df.columns:
        path_df["patron_tactico"] = path_df.get("tipologia", path_df.get("cluster_trayectoria", "-")).astype(str)
    return path_df.dropna(subset=["ball_x_m", "ball_y_m", "cluster_trayectoria"])


def _plot_trajectory_fields(match_id: int, meta: dict, seq: pd.DataFrame, height: int = 500) -> bool:
    path_df = _path_table_for_plot(match_id, meta, seq)
    if path_df.empty:
        return False
    clusters = sorted(path_df["cluster_trayectoria"].dropna().unique())
    if not clusters:
        return False
    ncols = min(len(clusters), 4)
    clusters = clusters[:ncols]
    titles = []
    for cluster in clusters:
        gr = path_df[path_df["cluster_trayectoria"].eq(cluster)]
        label = gr["patron_tactico"].dropna().astype(str).head(1)
        titles.append(f"{label.iloc[0] if not label.empty else _as_tipologia(cluster)} ({gr['secuencia_rival_id'].nunique()} sec.)")
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles, horizontal_spacing=0.012)
    color_map = _category_color_map([str(v) for v in path_df["patron_tactico"].dropna().unique()])
    for idx, cluster in enumerate(clusters, start=1):
        gr_c = path_df[path_df["cluster_trayectoria"].eq(cluster)]
        _add_plotly_pitch(fig, row=1, col=idx)
        for seq_id, gr_seq in gr_c.groupby("secuencia_rival_id"):
            gr_seq = gr_seq.sort_values(["point_order", "match_time"], na_position="last")
            label = str(gr_seq["patron_tactico"].dropna().iloc[0]) if gr_seq["patron_tactico"].notna().any() else _as_tipologia(cluster)
            fig.add_trace(
                go.Scatter(
                    x=gr_seq["ball_y_m"],
                    y=gr_seq["ball_x_m"],
                    mode="lines+markers",
                    line=dict(color=color_map.get(label, APP_COLOR_SEQUENCE[(idx - 1) % len(APP_COLOR_SEQUENCE)]), width=2.2),
                    marker=dict(size=4, color=color_map.get(label, APP_COLOR_SEQUENCE[(idx - 1) % len(APP_COLOR_SEQUENCE)]), opacity=0.65),
                    opacity=0.72,
                    hovertemplate=f"Secuencia {int(seq_id)}<br>{html.escape(label)}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )
        first_points = gr_c.sort_values(["point_order", "match_time"], na_position="last").groupby("secuencia_rival_id").head(1)
        fig.add_trace(
            go.Scatter(
                x=first_points["ball_y_m"],
                y=first_points["ball_x_m"],
                mode="markers",
                marker=dict(size=7, color="#111827", opacity=0.72, line=dict(color="rgba(255,255,255,0.35)", width=0.8)),
                hovertemplate="Inicio secuencia %{customdata}<extra></extra>",
                customdata=first_points["secuencia_rival_id"].astype(int),
                showlegend=False,
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(range=[0, FIELD_WIDTH_M], visible=False, row=1, col=idx)
        fig.update_yaxes(range=[0, FIELD_LENGTH_M], visible=False, scaleanchor=f"x{idx}" if idx > 1 else "x", scaleratio=1, row=1, col=idx)
    fig.update_layout(height=height, plot_bgcolor="#5f7668", paper_bgcolor="#202633", margin=dict(l=5, r=5, t=48, b=8))
    for ann in fig.layout.annotations:
        ann.font = dict(color="#e8edf7", size=13)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    return True


def _plot_density_fields(match_id: int, meta: dict, seq: pd.DataFrame, height: int = 500) -> bool:
    path_df = _path_table_for_plot(match_id, meta, seq)
    if path_df.empty:
        return False
    clusters = sorted(path_df["cluster_trayectoria"].dropna().unique())
    if not clusters:
        return False
    ncols = min(len(clusters), 4)
    clusters = clusters[:ncols]
    titles = []
    for cluster in clusters:
        gr = path_df[path_df["cluster_trayectoria"].eq(cluster)]
        label = gr["patron_tactico"].dropna().astype(str).head(1)
        titles.append(f"Mapa de calor | {label.iloc[0] if not label.empty else _as_tipologia(cluster)}")
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles, horizontal_spacing=0.012)
    for idx, cluster in enumerate(clusters, start=1):
        gr = path_df[path_df["cluster_trayectoria"].eq(cluster)]
        _add_plotly_pitch(fig, row=1, col=idx)
        contour_kwargs = {}
        if idx == ncols:
            contour_kwargs = {"showscale": True, "colorbar": dict(title="Densidad", len=0.70)}
        else:
            contour_kwargs = {"showscale": False}
        fig.add_trace(
            go.Histogram2dContour(
                x=gr["ball_y_m"],
                y=gr["ball_x_m"],
                colorscale=[[0, "rgba(255,255,255,0)"], [0.35, "#f2c94c"], [0.70, "#e45a67"], [1, "#c8102e"]],
                contours=dict(coloring="heatmap"),
                ncontours=18,
                opacity=0.78,
                hovertemplate="Densidad<br>Carril %{x:.1f}<br>Profundidad %{y:.1f}<extra></extra>",
                **contour_kwargs,
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(range=[0, FIELD_WIDTH_M], visible=False, row=1, col=idx)
        fig.update_yaxes(range=[0, FIELD_LENGTH_M], visible=False, scaleanchor=f"x{idx}" if idx > 1 else "x", scaleratio=1, row=1, col=idx)
    fig.update_layout(height=height, plot_bgcolor="#5f7668", paper_bgcolor="#202633", margin=dict(l=5, r=5, t=48, b=8))
    for ann in fig.layout.annotations:
        ann.font = dict(color="#e8edf7", size=13)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    return True


def _plot_xt_evolution(match_id: int, meta: dict, clusters: pd.DataFrame, height: int = 520) -> bool:
    table_name = meta.get("tables", {}).get("amenaza_media_cluster", "amenaza_media_cluster.csv")
    threat = _load_table(match_id, table_name)
    if threat.empty or not {"cluster_trayectoria", "punto", "amenaza_media"}.issubset(threat.columns):
        return False
    threat["cluster_trayectoria"] = pd.to_numeric(threat["cluster_trayectoria"], errors="coerce")
    threat["punto"] = pd.to_numeric(threat["punto"], errors="coerce")
    threat["amenaza_media"] = pd.to_numeric(threat["amenaza_media"], errors="coerce")
    threat = threat.dropna(subset=["cluster_trayectoria", "punto", "amenaza_media"])
    if threat.empty:
        return False
    label_map = {}
    if not clusters.empty and {"cluster_trayectoria", "patron_tactico"}.issubset(clusters.columns):
        for _, row in clusters.dropna(subset=["cluster_trayectoria"]).iterrows():
            label_map[int(float(row["cluster_trayectoria"]))] = str(row["patron_tactico"])
    x_max = int(threat["punto"].max())
    inicio_fin = x_max * 0.33
    progresion_fin = x_max * 0.67
    third_entry_x = progresion_fin
    third_zone_x = min(x_max - 0.5, third_entry_x + 0.22)
    shot_x = x_max * 0.88
    y_max = max(float(threat["amenaza_media"].max()) * 1.14, 0.02)
    fig = go.Figure()
    steps = 42
    for idx in range(steps):
        x0 = third_zone_x + (x_max - third_zone_x) * idx / steps
        x1 = third_zone_x + (x_max - third_zone_x) * (idx + 1) / steps
        midpoint = (x0 + x1) / 2
        frac = (midpoint - third_zone_x) / max(x_max - third_zone_x, 1)
        alpha = 0.10 + 0.34 * frac
        if midpoint >= shot_x:
            alpha = 0.42 + 0.18 * frac
        fig.add_vrect(x0=x0, x1=x1, fillcolor=f"rgba(200,16,46,{alpha:.3f})", line_width=0, layer="below")
    fig.add_vline(x=third_zone_x, line_color="rgba(255,120,138,0.92)", line_width=4)
    fig.add_vline(x=shot_x, line_color="#c8102e", line_width=6)
    fig.add_vline(x=inicio_fin, line_dash="dash", line_color="rgba(255,255,255,0.92)", line_width=2.4)
    fig.add_vline(x=progresion_fin, line_dash="dash", line_color="rgba(255,255,255,0.96)", line_width=2.6)
    colors = _category_color_map([label_map.get(int(c), _as_tipologia(c)) for c in sorted(threat["cluster_trayectoria"].unique())])

    seq_detail = _load_sequence_detail(match_id, meta)
    survival_by_cluster: dict[int, tuple[float, float]] = {}
    if not seq_detail.empty and {"cluster_trayectoria", "x_max_norm_m"}.issubset(seq_detail.columns):
        seq_survival = seq_detail.copy()
        seq_survival["cluster_trayectoria"] = pd.to_numeric(seq_survival["cluster_trayectoria"], errors="coerce")
        seq_survival["x_max_norm_m"] = pd.to_numeric(seq_survival["x_max_norm_m"], errors="coerce")
        seq_survival = seq_survival.dropna(subset=["cluster_trayectoria", "x_max_norm_m"])
        last_third_threshold = FIELD_LENGTH_M * (2 / 3)
        final_zone_threshold = FIELD_LENGTH_M * 0.90
        for cluster_id, gr_seq in seq_survival.groupby("cluster_trayectoria"):
            survival_by_cluster[int(cluster_id)] = (
                float(gr_seq["x_max_norm_m"].ge(last_third_threshold).mean() * 100),
                float(gr_seq["x_max_norm_m"].ge(final_zone_threshold).mean() * 100),
            )

    line_lookup: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    for cluster, gr in threat.groupby("cluster_trayectoria"):
        label = label_map.get(int(cluster), _as_tipologia(cluster))
        gr_sorted = gr.sort_values("punto")
        line_color = colors.get(label, APP_COLOR_SEQUENCE[int(cluster) % len(APP_COLOR_SEQUENCE)])
        line_lookup[int(cluster)] = (
            gr_sorted["punto"].to_numpy(dtype=float),
            gr_sorted["amenaza_media"].to_numpy(dtype=float),
            line_color,
        )
        fig.add_trace(
            go.Scatter(
                x=gr_sorted["punto"],
                y=gr_sorted["amenaza_media"],
                mode="lines+markers",
                name=label,
                line=dict(color=line_color, width=3),
                marker=dict(size=8),
                hovertemplate=f"{html.escape(label)}<br>Momento %{{x}}<br>xT %{{y:.3f}}<extra></extra>",
            )
        )
    phase_y = y_max * 0.14
    fig.add_annotation(x=inicio_fin / 2, y=phase_y, text="INICIO", showarrow=False, font=dict(color="#ffffff", size=13), bgcolor="rgba(32,37,50,0.72)")
    fig.add_annotation(x=(inicio_fin + progresion_fin) / 2, y=phase_y, text="PROGRESION", showarrow=False, font=dict(color="#ffffff", size=13), bgcolor="rgba(32,37,50,0.72)")
    fig.add_annotation(x=(progresion_fin + x_max) / 2, y=phase_y, text="FINALIZACION", showarrow=False, font=dict(color="#ffffff", size=13), bgcolor="rgba(32,37,50,0.72)")
    fig.add_annotation(x=third_zone_x + 0.25, y=y_max * 0.90, text="ENTRADA ULTIMO TERCIO", showarrow=False, font=dict(color="#ffffff", size=13), bgcolor="rgba(169,15,40,0.92)", bordercolor="#ffffff", borderwidth=1)
    fig.add_annotation(x=shot_x + 0.25, y=y_max * 0.78, text="ZONA MEDIA DE TIRO", showarrow=False, font=dict(color="#ffffff", size=13), bgcolor="rgba(200,16,46,0.98)", bordercolor="#ffffff", borderwidth=1)
    label_columns = {third_zone_x + 0.08: 0, shot_x + 0.08: 1}
    for x_label, value_idx in label_columns.items():
        labels_to_place = []
        x_line = third_zone_x if value_idx == 0 else shot_x
        for cluster in sorted(survival_by_cluster):
            if int(cluster) not in line_lookup:
                continue
            values = survival_by_cluster[cluster]
            xs, ys, line_color = line_lookup[int(cluster)]
            labels_to_place.append(
                {
                    "y_raw": float(np.interp(x_line, xs, ys)),
                    "text": f"{values[value_idx]:.0f}%",
                    "color": line_color,
                }
            )
        labels_to_place.sort(key=lambda item: item["y_raw"])
        min_gap = y_max * 0.045
        placed = []
        for item in labels_to_place:
            y_value = item["y_raw"]
            if placed and y_value < placed[-1]["y"] + min_gap:
                y_value = placed[-1]["y"] + min_gap
            placed.append({**item, "y": y_value})
        overflow = placed[-1]["y"] - y_max * 0.92 if placed else 0
        if overflow > 0:
            for item in placed:
                item["y"] = max(y_max * 0.055, item["y"] - overflow)
        for item in placed:
            fig.add_annotation(
                x=x_label,
                y=item["y"],
                text=item["text"],
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(color=item["color"], size=13),
                bgcolor="rgba(32,37,50,0.74)",
                bordercolor=item["color"],
                borderwidth=1.4,
            )
    fig.update_layout(
        title="Evolución media de la amenaza concedida por tipología",
        height=height + 60,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=10, r=10, t=54, b=38),
        hoverlabel=dict(bgcolor="#202532", bordercolor="rgba(255,255,255,0.16)", font=dict(color="#e8edf7")),
    )
    _apply_plotly_theme(fig)
    fig.update_xaxes(title="Momento normalizado de la secuencia", range=[-0.6, x_max + 0.6])
    fig.update_yaxes(title="Amenaza concedida xT", range=[0, y_max])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    return True


def _mean_numeric(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return float("nan")
    return pd.to_numeric(df[col], errors="coerce").mean()


def _max_numeric(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return float("nan")
    return pd.to_numeric(df[col], errors="coerce").max()


def _sum_numeric(df: pd.DataFrame, col: str):
    if df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _time_window_summary(seq: pd.DataFrame) -> tuple[str, int]:
    if seq.empty or "minuto_partido" not in seq.columns:
        return "-", 0
    plot_df = seq.dropna(subset=["minuto_partido"]).copy()
    plot_df["minuto_partido"] = pd.to_numeric(plot_df["minuto_partido"], errors="coerce")
    plot_df = plot_df.dropna(subset=["minuto_partido"])
    if plot_df.empty:
        return "-", 0
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    plot_df["tramo"] = pd.cut(plot_df["minuto_partido"], bins=bins, labels=labels, include_lowest=True)
    counts = plot_df["tramo"].value_counts(sort=False)
    if counts.empty:
        return "-", 0
    label = str(counts.idxmax())
    return label, int(counts.max())


def _most_damaged_lane(seq: pd.DataFrame) -> str:
    if seq.empty:
        return "-"
    y_col = "y_fin" if "y_fin" in seq.columns else "ball_y_fin_m"
    if y_col not in seq.columns:
        return "-"
    weight_col = "score_critico" if "score_critico" in seq.columns else "indice_peligrosidad_accion"
    cols = [y_col] + ([weight_col] if weight_col in seq.columns else [])
    tmp = seq[cols].copy()
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[y_col])
    if tmp.empty:
        return "-"
    tmp["zona"] = tmp[y_col].apply(_lane_from_y)
    tmp["impacto"] = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0) if weight_col in tmp.columns else 1
    if tmp["impacto"].sum() <= 0:
        tmp["impacto"] = 1
    damaged = tmp.groupby("zona")["impacto"].sum().sort_values(ascending=False)
    return damaged.index[0] if not damaged.empty else "-"


def _main_disorganization_cause(seq: pd.DataFrame) -> str:
    if seq.empty:
        return "-"
    if "causa_tactica" in seq.columns:
        series = seq["causa_tactica"]
    elif "tipo_desorganizacion_principal" in seq.columns:
        series = seq["tipo_desorganizacion_principal"].apply(_cause_label)
    else:
        return "-"
    series = series.dropna().astype(str)
    series = series[~series.str.lower().isin(["", "-", "nan"])]
    if series.empty:
        return "-"
    return series.value_counts().idxmax()


def _summary_metric_cards(seq: pd.DataFrame, clusters: pd.DataFrame, zona: pd.DataFrame):
    tramo, tramo_n = _time_window_summary(seq)
    cluster_mas_repetido = "-"
    if not clusters.empty and {"cluster_trayectoria", "secuencias"}.issubset(clusters.columns):
        row = clusters.sort_values("secuencias", ascending=False).iloc[0]
        cluster_mas_repetido = row.get("patron_tactico", _as_tipologia(row["cluster_trayectoria"]))
    patron_peligroso = "-"
    if not seq.empty and {"patron_tactico", "indice_peligrosidad_accion"}.issubset(seq.columns):
        tmp = (
            seq.groupby("patron_tactico", as_index=False)
            .agg(ipar_medio=("indice_peligrosidad_accion", "mean"), secuencias=("secuencia_rival_id", "count"))
            .query("secuencias >= 2")
        )
        if not tmp.empty:
            patron_peligroso = tmp.sort_values("ipar_medio", ascending=False).iloc[0]["patron_tactico"]
    zona_danada = _most_damaged_lane(seq)
    if zona_danada == "-":
        zona_danada = _first(zona, "zona_ataque")
    causa_principal = _main_disorganization_cause(seq)
    entradas_ultimo_tercio = "-"
    if not seq.empty and "x_max_norm_m" in seq.columns:
        x_max = pd.to_numeric(seq["x_max_norm_m"], errors="coerce")
        entradas_ultimo_tercio = int(x_max.ge(FIELD_LENGTH_M * (2 / 3)).sum())
    tramo_critico = "-"
    try:
        chrono = _chronology_base(seq)
        windows = _chronology_windows(chrono)
        windows = windows[pd.to_numeric(windows.get("secuencias", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0)]
        if not windows.empty:
            critical = windows.sort_values("stress", ascending=False).iloc[0]
            tramo_critico = f"{critical['tramo']} ({int(critical['secuencias'])})"
    except Exception:
        tramo_critico = "-"
    return [
        ("Secuencias Rivales", len(seq)),
        ("Tiros Asociados", _sum_numeric(seq, "tipo_finalizacion_tiro")),
        ("Tiros A Puerta", _sum_numeric(seq, "tipo_finalizacion_tiro_puerta")),
        ("Goles Asociados", _sum_numeric(seq, "es_gol")),
        ("Duración Media", f"{_format_metric(_mean_numeric(seq, 'duracion_seg'))} s"),
        ("Pases Medios", _mean_numeric(seq, "num_pases_bepro")),
        ("IDD Medio", _mean_numeric(seq, "indice_desorganizacion")),
        ("IPO Medio", _mean_numeric(seq, "indice_peligrosidad_accion")),
        ("IDD Máximo", _max_numeric(seq, "indice_desorganizacion")),
        ("IPO Máximo", _max_numeric(seq, "indice_peligrosidad_accion")),
        ("xT Máximo", _max_numeric(seq, "xT_max")),
        ("Pitch Control Rival Medio", _mean_numeric(seq, "control_campo_rival_medio")),
        ("Pitch Control Rival Zona Peligrosa", _mean_numeric(seq, "control_zona_peligrosa_rival_medio")),
        ("Tipología Más Repetida", cluster_mas_repetido),
        ("Tipología Más Peligrosa", patron_peligroso),
        ("Zona Más Dañada", zona_danada),
        ("Causa Principal", causa_principal),
        ("Tramo Más Atacado", f"{tramo} ({tramo_n})" if tramo != "-" else "-"),
        ("Tramo Más Crítico", tramo_critico),
        ("Entradas Último Tercio", entradas_ultimo_tercio),
    ]


def _lane_from_y(y: float) -> str:
    if pd.isna(y):
        return "Desconocido"
    if y < FIELD_WIDTH_M * 0.20:
        return "Banda izquierda"
    if y < FIELD_WIDTH_M * 0.40:
        return "Half-space izquierdo"
    if y < FIELD_WIDTH_M * 0.60:
        return "Carril central"
    if y < FIELD_WIDTH_M * 0.80:
        return "Half-space derecho"
    return "Banda derecha"


def _lane_summary(seq: pd.DataFrame) -> pd.DataFrame:
    if seq.empty:
        return pd.DataFrame(columns=["carril", "secuencias"])
    y_col = "y_fin" if "y_fin" in seq.columns else "ball_y_fin_m"
    if y_col not in seq.columns:
        return pd.DataFrame(columns=["carril", "secuencias"])
    out = seq.copy()
    out["carril"] = pd.to_numeric(out[y_col], errors="coerce").apply(_lane_from_y)
    order = ["Banda izquierda", "Half-space izquierdo", "Carril central", "Half-space derecho", "Banda derecha"]
    counts = out["carril"].value_counts().reindex(order, fill_value=0).reset_index()
    counts.columns = ["carril", "secuencias"]
    return counts


def _plot_lane_pie(seq: pd.DataFrame, title: str = "Distribucion de secuencias por carril", height: int = 390):
    lanes = _lane_summary(seq)
    lanes = lanes[lanes["secuencias"].gt(0)].copy()
    if lanes.empty:
        st.info("No hay datos de carril para esta gráfica.")
        return
    lanes = lanes.sort_values("secuencias", ascending=False)
    red_scale = ["#8f0d23", "#c8102e", "#e45a67", "#f4b4b8", "#f7f0df"]
    fig = px.pie(
        lanes,
        names="carril",
        values="secuencias",
        hole=0.42,
        title=title,
        color_discrete_sequence=red_scale,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(line=dict(color="white", width=2)))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=54, b=20),
        showlegend=False,
    )
    _apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _plot_mean_passes_by_tipology(seq: pd.DataFrame, height: int = 390):
    needed = {"patron_tactico", "num_pases_bepro"}
    if seq.empty or not needed.issubset(seq.columns):
        st.info("No hay datos de pases por tipología para esta gráfica.")
        return
    plot_df = seq.copy()
    plot_df["num_pases_bepro"] = pd.to_numeric(plot_df["num_pases_bepro"], errors="coerce")
    plot_df = (
        plot_df.dropna(subset=["num_pases_bepro"])
        .groupby("patron_tactico", as_index=False)
        .agg(pases_medios=("num_pases_bepro", "mean"), secuencias=("secuencia_rival_id", "count"))
        .sort_values("pases_medios", ascending=False)
    )
    if plot_df.empty:
        st.info("No hay datos de pases por tipología para esta gráfica.")
        return
    _plot_bar(
        plot_df,
        "patron_tactico",
        "pases_medios",
        "Número medio de pases por tipología",
        color="patron_tactico",
        labels={"patron_tactico": "Tipología", "pases_medios": "Pases medios"},
        height=height,
    )


def _story_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="story-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_metric(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return value
        return f"{numeric:.2f}" if "." in value else str(int(numeric))
    return str(value)


def _format_metric_precise(value, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def _format_percent(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(numeric):
        return "-"
    return f"{numeric * 100:.0f}%"


def _metric_key_from_label(label: object) -> str | None:
    text = _display_text(label).upper()
    if text.startswith("DIF"):
        return None
    tokens = set(re.findall(r"[A-ZÁÉÍÓÚÜÑ]+", text))
    if "IDD" in tokens:
        return "indice_desorganizacion"
    if "IPO" in tokens or "IPAR" in tokens:
        return "indice_peligrosidad_accion"
    return None


def _index_badge_html(metric: str, value, compact: bool = False) -> str:
    reference = _interpretability_reference()
    thresholds = reference.get("thresholds", {}).get(metric, {})
    info = classify_index_value(metric, value, thresholds)
    label = html.escape(str(info.get("label", "-")))
    number = _format_metric_precise(info.get("value"), 3)
    color = html.escape(str(info.get("color", "#6b7280")))
    compact_class = " compact" if compact else ""
    return (
        f'<span class="metric-badge{compact_class}">'
        f'<span class="metric-dot" style="background:{color};"></span>'
        f'<span class="metric-label">{label}</span>'
        f'<span class="metric-number">({number})</span>'
        "</span>"
    )


def _metric_value_html(label: object, value) -> str:
    metric = _metric_key_from_label(label)
    if metric is not None:
        return _index_badge_html(metric, value, compact=True)
    if "PITCH CONTROL" in _display_text(label).upper():
        return html.escape(_format_percent(value))
    return html.escape(_format_metric(value))


def _first(df: pd.DataFrame, col: str, default: str = "-"):
    if df.empty or col not in df.columns:
        return default
    return df.iloc[0][col]


def _as_cluster(value) -> str:
    if value in (None, "-") or pd.isna(value):
        return "-"
    text = str(value)
    return text if text.upper().startswith("C") else f"C{text}"


def _as_tipologia(value) -> str:
    if value in (None, "-") or pd.isna(value):
        return "-"
    text = str(value)
    if text.lower().startswith("tipologia"):
        return text.replace("Tipologia", "Tipología").replace("tipologia", "tipología")
    if text.upper().startswith("T") and text[1:].isdigit():
        return f"Tipología {int(text[1:]) + 1}"
    try:
        return f"Tipología {int(float(text)) + 1}"
    except ValueError:
        return text


def _clusters_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    display = df.copy()
    if "tipologia" not in display.columns and "cluster_trayectoria" in display.columns:
        display["tipologia"] = display["cluster_trayectoria"].map(_as_tipologia)
    if "patron_tactico" not in display.columns and "cluster_trayectoria" in display.columns:
        display["patron_tactico"] = display["cluster_trayectoria"].map(_as_tipologia)
    display = display.rename(
        columns={
            "patron_tactico": "nombre tipología",
            "porcentaje": "% secuencias",
            "duracion_media": "duración media",
            "tiro_pct": "% con tiro",
            "zona_dominante": "zona dominante",
            "carril_dominante": "carril dominante",
        }
    )
    cols = [
        "tipologia",
        "nombre tipología",
        "secuencias",
        "% secuencias",
        "duración media",
        "tiros",
        "% con tiro",
        "zona dominante",
        "carril dominante",
    ]
    return display[[c for c in cols if c in display.columns]]


def _risk_level(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if numeric >= 0.66:
        return "alto"
    if numeric >= 0.33:
        return "medio"
    return "bajo"


def _kpi_value(kpis: pd.DataFrame, name: str, default: str = "-"):
    if kpis.empty or "kpi" not in kpis.columns or "valor" not in kpis.columns:
        return default
    rows = kpis.loc[kpis["kpi"].astype(str).eq(name), "valor"]
    if rows.empty:
        return default
    return rows.iloc[0]


def _sum_column(df: pd.DataFrame, col: str, default: str = "-"):
    if df.empty or col not in df.columns:
        return default
    values = pd.to_numeric(df[col], errors="coerce").fillna(0)
    total = values.sum()
    return int(total) if float(total).is_integer() else total


def _card(label: str, value, compact: bool = False):
    value_text = _metric_value_html(label, value)
    extra_class = " compact" if compact else ""
    st.markdown(
        f"""
        <div class="analysis-card{extra_class}">
            <div class="label">{html.escape(str(label))}</div>
            <div class="value">{value_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _summary_card_html(label: str, value) -> str:
    return (
        '<div class="analysis-card compact">'
        f'<div class="label">{html.escape(str(label))}</div>'
        f'<div class="value">{_metric_value_html(label, value)}</div>'
        "</div>"
    )


def _summary_combo_card_html(items: list[tuple[str, object]]) -> str:
    cols = max(1, len(items))
    extra_class = " triple cols-3" if cols >= 3 else f" cols-{cols}"
    parts = []
    for label, value in items:
        parts.append(
            '<div class="summary-combo-item">'
            f'<div class="label">{html.escape(str(label))}</div>'
            f'<div class="value">{_metric_value_html(label, value)}</div>'
            "</div>"
        )
    return (
        f'<div class="summary-combo-card{extra_class}">'
        f'<div class="summary-combo-grid" style="--combo-cols:{cols};">'
        f'{"".join(parts)}'
        "</div></div>"
    )


def _info_tip_html(text: str) -> str:
    text = str(_clean_app_text(text) or "")
    raw_lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = [text.strip()]
    rows_source = raw_lines[:]
    rows = []
    for idx, line in enumerate(rows_source):
        if ":" in line:
            label, body = line.split(":", 1)
            label = label.strip()
            body = body.strip()
        else:
            label, body = ("Resumen" if idx == 0 else "Clave"), line.strip()
        label_upper = label.upper()
        row_class = "info-tip-row"
        if label_upper in {"IDD", "IPO"}:
            row_class += " index-row"
        elif "·" in label or "%" in label:
            row_class += " child-row"
        elif label_upper in {"PITCH CONTROL", "XTHREAT"}:
            row_class += " glossary-row"
        rows.append(
            f'<span class="{row_class}">'
            f"<b>{html.escape(str(_clean_app_text(label) or label))}</b>"
            f"<span>{html.escape(str(_clean_app_text(body) or body))}</span>"
            "</span>"
        )
    return (
        '<span class="info-tip" tabindex="0" aria-label="Información">i'
        '<span class="info-tip-panel">'
        '<span class="info-tip-title">INFORMACIÓN</span>'
        f'<span class="info-tip-grid">{"".join(rows)}</span>'
        "</span>"
        "</span>"
    )


def _section_intro(title: str, body: str):
    title = str(_clean_app_text(title) or "")
    body = str(_clean_app_text(body) or "")
    st.markdown(
        f"""
        <div class="section-intro">
            <strong>{html.escape(title)}</strong>
            <span>{html.escape(str(body))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_heading(title: str):
    title = str(_clean_app_text(title) or "")
    st.markdown(f'<h2 class="app-page-heading">{html.escape(title)}</h2>', unsafe_allow_html=True)


def _subsection_heading(title: str):
    title = str(_clean_app_text(title) or "")
    st.markdown(f'<h3 class="app-subsection-heading">{html.escape(title)}</h3>', unsafe_allow_html=True)


def _selected_heading(title: str):
    title = str(_clean_app_text(title) or "")
    st.markdown(f'<h4 class="app-selected-heading">{html.escape(title)}</h4>', unsafe_allow_html=True)


def _info_heading(title: str, info: str, level: int = 3):
    tag = "h4" if level >= 4 else "h3"
    title = str(_clean_app_text(title) or "")
    st.markdown(
        f'<div class="section-info-heading"><{tag}>{html.escape(title)}</{tag}>{_info_tip_html(info)}</div>',
        unsafe_allow_html=True,
    )


def _display_text(value) -> str:
    text = str(value if value is not None else "-")
    return str(_clean_app_text(text) or "-")


def _summary_group_html(title: str, items: list[tuple[str, object] | list[tuple[str, object]]], cols: int = 2, note: str = "") -> str:
    cards = ""
    for item in items:
        if isinstance(item, list):
            cards += _summary_combo_card_html(item)
        else:
            label, value = item
            cards += _summary_card_html(label, value)
    info_html = _info_tip_html(note) if str(note).strip() else ""
    return (
        '<section class="summary-section wide">'
        '<div class="summary-section-head">'
        f'<div class="summary-section-title">{html.escape(title)}{info_html}</div>'
        '</div>'
        f'<div class="summary-card-grid" style="--summary-cols:{int(cols)};">{cards}</div>'
        "</section>"
    )


SUMMARY_FAMILY_TITLES = (
    "Volumen Ofensivo Rival",
    "Nivel De Desorganización Defensiva",
    "Contexto Táctico",
    "Contexto Temporal",
)
SUMMARY_INFO_TEXTS = {
    "Volumen Ofensivo Rival": (
        "Lectura Rápida: Resume si el rival acumuló presencia ofensiva y si esa presencia acabó en amenaza real.\n"
        "Volumen: Número de secuencias ofensivas evaluables del rival.\n"
        "Continuidad: Duración media y pases medios indican si pudo instalarse y circular.\n"
        "Finalización: Tiros, tiros a puerta y goles muestran si el volumen terminó en amenaza real."
    ),
    "Nivel De Desorganización Defensiva": (
        "Lectura Rápida: Explica cuánto se deformó la estructura y cuánto peligro real generó esa deformación.\n"
        "IDD: Índice De Desorganización Defensiva. Mide cuánto se desordena el bloque propio.\n"
        "10% Anchura Del Bloque: Penaliza que la línea defensiva se estire o pierda compactación horizontal.\n"
        "10% Distancia Al Balón: Penaliza que el defensor más cercano quede demasiado lejos del poseedor.\n"
        "30% Retroceso Defensivo: Mide cuánto tuvo que retroceder el bloque durante la secuencia.\n"
        "50% Pitch Control Rival: Mide cuánto territorio pasó a controlar el rival durante la acción.\n"
        "IPO: Índice De Peligrosidad Ofensiva. Mide cuánto peligro ofensivo genera la secuencia rival.\n"
        "15% Pitch Control En Zona Peligrosa: Valora si el rival controla zonas cercanas al área o espacios sensibles.\n"
        "35% xThreat Máximo: Valora el pico de amenaza esperada alcanzado por la posesión rival.\n"
        "50% Finalización: Premia el resultado de la acción; gol vale 1.00, tiro con xG vale 0.30 + 0.70*xG, tiro a puerta sin xG vale 0.75, tiro fuera sin xG vale 0.60, centro peligroso vale 0.30, pérdida rival a partir del 66% del campo vale 0.20, pérdida rival por debajo de ese umbral vale 0.10 y acción sin finalización vale 0.05.\n"
        "Pitch Control: Probabilidad territorial de que un equipo llegue antes o controle una zona.\n"
        "xThreat: Amenaza esperada según la posición del balón y la capacidad de generar peligro desde ahí."
    ),
    "Contexto Táctico": (
        "Lectura Rápida: Traduce los números a comportamientos entrenables.\n"
        "Tipología: Detecta qué patrón ofensivo se repite y cuál castiga más.\n"
        "Zona: Localiza por dónde se desajusta el bloque con más frecuencia.\n"
        "Causa: Resume el comportamiento defensivo que conviene corregir primero."
    ),
    "Contexto Temporal": (
        "Lectura Rápida: Sitúa los problemas dentro del reloj del partido.\n"
        "Tramo Más Atacado: Ventana donde el rival acumuló más secuencias.\n"
        "Tramo Más Crítico: Ventana donde se combinan volumen, IDD, IPO, tiros y entradas al último tercio.\n"
        "Tramo Unificado: Si ambos coinciden, se muestra una sola KPI porque el mayor volumen y el mayor riesgo aparecen en la misma fase.\n"
        "Uso Práctico: Separa una fase de dominio rival de una fase realmente peligrosa."
    ),
}
MOMENTUM_INFO = (
    "Construcción Del Momentum Defensivo: Acumula la presión rival minuto a minuto.\n"
    "35% IDD: Índice De Desorganización Defensiva. Aporta el peso de la deformación del bloque.\n"
    "35% IPO: Índice De Peligrosidad Ofensiva. Aporta el peso del peligro ofensivo generado por el rival.\n"
    "20% xThreat: Añade la amenaza esperada máxima alcanzada por la posesión.\n"
    "10% Último Tercio: Añade valor cuando la secuencia entra en zonas profundas.\n"
    "Memoria 88%: La curva conserva parte del valor anterior para que varias acciones seguidas mantengan la presión.\n"
    "Uso: Primero mira la curva, después revisa las alertas y las secuencias del tramo crítico."
)


MOMENTUM_INFO = (
    "Construccion del momentum defensivo: acumula la presion rival minuto a minuto.\n"
    "40% IDD: Indice de Desorganizacion Defensiva. Aporta el peso de la deformacion del bloque.\n"
    "60% IPO: Indice de Peligrosidad Ofensiva Rival. Tiene mayor peso porque el modelo mide presion acumulada rival.\n"
    "EWMA: M(t)=0.88*M(t-1)+0.12*(0.40*IDD+0.60*IPO).\n"
    "Memoria 88%: la curva conserva parte del valor anterior para que varias acciones seguidas mantengan la presion.\n"
    "Uso: primero mira la curva, despues revisa las alertas y las secuencias del tramo critico."
)


def _fallback_summary_family_notes(metric_map: dict[str, object]) -> dict[str, str]:
    pattern = _format_metric(metric_map.get("Tipología Más Peligrosa", "-"))
    zone = _format_metric(metric_map.get("Zona Más Dañada", "-"))
    cause = _format_metric(metric_map.get("Causa Principal", "-"))
    tramo = _format_metric(metric_map.get("Tramo Más Atacado", "-"))
    tramo_critico = _format_metric(metric_map.get("Tramo Más Crítico", "-"))
    return {
        "Volumen Ofensivo Rival": "Sitúa El Peso Del Partido: Permite leer si el rival acumuló presencia y si esa presencia se transformó en finalizaciones claras.",
        "Nivel De Desorganización Defensiva": "Resume Cuánto Se Deformó La Estructura: Cruza control territorial cedido, IDD, IPO y llegada a zonas sensibles.",
        "Contexto Táctico": f"Lectura Táctica: La lectura principal apunta a {pattern}, con especial atención a {zone}; la causa dominante fue {cause}.",
        "Contexto Temporal": f"Lectura Temporal: El tramo más atacado fue {tramo} y el tramo más crítico fue {tramo_critico}.",
    }


def _summary_family_prompt(metrics_json: str) -> str:
    return (
        "Actua como analista tactico profesional de futbol. "
        "Necesito 4 microlecturas para el apartado Resumen de una app de analisis defensivo. "
        "Devuelve SOLO JSON valido, sin markdown, con estas claves exactas: "
        f"{', '.join(SUMMARY_FAMILY_TITLES)}. "
        "Cada valor debe ser una lectura global de 1 o 2 frases, en espanol, clara y profesional. "
        "No repitas literalmente los datos que ya aparecen en los KPIs salvo que sea imprescindible. "
        "Enfocate en interpretar que significa cada familia para un entrenador con poco tiempo.\n\n"
        f"KPIS:\n{metrics_json}"
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _call_ollama_summary_notes_cached(metrics_json: str) -> dict[str, str]:
    text = _call_ollama_report(_summary_family_prompt(metrics_json))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    return {title: str(parsed.get(title, "")).strip() for title in SUMMARY_FAMILY_TITLES}


def _summary_family_notes(metric_map: dict[str, object]) -> dict[str, str]:
    fallback = _fallback_summary_family_notes(metric_map)
    metrics_json = json.dumps(
        {str(key): _format_metric(value) for key, value in metric_map.items()},
        ensure_ascii=False,
        sort_keys=True,
    )
    try:
        ollama_notes = _call_ollama_summary_notes_cached(metrics_json)
    except Exception:
        return fallback
    return {title: ollama_notes.get(title) or fallback[title] for title in SUMMARY_FAMILY_TITLES}


def _render_summary_metric_blocks(metrics: list[tuple[str, object]]):
    metric_map = {str(label): value for label, value in metrics}
    notes = SUMMARY_INFO_TEXTS

    def item(label: str) -> tuple[str, object]:
        return label, metric_map.get(label, "-")

    tramo_atacado = _format_metric(metric_map.get("Tramo Más Atacado", "-"))
    tramo_critico = _format_metric(metric_map.get("Tramo Más Crítico", "-"))
    if tramo_atacado != "-" and tramo_atacado == tramo_critico:
        temporal_items = [("Tramo Más Atacado Y Crítico", tramo_atacado)]
        temporal_cols = 1
    else:
        temporal_items = [item("Tramo Más Atacado"), item("Tramo Más Crítico")]
        temporal_cols = 2

    groups = [
        (
            "Volumen Ofensivo Rival",
            [
                item("Secuencias Rivales"),
                [item("Duración Media"), item("Pases Medios")],
                [item("Tiros Asociados"), item("Tiros A Puerta"), item("Goles Asociados")],
            ],
            3,
            notes["Volumen Ofensivo Rival"],
        ),
        (
            "Nivel De Desorganización Defensiva",
            [
                [item("IPO Medio"), item("IDD Medio")],
                [item("IPO Máximo"), item("IDD Máximo")],
                item("xT Máximo"),
                [item("Pitch Control Rival Medio"), item("Pitch Control Rival Zona Peligrosa")],
            ],
            4,
            notes["Nivel De Desorganización Defensiva"],
        ),
        (
            "Contexto Táctico",
            [
                item("Tipología Más Repetida"),
                item("Tipología Más Peligrosa"),
                item("Zona Más Dañada"),
                item("Causa Principal"),
            ],
            2,
            notes["Contexto Táctico"],
        ),
        (
            "Contexto Temporal",
            temporal_items,
            temporal_cols,
            notes["Contexto Temporal"],
        ),
    ]
    html_groups = "".join(_summary_group_html(title, items, cols=cols, note=note) for title, items, cols, note in groups)
    st.markdown(f'<div class="summary-section-grid">{html_groups}</div>', unsafe_allow_html=True)


def _render_desorg_metric_blocks(current: pd.DataFrame, diff_ddi: str, diff_ipar: str):
    groups = [
        (
            "Volumen Y Finalización",
            [
                ("Secuencias", len(current)),
                [
                    ("Tiros", _sum_numeric(current, "tipo_finalizacion_tiro")),
                    ("A Puerta", _sum_numeric(current, "tipo_finalizacion_tiro_puerta")),
                    ("Goles", _sum_numeric(current, "es_gol")),
                ],
            ],
            2,
            "",
        ),
        (
            "Nivel De Desorganización",
            [
                [
                    ("IPO Medio", _mean_numeric(current, "indice_peligrosidad_accion")),
                    ("IDD Medio", _mean_numeric(current, "indice_desorganizacion")),
                ],
                [
                    ("IPO Máximo", _max_numeric(current, "indice_peligrosidad_accion")),
                    ("IDD Máximo", _max_numeric(current, "indice_desorganizacion")),
                ],
                ("xT Máximo", _max_numeric(current, "xT_max")),
                [
                    ("Pitch Control Rival Medio", _mean_numeric(current, "control_campo_rival_medio")),
                    ("Pitch Control Rival Zona Peligrosa", _mean_numeric(current, "control_zona_peligrosa_rival_medio")),
                ],
            ],
            4,
            "",
        ),
        (
            "Comparación Con La Media",
            [
                [("Dif. IDD", diff_ddi), ("Dif. IPO", diff_ipar)],
            ],
            1,
            "",
        ),
    ]
    html_groups = "".join(_summary_group_html(title, items, cols=cols, note=note) for title, items, cols, note in groups)
    st.markdown(f'<div class="summary-section-grid">{html_groups}</div>', unsafe_allow_html=True)


def render_login() -> bool:
    if st.session_state.get("logged_in"):
        return True
    left, right = st.columns([1.0, 0.82])
    with left:
        st.markdown(
            """
            <div class="login-visual">
                <div class="login-brand">
                    <div>Defensive Intelligence</div>
                    <span>v1.0</span>
                </div>
                <div class="login-copy">
                    <span class="login-kicker">Tracking · Eventing · Tactical Intelligence</span>
                    <h1>Análisis defensivo avanzado</h1>
                    <p>Visualiza secuencias críticas, patrones ofensivos rivales y momentos de presión acumulada
                    a partir de datos de tracking y eventing.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="login-access-heading">
                <span>Acceso privado</span>
                <strong>Entrar al panel</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Tipo de acceso",
            ["Iniciar sesion", "Registrarse"],
            horizontal=True,
            label_visibility="collapsed",
        )
        with st.form("login_form"):
            user = st.text_input("Correo electrónico")
            password = st.text_input("Contraseña", type="password")
            submitted = st.form_submit_button(mode, type="primary", use_container_width=True)
        st.markdown(
            '<div class="login-access-note">Acceso restringido para cuerpo técnico y analistas.</div>',
            unsafe_allow_html=True,
        )
        if submitted:
            clean_user = user.strip().lower()
            if not clean_user or not password:
                st.error("Introduce usuario/email y contrasena.")
            elif mode == "Registrarse":
                if not _firebase_enabled():
                    st.error("Para registrar usuarios necesitas configurar Firebase en .streamlit/secrets.toml.")
                else:
                    try:
                        _register_guest(clean_user, password)
                        st.success("Cuenta creada como invitado. Un administrador podra asignarte permisos.")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))
            elif _firebase_enabled():
                try:
                    _login_firebase(clean_user, password)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
            elif _login_local(clean_user, password):
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")
    return False


def render_topbar(team: dict | None = None):
    title = "Plataforma de análisis defensivo"
    if team:
        title = f"Panel de análisis de la estructura defensiva | {_team_display_name(team)}"
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="page-title-lockup">{html.escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_sidebar():
    user, role = _current_user_label()
    view = str(st.session_state.get("app_view", "portal"))
    team = _team_by_id(st.session_state.get("selected_team_id"))
    st.sidebar.markdown(
        f"""
        <div class="sidebar-account">
            <strong>{html.escape(user)}</strong>
            <span>{html.escape(role)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if _is_admin() and st.sidebar.button("Gestionar usuarios", use_container_width=True):
        _set_app_view("admin_users")

    if view == "admin_users":
        if st.sidebar.button("Menu principal", use_container_width=True):
            _set_app_view("portal")
    elif view == "club":
        if st.sidebar.button("Menú principal", use_container_width=True):
            _set_app_view("portal")
    elif view == "analysis":
        if st.sidebar.button("Menú principal", use_container_width=True):
            _set_app_view("portal")
        if st.sidebar.button("Espacio del club", use_container_width=True):
            _set_app_view("club", int(team["team_id"]))

    if st.sidebar.button("Cerrar sesión", use_container_width=True):
        _logout()

    if view == "analysis" and st.session_state.get("selected_match_id") is not None:
        st.sidebar.divider()
        team_name = _team_display_name(team)
        logo = _team_logo_html(team.get("team_id"), _initials(team_name), "sidebar-crest")
        st.sidebar.markdown(
            f"""
            <div class="sidebar-club">
                {logo}
                <h2>{html.escape(team_name)}</h2>
                <p>Análisis defensivo IDD/IPO</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        matches = _team_matches(int(team["team_id"]))
        labels = _match_label_map(matches)
        if labels:
            current_match = int(st.session_state.get("selected_match_id"))
            label_list = list(labels.keys())
            current_label = next((label for label, mid in labels.items() if mid == current_match), label_list[0])
            selected_label = st.sidebar.selectbox(
                "Partido activo",
                label_list,
                index=label_list.index(current_label),
                key=f"sidebar_match_{int(team['team_id'])}_{current_match}",
            )
            selected_match = labels[selected_label]
            if selected_match != current_match:
                st.session_state["selected_match_id"] = int(selected_match)
                st.rerun()


def _int_flag(value) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _si_no(value) -> str:
    return "si" if _int_flag(value) else "no"


def _load_sequence_detail(match_id: int, meta: dict) -> pd.DataFrame:
    tables = meta.get("tables", {})
    filename = tables.get("secuencias_detalle", "secuencias_detalle.csv")
    df = _load_table(match_id, filename)
    if df.empty:
        df = _load_table(match_id, tables.get("ranking_secuencias", "ranking_secuencias.csv"))
        if not df.empty and "cluster_trayectoria" in df.columns:
            df["tipologia"] = df["cluster_trayectoria"].map(_as_tipologia)
    if "tipologia" not in df.columns and "cluster_trayectoria" in df.columns:
        df["tipologia"] = df["cluster_trayectoria"].map(_as_tipologia)
    for col in [
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "score_critico",
        "xT_max",
        "xT_added",
        "num_pases_bepro",
        "num_eventos",
        "num_eventos_rivales",
        "minuto_partido",
        "duracion_seg",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
        "control_campo_subiza_medio",
        "control_campo_rival_medio",
        "control_zona_peligrosa_subiza_medio",
        "control_zona_peligrosa_rival_medio",
        "pitch_control_rival_detras_linea",
        "perdida_control_campo_subiza",
        "incremento_control_zona_peligrosa_rival",
        "distancia_tiro_seq_ms",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "es_gol" in df.columns:
        if df["es_gol"].dtype == bool:
            df["es_gol"] = df["es_gol"].astype(int)
        else:
            df["es_gol"] = df["es_gol"].map(_clean_app_text).astype(str).str.lower().isin({"true", "1", "si", "sí", "yes"}).astype(int)
    return _clean_text_columns(df)


def _tipology_options(df: pd.DataFrame) -> list[str]:
    if df.empty or "tipologia" not in df.columns:
        return ["Todas"]
    def _tipology_sort_key(value: str) -> tuple[int, str]:
        match = re.search(r"(\d+)", str(value))
        return (int(match.group(1)) if match else 999, str(value))

    values = sorted(df["tipologia"].dropna().astype(str).unique(), key=_tipology_sort_key)
    return ["Todas", *values]


def _filter_tipology(df: pd.DataFrame, tipologia: str) -> pd.DataFrame:
    if tipologia == "Todas" or df.empty or "tipologia" not in df.columns:
        return df
    return df[df["tipologia"].astype(str).eq(tipologia)].copy()


def _hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _hms_from_value(value) -> str:
    try:
        return _hms(int(float(value)))
    except (TypeError, ValueError):
        return "00:00:00"


def _bepro_schedule_id(match_id: int) -> int | None:
    env_key = f"BEPRO_SCHEDULE_ID_{int(match_id)}"
    value = os.getenv(env_key, "").strip()
    if value:
        try:
            return int(value)
        except ValueError:
            return None
    return BEPRO_SCHEDULE_IDS.get(int(match_id))


def _video_url(match_id: int, row: pd.Series) -> tuple[str, bool] | None:
    template = os.getenv("BEPRO_VIDEO_URL_TEMPLATE", "").strip() or BEPRO_MATCH_URL_TEMPLATE_DEFAULT
    using_custom_template = bool(os.getenv("BEPRO_VIDEO_URL_TEMPLATE", "").strip())
    seconds = row.get("start_time_seg", row.get("end_time_seg", 0))
    try:
        seconds = int(float(seconds))
    except (TypeError, ValueError):
        seconds = 0
    minute = row.get("minuto_partido", 0)
    try:
        minute = float(minute)
    except (TypeError, ValueError):
        minute = 0
    end_seconds = row.get("end_time_seg", seconds)
    try:
        end_seconds = int(float(end_seconds))
    except (TypeError, ValueError):
        end_seconds = seconds
    bepro_match_id = int(os.getenv(f"BEPRO_MATCH_ID_{int(match_id)}", str(int(match_id))).strip() or int(match_id))
    schedule_id = _bepro_schedule_id(match_id) or ""
    url = template.format(
        match_id=int(match_id),
        bepro_match_id=bepro_match_id,
        schedule_id=schedule_id,
        secuencia=int(row.get("secuencia_rival_id", 0)),
        seconds=seconds,
        end_seconds=end_seconds,
        start_ms=seconds * 1000,
        end_ms=end_seconds * 1000,
        start_hms=_hms(seconds),
        minute=minute,
    )
    return url, using_custom_template


def _bepro_library_url(library_item_id: int | None = None, clip_id: int | None = None) -> str:
    template = os.getenv("BEPRO_LIBRARY_URL_TEMPLATE", "").strip() or BEPRO_LIBRARY_URL_TEMPLATE_DEFAULT
    try:
        return template.format(
            library_item_id=int(library_item_id or 0),
            clip_id=int(clip_id or 0),
        )
    except (KeyError, ValueError):
        return BEPRO_LIBRARY_URL_TEMPLATE_DEFAULT


def _bepro_hls_url(match_id: int) -> str | None:
    videos = _load_bepro_video_index()
    if videos.empty or "match_id" not in videos.columns or "hls_url" not in videos.columns:
        return None
    current = videos[videos["match_id"].astype(int).eq(int(match_id))]
    if current.empty:
        return None
    url = current.iloc[0].get("hls_url")
    if pd.isna(url):
        return None
    return str(url).strip() or None


def _bepro_video_row(match_id: int) -> pd.Series | None:
    videos = _load_bepro_video_index()
    if videos.empty or "match_id" not in videos.columns:
        return None
    current = videos[videos["match_id"].astype(int).eq(int(match_id))]
    if current.empty:
        return None
    return current.iloc[0]


def _sequence_match_bounds(row: pd.Series) -> tuple[float, float]:
    start = row.get("start_time_seg", row.get("end_time_seg", 0))
    end = row.get("end_time_seg", start)
    try:
        start = float(start)
    except (TypeError, ValueError):
        start = 0.0
    try:
        end = float(end)
    except (TypeError, ValueError):
        end = start + 8.0
    if end <= start:
        end = start + 8.0
    # Two seconds of context help the action make sense without losing the exact segment.
    return max(start - 2.0, 0.0), end + 2.0


def _match_second_to_video_second(match_id: int, period: int, match_second: float) -> float:
    info = _bepro_video_row(match_id)
    if info is None:
        return match_second
    try:
        period = int(period)
    except (TypeError, ValueError):
        period = 0
    padding_ms = pd.to_numeric(info.get(f"period{period}_padding_ms"), errors="coerce")
    period_start_ms = pd.to_numeric(info.get(f"period{period}_start_ms"), errors="coerce")
    if pd.isna(padding_ms) or pd.isna(period_start_ms):
        return match_second
    video_ms = float(match_second) * 1000 - float(period_start_ms) + float(padding_ms)
    return max(video_ms / 1000, 0.0)


def _sequence_bounds(match_id: int, row: pd.Series) -> tuple[float, float, float, float]:
    match_start, match_end = _sequence_match_bounds(row)
    period = row.get("period", 0)
    video_start = _match_second_to_video_second(match_id, period, match_start)
    video_end = _match_second_to_video_second(match_id, period, match_end)
    if video_end <= video_start:
        video_end = video_start + max(match_end - match_start, 8.0)
    return video_start, video_end, match_start, match_end


def _infer_sequence_attack_left(row: pd.Series) -> bool | None:
    """Infer the rival attacking goal from precomputed normalized coordinates."""
    votes: list[bool] = []
    for raw_col, norm_col in [
        ("ball_x_fin_m", "ball_x_fin_norm_m"),
        ("ball_x_ini_m", "ball_x_ini_norm_m"),
        ("x_fin", "ball_x_fin_norm_m"),
    ]:
        if raw_col not in row.index or norm_col not in row.index:
            continue
        raw = pd.to_numeric(row.get(raw_col), errors="coerce")
        norm = pd.to_numeric(row.get(norm_col), errors="coerce")
        if pd.isna(raw) or pd.isna(norm):
            continue
        raw_f = float(np.clip(raw, 0, FIELD_LENGTH_M))
        norm_f = float(np.clip(norm, 0, FIELD_LENGTH_M))
        dist_attack_right = abs(norm_f - raw_f)
        dist_attack_left = abs(norm_f - (FIELD_LENGTH_M - raw_f))
        if abs(dist_attack_left - dist_attack_right) >= 1.0:
            votes.append(dist_attack_left < dist_attack_right)
    if votes:
        return sum(votes) > (len(votes) / 2)
    return None


def _infer_period_attack_left(match_id: int, period: int) -> bool | None:
    seq_detail = _load_sequence_detail(match_id, {})
    needed = {"period", "ball_x_fin_m", "ball_x_fin_norm_m"}
    if seq_detail.empty or not needed.issubset(seq_detail.columns):
        return None
    tmp = seq_detail[list(needed)].copy()
    tmp["period"] = pd.to_numeric(tmp["period"], errors="coerce")
    tmp["ball_x_fin_m"] = pd.to_numeric(tmp["ball_x_fin_m"], errors="coerce")
    tmp["ball_x_fin_norm_m"] = pd.to_numeric(tmp["ball_x_fin_norm_m"], errors="coerce")
    tmp = tmp[tmp["period"].eq(period)].dropna()
    if tmp.empty:
        return None
    raw = tmp["ball_x_fin_m"].clip(0, FIELD_LENGTH_M)
    norm = tmp["ball_x_fin_norm_m"].clip(0, FIELD_LENGTH_M)
    left_like = (norm - (FIELD_LENGTH_M - raw)).abs()
    right_like = (norm - raw).abs()
    clear = (left_like - right_like).abs().ge(1.0)
    if not clear.any():
        return None
    return bool((left_like[clear] < right_like[clear]).mean() >= 0.5)


def _sequence_minimap_payload(match_id: int, row: pd.Series) -> dict:
    seq_id = int(row.get("secuencia_rival_id", 0))
    paths = _load_table(match_id, "trayectorias_ligeras.csv")
    if paths.empty or "secuencia_rival_id" not in paths.columns:
        return {}

    paths["secuencia_rival_id"] = pd.to_numeric(paths["secuencia_rival_id"], errors="coerce")
    current = paths[paths["secuencia_rival_id"].eq(seq_id)].copy()
    needed = {"match_time", "ball_x_m", "ball_y_m"}
    if current.empty or not needed.issubset(current.columns):
        return {}

    for col in ["match_time", "ball_x_m", "ball_y_m", "period", "point_order"]:
        if col in current.columns:
            current[col] = pd.to_numeric(current[col], errors="coerce")
    current = (
        current.dropna(subset=["match_time", "ball_x_m", "ball_y_m"])
        .sort_values(["point_order", "match_time"], na_position="last")
        .drop_duplicates("match_time")
    )
    if current.empty:
        return {}

    period_value = pd.to_numeric(
        row.get("period", current["period"].dropna().iloc[0] if "period" in current.columns and not current["period"].dropna().empty else 0),
        errors="coerce",
    )
    period = int(period_value) if pd.notna(period_value) else 0
    n_x, n_y = 32, 24
    xt_grid = _crear_xt_grid(n_x=n_x, n_y=n_y)

    def _xt_for(x_value: float, y_value: float, attack_left: bool) -> float:
        xt_x = FIELD_LENGTH_M - x_value if attack_left else x_value
        cell_x, cell_y = _asignar_celda_xt(
            float(np.clip(xt_x, 0, FIELD_LENGTH_M)),
            float(np.clip(y_value, 0, FIELD_WIDTH_M)),
            n_x,
            n_y,
        )
        return float(xt_grid[cell_x, cell_y])

    raw_points = []
    for _, point in current.iterrows():
        raw_x = float(np.clip(point["ball_x_m"], 0, FIELD_LENGTH_M))
        raw_y = float(np.clip(point["ball_y_m"], 0, FIELD_WIDTH_M))
        raw_points.append((raw_x, raw_y, float(point["match_time"]) / 1000))
    if len(raw_points) < 2:
        return {}

    attack_left = _infer_sequence_attack_left(row)
    if attack_left is None:
        attack_left = _infer_period_attack_left(match_id, period)
    if attack_left is None:
        tail = raw_points[max(0, len(raw_points) - 5):]
        right_score = max(_xt_for(x, y, False) for x, y, _ in raw_points) + 1.35 * np.mean([_xt_for(x, y, False) for x, y, _ in tail])
        left_score = max(_xt_for(x, y, True) for x, y, _ in raw_points) + 1.35 * np.mean([_xt_for(x, y, True) for x, y, _ in tail])
        attack_left = bool(left_score > right_score)
    mirror = attack_left
    grid_display = xt_grid.T[:, ::-1] if mirror else xt_grid.T
    points = []
    for raw_x, raw_y, match_second in raw_points:
        display_x = raw_x
        display_y = raw_y
        points.append(
            {
                "videoTime": round(_match_second_to_video_second(match_id, period, match_second), 3),
                "matchSecond": round(match_second, 3),
                "x": round(display_x, 3),
                "y": round(display_y, 3),
                "xPct": round(display_x / FIELD_LENGTH_M * 100, 3),
                "yPct": round(display_y / FIELD_WIDTH_M * 100, 3),
                "xT": round(_xt_for(raw_x, raw_y, attack_left), 4),
            }
        )

    return {
        "fieldLength": FIELD_LENGTH_M,
        "fieldWidth": FIELD_WIDTH_M,
        "gridNx": n_x,
        "gridNy": n_y,
        "attackDirection": "left" if mirror else "right",
        "maxXT": round(float(np.nanmax(xt_grid)), 4),
        "grid": np.round(grid_display, 4).tolist(),
        "points": points,
    }


def _bepro_credentials() -> tuple[str | None, str | None]:
    email = (
        os.getenv("BEPRO_EMAIL", "").strip()
        or str(st.session_state.get("bepro_email", "")).strip()
        or str(st.session_state.get("bepro_email_inline", "")).strip()
    )
    password = (
        os.getenv("BEPRO_PASSWORD", "").strip()
        or str(st.session_state.get("bepro_password", "")).strip()
        or str(st.session_state.get("bepro_password_inline", "")).strip()
    )
    return email or None, password or None


def _bepro_token() -> str | None:
    env_token = os.getenv("BEPRO_AUTH_TOKEN", "").strip()
    if env_token:
        return env_token
    cached = st.session_state.get("bepro_auth_token")
    if cached:
        return str(cached)
    email, password = _bepro_credentials()
    if not email or not password:
        return None
    response = requests.post(
        f"{BEPRO_API_BASE}/login",
        json={"email": email, "password": password},
        headers={"X-Bepro-Client": "space"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    token = (
        data.get("token")
        or data.get("accessToken")
        or data.get("access_token")
        or data.get("authenticationToken")
    )
    if isinstance(data.get("data"), dict):
        token = token or data["data"].get("token") or data["data"].get("accessToken")
    if token:
        st.session_state["bepro_auth_token"] = str(token)
    return str(token) if token else None


def _bepro_headers() -> dict[str, str] | None:
    token = _bepro_token()
    if not token:
        return None
    return {
        "Authorization": token,
        "X-Bepro-Client": "space",
        "Content-Type": "application/json",
    }


def _bepro_clip_payload(match_id: int, row: pd.Series) -> list[dict]:
    info = _bepro_video_row(match_id)
    if info is None:
        raise ValueError("No hay datos BePro para este partido en outputs/app_data/bepro_videos.csv.")
    video_start, video_end, match_start, match_end = _sequence_bounds(match_id, row)
    seq_id = int(row.get("secuencia_rival_id", 0))
    tipologia = str(row.get("tipologia", "-"))
    name = (
        f"TFG rival | partido {int(match_id)} | secuencia {seq_id} | "
        f"{tipologia} | {_hms_from_value(match_start)}-{_hms_from_value(match_end)}"
    )
    return [
        {
            "name": name,
            "clips": [
                {
                    "scheduleVideoId": int(info["schedule_video_id"]),
                    "scheduleId": int(info["schedule_id"]),
                    "videoId": int(info["video_id"]),
                    "startVideoTime": int(round(video_start * 1000)),
                    "endVideoTime": int(round(video_end * 1000)),
                    "sourceType": "FULL_MATCH",
                }
            ],
        }
    ]


def _create_bepro_clip(match_id: int, row: pd.Series) -> dict:
    headers = _bepro_headers()
    if headers is None:
        raise RuntimeError("Faltan credenciales de BePro.")
    payload = _bepro_clip_payload(match_id, row)
    response = requests.post(
        f"{BEPRO_API_BASE}/library/items/bulk",
        json=payload,
        headers=headers,
        timeout=30,
    )
    if response.status_code == 401 and not headers["Authorization"].lower().startswith("bearer "):
        headers = {**headers, "Authorization": f"Bearer {headers['Authorization']}"}
        response = requests.post(
            f"{BEPRO_API_BASE}/library/items/bulk",
            json=payload,
            headers=headers,
            timeout=30,
        )
    response.raise_for_status()
    data = response.json()
    item = data[0] if isinstance(data, list) and data else data
    if isinstance(item, dict) and isinstance(item.get("data"), dict):
        item = item["data"]
    if not isinstance(item, dict):
        raise RuntimeError("BePro ha respondido con un formato inesperado al crear el clip.")
    clips = item.get("clips") or []
    clip = clips[0] if clips and isinstance(clips[0], dict) else {}
    return {
        "library_item_id": item.get("id"),
        "clip_id": clip.get("id"),
        "name": item.get("name") or payload[0]["name"],
        "url": _bepro_library_url(item.get("id"), clip.get("id")),
    }


def _render_sequence_player(match_id: int, row: pd.Series):
    hls_url = _bepro_hls_url(match_id)
    if not hls_url:
        st.warning("No encuentro la URL directa del vídeo BePro para este partido. Revisa outputs/app_data/bepro_videos.csv.")
        return
    start, end, match_start, match_end = _sequence_bounds(match_id, row)
    minimap_payload = _sequence_minimap_payload(match_id, row)
    title = (
        f"Secuencia {int(row.get('secuencia_rival_id', 0))} | "
        f"partido {_hms_from_value(match_start)} - {_hms_from_value(match_end)}"
    )
    html = f"""
    <style>
      #video-stage:fullscreen {{
        width: 100vw !important;
        height: 100vh !important;
        max-height: none !important;
        border-radius: 0 !important;
        background: #000 !important;
      }}
      #video-stage:fullscreen video {{
        width: 100vw !important;
        height: 100vh !important;
        object-fit: cover !important;
      }}
      #xt-minimap {{
        position: absolute;
        left: 8px;
        bottom: 72px;
        width: min(305px, 22vw);
        min-width: 215px;
        aspect-ratio: 105 / 68;
        background: rgba(11, 16, 32, 0.08);
        border: 1px solid rgba(255,255,255,0.38);
        border-radius: 8px;
        box-shadow: 0 14px 34px rgba(0,0,0,0.34);
        overflow: hidden;
        backdrop-filter: blur(3px);
      }}
      #video-stage:fullscreen #xt-minimap {{
        width: min(350px, 18vw);
        left: 18px;
        bottom: 86px;
      }}
      #xt-minimap-canvas {{
        width: 100%;
        height: 100%;
        display: block;
      }}
      #xt-minimap-value {{
        position: absolute;
        left: 10px;
        right: 10px;
        bottom: 4px;
        color: #ffffff;
        font-size: 11px;
        font-weight: 800;
        text-align: center;
        padding: 2px 5px;
        border-radius: 999px;
        background: rgba(10,16,32,.46);
        text-shadow: 0 1px 3px rgba(0,0,0,.65);
      }}
      #video-controls {{
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        display: grid;
        grid-template-columns: auto auto auto auto minmax(120px, 1fr) auto auto;
        align-items: center;
        gap: 8px;
        padding: 28px 14px 12px;
        background: linear-gradient(180deg, rgba(0,0,0,0), rgba(0,0,0,.72));
        opacity: .96;
        transition: opacity .18s ease;
        z-index: 8;
        pointer-events: auto;
      }}
      #video-controls button {{
        width: 34px;
        height: 30px;
        border: 0;
        border-radius: 999px;
        background: rgba(255,255,255,.16);
        color: #fff;
        font-size: 15px;
        font-weight: 900;
        cursor: pointer;
        display: grid;
        place-items: center;
      }}
      #video-controls button:hover {{
        background: rgba(255,255,255,.26);
      }}
      #sequence-progress {{
        width: 100%;
        accent-color: #ffffff;
        cursor: pointer;
      }}
      #sequence-time {{
        color: #ffffff;
        font-size: 12px;
        font-weight: 800;
        text-shadow: 0 1px 4px rgba(0,0,0,.75);
        white-space: nowrap;
      }}
    </style>
    <div style="background:#30384a;border:1px solid rgba(255,255,255,0.14);border-radius:8px;padding:10px;font-family:Arial,sans-serif;">
      <div style="color:#f5f7fb;font-weight:700;margin:0 0 8px 0;">{title}</div>
      <div id="video-stage" style="position:relative;width:100%;aspect-ratio:16/9;max-height:720px;overflow:hidden;background:#30384a;border-radius:6px;cursor:grab;touch-action:none;">
        <video id="bepro-sequence-player" autoplay muted playsinline style="width:100%;height:100%;object-fit:cover;object-position:50% 50%;transform-origin:center center;will-change:transform,object-position;"></video>
        <div id="xt-minimap" aria-label="Mapa xThreat sincronizado">
          <canvas id="xt-minimap-canvas"></canvas>
          <div id="xt-minimap-value">xThreat en tiempo real: --</div>
        </div>
        <div id="video-controls" aria-label="Controles de reproducción">
          <button id="play-toggle" type="button" title="Reproducir/Pausar">&#9654;</button>
          <button id="replay-sequence" type="button" title="Ver de nuevo">&#8634;</button>
          <button id="zoom-out-view" type="button" title="Empequeñecer">&minus;</button>
          <button id="zoom-in-view" type="button" title="Agrandar">+</button>
          <input id="sequence-progress" type="range" min="0" max="1000" value="0" aria-label="Progreso de la secuencia">
          <span id="sequence-time">0s / 0s</span>
          <button id="fullscreen-view" type="button" title="Pantalla completa">&#9974;</button>
        </div>
      </div>
      <div id="bepro-sequence-status" style="color:#cbd2df;font-size:12px;margin-top:7px;">Cargando secuencia...</div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <script>
      const video = document.getElementById("bepro-sequence-player");
      const stage = document.getElementById("video-stage");
      const status = document.getElementById("bepro-sequence-status");
      const fullscreenView = document.getElementById("fullscreen-view");
      const playToggle = document.getElementById("play-toggle");
      const replaySequence = document.getElementById("replay-sequence");
      const zoomInView = document.getElementById("zoom-in-view");
      const zoomOutView = document.getElementById("zoom-out-view");
      const sequenceProgress = document.getElementById("sequence-progress");
      const sequenceTime = document.getElementById("sequence-time");
      const source = {json.dumps(hls_url)};
      const minimap = {json.dumps(minimap_payload)};
      const start = {start:.3f};
      const end = {end:.3f};
      const baseZoom = 1.34;
      const minVisibleX = 0;
      const maxVisibleX = 100;
      const minVisibleY = 56;
      const maxVisibleY = 56;
      let zoom = baseZoom;
      let panX = 0;
      let panY = 0;
      let posX = 50;
      let posY = 56;
      let dragging = false;
      let dragStartX = 0;
      let dragStartY = 0;
      let basePanX = 0;
      let basePanY = 0;
      let basePosX = 50;
      let basePosY = 50;
      const minimapBox = document.getElementById("xt-minimap");
      const minimapCanvas = document.getElementById("xt-minimap-canvas");
      const minimapValue = document.getElementById("xt-minimap-value");
      const minimapCtx = minimapCanvas ? minimapCanvas.getContext("2d") : null;
      let minimapReady = !!(minimap && minimap.points && minimap.points.length > 1 && minimap.grid);
      if (!minimapReady && minimapBox) {{
        minimapBox.style.display = "none";
      }}

      function updateView() {{
        video.style.transform = "translate(" + panX + "px, " + panY + "px) scale(" + zoom + ")";
        video.style.objectPosition = posX + "% " + posY + "%";
      }}
      function setZoom(nextZoom) {{
        zoom = Math.max(baseZoom, Math.min(4.0, nextZoom));
        panX = 0;
        panY = 0;
        clampPan();
        updateView();
      }}
      function threatColor(value) {{
        const maxXT = Math.max(minimap.maxXT || 0.32, 0.001);
        const t = Math.max(0, Math.min(1, value / maxXT));
        let r, g, b;
        if (t < 0.48) {{
          const k = t / 0.48;
          r = Math.round(31 + (242 - 31) * k);
          g = Math.round(64 + (201 - 64) * k);
          b = Math.round(122 + (76 - 122) * k);
        }} else {{
          const k = (t - 0.48) / 0.52;
          r = Math.round(242 + (200 - 242) * k);
          g = Math.round(201 + (16 - 201) * k);
          b = Math.round(76 + (46 - 76) * k);
        }}
        return "rgba(" + r + "," + g + "," + b + ",0.68)";
      }}
      function pitchRect(width, height) {{
        const pad = 10;
        const labelH = 18;
        return {{
          x: pad,
          y: 8,
          w: width - pad * 2,
          h: Math.max(1, height - labelH - 14)
        }};
      }}
      function mapPoint(point, rect) {{
        const px = rect.x + (point.x / minimap.fieldLength) * rect.w;
        const py = rect.y + (1 - point.y / minimap.fieldWidth) * rect.h;
        return [px, py];
      }}
      function interpolatedBall(currentTime) {{
        if (!minimapReady) return null;
        const pts = minimap.points;
        if (currentTime <= pts[0].videoTime) return pts[0];
        if (currentTime >= pts[pts.length - 1].videoTime) return pts[pts.length - 1];
        for (let i = 0; i < pts.length - 1; i++) {{
          const a = pts[i];
          const b = pts[i + 1];
          if (currentTime >= a.videoTime && currentTime <= b.videoTime) {{
            const span = Math.max(0.001, b.videoTime - a.videoTime);
            const k = (currentTime - a.videoTime) / span;
            return {{
              videoTime: currentTime,
              x: a.x + (b.x - a.x) * k,
              y: a.y + (b.y - a.y) * k,
              xT: a.xT + (b.xT - a.xT) * k,
              index: i + k
            }};
          }}
        }}
        return pts[pts.length - 1];
      }}
      function drawPitchLines(ctx, rect) {{
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,0.82)";
        ctx.lineWidth = 1.4;
        ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
        const xMid = rect.x + rect.w * 0.5;
        const yMid = rect.y + rect.h * 0.5;
        ctx.beginPath();
        ctx.moveTo(xMid, rect.y);
        ctx.lineTo(xMid, rect.y + rect.h);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(xMid, yMid, rect.h * 0.205, 0, Math.PI * 2);
        ctx.stroke();
        const boxW = rect.w * 0.155;
        const boxH = rect.h * 0.60;
        ctx.strokeRect(rect.x + 0.5, rect.y + (rect.h - boxH) / 2, boxW, boxH);
        ctx.strokeRect(rect.x + rect.w - boxW - 0.5, rect.y + (rect.h - boxH) / 2, boxW, boxH);
        const smallW = rect.w * 0.058;
        const smallH = rect.h * 0.31;
        ctx.strokeRect(rect.x + 0.5, rect.y + (rect.h - smallH) / 2, smallW, smallH);
        ctx.strokeRect(rect.x + rect.w - smallW - 0.5, rect.y + (rect.h - smallH) / 2, smallW, smallH);
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.beginPath();
        ctx.arc(rect.x + rect.w * 0.11, yMid, 1.45, 0, Math.PI * 2);
        ctx.arc(rect.x + rect.w * 0.89, yMid, 1.45, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.70)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (minimap.attackDirection === "left") {{
          ctx.moveTo(rect.x + 15, rect.y + 10);
          ctx.lineTo(rect.x + 6, rect.y + 10);
          ctx.lineTo(rect.x + 10, rect.y + 6);
          ctx.moveTo(rect.x + 6, rect.y + 10);
          ctx.lineTo(rect.x + 10, rect.y + 14);
        }} else {{
          ctx.moveTo(rect.x + rect.w - 15, rect.y + 10);
          ctx.lineTo(rect.x + rect.w - 6, rect.y + 10);
          ctx.lineTo(rect.x + rect.w - 10, rect.y + 6);
          ctx.moveTo(rect.x + rect.w - 6, rect.y + 10);
          ctx.lineTo(rect.x + rect.w - 10, rect.y + 14);
        }}
        ctx.stroke();
        ctx.restore();
      }}
      function drawSoccerBall(ctx, x, y, radius) {{
        ctx.save();
        ctx.shadowColor = "rgba(0,0,0,0.45)";
        ctx.shadowBlur = 3;
        ctx.shadowOffsetY = 1;
        const grad = ctx.createRadialGradient(x - radius * 0.35, y - radius * 0.45, radius * 0.15, x, y, radius);
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.58, "#f4f4f4");
        grad.addColorStop(1, "#cfd3da");
        ctx.fillStyle = grad;
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.shadowColor = "transparent";

        ctx.fillStyle = "#111827";
        ctx.beginPath();
        for (let i = 0; i < 5; i++) {{
          const a = -Math.PI / 2 + i * Math.PI * 2 / 5;
          const px = x + Math.cos(a) * radius * 0.34;
          const py = y + Math.sin(a) * radius * 0.34;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }}
        ctx.closePath();
        ctx.fill();

        for (let i = 0; i < 5; i++) {{
          const a = -Math.PI / 2 + i * Math.PI * 2 / 5;
          const cx = x + Math.cos(a) * radius * 0.72;
          const cy = y + Math.sin(a) * radius * 0.72;
          ctx.beginPath();
          for (let j = 0; j < 5; j++) {{
            const aa = a + Math.PI / 5 + j * Math.PI * 2 / 5;
            const px = cx + Math.cos(aa) * radius * 0.22;
            const py = cy + Math.sin(aa) * radius * 0.22;
            if (j === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }}
          ctx.closePath();
          ctx.fill();
        }}

        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 0.9;
        for (let i = 0; i < 5; i++) {{
          const a = -Math.PI / 2 + i * Math.PI * 2 / 5;
          ctx.beginPath();
          ctx.moveTo(x + Math.cos(a) * radius * 0.36, y + Math.sin(a) * radius * 0.36);
          ctx.quadraticCurveTo(
            x + Math.cos(a + 0.22) * radius * 0.62,
            y + Math.sin(a + 0.22) * radius * 0.62,
            x + Math.cos(a) * radius * 0.92,
            y + Math.sin(a) * radius * 0.92
          );
          ctx.stroke();
        }}
        ctx.restore();
      }}
      function drawMinimap() {{
        if (!minimapReady || !minimapCanvas || !minimapCtx) return;
        const rect = minimapCanvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));
        if (minimapCanvas.width !== Math.round(width * dpr) || minimapCanvas.height !== Math.round(height * dpr)) {{
          minimapCanvas.width = Math.round(width * dpr);
          minimapCanvas.height = Math.round(height * dpr);
        }}
        const ctx = minimapCtx;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);
        const rectPitch = pitchRect(width, height);
        const rows = minimap.grid.length;
        const cols = minimap.grid[0].length;
        const cellW = rectPitch.w / cols;
        const cellH = rectPitch.h / rows;
        for (let y = 0; y < rows; y++) {{
          for (let x = 0; x < cols; x++) {{
            ctx.fillStyle = threatColor(minimap.grid[y][x]);
            ctx.fillRect(rectPitch.x + x * cellW, rectPitch.y + y * cellH, cellW + 0.75, cellH + 0.75);
          }}
        }}
        drawPitchLines(ctx, rectPitch);
        const ball = interpolatedBall(video.currentTime || start);
        if (!ball) return;
        ctx.save();
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "rgba(255,255,255,0.26)";
        ctx.lineWidth = 4.4;
        ctx.beginPath();
        minimap.points.forEach((point, idx) => {{
          const [px, py] = mapPoint(point, rectPitch);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }});
        ctx.stroke();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        minimap.points.forEach((point, idx) => {{
          if (point.videoTime > ball.videoTime) return;
          const [px, py] = mapPoint(point, rectPitch);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }});
        const [bx, by] = mapPoint(ball, rectPitch);
        ctx.lineTo(bx, by);
        ctx.stroke();
        drawSoccerBall(ctx, bx, by, 6.8);
        ctx.restore();
        if (minimapValue) {{
          minimapValue.textContent = "xThreat en tiempo real: " + (ball.xT || 0).toFixed(3);
        }}
      }}
      function syncPlaybackButton() {{
        if (!playToggle) return;
        playToggle.innerHTML = video.paused ? "&#9654;" : "&#10074;&#10074;";
      }}
      function playSequence() {{
        if (video.currentTime < start || video.currentTime >= end) {{
          video.currentTime = start;
        }}
        video.play().then(syncPlaybackButton).catch(() => {{
          status.textContent = "Pulsa Reproducir si el navegador bloquea la reproduccion automatica.";
          syncPlaybackButton();
        }});
      }}
      function pauseSequence() {{
        video.pause();
        syncPlaybackButton();
      }}
      function replayFromStart() {{
        video.currentTime = start;
        playSequence();
      }}
      function resetViewState() {{
        zoom = baseZoom;
        panX = 0;
        panY = 0;
        posX = 50;
        posY = 56;
        clampPan();
        updateView();
      }}
      function clampPan() {{
        const rect = stage.getBoundingClientRect();
        const maxX = Math.max(0, rect.width * (zoom - 1) * 0.5);
        const maxY = Math.max(0, rect.height * (zoom - 1) * 0.5);
        panX = Math.max(-maxX, Math.min(maxX, panX));
        panY = Math.max(-maxY, Math.min(maxY, panY));
        posX = Math.max(minVisibleX, Math.min(maxVisibleX, posX));
        posY = Math.max(minVisibleY, Math.min(maxVisibleY, posY));
      }}
      stage.addEventListener("wheel", (event) => {{
        event.preventDefault();
        const delta = event.deltaY < 0 ? 0.12 : -0.12;
        setZoom(zoom + delta);
      }}, {{ passive: false }});
      stage.addEventListener("pointerdown", (event) => {{
        dragging = true;
        dragStartX = event.clientX;
        dragStartY = event.clientY;
        basePanX = panX;
        basePanY = panY;
        basePosX = posX;
        basePosY = posY;
        stage.setPointerCapture(event.pointerId);
        stage.style.cursor = "grabbing";
      }});
      stage.addEventListener("pointermove", (event) => {{
        if (!dragging) return;
        const dx = event.clientX - dragStartX;
        const rect = stage.getBoundingClientRect();
        posX = basePosX - (dx / rect.width) * 130;
        posY = 56;
        panX = 0;
        panY = 0;
        clampPan();
        updateView();
      }});
      stage.addEventListener("pointerup", (event) => {{
        dragging = false;
        stage.releasePointerCapture(event.pointerId);
        stage.style.cursor = "grab";
      }});
      stage.addEventListener("dblclick", () => {{
        resetViewState();
      }});
      const controlBar = document.getElementById("video-controls");
      if (controlBar) {{
        ["pointerdown", "pointermove", "pointerup", "click", "dblclick", "wheel"].forEach((eventName) => {{
          controlBar.addEventListener(eventName, (event) => {{
            event.stopPropagation();
          }}, {{ passive: eventName === "wheel" ? false : true }});
        }});
      }}
      playToggle.addEventListener("click", () => {{
        if (video.paused) {{
          playSequence();
        }} else {{
          pauseSequence();
        }}
      }});
      replaySequence.addEventListener("click", () => {{
        replayFromStart();
      }});
      zoomInView.addEventListener("click", () => {{
        setZoom(zoom + 0.18);
      }});
      zoomOutView.addEventListener("click", () => {{
        setZoom(zoom - 0.18);
      }});
      sequenceProgress.addEventListener("input", () => {{
        const pct = Number(sequenceProgress.value || 0) / 1000;
        video.currentTime = start + (end - start) * pct;
        drawMinimap();
      }});
      fullscreenView.addEventListener("click", async () => {{
        try {{
          if (!document.fullscreenElement) {{
            await stage.requestFullscreen();
          }} else {{
            await document.exitFullscreen();
          }}
          setTimeout(() => {{ clampPan(); updateView(); }}, 120);
        }} catch (err) {{
          status.textContent = "El navegador no ha permitido abrir pantalla completa desde aqui.";
        }}
      }});
      resetViewState();

      function playFromStart() {{
        replayFromStart();
        drawMinimap();
      }}

      function monitorEnd() {{
        const duration = Math.max(0.001, end - start);
        const elapsed = Math.max(0, Math.min(duration, video.currentTime - start));
        if (sequenceProgress) {{
          sequenceProgress.value = String(Math.round(elapsed / duration * 1000));
        }}
        if (sequenceTime) {{
          sequenceTime.textContent = Math.floor(elapsed) + "s / " + Math.floor(duration) + "s";
        }}
        if (video.currentTime >= end) {{
          pauseSequence();
          video.currentTime = start;
          status.textContent = "Secuencia terminada. Pulsa Ver de nuevo para repetirla.";
        }} else {{
          status.textContent = "Reproduciendo " + Math.floor(video.currentTime) + "s / fin " + Math.floor(end) + "s";
        }}
        drawMinimap();
      }}

      function animateMinimap() {{
        drawMinimap();
        if (!video.paused && !video.ended) {{
          window.requestAnimationFrame(animateMinimap);
        }}
      }}

      if (window.Hls && Hls.isSupported()) {{
        const hls = new Hls({{ startPosition: start }});
        hls.loadSource(source);
        hls.attachMedia(video);
        hls.on(Hls.Events.MANIFEST_PARSED, playFromStart);
      }} else if (video.canPlayType("application/vnd.apple.mpegurl")) {{
        video.src = source;
        video.addEventListener("loadedmetadata", playFromStart, {{ once: true }});
      }} else {{
        status.textContent = "Este navegador no puede reproducir HLS directamente.";
      }}
      video.addEventListener("timeupdate", monitorEnd);
      video.addEventListener("play", () => {{
        syncPlaybackButton();
        animateMinimap();
      }});
      video.addEventListener("pause", () => {{
        syncPlaybackButton();
        drawMinimap();
      }});
      video.addEventListener("seeking", () => {{
        if (video.currentTime < start || video.currentTime > end) {{
          status.textContent = "Tramo objetivo: " + Math.floor(start) + "s - " + Math.floor(end) + "s";
        }}
        drawMinimap();
      }});
      window.addEventListener("resize", drawMinimap);
      drawMinimap();
    </script>
    """
    components.html(html, height=780, scrolling=False)


def _sequence_display_table(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    source = df.copy()
    if not source.empty:
        source["tiro/a puerta/gol"] = (
            source.get("tipo_finalizacion_tiro", pd.Series(0, index=source.index)).map(_int_flag).astype(str)
            + " / "
            + source.get("tipo_finalizacion_tiro_puerta", pd.Series(0, index=source.index)).map(_int_flag).astype(str)
            + " / "
            + source.get("es_gol", pd.Series(0, index=source.index)).map(_int_flag).astype(str)
        )
        reference = _interpretability_reference().get("thresholds", {})
        if "indice_desorganizacion" in source.columns:
            source["idd_lectura"] = source["indice_desorganizacion"].apply(
                lambda value: f"{classify_index_value('indice_desorganizacion', value, reference.get('indice_desorganizacion', {}))['label']} ({_format_metric_precise(value, 3)})"
            )
        if "indice_peligrosidad_accion" in source.columns:
            source["ipo_lectura"] = source["indice_peligrosidad_accion"].apply(
                lambda value: f"{classify_index_value('indice_peligrosidad_accion', value, reference.get('indice_peligrosidad_accion', {}))['label']} ({_format_metric_precise(value, 3)})"
            )
    cols = [
        "secuencia_rival_id",
        "tipologia_id",
        "minuto_partido",
        "duracion_seg",
        "idd_lectura",
        "ipo_lectura",
        "xT_max",
        "tiro/a puerta/gol",
        "causa_tactica",
    ]
    display = source[[c for c in cols if c in source.columns]].head(n).copy()
    rename = {
        "secuencia_rival_id": "secuencia",
        "tipologia_id": "tipologia",
        "minuto_partido": "min",
        "duracion_seg": "duracion",
        "idd_lectura": "IDD",
        "ipo_lectura": "IPO",
        "xT_max": "xT max",
        "causa_tactica": "causa",
    }
    return display.rename(columns=rename)


def _tipology_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    for col in [
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "xT_max",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
        "control_campo_rival_medio",
        "control_zona_peligrosa_rival_medio",
    ]:
        if col not in df.columns:
            df[col] = 0
    group = df.groupby("tipologia", as_index=False)
    out = group.agg(
        secuencias=("secuencia_rival_id", "count"),
        ddi_medio=("indice_desorganizacion", "mean"),
        ddi_max=("indice_desorganizacion", "max"),
        ipar_medio=("indice_peligrosidad_accion", "mean"),
        ipar_max=("indice_peligrosidad_accion", "max"),
        xt_max=("xT_max", "max"),
        pc_rival_medio=("control_campo_rival_medio", "mean"),
        pc_zona_peligrosa_rival=("control_zona_peligrosa_rival_medio", "mean"),
        tiros=("tipo_finalizacion_tiro", "sum"),
        tiros_puerta=("tipo_finalizacion_tiro_puerta", "sum"),
    )
    if {"categoria_peligrosidad_auto", "tipologia"}.issubset(df.columns):
        high = (
            df[df["categoria_peligrosidad_auto"].isin(["muy_alta", "critica"])]
            .groupby("tipologia")
            .size()
            .rename("sec_peligrosas")
        )
        out = out.merge(high, on="tipologia", how="left")
    if {"categoria_desorganizacion_auto", "tipologia"}.issubset(df.columns):
        high_dd = (
            df[df["categoria_desorganizacion_auto"].isin(["muy_alta", "critica"])]
            .groupby("tipologia")
            .size()
            .rename("sec_desorganizacion_alta")
        )
        out = out.merge(high_dd, on="tipologia", how="left")
    for col in ["sec_peligrosas", "sec_desorganizacion_alta"]:
        if col not in out.columns:
            out[col] = 0
    return out.fillna(0).round(3)


def _temporal_interval_data(seq: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_df = seq.dropna(subset=["minuto_partido"]).copy()
    plot_df["minuto_partido"] = pd.to_numeric(plot_df["minuto_partido"], errors="coerce")
    plot_df[metric] = pd.to_numeric(plot_df.get(metric, 0), errors="coerce")
    plot_df = plot_df.dropna(subset=["minuto_partido", metric])
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    centers = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5]
    plot_df["tramo"] = pd.cut(plot_df["minuto_partido"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        plot_df.groupby("tramo", observed=False)
        .agg(
            secuencias=("secuencia_rival_id", "count"),
            valor=(metric, "mean"),
        )
        .reset_index()
    )
    grouped["centro"] = centers[: len(grouped)]
    grouped["tramo"] = grouped["tramo"].astype(str)
    return plot_df, grouped


def _plot_temporal_metric(seq: pd.DataFrame, metric: str, title: str, y_label: str, color: str, baseline_seq: pd.DataFrame | None = None):
    if seq.empty or "minuto_partido" not in seq.columns or metric not in seq.columns:
        st.info("No hay datos suficientes para este grafico temporal.")
        return
    plot_df, grouped = _temporal_interval_data(seq, metric)
    if plot_df.empty or grouped.empty:
        st.info("No hay datos suficientes para este grafico temporal.")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=grouped["tramo"],
            y=grouped["secuencias"],
            marker_color="rgba(100,112,132,0.22)",
            name="Secuencias",
            hovertemplate="%{customdata}<br>Secuencias: %{y}<extra></extra>",
            customdata=grouped["tramo"],
        ),
        secondary_y=True,
    )
    if baseline_seq is not None and not baseline_seq.empty:
        _, baseline_grouped = _temporal_interval_data(baseline_seq, metric)
        if not baseline_grouped.empty:
            fig.add_trace(
                go.Scatter(
                    x=baseline_grouped["tramo"],
                    y=baseline_grouped["valor"],
                    mode="lines",
                    line=dict(color="#e8edf7", width=2.2, dash="dash"),
                    name=f"{y_label} media total",
                    hovertemplate="Tramo %{x}<br>Media total: %{y:.3f}<extra></extra>",
                ),
                secondary_y=False,
            )
    line_color = color
    fig.add_trace(
        go.Scatter(
            x=grouped["tramo"],
            y=grouped["valor"],
            mode="lines+markers",
            line=dict(color=line_color, width=2.6),
            marker=dict(size=8, color=line_color),
            name=f"{y_label} medio",
            hovertemplate="Tramo %{x}<br>Media: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )

    events = [
        ("tipo_finalizacion_tiro", "Tiro", "circle", "#f2c94c", 13),
        ("tipo_finalizacion_tiro_puerta", "Tiro a puerta", "diamond", "#ffffff", 15),
        ("es_gol", "Gol", "star", "#c8102e", 18),
    ]
    for flag, name, symbol, marker_color, size in events:
        if flag not in plot_df.columns:
            continue
        event_df = plot_df[pd.to_numeric(plot_df[flag], errors="coerce").fillna(0).astype(int).eq(1)].copy()
        if event_df.empty:
            continue
        event_df["tramo"] = pd.cut(
            pd.to_numeric(event_df["minuto_partido"], errors="coerce"),
            bins=[0, 15, 30, 45, 60, 75, 90, 120],
            labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"],
            include_lowest=True,
        ).astype(str)
        fig.add_trace(
            go.Scatter(
                x=event_df["tramo"],
                y=event_df[metric],
                mode="markers",
                marker=dict(symbol=symbol, size=size, color=marker_color, line=dict(width=2.5, color="#101522")),
                name=name,
                customdata=event_df[["secuencia_rival_id", "tipologia"]],
                hovertemplate="%{customdata[1]} | secuencia %{customdata[0]}<br>Tramo %{x}<br>" + y_label + " %{y:.3f}<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=17, color="#f4f6fb")),
        height=390,
        plot_bgcolor="#202633",
        paper_bgcolor="#202633",
        font=dict(color="#e8edf7"),
        margin=dict(l=10, r=10, t=54, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff")),
        barmode="overlay",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"],
        title_text="Tramo",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(title_text=y_label, range=[0, 1.02], showgrid=True, gridcolor="rgba(255,255,255,0.08)", secondary_y=False)
    fig.update_yaxes(title_text="N secuencias", rangemode="tozero", showgrid=False, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _build_command(match_id: int | None = None, team_id: int = TEAM_ID_DEFAULT) -> str:
    match_part = f" --match-id {int(match_id)}" if match_id is not None else ""
    return (
        "$env:PYTHONPATH='src'\n"
        f".\\.venv\\Scripts\\python.exe scripts\\build_app_data.py --team-id {team_id}{match_part}"
    )


def render_admin_users():
    st.markdown('<div class="portal-section-title">Gestion de usuarios</div>', unsafe_allow_html=True)
    if not _firebase_enabled():
        st.warning("Firebase no esta configurado. Anade FIREBASE_API_KEY y FIREBASE_PROJECT_ID en .streamlit/secrets.toml.")
        return
    try:
        users = _list_firestore_users()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not users:
        st.info("Todavia no hay usuarios registrados en Firestore.")
        return

    team_names = [_team_display_name(team) for team in _registered_teams()]
    default_team = team_names[0] if team_names else ANON_TEAM_NAME
    for user in users:
        doc_id = str(user.get("_doc_id", ""))
        email = str(user.get("email", ""))
        role = str(user.get("rol", "guest")).lower()
        team = str(user.get("equipo", ""))
        with st.expander(email or doc_id, expanded=False):
            role_index = USER_ROLES.index(role) if role in USER_ROLES else 0
            selected_role = st.selectbox("Rol", USER_ROLES, index=role_index, key=f"user_role_{doc_id}")
            selected_team = ""
            if selected_role == "analista":
                options = team_names or [default_team]
                index = options.index(team) if team in options else 0
                selected_team = st.selectbox("Equipo asignado", options, index=index, key=f"user_team_{doc_id}")
            if st.button("Guardar cambios", key=f"save_user_{doc_id}", type="primary"):
                try:
                    _update_firestore_user(doc_id, email, selected_role, selected_team)
                    st.success("Usuario actualizado.")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))


def render_portal_home():
    teams = _registered_teams()
    allowed_team_ids = _user_allowed_team_ids()
    visible_teams = teams if allowed_team_ids is None or _is_guest() else [
        team for team in teams if int(team["team_id"]) in allowed_team_ids
    ]
    st.markdown(
        """
        <div class="portal-landing">
            <div class="portal-welcome">
                <h1>Bienvenido a la plataforma de análisis defensivo</h1>
                <p>
                    Esta aplicación está pensada para cualquier club que disponga de tracking y eventing.
                    Cada equipo registrado accede a su propio espacio, selecciona el partido que quiere revisar
                    y abre un panel táctico con métricas, mapas, secuencias críticas, momentum e informe final.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="portal-section-title">Módulos de la aplicación</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="portal-module-grid">
            <div class="portal-module-card">
                <strong>Gestión de datos deportivos</strong>
                <span>Base para registrar partidos con tracking y eventing ya procesados.</span>
                <ul>
                    <li>Selección de club registrado</li>
                    <li>Lectura de partidos disponibles</li>
                    <li>Actualización de resultados generados</li>
                </ul>
            </div>
            <div class="portal-module-card">
                <strong>Análisis táctico defensivo</strong>
                <span>Panel de lectura para transformar posesiones rivales en decisiones de entrenamiento.</span>
                <ul>
                    <li>Tipologías ofensivas rivales</li>
                    <li>IDD, IPO, xThreat y Pitch Control</li>
                    <li>Causas y zonas de desajuste</li>
                </ul>
            </div>
            <div class="portal-module-card">
                <strong>Revisión y entrega</strong>
                <span>Herramientas para priorizar vídeo y comunicar conclusiones al cuerpo técnico.</span>
                <ul>
                    <li>Secuencias críticas interactivas</li>
                    <li>Momentum temporal del partido</li>
                    <li>Informe PDF del análisis</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="portal-section-title">Manual rápido de uso</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="portal-steps-grid">
            <div class="portal-step-card"><strong>1. Entra a la plataforma</strong><span>Accede con tu cuenta para consultar los equipos registrados.</span></div>
            <div class="portal-step-card"><strong>2. Selecciona equipo</strong><span>Elige el club del que quieres abrir el entorno de análisis.</span></div>
            <div class="portal-step-card"><strong>3. Elige partido</strong><span>Dentro del club, selecciona el partido procesado que quieres revisar.</span></div>
            <div class="portal-step-card"><strong>4. Revisa el panel</strong><span>Navega por resumen, tipologías, estructura defensiva, secuencias, momentum e informe.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="portal-section-title">Equipos registrados</div>', unsafe_allow_html=True)
    if _is_guest():
        st.info("Tu cuenta esta en modo invitado. Puedes ver la portada y los equipos registrados; un administrador puede convertirte en analista y asignarte un club.")
    elif not visible_teams:
        st.warning("Tu usuario aun no tiene equipo asignado. Pide a un administrador que revise tu perfil.")

    for team in visible_teams:
        team_name = _team_display_name(team)
        logo = _team_logo_html(team.get("team_id"), _initials(team_name), "sidebar-crest")
        cols = st.columns([0.82, 0.18], vertical_alignment="center")
        with cols[0]:
            st.markdown(
                f"""
                <span class="registered-team-card-marker"></span>
                <div class="team-card">
                    <div class="team-card-logo">{logo}</div>
                    <div>
                        <strong>{html.escape(team_name)}</strong>
                        <span class="team-pill">{html.escape(str(team.get("status", "Activo")))} · {int(team.get("matches", 0))} partidos</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            disabled = _is_guest() or not _can_access_team(int(team["team_id"]))
            if st.button("Seleccionar", key=f"pick_team_{int(team['team_id'])}", type="primary", use_container_width=True, disabled=disabled):
                st.session_state["portal_selected_team_id"] = int(team["team_id"])
                st.session_state.pop("pending_team_access_id", None)
                st.session_state.pop("team_access_error", None)
                st.rerun()

    selected_team_id = st.session_state.get("portal_selected_team_id")
    if selected_team_id is not None:
        selected_team = _team_by_id(int(selected_team_id))
        selected_team_name = _team_display_name(selected_team)
        st.markdown(
            f"""
            <div class="match-picker-card">
                <strong>Equipo seleccionado: {html.escape(selected_team_name)}</strong>
                <span>{html.escape(str(selected_team.get("status", "Activo")))} · {int(selected_team.get("matches", 0))} partidos disponibles</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Entrar en el espacio del equipo", type="primary", use_container_width=True):
            if _can_access_team(int(selected_team_id)):
                _set_app_view("club", int(selected_team_id))
            else:
                st.session_state["pending_team_access_id"] = int(selected_team_id)
                st.session_state.pop("team_access_error", None)
    pending_team_id = st.session_state.get("pending_team_access_id")
    if pending_team_id is not None:
        pending_team = _team_by_id(int(pending_team_id))
        with st.form(f"team_code_form_{int(pending_team_id)}"):
            code = st.text_input("Código del equipo", type="password")
            validate = st.form_submit_button("Validar código y entrar", type="primary", use_container_width=True)
        if validate:
            expected = str(pending_team.get("access_code", "")).strip().lower()
            if expected and code.strip().lower() == expected:
                st.session_state.pop("pending_team_access_id", None)
                st.session_state.pop("team_access_error", None)
                _set_app_view("club", int(pending_team_id))
            else:
                st.session_state["team_access_error"] = "Código incorrecto para este equipo."
        if st.session_state.get("team_access_error"):
            st.error(st.session_state["team_access_error"])


def render_club_home(team: dict):
    team_id = int(team["team_id"])
    matches = _team_matches(team_id)
    team_name = _team_display_name(team)
    logo = _team_logo_html(team_id, _initials(team_name), "sidebar-crest")
    st.markdown(
        f"""
        <div class="club-home-card">
            <div>{logo}</div>
            <div>
                <div class="club-name-large">{html.escape(team_name)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    total_matches = len(matches)
    total_sequences = "-"
    last_date = "-"
    if not matches.empty:
        numeric_seq = pd.to_numeric(matches.get("secuencias", pd.Series(dtype=float)), errors="coerce")
        total_sequences = int(numeric_seq.fillna(0).sum())
        last_date = str(matches.iloc[0].get("fecha", "-"))
    st.markdown(
        f"""
        <div class="club-stat-grid">
            <div class="club-stat"><b>{total_matches}</b><span>Partidos disponibles</span></div>
            <div class="club-stat"><b>{total_sequences}</b><span>Secuencias analizadas</span></div>
            <div class="club-stat"><b>{html.escape(str(last_date))}</b><span>Último partido analizado</span></div>
            <div class="club-stat"><b>{html.escape(str(team.get("status", "Activo")))}</b><span>Estado del club</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="portal-section-title">Seleccionar partido</div>', unsafe_allow_html=True)
    if matches.empty:
        st.warning("Este equipo todavía no tiene partidos web generados.")
        st.code(_build_command(team_id=team_id), language="powershell")
        return

    label_map = _match_label_map(matches)
    selected_label = st.selectbox("Partido disponible", list(label_map.keys()), key=f"club_match_select_{team_id}")
    selected_match_id = label_map[selected_label]
    selected_row = matches[matches["match_id"].astype(int).eq(int(selected_match_id))].iloc[0]
    st.markdown(
        f"""
        <div class="match-picker-card">
            <strong>{html.escape(str(selected_row.get("rival", "Rival")))}</strong>
            <span>Fecha: {html.escape(str(selected_row.get("fecha", "-")))} · Resultado: {html.escape(str(selected_row.get("score", "vs")))} · Secuencias: {html.escape(str(selected_row.get("secuencias", "-")))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Abrir análisis del partido", type="primary", use_container_width=True):
        _set_app_view("analysis", team_id, selected_match_id)


def render_analysis_workspace(team: dict, match_id: int):
    meta = _load_metadata(match_id)
    render_topbar(team)
    team_name = _team_display_name(team)
    st.markdown(
        f'<div class="app-breadcrumb">Menú principal / {html.escape(team_name)} / <b>Partido {int(match_id)}</b></div>',
        unsafe_allow_html=True,
    )
    render_hero(meta)
    render_match_header(match_id, meta)

    sections = [
        "Resumen",
        "Tipologías ofensivas",
        "Estructura defensiva",
        "Secuencias críticas",
        "Momentum",
        "Informe",
    ]
    section_aliases = {
        "Ataque rival": "Tipologías ofensivas",
        "Tipologias ofensivas": "Tipologías ofensivas",
        "Desorganizacion defensiva": "Estructura defensiva",
        "Secuencias criticas": "Secuencias críticas",
        "Cronologia": "Momentum",
    }
    nav_key = f"main_section_{int(team['team_id'])}_{int(match_id)}"
    if st.session_state.get(nav_key) in section_aliases:
        st.session_state[nav_key] = section_aliases[st.session_state[nav_key]]
    if nav_key not in st.session_state or st.session_state[nav_key] not in sections:
        st.session_state[nav_key] = "Resumen"

    active_section = st.radio(
        "Sección principal",
        options=sections,
        key=nav_key,
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_section == "Resumen":
        render_resumen(match_id, meta)
    elif active_section == "Tipologías ofensivas":
        render_ataque_rival(match_id, meta)
    elif active_section == "Estructura defensiva":
        render_desorganizacion(match_id, meta)
    elif active_section == "Secuencias críticas":
        render_secuencias_criticas(match_id, meta)
    elif active_section == "Momentum":
        render_cronologia(match_id, meta)
    elif active_section == "Informe":
        render_informe(match_id, meta)


def render_hero(meta: dict):
    st.markdown(
        '<div class="portal-hero"></div>',
        unsafe_allow_html=True,
    )


def sidebar_selector() -> int | None:
    default_team = _team_by_id(TEAM_ID_DEFAULT)
    default_team_name = _team_display_name(default_team)
    sidebar_logo = _team_logo_html(TEAM_ID_DEFAULT, _initials(default_team_name), "sidebar-crest")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-club">
            {sidebar_logo}
            <h2>{html.escape(default_team_name)}</h2>
            <p>Análisis Defensivo IDD/IPO</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    app_data = _available_app_data()
    tracking = _available_tracking()

    if app_data.empty:
        st.sidebar.warning("Sin resultados web generados")
        st.sidebar.divider()
        st.sidebar.write("Partidos detectados")
        if not tracking.empty:
            st.sidebar.dataframe(
                tracking[["match_id", "size_mb", "modified"]],
                use_container_width=True,
                hide_index=True,
                height=170,
            )
        return None

    options = app_data["match_id"].dropna().astype(int).tolist()
    match_id = st.sidebar.selectbox("Partido seleccionado", options, index=0)
    row = app_data.loc[app_data["match_id"].astype(int).eq(int(match_id))].iloc[0]
    row_meta = _load_metadata(int(match_id))
    anon_rival = _team_name_for_id(row_meta, row_meta.get("rival_team_id"), "Equipo Rival")
    st.sidebar.success("Resultado web disponible")
    st.sidebar.caption(f"Rival: {anon_rival}")

    st.sidebar.divider()
    st.sidebar.write("Partidos descargados")
    sidebar_table = app_data[["match_id", "rival", "modified"]].copy()
    sidebar_table["rival"] = sidebar_table["match_id"].map(
        lambda mid: _team_name_for_id(_load_metadata(int(mid)), _load_metadata(int(mid)).get("rival_team_id"), "Equipo Rival")
    )
    st.sidebar.dataframe(
        sidebar_table,
        use_container_width=True,
        hide_index=True,
        height=170,
    )
    if st.sidebar.button("Actualizar lista", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    return int(match_id)


def render_match_header(match_id: int, meta: dict):
    teams = _presentation_teams(match_id, meta)
    resumen = meta.get("resumen", {})
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(
        match_id,
        teams["home_id"],
        teams["away_id"],
    )
    st.markdown(
        f"""
        <div class="match-card">
            <div class="match-grid">
                <div class="match-team">
                    {teams["home_logo"]}
                    <div>
                        <div class="team-label">Equipo local</div>
                        <div class="team-name">{teams["home_name"]}</div>
                    </div>
                </div>
                <div class="score-pill">{score}</div>
                <div class="match-team away">
                    <div>
                        <div class="team-label">Equipo visitante</div>
                        <div class="team-name">{teams["away_name"]}</div>
                    </div>
                    {teams["away_logo"]}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    scope = str(resumen.get("cluster_scope", "partido_auto"))
    n_activas = resumen.get("n_tipologias_detectadas", "-")
    n_modelo = resumen.get("n_clusters_modelo", n_activas)
    if scope == "global":
        cluster_txt = f"tipologías activas {n_activas}/{n_modelo} del modelo global"
    else:
        cluster_txt = f"tipologías detectadas {n_activas}"
    return cluster_txt


def render_resumen(match_id: int, meta: dict):
    seq, clusters, zona, _ = _prepare_app_data(match_id, meta)

    _page_heading("Resumen")
    _section_intro(
        "Lectura ejecutiva del partido",
        "Esta vista ofrece una lectura rápida del partido para explicar en pocos segundos el volumen ofensivo rival, la amenaza concedida, la respuesta defensiva del bloque y los momentos clave del análisis.",
    )
    metrics = _summary_metric_cards(seq, clusters, zona)
    _render_summary_metric_blocks(metrics)


def render_ataque_rival(match_id: int, meta: dict):
    figs = meta["figures"]
    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    _page_heading("Tipologías ofensivas")
    _section_intro(
        "Lectura general del ataque rival",
        "Esta pestaña resume cómo atacó el rival y qué tipologías ofensivas se repitieron. Sirve para comparar rutas de balón, zonas de posesión, volumen, ritmo y amenaza generada por cada patrón.",
    )
    _subsection_heading("Trayectorias por tipología de ataque rival")
    st.caption("Representa la ruta del balón en cada tipología para reconocer por dónde progresa el rival.")
    _show_image(match_id, figs["trayectorias_cluster"], max_width=1180)
    _subsection_heading("Mapas de calor de localización del balón")
    st.caption("Muestra las zonas donde se concentra la posesión rival dentro de cada tipología.")
    _show_image(match_id, figs["mapa_calor_notebook"], max_width=1180)

    if not clusters.empty:
        _subsection_heading("Asignación de tipologías")
        st.caption("Leyenda fija de colores y nombres: cada tipología mantiene el mismo color en todos los gráficos.")
        _render_tipology_assignment(clusters)

        clusters_plot = clusters.copy()
        _subsection_heading("Resumen cuantitativo por tipología de ataque rival")
        st.caption(
            "Agrupa volumen, eficacia y reparto territorial para comparar las tipologías con el mismo código de color."
        )
        c1, c2 = st.columns([1.1, 1])
        with c1:
            _plot_bar(
                clusters_plot,
                "patron_tactico",
                "secuencias",
                "Volumen de ataques por tipología de ataque rival",
                color="patron_tactico",
                labels={"patron_tactico": "Tipología", "secuencias": "Secuencias"},
                height=390,
            )
        with c2:
            _plot_mean_passes_by_tipology(seq, height=390)
        c1, c2 = st.columns([1, 1])
        with c1:
            _plot_bar(
                clusters_plot.sort_values("tiro_pct", ascending=False),
                "patron_tactico",
                "tiro_pct",
                "Probabilidad de tiro por tipología",
                color="patron_tactico",
                labels={"patron_tactico": "Tipología", "tiro_pct": "% con tiro"},
                height=390,
            )
        with c2:
            _plot_bar(
                clusters_plot.sort_values("duracion_media", ascending=False),
                "patron_tactico",
                "duracion_media",
                "Duración media del ataque",
                color="patron_tactico",
                labels={"patron_tactico": "Tipología", "duracion_media": "segundos"},
                height=390,
            )
        _subsection_heading("Evolución media de la amenaza concedida por tipología")
        st.caption("Lectura táctica: compara cuándo escala el xT de cada tipología. La zona final de la curva suele señalar el momento de aceleración o finalización.")
        _render_xt_reference_panel()
        if not _plot_xt_evolution(match_id, meta, clusters):
            _show_image(match_id, figs["amenaza_media_cluster"], max_width=1040)
    with st.expander("Ver ficha numérica de tipologías"):
        _show_static_table(_clusters_for_display(clusters))


def render_desorganizacion(match_id: int, meta: dict):
    seq, _, _, _ = _prepare_app_data(match_id, meta)
    if seq.empty:
        st.info("No hay detalle por secuencia para este partido. Regenera outputs/app_data para activar esta vista.")
        return
    team_name = _team_name_for_id(meta, meta.get("team_id", TEAM_ID_DEFAULT), "el equipo analizado")
    _page_heading("Estructura defensiva")
    _section_intro(
        "Lectura general de la estructura defensiva",
        f"Esta pestaña explica cómo respondió el bloque de {team_name} ante cada tipología ofensiva rival. Permite cruzar IDD, IPO, Pitch Control, xThreat y causas defensivas para priorizar correcciones.",
    )

    options = ["Seleccione una tipología"] + [value for value in _tipology_options(seq) if value != "Todas"]
    tipologia = st.selectbox(
        "Seleccione una tipología",
        options,
        key=f"desorg_tip_{match_id}",
        label_visibility="collapsed",
    )
    if tipologia == "Seleccione una tipología":
        return

    _subsection_heading("Desorganización defensiva propia por tipología de ataque rival")
    st.caption("KPIs defensivos, relación IDD/IPO y secuencias prioritarias de la tipología seleccionada.")
    summary = _tipology_summary(seq)
    current = _filter_tipology(seq, tipologia)
    _selected_heading(tipologia)
    global_ddi = pd.to_numeric(seq.get("indice_desorganizacion", pd.Series(dtype=float)), errors="coerce").mean()
    global_ipar = pd.to_numeric(seq.get("indice_peligrosidad_accion", pd.Series(dtype=float)), errors="coerce").mean()
    current_ddi = pd.to_numeric(current.get("indice_desorganizacion", pd.Series(dtype=float)), errors="coerce").mean()
    current_ipar = pd.to_numeric(current.get("indice_peligrosidad_accion", pd.Series(dtype=float)), errors="coerce").mean()
    diff_ddi = "-"
    diff_ipar = "-"
    if pd.notna(global_ddi) and global_ddi != 0 and pd.notna(current_ddi):
        diff_ddi = f"{((current_ddi / global_ddi) - 1) * 100:+.0f}% vs media"
    if pd.notna(global_ipar) and global_ipar != 0 and pd.notna(current_ipar):
        diff_ipar = f"{((current_ipar / global_ipar) - 1) * 100:+.0f}% vs media"
    _render_desorg_metric_blocks(current, diff_ddi, diff_ipar)
    st.markdown('<div class="graph-section-gap"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.caption("Cruza la desorganización concedida con la peligrosidad de cada secuencia filtrada.")
        _plot_seq_scatter(current, "Relación IDD/IPO del filtro")
    with c2:
        st.caption("Resume las causas defensivas más repetidas dentro de la tipología seleccionada.")
        top_causes = current.get("causa_tactica", pd.Series(dtype=str)).fillna("sin causa dominante").value_counts().reset_index()
        top_causes.columns = ["causa", "secuencias"]
        _plot_bar(
            top_causes,
            "causa",
            "secuencias",
            f"Causas principales | {tipologia}",
            labels={"causa": "Causa", "secuencias": "Secuencias"},
            height=390,
            red_gradient=True,
            gradient_scale=_tipology_gradient(tipologia),
        )

    top_ddi = current.sort_values("indice_desorganizacion", ascending=False)
    with st.expander("Secuencias que más rompen la estructura", expanded=False):
        st.caption("Ordena las acciones por IDD para detectar dónde se rompe antes la organización defensiva.")
        _show_static_table(_sequence_display_table(top_ddi, n=6))

    top_ipar = current.sort_values("indice_peligrosidad_accion", ascending=False)
    with st.expander("Secuencias con más amenaza/peligro", expanded=False):
        st.caption("Ordena las acciones por IPO para priorizar las que generan más peligro real.")
        _show_static_table(_sequence_display_table(top_ipar, n=6))


def render_secuencias_criticas(match_id: int, meta: dict):
    seq, _, _, _ = _prepare_app_data(match_id, meta)
    if seq.empty:
        st.info("No hay detalle por secuencia para este partido.")
        return
    team_name = _team_name_for_id(meta, meta.get("team_id", TEAM_ID_DEFAULT), "el equipo analizado")

    _page_heading("Secuencias críticas")
    _section_intro(
        "Lectura general de las secuencias críticas",
        "Esta pestaña concentra las jugadas que conviene revisar en vídeo primero. Puedes filtrar, ordenar por prioridad, abrir la lectura táctica de cada barra y cargar el vídeo solo cuando lo necesites.",
    )
    _subsection_heading("Jugada crítica: filtrar, ordenar y revisar")
    st.caption("Filtra por todas las tipologías o por una concreta, y ordena solo por IDD, IPO o prioridad combinada. La gráfica muestra el top 5 de acciones para revisar primero.")
    filter_col, sort_col_ui = st.columns([1.15, 1])
    with filter_col:
        tipologia_filter = st.selectbox(
            "Tipología",
            _tipology_options(seq),
            key=f"crit_tipologia_{match_id}",
        )
    with sort_col_ui:
        criterio = st.selectbox(
            "Ordenar por",
            ["Combinada", "IPO", "IDD"],
            key=f"crit_sort_{match_id}",
        )
    current = _filter_tipology(seq, tipologia_filter)
    if "score_critico" not in current.columns:
        current["score_critico"] = (
            pd.to_numeric(current.get("indice_desorganizacion", pd.Series(0, index=current.index)), errors="coerce").fillna(0)
            + pd.to_numeric(current.get("indice_peligrosidad_accion", pd.Series(0, index=current.index)), errors="coerce").fillna(0)
        ) / 2
    sort_col = {
        "Combinada": "score_critico",
        "IPO": "indice_peligrosidad_accion",
        "IDD": "indice_desorganizacion",
    }[criterio]
    if sort_col in current.columns:
        current = current.sort_values(sort_col, ascending=False)
    all_patterns = [v for v in _tipology_options(seq) if v != "Todas"]
    pattern_color_map = _tipology_color_map(all_patterns)
    top_for_chart = current.head(5).copy()
    selected_key = f"crit_seq_selected_{match_id}_{tipologia_filter}_{criterio}"
    options = top_for_chart["secuencia_rival_id"].dropna().astype(int).tolist()
    selected = st.session_state.get(selected_key)
    try:
        selected = int(selected) if selected is not None else None
    except (TypeError, ValueError):
        selected = None
    if selected is not None and selected not in options:
        selected = None
        st.session_state[selected_key] = None
    if top_for_chart.empty:
        st.info("No hay secuencias para el filtro seleccionado.")
        return
    clicked = _plot_selectable_sequence_bar(
        top_for_chart,
        sort_col,
        f"Top 5 secuencias por {criterio}",
        color="tipologia",
        selected_id=int(selected) if selected is not None else None,
        key=f"crit_bar_select_{match_id}_{tipologia_filter}_{criterio}",
        labels={"secuencia": "Secuencia", sort_col: criterio, "tipologia": "Tipología"},
        height=500,
        color_map=pattern_color_map,
    )
    if clicked is not None and int(clicked) in options:
        selected = int(clicked)
        st.session_state[selected_key] = selected
    if selected is None:
        return
    row = current[current["secuencia_rival_id"].astype(int).eq(int(selected))].iloc[0]
    row_tipologia = str(row.get("tipologia", "-"))
    pattern_group = seq[seq["tipologia"].astype(str).eq(row_tipologia)].copy() if "tipologia" in seq.columns else current
    pattern_means = pattern_group[["indice_desorganizacion", "indice_peligrosidad_accion", "xT_max"]].mean(numeric_only=True)
    ddi = float(row.get("indice_desorganizacion", 0) or 0)
    ipar = float(row.get("indice_peligrosidad_accion", 0) or 0)
    xt = float(row.get("xT_max", 0) or 0)
    ddi_base = pattern_means.get("indice_desorganizacion", np.nan)
    ipar_base = pattern_means.get("indice_peligrosidad_accion", np.nan)
    xt_base = pattern_means.get("xT_max", np.nan)
    ddi_vs = ((ddi / ddi_base) - 1) * 100 if pd.notna(ddi_base) and ddi_base else np.nan
    ipar_vs = ((ipar / ipar_base) - 1) * 100 if pd.notna(ipar_base) and ipar_base else np.nan
    xt_vs = ((xt / xt_base) - 1) * 100 if pd.notna(xt_base) and xt_base else np.nan
    def _delta_text(value) -> str:
        return f"{value:+.0f}%" if pd.notna(value) else "-"
    causa = row.get("causa_tactica", _cause_label(row.get("tipo_desorganizacion_principal", "-")))
    if causa == "dominio rival en zonas peligrosas":
        narrative = "El rival consiguio dominar espacio util cerca de zonas peligrosas y obligo al bloque a defender hacia atras."
    elif causa == "problemas de repliegue":
        narrative = f"La secuencia obligó a {team_name} a replegar con urgencia y elevó la desorganización del bloque."
    elif causa == "bloque demasiado abierto":
        narrative = "El rival encontró separación entre defensores y pudo atacar intervalos con más facilidad."
    elif causa == "poca presion sobre poseedor" or causa == "poca presión sobre poseedor":
        narrative = "El poseedor rival tuvo demasiado tiempo para orientar la accion y progresar."
    else:
        narrative = "La secuencia combina ruptura estructural y amenaza suficiente para ser revisada en vídeo."
    tipologia_txt = html.escape(_display_text(row.get("tipologia", "-")))
    causa_txt = html.escape(_display_text(causa))
    narrative_txt = html.escape(_display_text(narrative))
    finalizacion_txt = (
        f"tiro {_si_no(row.get('tipo_finalizacion_tiro'))}, "
        f"puerta {_si_no(row.get('tipo_finalizacion_tiro_puerta'))}, "
        f"gol {_si_no(row.get('es_gol'))}"
    )
    url_info = _video_url(match_id, row)
    st.markdown(
        f"""
        <div class="sequence-card">
            <h4>Secuencia {int(selected)} &middot; {tipologia_txt} &middot; minuto {_format_metric(row.get('minuto_partido'))}</h4>
            <div class="sequence-kpi-row">
                <div class="sequence-kpi"><span>IDD</span><strong>{_index_badge_html('indice_desorganizacion', row.get('indice_desorganizacion'), compact=True)}</strong></div>
                <div class="sequence-kpi"><span>IPO</span><strong>{_index_badge_html('indice_peligrosidad_accion', row.get('indice_peligrosidad_accion'), compact=True)}</strong></div>
                <div class="sequence-kpi"><span>xT max</span><strong>{_format_metric(row.get('xT_max'))}</strong></div>
                <div class="sequence-kpi"><span>Incremento xT</span><strong>{_format_metric(row.get('xT_added'))}</strong></div>
            </div>
            <div class="sequence-read-box"><strong>Lectura táctica:</strong> {narrative_txt}</div>
            <div class="sequence-tag-grid">
                <div class="sequence-tag"><span>Comparación tipología</span><strong>IDD {_delta_text(ddi_vs)} &middot; IPO {_delta_text(ipar_vs)} &middot; xT {_delta_text(xt_vs)}</strong></div>
                <div class="sequence-tag"><span>Causa defensiva</span><strong>{causa_txt}</strong></div>
                <div class="sequence-tag"><span>Finalizacion</span><strong>{finalizacion_txt}</strong></div>
                <div class="sequence-tag"><span>Eventos accion</span><strong>{_format_metric(row.get('num_eventos'))}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if url_info:
        video_key = f"crit_video_open_{match_id}_{int(selected)}"
        if st.button("Ver secuencia", type="primary", use_container_width=True, key=f"show_video_{match_id}_{int(selected)}"):
            st.session_state[video_key] = True
        if st.session_state.get(video_key, False):
            _render_sequence_player(match_id, row)
        if False:
            _, _, start, end = _sequence_bounds(match_id, row)
            st.caption(
                "Tramo objetivo en BePro: "
                f"{_hms_from_value(start)} - {_hms_from_value(end)}. "
                "Abre el partido nativo de BePro y usa este tiempo como referencia."
            )
        elif False:
            _, _, start, end = _sequence_bounds(match_id, row)
            st.caption(
                "Abre la pestaña de vídeo del partido en BePro. Segundo objetivo: "
                f"{_hms_from_value(start)} - {_hms_from_value(end)}. "
                "Abre el partido nativo de BePro y usa este tiempo como referencia."
            )
    else:
        st.caption(
            "El salto al vídeo queda preparado. Falta configurar BEPRO_VIDEO_URL_TEMPLATE con el patrón real de enlace "
            "(por ejemplo usando match_id y seconds)."
        )


MOMENTUM_ALPHA = 0.88
MOMENTUM_IDD_WEIGHT = 0.40
MOMENTUM_IPO_WEIGHT = 0.60
MOMENTUM_ALERT_THRESHOLD = 0.20
MOMENTUM_STRESS_THRESHOLD = 0.32


def _chronology_base(seq: pd.DataFrame) -> pd.DataFrame:
    required = ["minuto_partido", "secuencia_rival_id"]
    if seq.empty or not set(required).issubset(seq.columns):
        return pd.DataFrame()
    df = seq.copy()
    for col in [
        "minuto_partido",
        "duracion_seg",
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "xT_max",
        "x_max_norm_m",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
        "es_gol",
        "score_critico",
        "y_fin",
        "ball_y_fin_m",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["minuto_partido"]).sort_values("minuto_partido").copy()
    df["duracion_seg"] = df.get("duracion_seg", pd.Series(6, index=df.index)).fillna(6).clip(lower=2)
    df["inicio_min"] = df["minuto_partido"]
    df["fin_min"] = df["inicio_min"] + df["duracion_seg"] / 60
    df["ddi"] = df.get("indice_desorganizacion", pd.Series(0, index=df.index)).fillna(0)
    df["ipar"] = df.get("indice_peligrosidad_accion", pd.Series(0, index=df.index)).fillna(0)
    xt = df.get("xT_max", pd.Series(0, index=df.index)).fillna(0)
    df["xt_norm"] = (xt / max(float(xt.quantile(0.90)) if len(xt) else 0.22, 0.12)).clip(0, 1)
    df["entra_ultimo_tercio"] = df.get("x_max_norm_m", pd.Series(0, index=df.index)).fillna(0).ge(FIELD_LENGTH_M * (2 / 3)).astype(float)
    df["momentum_evento"] = (
        MOMENTUM_IDD_WEIGHT * df["ddi"] + MOMENTUM_IPO_WEIGHT * df["ipar"]
    ).clip(0, 1)
    df["tipologia"] = df.get("tipologia", pd.Series("Tipología no identificada", index=df.index)).fillna("Tipología no identificada").astype(str)
    lane_col = "y_fin" if "y_fin" in df.columns else "ball_y_fin_m" if "ball_y_fin_m" in df.columns else None
    df["carril"] = df[lane_col].apply(_lane_from_y) if lane_col else "Desconocido"
    return df


def _defensive_momentum_timeline(df: pd.DataFrame, alpha: float = MOMENTUM_ALPHA) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["minuto", "senal", "momentum"])
    max_minute = int(max(95, np.ceil(float(df["fin_min"].max()) + 2)))
    timeline = pd.DataFrame({"minuto": np.arange(0, max_minute + 1, 1)})
    event_df = df.dropna(subset=["minuto_partido"]).copy()
    if event_df.empty:
        timeline["senal"] = 0.0
        timeline["momentum"] = 0.0
        return timeline
    event_df["minuto_modelo"] = np.floor(event_df["minuto_partido"]).astype(int).clip(0, max_minute)
    signal_by_minute = (
        event_df.groupby("minuto_modelo")["momentum_evento"]
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
    return timeline


def _chronology_windows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    bins = [0, 15, 30, 45, 60, 75, 90, 120]
    labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    tmp = df.copy()
    tmp["tramo"] = pd.cut(tmp["minuto_partido"], bins=bins, labels=labels, include_lowest=True, right=False)
    grouped = (
        tmp.groupby("tramo", observed=False)
        .agg(
            secuencias=("secuencia_rival_id", "count"),
            ddi=("ddi", "mean"),
            ipar=("ipar", "mean"),
            tiros=("tipo_finalizacion_tiro", "sum") if "tipo_finalizacion_tiro" in tmp.columns else ("momentum_evento", "size"),
            tiros_puerta=("tipo_finalizacion_tiro_puerta", "sum") if "tipo_finalizacion_tiro_puerta" in tmp.columns else ("momentum_evento", "size"),
            goles=("es_gol", "sum") if "es_gol" in tmp.columns else ("momentum_evento", "size"),
            entradas_ultimo_tercio=("entra_ultimo_tercio", "sum"),
        )
        .reset_index()
    )
    timeline = _defensive_momentum_timeline(tmp)
    if not timeline.empty:
        timeline["tramo"] = pd.cut(timeline["minuto"], bins=bins, labels=labels, include_lowest=True, right=False)
        momentum_windows = (
            timeline.groupby("tramo", observed=False)
            .agg(
                momentum=("momentum", "mean"),
                momentum_p90=("momentum", lambda s: float(pd.Series(s).quantile(0.90))),
                momentum_max=("momentum", "max"),
            )
            .reset_index()
        )
        grouped = grouped.merge(momentum_windows, on="tramo", how="left")
    else:
        grouped["momentum"] = 0.0
        grouped["momentum_p90"] = 0.0
        grouped["momentum_max"] = 0.0
    grouped["stress"] = (
        0.60 * grouped["momentum_p90"].fillna(0)
        + 0.40 * grouped["momentum"].fillna(0)
    ).clip(0, 1)
    return grouped


def _dominant_value(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df.columns:
        return "-"
    values = df[col].dropna().astype(str)
    values = values[~values.str.lower().isin(["", "nan", "-"])]
    return values.mode().iloc[0] if not values.mode().empty else "-"


def _window_interval(window: pd.Series) -> tuple[float, float, str]:
    tramo = str(window.get("tramo", "-"))
    football_windows = {
        "0-15": (0.0, 15.0),
        "16-30": (15.0, 30.0),
        "31-45": (30.0, 45.0),
        "46-60": (45.0, 60.0),
        "61-75": (60.0, 75.0),
        "76-90": (75.0, 90.0),
        "90+": (90.0, 120.0),
    }
    if tramo in football_windows:
        start, end = football_windows[tramo]
        return start, end, tramo
    if tramo.endswith("+"):
        start_txt = tramo[:-1].strip()
        start = float(start_txt) if start_txt else 90.0
        return start, 120.0, tramo
    if "-" not in tramo:
        return 0.0, 120.0, tramo
    start_txt, end_txt = tramo.split("-", 1)
    start = float(start_txt.replace("+", "") or 0)
    end = 120.0 if end_txt.endswith("+") else float(end_txt)
    return start, end, tramo


def _window_sequences(df: pd.DataFrame, window: pd.Series) -> pd.DataFrame:
    start, end, _ = _window_interval(window)
    return df[df["minuto_partido"].between(start, end, inclusive="left")].copy()


def _selected_temporal_windows(windows: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    readable = windows.copy()
    readable = readable[pd.to_numeric(readable["secuencias"], errors="coerce").fillna(0).gt(0)]
    if readable.empty:
        return []
    ranked = readable.sort_values("stress", ascending=False)
    critical = ranked.iloc[0]
    selected: list[tuple[str, str, pd.Series]] = [("critical", "TRAMO CRITICO DETECTADO", critical)]
    used = {str(critical["tramo"])}
    alert_candidates = ranked[~ranked["tramo"].astype(str).isin(used)]
    if not alert_candidates.empty:
        alert = alert_candidates.iloc[0]
        selected.append(("warning", "TRAMO DE ALERTA", alert))
        used.add(str(alert["tramo"]))
    stable_candidates = readable[~readable["tramo"].astype(str).isin(used)].sort_values("stress", ascending=True)
    if not stable_candidates.empty:
        selected.append(("stable", "TRAMO MAS ESTABLE", stable_candidates.iloc[0]))
    return selected


def _render_temporal_alert_card(df: pd.DataFrame, window: pd.Series, tone: str, title: str) -> pd.DataFrame:
    part = _window_sequences(df, window)
    _, _, tramo = _window_interval(window)
    pattern_txt = _dominant_value(part, "tipologia")
    lane_txt = _dominant_value(part, "carril")
    shots = int(pd.to_numeric(part.get("tipo_finalizacion_tiro", pd.Series(0, index=part.index)), errors="coerce").fillna(0).sum())
    shot_zone = int(part["entra_ultimo_tercio"].sum()) if "entra_ultimo_tercio" in part.columns else 0
    ddi = float(part["ddi"].mean()) if not part.empty else 0.0
    ipar = float(part["ipar"].mean()) if not part.empty else 0.0
    ddi_html = _index_badge_html("indice_desorganizacion", ddi, compact=True)
    ipar_html = _index_badge_html("indice_peligrosidad_accion", ipar, compact=True)
    sec = int(len(part))
    stress = float(window.get("stress", 0) or 0) * 100
    tone_text = {
        "critical": "Prioridad máxima: tramo con mayor mezcla de volumen, desorganización, peligro y finalización.",
        "warning": "Tramo a vigilar: no es el peor pico, pero acumula señales suficientes para revisar contexto.",
        "stable": "Referencia estable: tramo con menor estrés relativo y mejor control defensivo dentro del partido.",
    }.get(tone, "")
    if tone == "critical":
        tone_text = "Prioridad maxima: tramo donde la curva EWMA de presion rival se mantiene mas alta."
    elif tone == "warning":
        tone_text = "Tramo a vigilar: no es el peor pico, pero conserva presion acumulada suficiente para revisar contexto."
    elif tone == "stable":
        tone_text = "Referencia estable: tramo con menor estres relativo y mejor control defensivo dentro del partido."
    st.markdown(
        f"""
        <div class="temporal-alert-card temporal-alert-{tone}">
          <div class="temporal-alert-kicker">{html.escape(title)}</div>
          <div class="temporal-alert-title">{html.escape(tramo)}'</div>
          <div class="temporal-alert-grid">
            <div><span>{sec}</span><small>secuencias</small></div>
            <div><span>{stress:.0f}%</span><small>indice temporal</small></div>
            <div><span>{ddi_html}</span><small>IDD medio</small></div>
            <div><span>{ipar_html}</span><small>IPO medio</small></div>
            <div><span>{shots}</span><small>tiros</small></div>
            <div><span>{shot_zone}</span><small>entradas último tercio</small></div>
            <div><span>{html.escape(_short_pattern_name(pattern_txt, 25))}</span><small>tipología dominante</small></div>
            <div><span>{html.escape(lane_txt)}</span><small>carril dominante</small></div>
          </div>
          <p>{html.escape(tone_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return part


def _render_critical_window_card(df: pd.DataFrame, windows: pd.DataFrame) -> pd.Series | None:
    if df.empty or windows.empty:
        return None
    top = windows.sort_values("stress", ascending=False).iloc[0]
    start, end, tramo = _window_interval(top)
    part = df[df["minuto_partido"].between(start, end, inclusive="left")].copy()
    pattern_txt = _dominant_value(part, "tipologia")
    lane_txt = _dominant_value(part, "carril")
    shots = int(pd.to_numeric(part.get("tipo_finalizacion_tiro", pd.Series(0, index=part.index)), errors="coerce").fillna(0).sum())
    shot_zone = int(part["entra_ultimo_tercio"].sum())
    ddi = float(part["ddi"].mean()) if not part.empty else 0.0
    ipar = float(part["ipar"].mean()) if not part.empty else 0.0
    ddi_html = _index_badge_html("indice_desorganizacion", ddi, compact=True)
    ipar_html = _index_badge_html("indice_peligrosidad_accion", ipar, compact=True)
    sec = int(len(part))
    collapse = float(top.get("stress", 0) or 0) * 100
    interpretation = (
        f"El rival concentró {sec} secuencias en este tramo y elevó el estrés defensivo con "
        f"{_short_pattern_name(pattern_txt, 74)}. La zona más repetida fue {lane_txt}; "
        f"la mezcla de IDD {ddi:.2f}, IPO {ipar:.2f} y {shot_zone} entradas al último tercio "
        "marca el momento donde el partido exige revisión prioritaria."
    )
    interpretation = (
        f"El rival concentro {sec} secuencias en este tramo y elevo el momentum defensivo con "
        f"{_short_pattern_name(pattern_txt, 74)}. La zona mas repetida fue {lane_txt}; "
        f"la combinacion media de IDD {ddi:.2f} e IPO {ipar:.2f} explica el periodo donde "
        "la presion acumulada exige revision prioritaria."
    )
    st.markdown(
        f"""
        <div class="chrono-critical-card">
          <div class="chrono-critical-kicker">TRAMO CRÍTICO DETECTADO</div>
          <div class="chrono-critical-title">{html.escape(tramo)}'</div>
          <div class="chrono-critical-grid">
            <div><span>{sec}</span><small>secuencias rivales</small></div>
            <div><span>{ddi_html}</span><small>IDD medio</small></div>
            <div><span>{ipar_html}</span><small>IPO medio</small></div>
            <div><span>{shots}</span><small>tiros</small></div>
            <div><span>{shot_zone}</span><small>entradas último tercio</small></div>
            <div><span>{collapse:.0f}%</span><small>Momentum defensivo</small></div>
            <div><span>{html.escape(_short_pattern_name(pattern_txt, 24))}</span><small>tipología dominante</small></div>
            <div><span>{html.escape(lane_txt)}</span><small>carril dominante</small></div>
          </div>
          <p>{html.escape(interpretation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return top


def _plot_defensive_momentum(df: pd.DataFrame, critical: pd.Series | None):
    if df.empty:
        return
    timeline = _defensive_momentum_timeline(df)
    if timeline.empty:
        return
    max_minute = int(timeline["minuto"].max())
    y_upper = max(0.46, min(1.0, float(timeline["momentum"].max()) + 0.08))

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=MOMENTUM_ALERT_THRESHOLD, fillcolor="rgba(46,160,90,0.12)", line_width=0, layer="below")
    fig.add_hrect(y0=MOMENTUM_ALERT_THRESHOLD, y1=MOMENTUM_STRESS_THRESHOLD, fillcolor="rgba(242,201,76,0.12)", line_width=0, layer="below")
    fig.add_hrect(y0=MOMENTUM_STRESS_THRESHOLD, y1=1.0, fillcolor="rgba(200,16,46,0.12)", line_width=0, layer="below")
    fig.add_hline(y=MOMENTUM_ALERT_THRESHOLD, line_dash="dot", line_color="rgba(242,201,76,0.65)", line_width=1)
    fig.add_hline(y=MOMENTUM_STRESS_THRESHOLD, line_dash="dot", line_color="rgba(200,16,46,0.72)", line_width=1)
    fig.add_annotation(x=1, y=MOMENTUM_ALERT_THRESHOLD / 2, text="CONTROLADO", showarrow=False, font=dict(color="#8ee1ad", size=11), bgcolor="rgba(32,37,50,0.54)")
    fig.add_annotation(x=1, y=(MOMENTUM_ALERT_THRESHOLD + MOMENTUM_STRESS_THRESHOLD) / 2, text="ALERTA", showarrow=False, font=dict(color="#f2c94c", size=11), bgcolor="rgba(32,37,50,0.54)")
    fig.add_annotation(x=1, y=min(y_upper - 0.025, MOMENTUM_STRESS_THRESHOLD + 0.035), text="ESTRES ALTO", showarrow=False, font=dict(color="#ff8a9b", size=11), bgcolor="rgba(32,37,50,0.54)")
    if critical is not None and "tramo" in critical:
        start, end, _ = _window_interval(critical)
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(200,16,46,0.08)", line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=timeline["minuto"],
            y=timeline["momentum"],
            mode="lines",
            line=dict(color="#f4f6fb", width=4),
            fill="tozeroy",
            fillcolor="rgba(255,255,255,0.06)",
            hovertemplate="Minuto %{x}<br>Momentum defensivo %{y:.2f}<extra></extra>",
            name="Momentum defensivo",
        )
    )
    def _event_flag(column: str) -> pd.Series:
        return pd.to_numeric(df.get(column, pd.Series(0, index=df.index)), errors="coerce").fillna(0).gt(0)

    goal_mask = _event_flag("es_gol")
    shot_on_target_mask = _event_flag("tipo_finalizacion_tiro_puerta") & ~goal_mask
    shot_mask = _event_flag("tipo_finalizacion_tiro") & ~shot_on_target_mask & ~goal_mask
    marker_specs = [
        ("Tiro", shot_mask, "#2ea05a", 22, 0.026),
        ("Tiro a puerta", shot_on_target_mask, "#f2c94c", 24, 0.034),
        ("Gol", goal_mask, "#c8102e", 26, 0.042),
    ]
    for label, mask, ring_color, size, y_offset in marker_specs:
        events = df[mask].copy()
        if events.empty:
            x_vals = [None]
            y_vals = [None]
            text_vals = [""]
        else:
            x_vals = events["minuto_partido"]
            y_vals = np.interp(events["minuto_partido"], timeline["minuto"], timeline["momentum"]) + y_offset
            text_vals = ["⚽"] * len(events)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    color="#f8fafc",
                    size=size,
                    line=dict(color=ring_color, width=4),
                ),
                text=text_vals,
                textfont=dict(size=max(10, size - 8), color="#111827"),
                textposition="middle center",
                name=label,
                hovertemplate=f"{label}<br>Minuto %{{x:.1f}}<extra></extra>",
                showlegend=True,
            )
        )
    fig.update_layout(
        title="Curva de momentum defensivo (EWMA)",
        height=420,
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0,
            font=dict(color="#f4f6fb", size=13),
            bgcolor="rgba(32,37,50,0.82)",
            bordercolor="rgba(255,255,255,0.16)",
            borderwidth=1,
        ),
    )
    _apply_plotly_theme(fig)
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=-0.22,
            x=0,
            font=dict(color="#f4f6fb", size=13),
            bgcolor="rgba(32,37,50,0.82)",
            bordercolor="rgba(255,255,255,0.16)",
            borderwidth=1,
        )
    )
    fig.update_xaxes(title="Minuto de partido", range=[0, max_minute])
    fig.update_yaxes(title="Presion acumulada", range=[0, y_upper])
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _plot_typology_timeline(df: pd.DataFrame):
    if df.empty:
        return
    labels = [str(v) for v in df["tipologia"].dropna().unique()]
    colors = _category_color_map(labels)
    plot_df = df.copy()
    plot_df["dur_min"] = (plot_df["fin_min"] - plot_df["inicio_min"]).clip(lower=0.05)
    plot_df["tipologia_corta"] = plot_df["tipologia"].map(lambda x: _short_pattern_name(x, 46))
    shot = pd.to_numeric(plot_df.get("tipo_finalizacion_tiro", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0).gt(0)
    shot_on = pd.to_numeric(plot_df.get("tipo_finalizacion_tiro_puerta", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0).gt(0)
    goal = pd.to_numeric(plot_df.get("es_gol", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0).gt(0)
    line_colors = np.where(goal, "#ff405c", np.where(shot_on, "#ff405c", np.where(shot, "#ffffff", "rgba(255,255,255,0.20)")))
    line_widths = np.where(goal, 4, np.where(shot_on, 3, np.where(shot, 2.2, 0.7)))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["dur_min"],
            y=plot_df["tipologia_corta"],
            base=plot_df["inicio_min"],
            orientation="h",
            marker=dict(
                color=[colors.get(v, "#d7dde8") for v in plot_df["tipologia"]],
                line=dict(color=line_colors.tolist(), width=line_widths.tolist()),
            ),
            customdata=np.stack(
                [
                    plot_df["secuencia_rival_id"].astype(int),
                    plot_df["tipologia"].map(lambda x: _short_pattern_name(x, 52)),
                    plot_df["ddi"].round(2),
                    plot_df["ipar"].round(2),
                    plot_df.get("xT_max", pd.Series(0, index=plot_df.index)).fillna(0).round(3),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Secuencia %{customdata[0]}<br>%{customdata[1]}"
                "<br>Minuto %{base:.1f}<br>Duración %{x:.1f} min"
                "<br>IDD %{customdata[2]} | IPO %{customdata[3]} | xT %{customdata[4]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    for minute in [15, 30, 45, 60, 75, 90]:
        fig.add_vline(x=minute, line_dash="dot", line_color="rgba(255,255,255,0.16)", line_width=1)
    fig.update_layout(title="Timeline de tipologías: insistencia, duración y finalización", height=330, bargap=0.22, showlegend=False)
    _apply_plotly_theme(fig)
    fig.update_xaxes(title="Minuto de partido")
    fig.update_yaxes(title="", automargin=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _render_momentum_formula_note():
    st.markdown(
        """
        <div class="momentum-formula-card">
          <strong>Como se calcula el momentum defensivo</strong>
          <span>Cada secuencia rival se transforma primero en una senal de presion defensiva. Esa senal combina desorganizacion del bloque y peligrosidad ofensiva rival.</span>
          <div class="momentum-formula-grid">
            <div><b>IDD</b><em>40%</em><span>Cuanto mas se rompe la estructura, mas sube la curva.</span></div>
            <div><b>IPO</b><em>60%</em><span>Prioriza el agobio rival y el peligro ofensivo generado.</span></div>
            <div><b>alpha</b><em>0.88</em><span>Conserva memoria temporal de la presion anterior.</span></div>
            <div><b>EWMA</b><em>12%</em><span>Integra la nueva senal sin que una accion aislada distorsione la lectura.</span></div>
          </div>
          <span>La curva se actualiza como M(t)=0.88*M(t-1)+0.12*(0.40*IDD+0.60*IPO). Por eso una accion aislada genera un pico contenido, mientras que varias llegadas seguidas sostienen el momentum rival.</span>
          <span>Las alertas temporales se calculan sobre la curva acumulada en tramos de 15 minutos, priorizando los intervalos donde la presion se mantiene mas alta.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_window_reading(windows: pd.DataFrame):
    if windows.empty:
        return
    readable = windows.copy()
    readable = readable[pd.to_numeric(readable["secuencias"], errors="coerce").fillna(0).gt(0)]
    if readable.empty:
        return
    top = readable.sort_values("stress", ascending=False).iloc[0]
    calm = readable.sort_values("stress", ascending=True).iloc[0]
    shot_window = readable.sort_values(["tiros", "stress"], ascending=False).iloc[0]
    cols = st.columns(3)
    cols[0].metric("Tramo más exigente", f"{top['tramo']}'", f"Presión {float(top['stress']) * 100:.0f}%")
    cols[1].metric("Tramo más estable", f"{calm['tramo']}'", f"{int(calm['secuencias'])} secuencias")
    cols[2].metric("Tramo con más remate", f"{shot_window['tramo']}'", f"{int(shot_window['tiros'])} tiros")


def _phase_label(row: pd.Series) -> tuple[str, str]:
    stress = float(row.get("stress", 0) or 0)
    shots = float(row.get("tiros", 0) or 0)
    entries = float(row.get("entradas_ultimo_tercio", 0) or 0)
    if stress >= MOMENTUM_STRESS_THRESHOLD or shots >= 2:
        return "ESTRES ALTO", "Bloque bajo presion acumulada: el rival sostiene amenaza y obliga a defender cerca del area."
    if stress >= MOMENTUM_ALERT_THRESHOLD or entries >= 2:
        return "ALERTA", "Presion rival creciente: conviene revisar ajustes de carril, saltos y coberturas."
    return "CONTROLADO", "Control defensivo razonable: pocas secuencias de alto impacto acumulado en el tramo."


def _render_phase_blocks(windows: pd.DataFrame):
    if windows.empty:
        return
    blocks = []
    for _, row in windows.iterrows():
        if int(row.get("secuencias", 0) or 0) == 0:
            continue
        label, text = _phase_label(row)
        color = {"CONTROLADO": "#2ea05a", "ALERTA": "#f2c94c", "ESTRES ALTO": "#c8102e"}[label]
        blocks.append(
            f"""
            <div class="chrono-phase-card" style="border-color:{color};">
              <div style="color:{color};">{html.escape(label)}</div>
              <strong>{html.escape(str(row['tramo']))}'</strong>
              <span>{html.escape(text)}</span>
            </div>
            """
        )
    if blocks:
        st.markdown('<div class="chrono-phase-grid">' + "".join(blocks) + "</div>", unsafe_allow_html=True)


def render_cronologia(match_id: int, meta: dict):
    seq, _, _, _ = _prepare_app_data(match_id, meta)
    if seq.empty or "minuto_partido" not in seq.columns:
        st.info("No hay cronología por secuencia para este partido.")
        return

    st.markdown(
        """
        <style>
          .chrono-critical-card {
            border: 1px solid rgba(255,255,255,.16);
            border-left: 7px solid #c8102e;
            border-top: 3px solid #ff405c;
            background: linear-gradient(135deg, rgba(48,56,74,.98), rgba(86,20,39,.92));
            border-radius: 8px;
            padding: 18px 20px 16px;
            box-shadow: 0 18px 42px rgba(0,0,0,.24), 0 0 24px rgba(200,16,46,.16);
            margin: 8px 0 20px;
          }
          .chrono-critical-kicker {
            color: #ffb3bf;
            font-size: .78rem;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
          }
          .chrono-critical-title {
            color: #fff;
            font-size: 2.3rem;
            font-weight: 950;
            line-height: 1;
            margin: 4px 0 14px;
          }
          .chrono-critical-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 12px;
          }
          .chrono-critical-grid div {
            background: rgba(10,16,32,.34);
            border: 1px solid rgba(255,255,255,.11);
            border-radius: 8px;
            padding: 10px;
            min-height: 74px;
          }
          .chrono-critical-grid span {
            display: block;
            color: #fff;
            font-size: 1.22rem;
            font-weight: 950;
            line-height: 1.08;
          }
          .chrono-critical-grid .metric-badge {
            font-size: .82rem;
          }
          .chrono-critical-grid .metric-number {
            font-size: .80em;
          }
          .chrono-critical-grid small {
            display: block;
            color: #cbd4e6;
            font-size: .76rem;
            margin-top: 6px;
            line-height: 1.15;
          }
          .chrono-critical-card p {
            color: #e7edf8;
            margin: 0;
            font-size: .98rem;
            line-height: 1.45;
          }
          .temporal-alert-card {
            border: 1px solid rgba(255,255,255,.14);
            border-left: 7px solid var(--alert-color);
            border-top: 3px solid var(--alert-color);
            background:
              linear-gradient(135deg, rgba(48,56,74,.98), var(--alert-bg));
            border-radius: 8px;
            padding: 18px 20px 16px;
            box-shadow: 0 18px 42px rgba(0,0,0,.22), 0 0 24px var(--alert-glow);
            margin: 8px 0 10px;
          }
          .temporal-alert-critical {
            --alert-color: #ff2448;
            --alert-bg: rgba(86,20,39,.64);
            --alert-glow: rgba(200,16,46,.11);
            --alert-soft: #ffb3bf;
          }
          .temporal-alert-warning {
            --alert-color: #f2c94c;
            --alert-bg: rgba(86,67,24,.82);
            --alert-glow: rgba(242,201,76,.12);
            --alert-soft: #ffe59a;
          }
          .temporal-alert-stable {
            --alert-color: #2ea05a;
            --alert-bg: rgba(22,73,54,.82);
            --alert-glow: rgba(46,160,90,.14);
            --alert-soft: #8ee1ad;
          }
          .temporal-alert-kicker {
            color: var(--alert-soft);
            font-size: .78rem;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
          }
          .temporal-alert-title {
            color: #fff;
            font-size: 2.05rem;
            font-weight: 950;
            line-height: 1;
            margin: 4px 0 14px;
          }
          .temporal-alert-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 12px;
          }
          .temporal-alert-grid div {
            background: rgba(10,16,32,.30);
            border: 1px solid rgba(255,255,255,.11);
            border-radius: 8px;
            padding: 10px;
            min-height: 74px;
          }
          .temporal-alert-grid span {
            display: block;
            color: #fff;
            font-size: 1.13rem;
            font-weight: 950;
            line-height: 1.08;
          }
          .temporal-alert-grid .metric-badge {
            font-size: .82rem;
          }
          .temporal-alert-grid .metric-number {
            font-size: .80em;
          }
          .temporal-alert-grid small {
            display: block;
            color: #d6deed;
            font-size: .76rem;
            margin-top: 6px;
            line-height: 1.15;
          }
          .temporal-alert-card p {
            color: #e7edf8;
            margin: 0;
            font-size: .95rem;
            line-height: 1.4;
          }
          .momentum-formula-card {
            background: #30384a;
            border: 1px solid rgba(255,255,255,.14);
            border-left: 5px solid #f2c94c;
            border-radius: 8px;
            padding: 12px 14px;
            margin: -4px 0 16px;
          }
          .momentum-formula-card strong {
            display: block;
            color: #ffffff;
            font-size: .95rem;
            margin-bottom: 4px;
          }
          .momentum-formula-card span {
            display: block;
            color: #cbd4e6;
            font-size: .90rem;
            line-height: 1.4;
          }
          .momentum-formula-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 8px;
            margin: 8px 0 9px;
          }
          .momentum-formula-grid div {
            background: rgba(10,16,32,.32);
            border: 1px solid rgba(255,255,255,.10);
            border-radius: 7px;
            padding: 8px 10px;
          }
          .momentum-formula-grid b {
            display: block;
            color: #ffffff;
            font-size: .82rem;
          }
          .momentum-formula-grid em {
            display: block;
            color: #f2c94c;
            font-style: normal;
            font-weight: 950;
            font-size: 1.12rem;
            margin-top: 2px;
          }
          .momentum-formula-grid span {
            color: #d8e0ef;
            font-size: .76rem;
            line-height: 1.25;
            margin-top: 5px;
          }
          .chrono-phase-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 10px;
            margin: 8px 0 18px;
          }
          .chrono-phase-card {
            background: #30384a;
            border: 1px solid;
            border-radius: 8px;
            padding: 12px;
          }
          .chrono-phase-card div {
            font-size: .72rem;
            font-weight: 950;
            letter-spacing: .07em;
          }
          .chrono-phase-card strong {
            display: block;
            color: #fff;
            font-size: 1.1rem;
            margin: 4px 0 6px;
          }
          .chrono-phase-card span {
            color: #cbd4e6;
            font-size: .86rem;
            line-height: 1.28;
          }
          @media (max-width: 1100px) {
            .chrono-critical-grid,
            .temporal-alert-grid,
            .momentum-formula-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df = _chronology_base(seq)
    if df.empty:
        st.info("No hay cronología por secuencia para este partido.")
        return
    _page_heading("Momentum")
    _section_intro(
        "Lectura general del momentum",
        "Esta pestaña ordena el partido en el tiempo para detectar cuándo se acumuló la presión rival. Primero aparece la curva de momentum y después los tramos y secuencias que conviene revisar.",
    )
    windows = _chronology_windows(df)
    selected_windows = _selected_temporal_windows(windows)
    critical = selected_windows[0][2] if selected_windows else None

    _info_heading("Momentum defensivo", MOMENTUM_INFO, level=3)
    _plot_defensive_momentum(df, critical)

    _subsection_heading("Alertas temporales del partido")
    st.caption("El partido se resume en tres tramos: el pico crítico, un tramo de alerta a vigilar y el intervalo más estable como referencia.")
    for tone, title, window in selected_windows:
        _render_temporal_alert_card(df, window, tone, title)
        start, end, tramo = _window_interval(window)
        review = seq[pd.to_numeric(seq["minuto_partido"], errors="coerce").between(start, end, inclusive="left")].copy()
        sort_col = "score_critico" if "score_critico" in review.columns else "indice_peligrosidad_accion"
        review = review.sort_values(sort_col, ascending=False) if sort_col in review.columns else review
        with st.expander(f"Ver secuencias del tramo {tramo}'"):
            _show_table(_sequence_display_table(review, n=12), height=330)

def render_riesgo(match_id: int, meta: dict):
    figs = meta["figures"]
    tables = meta["tables"]
    _page_heading("Riesgo estructural y peligrosidad")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        _show_image(match_id, figs["matriz_ddi_ipar"])
    with c2:
        _show_image(match_id, figs["matriz_5x5"])
    _subsection_heading("Ranking de secuencias criticas")
    _show_image(match_id, figs["ranking_secuencias"])
    _show_table(_load_table(match_id, tables["ranking_secuencias"]), height=280)
    _subsection_heading("Resumen de riesgo por tipologia")
    _show_table(_load_table(match_id, tables["riesgo_resumen"]), height=260)


def render_danio(match_id: int, meta: dict):
    figs = meta["figures"]
    tables = meta["tables"]
    _page_heading("Causas del dano real")
    c1, c2 = st.columns(2)
    with c1:
        _show_image(match_id, figs["causas_danio"])
        _show_table(_load_table(match_id, tables["causas_danio"]), height=220)
    with c2:
        _show_image(match_id, figs["zona_cluster_danio"])
        _show_table(_load_table(match_id, tables["zona_cluster_danio"]), height=220)
    _subsection_heading("Perfiles extremos")
    c3, c4 = st.columns(2)
    with c3:
        st.caption("Dano alto sin gran desorganizacion")
        _show_table(_load_table(match_id, tables["danio_sin_desorden_simple"]), height=240)
    with c4:
        st.caption("Caos estructural sin castigo")
        _show_table(_load_table(match_id, tables["caos_sin_castigo_simple"]), height=240)


def _top_pattern_text(seq: pd.DataFrame, metric: str, min_rows: int = 2) -> str:
    if seq.empty or "tipologia" not in seq.columns or metric not in seq.columns:
        return "patrón no identificado"
    grouped = (
        seq.groupby("tipologia", as_index=False)
        .agg(valor=(metric, "mean"), secuencias=("secuencia_rival_id", "count"))
        .query("secuencias >= @min_rows")
    )
    if grouped.empty:
        return "patrón no identificado"
    row = grouped.sort_values("valor", ascending=False).iloc[0]
    return f"{row['tipologia']} ({row['valor']:.2f})"


def _round_value(value, ndigits: int = 3):
    if value is None or pd.isna(value):
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def _records_for_report(df: pd.DataFrame, cols: list[str], n: int | None = None) -> list[dict]:
    if df.empty:
        return []
    out = df[[c for c in cols if c in df.columns]].copy()
    if n is not None:
        out = out.head(n)
    records = []
    for row in out.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                clean[key] = _round_value(value)
            elif isinstance(value, (int, np.integer)):
                clean[key] = int(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


def _report_payload(match_id: int, meta: dict) -> dict:
    seq, clusters, zona, _ = _prepare_app_data(match_id, meta)
    resumen = meta.get("resumen", {})
    teams = _presentation_teams(match_id, meta)
    equipo = _team_name_for_id(meta, meta.get("team_id", TEAM_ID_DEFAULT), "Tu Equipo")
    rival = _team_name_for_id(meta, meta.get("rival_team_id"), "Equipo Rival")
    if seq.empty:
        return {"match_id": int(match_id), "rival": rival, "error": "No hay secuencias suficientes para generar informe táctico."}

    pattern_summary = pd.DataFrame()
    if "tipologia" in seq.columns:
        pattern_summary = (
            seq.groupby("tipologia", as_index=False)
            .agg(
                secuencias=("secuencia_rival_id", "count"),
                ddi_medio=("indice_desorganizacion", "mean"),
                ddi_max=("indice_desorganizacion", "max"),
                ipar_medio=("indice_peligrosidad_accion", "mean"),
                ipar_max=("indice_peligrosidad_accion", "max"),
                xt_max=("xT_max", "max"),
                tiros=("tipo_finalizacion_tiro", "sum"),
                tiros_puerta=("tipo_finalizacion_tiro_puerta", "sum"),
                goles=("es_gol", "sum"),
                pc_rival_medio=("control_campo_rival_medio", "mean"),
                pc_zona_peligrosa=("control_zona_peligrosa_rival_medio", "mean"),
            )
            .sort_values(["ipar_medio", "ddi_medio"], ascending=False)
        )

    temporal = pd.DataFrame()
    if "minuto_partido" in seq.columns:
        tmp = seq.dropna(subset=["minuto_partido"]).copy()
        bins = [0, 15, 30, 45, 60, 75, 90, 120]
        labels = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
        tmp["tramo"] = pd.cut(tmp["minuto_partido"], bins=bins, labels=labels, include_lowest=True)
        temporal = (
            tmp.groupby("tramo", observed=False)
            .agg(
                secuencias=("secuencia_rival_id", "count"),
                ddi_medio=("indice_desorganizacion", "mean"),
                ipar_medio=("indice_peligrosidad_accion", "mean"),
                xt_max=("xT_max", "max"),
                tiros=("tipo_finalizacion_tiro", "sum"),
                tiros_puerta=("tipo_finalizacion_tiro_puerta", "sum"),
                goles=("es_gol", "sum"),
            )
            .reset_index()
        )

    causes = pd.DataFrame()
    if "causa_tactica" in seq.columns:
        causes = seq["causa_tactica"].fillna("Causa no identificada").value_counts().reset_index()
        causes.columns = ["causa", "secuencias"]

    critical = seq.sort_values("score_critico", ascending=False).copy() if "score_critico" in seq.columns else seq.copy()
    critical_cols = [
        "secuencia_rival_id",
        "minuto_partido",
        "tipologia",
        "tipo_secuencia_label",
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "score_critico",
        "xT_max",
        "causa_tactica",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
        "es_gol",
        "control_campo_rival_medio",
        "control_zona_peligrosa_rival_medio",
    ]

    pattern_records = pattern_summary.to_dict(orient="records") if not pattern_summary.empty else []
    top_freq = max(pattern_records, key=lambda x: x.get("secuencias") or 0, default={})
    top_danger = max(pattern_records, key=lambda x: x.get("ipar_medio") or 0, default={})
    top_ddi = max(pattern_records, key=lambda x: x.get("ddi_medio") or 0, default={})
    top_sequence = critical.iloc[0].to_dict() if not critical.empty else {}
    seq_top_idd = seq.sort_values("indice_desorganizacion", ascending=False).copy() if "indice_desorganizacion" in seq.columns else critical
    seq_top_ipo = seq.sort_values("indice_peligrosidad_accion", ascending=False).copy() if "indice_peligrosidad_accion" in seq.columns else critical
    momentum_critico = {}
    try:
        chrono_df = _chronology_base(seq)
        chrono_windows = _chronology_windows(chrono_df)
        chrono_windows = chrono_windows[pd.to_numeric(chrono_windows["secuencias"], errors="coerce").fillna(0).gt(0)]
        if not chrono_windows.empty:
            row = chrono_windows.sort_values("stress", ascending=False).iloc[0]
            momentum_critico = {
                "tramo": str(row.get("tramo", "-")),
                "indice_temporal": _round_value(float(row.get("stress", 0) or 0) * 100, 1),
                "secuencias": int(row.get("secuencias", 0) or 0),
                "idd_medio": _round_value(row.get("ddi")),
                "ipo_medio": _round_value(row.get("ipar")),
                "momentum_medio": _round_value(row.get("momentum")),
                "tiros": int(row.get("tiros", 0) or 0),
                "tiros_puerta": int(row.get("tiros_puerta", 0) or 0),
                "entradas_ultimo_tercio": int(row.get("entradas_ultimo_tercio", 0) or 0),
            }
    except Exception:
        momentum_critico = {}

    tramo, tramo_n = _time_window_summary(seq)
    score_text = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(
        match_id,
        teams.get("home_id"),
        teams.get("away_id"),
    )
    score_parts = [part.strip() for part in str(score_text).split("-")]
    home_goals = int(score_parts[0]) if len(score_parts) == 2 and score_parts[0].isdigit() else resumen.get("home_score")
    away_goals = int(score_parts[1]) if len(score_parts) == 2 and score_parts[1].isdigit() else resumen.get("away_score")
    return {
        "match_id": int(match_id),
        "rival": rival,
        "equipo": equipo,
        "marcador": {
            "local": teams.get("home_name"),
            "visitante": teams.get("away_name"),
            "resultado": score_text,
            "goles_local": home_goals,
            "goles_visitante": away_goals,
        },
        "metricas_globales": {
            "secuencias_rivales": int(len(seq)),
            "tiros": _sum_numeric(seq, "tipo_finalizacion_tiro"),
            "tiros_puerta": _sum_numeric(seq, "tipo_finalizacion_tiro_puerta"),
            "goles": _sum_numeric(seq, "es_gol"),
            "ddi_medio": _round_value(_mean_numeric(seq, "indice_desorganizacion")),
            "ddi_max": _round_value(_max_numeric(seq, "indice_desorganizacion")),
            "ipar_medio": _round_value(_mean_numeric(seq, "indice_peligrosidad_accion")),
            "ipar_max": _round_value(_max_numeric(seq, "indice_peligrosidad_accion")),
            "pc_rival_medio": _round_value(_mean_numeric(seq, "control_campo_rival_medio")),
            "pc_rival_zona_peligrosa": _round_value(_mean_numeric(seq, "control_zona_peligrosa_rival_medio")),
            "secuencias_alta_amenaza": int(pd.to_numeric(seq.get("indice_peligrosidad_accion", pd.Series(dtype=float)), errors="coerce").ge(0.60).sum()),
            "zona_mas_danada": _most_damaged_lane(seq),
            "tramo_mas_atacado": f"{tramo} ({tramo_n})" if tramo != "-" else "-",
        },
        "tipologias": _records_for_report(pattern_summary, list(pattern_summary.columns)),
        "causas_defensivas": _records_for_report(causes, ["causa", "secuencias"], n=8),
        "analisis_temporal": _records_for_report(temporal, list(temporal.columns)),
        "secuencias_criticas": _records_for_report(critical, critical_cols, n=8),
        "secuencias_top_idd": _records_for_report(seq_top_idd, critical_cols, n=5),
        "secuencias_top_ipo": _records_for_report(seq_top_ipo, critical_cols, n=5),
        "momentum_critico": momentum_critico,
        "resumen_ejecutivo": {
            "tipologia_mas_repetida": _short_pattern_name(top_freq.get("tipologia", "-"), 80),
            "tipologia_mas_peligrosa": _short_pattern_name(top_danger.get("tipologia", "-"), 80),
            "tipologia_mas_desorganizante": _short_pattern_name(top_ddi.get("tipologia", "-"), 80),
            "secuencia_prioritaria": int(top_sequence.get("secuencia_rival_id")) if top_sequence.get("secuencia_rival_id") is not None and not pd.isna(top_sequence.get("secuencia_rival_id")) else "-",
            "momento_critico": momentum_critico.get("tramo", f"{tramo}"),
        },
        "notas_metodologicas": {
            "idd": "Índice De Desorganización Defensiva 0-1: Anchura, distancia al balón, retroceso y Pitch Control rival.",
            "ipar": "Índice De Peligrosidad Ofensiva 0-1: Pitch Control peligroso, xT máximo y score de finalización.",
            "regla_ia": "La IA redacta e interpreta, pero no debe inventar información fuera del JSON.",
        },
    }


def _fallback_tactical_report(payload: dict) -> str:
    if payload.get("error"):
        return payload["error"]
    metrics = payload["metricas_globales"]
    rival = payload["rival"]
    equipo = payload.get("equipo", "Equipo analizado")
    tipologias = payload.get("tipologias", [])
    top_freq = max(tipologias, key=lambda x: x.get("secuencias", 0), default={}).get("tipologia", "Patron no identificado")
    top_danger = max(tipologias, key=lambda x: x.get("ipar_medio") or 0, default={}).get("tipologia", "Patron no identificado")
    top_ddi = max(tipologias, key=lambda x: x.get("ddi_medio") or 0, default={}).get("tipologia", "Patron no identificado")
    causes = payload.get("causas_defensivas", [])
    cause_txt = causes[0]["causa"] if causes else "Causa no identificada"
    critical = payload.get("secuencias_criticas", [])
    critical_lines = "\n".join(
        f"- Secuencia {s.get('secuencia_rival_id')} ({s.get('minuto_partido')}'): {s.get('tipologia')}. "
        f"IDD {s.get('indice_desorganizacion')}, IPO {s.get('indice_peligrosidad_accion')}, xT {s.get('xT_max')}. "
        f"Causa: {s.get('causa_tactica')}. Tiro/a puerta/gol: {s.get('tipo_finalizacion_tiro')}/{s.get('tipo_finalizacion_tiro_puerta')}/{s.get('es_gol')}."
        for s in critical[:5]
    )
    pattern_lines = "\n".join(
        f"| {p.get('tipologia')} | {p.get('secuencias')} | {p.get('ddi_medio')} | {p.get('ipar_medio')} | {p.get('tiros')} | {p.get('goles')} |"
        for p in tipologias
    )
    temporal_lines = "\n".join(
        f"| {t.get('tramo')} | {t.get('secuencias')} | {t.get('ddi_medio')} | {t.get('ipar_medio')} | {t.get('tiros')} | {t.get('goles')} |"
        for t in payload.get("analisis_temporal", [])
    )
    return f"""# Informe táctico postpartido: {equipo} vs {rival}

## 1. Resumen general
El rival generó **{metrics['secuencias_rivales']} secuencias ofensivas evaluables**, con **{metrics['tiros']} tiros**, **{metrics['tiros_puerta']} a puerta** y **{metrics['goles']} goles asociados**. La desorganización defensiva media fue **IDD {metrics['ddi_medio']}** y la peligrosidad media fue **IPO {metrics['ipar_medio']}**. Hubo **{metrics['secuencias_alta_amenaza']} secuencias de alta amenaza**.

La zona más dañada fue **{metrics['zona_mas_danada']}** y el tramo con mayor volumen rival fue **{metrics['tramo_mas_atacado']}**.

## 2. Tipologías de ataque rival
El patrón más repetido fue **{top_freq}**. El patrón más peligroso por IPO medio fue **{top_danger}**. El patrón que más desorganizó al bloque fue **{top_ddi}**.

| Patrón ofensivo | Secuencias | IDD medio | IPO medio | Tiros | Goles |
|---|---:|---:|---:|---:|---:|
{pattern_lines}

## 3. Organización defensiva de {equipo}
La causa defensiva más frecuente fue **{cause_txt}**. La lectura principal es que el equipo debe priorizar ajustes en protección de espacios, presión al poseedor y respuesta tras pérdida o reinicio, especialmente en los patrones con IDD alto.

## 4. Jugadas críticas
{critical_lines}

## 5. Análisis temporal
| Tramo | Secuencias | IDD medio | IPO medio | Tiros | Goles |
|---|---:|---:|---:|---:|---:|
{temporal_lines}

## 6. Conclusiones operativas
- Revisar en vídeo las secuencias con IDD e IPO altos, no solo las que terminan en tiro.
- Ajustar la protección de la zona más dañada: **{metrics['zona_mas_danada']}**.
- Preparar una corrección específica para **{top_danger}**, por ser el patrón que más amenaza genera.
- Reforzar mecanismos de reorganización tras pérdida cuando aparezcan transiciones o ataques directos.
"""


def _ollama_api_key() -> str | None:
    try:
        secret_value = st.secrets.get("OLLAMA_API_KEY")
    except Exception:
        secret_value = None
    return os.getenv("OLLAMA_API_KEY") or secret_value


def _ollama_model() -> str:
    try:
        secret_value = st.secrets.get("OLLAMA_MODEL")
    except Exception:
        secret_value = None
    return os.getenv("OLLAMA_MODEL") or secret_value or OLLAMA_MODEL_DEFAULT


def _ollama_prompt(payload: dict) -> str:
    equipo = payload.get("equipo", "equipo analizado")
    return (
        f"Actúa como analista táctico profesional de {equipo}. "
        "Genera un informe postpartido completo, claro, accionable y listo para imprimir. "
        "No inventes datos que no estén en el JSON. Si falta un dato, dilo de forma prudente. "
        "Traduce IDD, IPO, pitch control y xT a lenguaje táctico comprensible para entrenador. "
        "Estructura obligatoria en Markdown:\n"
        "1. Resumen general\n"
        "2. Tipologías de ataque rival\n"
        f"3. Organización defensiva de {equipo}\n"
        "4. Jugadas críticas prioritarias\n"
        "5. Análisis temporal del partido\n"
        "6. Conclusiones y tareas defensivas a mejorar\n\n"
        "Incluye tablas Markdown solo cuando ayuden a resumir patrones, tramos o secuencias críticas. "
        "El tono debe ser técnico, directo y útil para un entrenador con poco tiempo.\n\n"
        f"JSON DEL PARTIDO:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _call_ollama_report(payload: dict) -> str:
    api_key = _ollama_api_key()
    if not api_key:
        raise RuntimeError("No hay OLLAMA_API_KEY configurada.")
    model = _ollama_model()
    url = OLLAMA_API_URL_TEMPLATE.format(model=model)
    body = {
        "contents": [{"parts": [{"text": _ollama_prompt(payload)}]}],
        "generationConfig": {"temperature": 0.25, "topP": 0.9, "maxOutputTokens": 8192},
    }
    response = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, json=body, timeout=90)
    response.raise_for_status()
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "\n".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Ollama no devolvió texto para el informe.")
    return text


def _generate_tactical_report(match_id: int, meta: dict, use_ollama: bool = False) -> tuple[str, dict, str]:
    payload = _report_payload(match_id, meta)
    if use_ollama:
        try:
            return _call_ollama_report(payload), payload, "ollama"
        except Exception as exc:
            fallback = _fallback_tactical_report(payload)
            return f"{fallback}\n\n> Aviso: no se pudo generar con Ollama ({exc}). Se muestra informe determinista.", payload, "fallback"
    return _fallback_tactical_report(payload), payload, "fallback"


def _markdown_to_pdf_bytes(markdown_text: str, title: str) -> bytes:
    buffer = io.BytesIO()
    lines = []
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        line = line.replace("**", "").replace("__", "").replace("`", "")
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            lines.append(("H" + str(min(level, 3)), line.lstrip("#").strip()))
        else:
            lines.append(("P", line))

    with PdfPages(buffer) as pdf:
        fig = None
        ax = None
        y = 0.95

        def new_page():
            nonlocal fig, ax, y
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            y = 0.95
            ax.text(0.05, y, title, fontsize=15, weight="bold", color="#0f172a", transform=ax.transAxes)
            y -= 0.045

        new_page()
        for item in lines:
            if item == "":
                y -= 0.015
                continue
            kind, text = item
            font_size = 13 if kind == "H1" else 11.5 if kind in {"H2", "H3"} else 8.8
            weight = "bold" if kind.startswith("H") else "normal"
            width = 88 if kind == "P" else 72
            wrapped = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
            needed = 0.022 * len(wrapped) + (0.012 if kind.startswith("H") else 0)
            if y - needed < 0.06:
                new_page()
            for part in wrapped:
                ax.text(0.05, y, part, fontsize=font_size, weight=weight, color="#111827", transform=ax.transAxes, va="top")
                y -= 0.022 if kind == "P" else 0.028
            if kind.startswith("H"):
                y -= 0.008
        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return buffer.getvalue()


def _legacy_tactical_report_unused(match_id: int, meta: dict) -> str:
    if seq.empty:
        return "No hay secuencias suficientes para generar informe táctico."
    resumen = meta.get("resumen", {})
    rival = _team_name_for_id(meta, meta.get("rival_team_id"), "Equipo Rival")
    team_name = _team_name_for_id(meta, meta.get("team_id", TEAM_ID_DEFAULT), "el equipo analizado")
    ddi = pd.to_numeric(seq.get("indice_desorganizacion"), errors="coerce").mean()
    ipar = pd.to_numeric(seq.get("indice_peligrosidad_accion"), errors="coerce").mean()
    high = int(pd.to_numeric(seq.get("indice_peligrosidad_accion"), errors="coerce").gt(0.75).sum())
    top_freq = seq["tipologia"].mode().iloc[0] if "tipologia" in seq.columns and not seq["tipologia"].mode().empty else "patrón no identificado"
    top_danger = _top_pattern_text(seq, "indice_peligrosidad_accion")
    top_ddi = _top_pattern_text(seq, "indice_desorganizacion")
    cause = seq.get("causa_tactica", pd.Series(dtype=str)).mode()
    cause_txt = cause.iloc[0] if not cause.empty else "causa no identificada"
    tramo, tramo_n = _time_window_summary(seq)
    top_seqs = seq.sort_values("score_critico", ascending=False).head(4)
    bullets = []
    for _, row in top_seqs.iterrows():
        bullets.append(
            f"- Secuencia {int(row['secuencia_rival_id'])} ({row.get('minuto_partido', 0):.1f}'): "
            f"{row.get('tipologia', '-')}. IDD {row.get('indice_desorganizacion', 0):.2f}, "
            f"IPO {row.get('indice_peligrosidad_accion', 0):.2f}. Causa: {row.get('causa_tactica', '-')}. "
            f"Tiro/a puerta/gol: {int(row.get('tipo_finalizacion_tiro', 0))}/"
            f"{int(row.get('tipo_finalizacion_tiro_puerta', 0))}/{int(row.get('es_gol', 0))}."
        )
    strengths = f"{team_name} contuvo varias posesiones sin finalización clara" if _sum_numeric(seq, "tipo_finalizacion_tiro") < resumen.get("tiros_rival", 0) else f"{team_name} redujo parte de la amenaza antes del remate final"
    return f"""
### 1. Resumen ejecutivo
El rival analizado, **{rival}**, activó **{len(seq)} secuencias ofensivas evaluables**. El comportamiento defensivo medio dejó un **IDD {ddi:.2f}** y un **IPO {ipar:.2f}**, con **{high} secuencias de alta amenaza**. El patrón más frecuente fue **{top_freq}**, mientras que el más peligroso fue **{top_danger}**.

### 2. Identidad ofensiva del rival
El rival generó más volumen a través de **{top_freq}**. La amenaza no depende solo del número de ataques: el patrón con mayor daño medio fue **{top_danger}**, lo que indica qué tipo de acción conviene preparar en vídeo y charla.

### 3. Dónde sufrió la defensa
La causa defensiva dominante fue **{cause_txt}**. El patrón que más desorganizó al equipo fue **{top_ddi}**. Esto sugiere revisar especialmente la protección del espacio, la distancia al poseedor y la respuesta del bloque tras pérdida o reinicio.

### 4. Momentos del partido
El tramo con mayor volumen ofensivo rival fue **{tramo}**, con **{tramo_n} secuencias**. Este intervalo debe revisarse junto con las secuencias críticas para entender si el daño vino por fatiga, contexto de marcador, pérdidas o problemas de ajuste defensivo.

### 5. Secuencias críticas prioritarias
{chr(10).join(bullets)}

### 6. Conclusión operativa
- Mejorar protección tras pérdida cuando el rival activa transiciones.
- Reducir distancia entre líneas en acciones con {cause_txt}.
- Priorizar la revisión de vídeo de **{top_danger}**.
- Mantener como fortaleza: {strengths.lower()}.
"""


def _short_pattern_name(value: object, max_len: int = 34) -> str:
    text = str(_clean_app_text(value) or "-").replace("_", " ")
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "."


def _deep_fallback_report_sections(payload: dict) -> dict[str, str]:
    metrics = payload.get("metricas_globales", {})
    tipologias = payload.get("tipologias", [])
    temporal = payload.get("analisis_temporal", [])
    critical = payload.get("secuencias_criticas", [])
    causes = payload.get("causas_defensivas", [])
    top_freq = max(tipologias, key=lambda x: x.get("secuencias", 0), default={})
    top_danger = max(tipologias, key=lambda x: x.get("ipar_medio") or 0, default={})
    top_ddi = max(tipologias, key=lambda x: x.get("ddi_medio") or 0, default={})
    top_temporal = max(temporal, key=lambda x: x.get("secuencias") or 0, default={})
    cause = causes[0]["causa"] if causes else "causa no identificada"
    crit_txt = ", ".join(str(s.get("secuencia_rival_id")) for s in critical[:3]) or "sin secuencias destacadas"
    return {
        "resumen_global": (
            f"El rival produjo {metrics.get('secuencias_rivales', '-')} secuencias ofensivas evaluables, "
            f"con {metrics.get('tiros', '-')} tiros dentro de secuencia y {metrics.get('tiros_puerta', '-')} a puerta. "
            f"La lectura global combina un IDD medio de {metrics.get('ddi_medio', '-')} y un IPO medio de "
            f"{metrics.get('ipar_medio', '-')}, por lo que el foco no debe limitarse al remate final, sino a las "
            "situaciones en las que el rival consiguio progresar con control territorial y aumentar amenaza antes de finalizar."
        ),
        "ataque_rival": (
            f"El patron mas repetido fue {_short_pattern_name(top_freq.get('tipologia', '-'), 80)}, mientras que el patron "
            f"con mayor peligrosidad media fue {_short_pattern_name(top_danger.get('tipologia', '-'), 80)}. "
            "Esto indica que volumen y peligro no siempre aparecen en el mismo tipo de accion: conviene separar los patrones "
            "que el rival repite mucho de aquellos que, aunque aparezcan menos, castigan mas por calidad de llegada."
        ),
        "organizacion_defensiva": (
            f"El patron que mas desordeno al bloque fue {_short_pattern_name(top_ddi.get('tipologia', '-'), 80)}. "
            f"La causa defensiva dominante fue {cause}. En terminos practicos, el equipo debe revisar distancias entre lineas, "
            "orientacion de la presion sobre poseedor y proteccion de espalda cuando el rival entra en zonas de amenaza."
        ),
        "temporal": (
            f"El tramo con mayor volumen fue {top_temporal.get('tramo', metrics.get('tramo_mas_atacado', '-'))}. "
            "La lectura temporal sirve para detectar si el problema aparece por inicio de partido, fatiga, contexto de marcador "
            "o acumulacion de ataques rivales. Las secuencias con tiro deben cruzarse con los picos de IDD/IPO para priorizar video."
        ),
        "jugadas": (
            f"Las secuencias prioritarias para revisar son {crit_txt}. Deben verse atendiendo a cinco elementos: minuto, tipologia, "
            "IDD, IPO, xT maximo y causa defensiva. Si una accion combina IDD alto e IPO alto, es prioritaria aunque no termine en gol."
        ),
        "conclusiones": (
            f"Conclusiones: 1) preparar ajustes especificos para {_short_pattern_name(top_danger.get('tipologia', '-'), 80)}; "
            f"2) proteger {metrics.get('zona_mas_danada', 'la zona mas danada')} con mejores coberturas; "
            "3) reducir tiempo y espacio del poseedor en progresion; 4) usar el ranking de secuencias para seleccionar clips de video."
        ),
    }


def _ollama_structured_report(payload: dict) -> dict[str, str]:
    fallback = _deep_fallback_report_sections(payload)
    api_key = _ollama_api_key()
    if not api_key:
        return fallback
    equipo = payload.get("equipo", "equipo analizado")
    prompt = (
        f"Eres analista tactico profesional de {equipo}. Redacta un informe mas profundo y util para cuerpo tecnico. "
        "No inventes datos. Usa solo el JSON. Devuelve SOLO JSON valido con estas claves: "
        "resumen_global, ataque_rival, organizacion_defensiva, temporal, jugadas, conclusiones. "
        "Cada clave debe contener entre 4 y 7 lineas de analisis tactico, con precision, interpretacion y acciones practicas. "
        "No uses markdown, no uses tablas, no uses comillas tipograficas.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        model = _ollama_model()
        url = OLLAMA_API_URL_TEMPLATE.format(model=model)
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.25, "topP": 0.9, "maxOutputTokens": 4096},
        }
        response = requests.post(url, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, json=body, timeout=90)
        response.raise_for_status()
        data = response.json()
        text = "\n".join(part.get("text", "") for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        return {**fallback, **{k: str(v) for k, v in parsed.items() if k in fallback and v}}
    except Exception:
        return fallback


def _pdf_new_page(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#fbf3f4", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.045, 0.04), 0.91, 0.92, transform=ax.transAxes, facecolor="#fffafa", edgecolor="#e5d5d8", lw=0.8))
    ax.add_patch(plt.Rectangle((0.045, 0.895), 0.91, 0.065, transform=ax.transAxes, facecolor="#c8102e", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.045, 0.872), 0.91, 0.023, transform=ax.transAxes, facecolor="#15223b", edgecolor="none"))
    ax.text(0.075, 0.932, title, transform=ax.transAxes, fontsize=18, weight="bold", color="white", va="top")
    if subtitle:
        ax.text(0.075, 0.900, subtitle, transform=ax.transAxes, fontsize=8.8, color="white", va="top")
    return fig, ax


def _pdf_wrapped(ax, x, y, text, size=9.2, weight="normal", color="#111827", width=88, line_height=0.021):
    for paragraph in str(text).splitlines():
        lines = textwrap.wrap(paragraph, width=width) or [""]
        for line in lines:
            ax.text(x, y, line, transform=ax.transAxes, fontsize=size, weight=weight, color=color, va="top")
            y -= line_height
        y -= line_height * 0.25
    return y


def _pdf_section(ax, y, title):
    ax.text(0.075, y, title, transform=ax.transAxes, fontsize=12.5, weight="bold", color="#c8102e", va="top")
    ax.add_line(plt.Line2D([0.075, 0.925], [y - 0.012, y - 0.012], transform=ax.transAxes, color="#d6c4c7", lw=0.8))
    return y - 0.035


def _pdf_kpi_box(ax, x, y, w, h, label, value):
    ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="#ffffff", edgecolor="#d8dbe2", lw=0.7))
    ax.add_patch(plt.Rectangle((x, y + h - 0.007), w, 0.007, transform=ax.transAxes, facecolor="#c8102e", edgecolor="none"))
    ax.text(x + 0.008, y + h - 0.017, str(label).upper(), transform=ax.transAxes, fontsize=6.7, color="#6b7280", weight="bold", va="top")
    _pdf_wrapped(ax, x + 0.008, y + h - 0.035, str(value), size=8.7, weight="bold", width=max(10, int(w * 85)), line_height=0.016)


def _pdf_clean_table(ax, df: pd.DataFrame, columns: list[str], labels: list[str], bbox, font_size=6.7, max_rows=8):
    if df is None or df.empty:
        ax.text(bbox[0], bbox[1] + bbox[3] - 0.02, "Sin datos disponibles", transform=ax.transAxes, fontsize=8, color="#6b7280")
        return
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    table = df[cols].head(max_rows).copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}" if abs(float(v)) < 10 and not float(v).is_integer() else f"{int(v)}")
        else:
            table[col] = table[col].fillna("-").astype(str).map(lambda x: _short_pattern_name(x, 28))
    shown_labels = [labels[columns.index(c)] for c in cols]
    tbl = ax.table(cellText=table.values, colLabels=shown_labels, cellLoc="center", colLoc="center", bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd4d7")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#15223b")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f8eef0")
            cell.get_text().set_color("#111827")


def _build_visual_report_pdf(match_id: int, meta: dict) -> bytes:
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    analysis = _ollama_structured_report(payload)
    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    clusters_display = _clusters_for_display(clusters)
    riesgo = _load_table(match_id, meta["tables"].get("riesgo_resumen", "riesgo_resumen.csv"))
    ranking = _load_table(match_id, meta["tables"].get("ranking_secuencias", "ranking_secuencias.csv"))
    ctx = _match_report_context(match_id, meta)
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(match_id, teams["home_id"], teams["away_id"])
    metrics = payload["metricas_globales"]
    buffer = io.BytesIO()
    subtitle = f"{ctx['fecha']} | {ctx['competicion']}"

    with PdfPages(buffer) as pdf:
        fig, ax = _pdf_new_page(f"Informe tactico partido {match_id}", subtitle)
        _pdf_add_image(fig, _logo_path(teams["home_id"]), (0.09, 0.735, 0.12, 0.10))
        _pdf_add_image(fig, _logo_path(teams["away_id"]), (0.79, 0.735, 0.12, 0.10))
        ax.text(0.27, 0.79, teams["home_name"], transform=ax.transAxes, fontsize=14, weight="bold", color="#111827", ha="center")
        ax.text(0.50, 0.79, score, transform=ax.transAxes, fontsize=22, weight="bold", color="#c8102e", ha="center")
        ax.text(0.73, 0.79, teams["away_name"], transform=ax.transAxes, fontsize=14, weight="bold", color="#111827", ha="center")
        ax.text(0.10, 0.70, f"Fecha: {ctx['fecha']}", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#111827")
        ax.text(0.10, 0.675, f"Rival: {_team_name_for_id(meta, meta.get('rival_team_id'), 'Equipo Rival')}", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#111827")
        ax.text(0.10, 0.650, f"Competicion: {ctx['competicion']}", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#111827")
        y = _pdf_section(ax, 0.60, "Resumen global")
        cards = [
            ("Secuencias rivales", metrics.get("secuencias_rivales")),
            ("Patron + repetido", _short_pattern_name(_best_pattern(payload, "secuencias"))),
            ("Patron + peligroso", _short_pattern_name(_best_pattern(payload, "ipar_medio"))),
            ("Zona + dañada", metrics.get("zona_mas_danada")),
            ("Tramo + atacado", metrics.get("tramo_mas_atacado")),
            ("IDD | IPO | PC", f"{metrics.get('ddi_medio')} | {metrics.get('ipar_medio')} | {metrics.get('pc_rival_medio')}"),
        ]
        for idx, (label, value) in enumerate(cards):
            _pdf_kpi_box(ax, 0.10 + (idx % 3) * 0.285, y - 0.075 - (idx // 3) * 0.092, 0.245, 0.058, label, value)
        _pdf_wrapped(ax, 0.10, y - 0.225, analysis["resumen_global"], size=9.5, width=95, line_height=0.020)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Tipologías de ataque rival", "Mapa territorial, volumen y eficacia por patrón")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("mapa_calor_notebook", "mapa_calor_clusters.png")), (0.08, 0.55, 0.84, 0.29))
        _pdf_wrapped(ax, 0.10, 0.515, analysis["ataque_rival"], size=8.9, width=95, line_height=0.018)
        _pdf_clean_table(
            ax,
            clusters_display,
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "duracion_media", "zona_dominante", "carril_dominante"],
            ["Tipología", "N", "%", "Tiros", "A puerta", "Dur.", "Zona", "Carril"],
            [0.08, 0.13, 0.84, 0.25],
            max_rows=8,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Organización defensiva propia", "IDD, IPO, Pitch Control y causas principales")
        _pdf_clean_table(
            ax,
            riesgo,
            ["cluster_trayectoria", "secuencias", "ddi_medio", "ipar_medio", "tiros", "tiros_puerta"],
            ["Tipología", "N", "IDD med.", "IPO med.", "Tiros", "A puerta"],
            [0.08, 0.67, 0.84, 0.17],
            max_rows=6,
        )
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("matriz_ddi_ipar", "matriz_ddi_ipar.png")), (0.08, 0.36, 0.40, 0.25))
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("causas_danio", "causas_danio.png")), (0.53, 0.36, 0.39, 0.25))
        _pdf_wrapped(ax, 0.10, 0.30, analysis["organizacion_defensiva"], size=9.3, width=95, line_height=0.019)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Analisis temporal", "Tramos de partido, tiros y evolucion IDD/IPO")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("evolucion_temporal_tiros", "evolucion_temporal_tiros.png")), (0.09, 0.56, 0.82, 0.27))
        temporal = pd.DataFrame(payload.get("analisis_temporal", []))
        _pdf_clean_table(
            ax,
            temporal,
            ["tramo", "secuencias", "ddi_medio", "ipar_medio", "xt_max", "tiros", "tiros_puerta", "goles"],
            ["Tramo", "N", "IDD", "IPO", "xT max", "Tiros", "A puerta", "Goles"],
            [0.08, 0.32, 0.84, 0.18],
            max_rows=8,
        )
        _pdf_wrapped(ax, 0.10, 0.25, analysis["temporal"], size=9.3, width=95, line_height=0.019)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Jugadas criticas y plan de mejora", "Secuencias prioritarias para revisar en video")
        _pdf_clean_table(
            ax,
            ranking,
            ["secuencia_rival_id", "cluster_trayectoria", "indice_peligrosidad_accion", "indice_desorganizacion", "tipo_finalizacion_tiro", "tipo_finalizacion_tiro_puerta", "tipo_desorganizacion_principal"],
            ["ID", "Tip.", "IPO", "IDD", "Tiro", "A puerta", "Causa"],
            [0.08, 0.66, 0.84, 0.19],
            max_rows=8,
        )
        _pdf_wrapped(ax, 0.10, 0.60, analysis["jugadas"], size=9.3, width=95, line_height=0.019)
        y = _pdf_section(ax, 0.39, "Conclusiones y lineas a mejorar")
        _pdf_wrapped(ax, 0.10, y, analysis["conclusiones"], size=9.6, width=95, line_height=0.020)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_informe(match_id: int, meta: dict):
    _page_heading("Informe táctico automático")
    report_key = f"ollama_report_{match_id}"
    source_key = f"ollama_report_source_{match_id}"
    payload_key = f"ollama_report_payload_{match_id}"

    st.caption(
        "Genera un informe imprimible a partir de las metricas del partido: resumen, patrones ofensivos, "
        "organizacion defensiva, jugadas criticas, lectura temporal y tareas de mejora."
    )

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        generate = st.button("Generar informe con Ollama", type="primary", use_container_width=True)
    with c2:
        reset = st.button("Usar informe local", use_container_width=True)
    with c3:
        st.caption(f"Modelo: {_ollama_model()}")

    if reset:
        st.session_state.pop(report_key, None)
        st.session_state.pop(source_key, None)
        st.session_state.pop(payload_key, None)

    if generate:
        if not _ollama_api_key():
            st.warning("No hay clave de Ollama configurada. Se muestra el informe local.")
        with st.spinner("Ollama esta redactando el informe tactico..."):
            report_text, payload, source = _generate_tactical_report(match_id, meta, use_ollama=True)
            st.session_state[report_key] = report_text
            st.session_state[source_key] = source
            st.session_state[payload_key] = payload

    if report_key in st.session_state:
        report_text = st.session_state[report_key]
        payload = st.session_state.get(payload_key) or _report_payload(match_id, meta)
        source = st.session_state.get(source_key, "ollama")
    else:
        report_text, payload, source = _generate_tactical_report(match_id, meta, use_ollama=False)

    if source == "ollama":
        st.success("Informe generado con Ollama a partir del JSON tactico del partido.")
    else:
        st.info("Informe local disponible. Pulsa el boton de Ollama para generar una redaccion tactica mas completa.")

    pdf_bytes = _markdown_to_pdf_bytes(report_text, title=f"Informe tactico partido {match_id}")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Descargar informe en PDF",
            data=pdf_bytes,
            file_name=f"informe_tactico_{match_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Descargar informe en Markdown",
            data=report_text.encode("utf-8"),
            file_name=f"informe_tactico_{match_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown(report_text)

    with st.expander("Ver datos enviados al informe"):
        st.json(payload)

    _subsection_heading("Tablas clave")
    _, clusters, _, _ = _prepare_app_data(match_id, meta)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Patrones ofensivos")
        _show_static_table(_clusters_for_display(clusters))
    with c2:
        st.caption("Riesgo")
        _show_table(_load_table(match_id, meta["tables"]["riesgo_resumen"]), height=260)


def _fmt_report_value(value, ndigits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{ndigits}f}"
    return str(value)


def _report_table_html(df: pd.DataFrame, columns: list[str], labels: dict[str, str], n: int = 8) -> str:
    if df.empty:
        return '<p class="report-compact-text">No hay datos suficientes para esta tabla.</p>'
    table = df[[c for c in columns if c in df.columns]].head(n).copy()
    if table.empty:
        return '<p class="report-compact-text">No hay columnas disponibles para esta tabla.</p>'
    table = table.rename(columns=labels)
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda v: _fmt_report_value(v, 2))
    html_table = table.fillna("").to_html(index=False, escape=True, classes="report-table", border=0)
    return f'<div class="report-table-wrap">{html_table}</div>'


def _report_header_html(match_id: int, meta: dict) -> str:
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(
        match_id,
        teams["home_id"],
        teams["away_id"],
    )
    date_label = _match_date_label(match_id)
    return f"""
    <div class="report-head">
        <div class="report-team">
            {teams["home_logo"]}
            <div>
                <div class="report-team-label">Local</div>
                <div class="report-team-name">{html.escape(str(teams["home_name"]))}</div>
            </div>
        </div>
        <div class="report-title">
            <h2>Informe tactico defensivo</h2>
            <p>Partido {int(match_id)} | {html.escape(date_label)} | {html.escape(score)}</p>
        </div>
        <div class="report-team away">
            <div>
                <div class="report-team-label">Visitante</div>
                <div class="report-team-name">{html.escape(str(teams["away_name"]))}</div>
            </div>
            {teams["away_logo"]}
        </div>
    </div>
    """


def _report_kpis_html(payload: dict) -> str:
    metrics = payload.get("metricas_globales", {})
    items = [
        ("Secuencias", metrics.get("secuencias_rivales")),
        ("Tiros", metrics.get("tiros")),
        ("A puerta", metrics.get("tiros_puerta")),
        ("Goles", metrics.get("goles")),
        ("IDD medio", metrics.get("ddi_medio")),
        ("IPO medio", metrics.get("ipar_medio")),
    ]
    cards = "".join(
        f'<div class="report-kpi"><span>{html.escape(label)}</span><strong>{html.escape(_fmt_report_value(value))}</strong></div>'
        for label, value in items
    )
    return f'<div class="report-kpi-grid">{cards}</div>'


def _report_callouts_html(payload: dict) -> str:
    metrics = payload.get("metricas_globales", {})
    tipologias = payload.get("tipologias", [])
    top_freq = max(tipologias, key=lambda x: x.get("secuencias", 0), default={}).get("tipologia", "-")
    top_danger = max(tipologias, key=lambda x: x.get("ipar_medio") or 0, default={}).get("tipologia", "-")
    top_ddi = max(tipologias, key=lambda x: x.get("ddi_medio") or 0, default={}).get("tipologia", "-")
    items = [
        ("Patron mas repetido", top_freq),
        ("Mayor peligro IPO", top_danger),
        ("Mayor desorganizacion IDD", top_ddi),
        ("Zona mas danada", metrics.get("zona_mas_danada", "-")),
        ("Tramo mas atacado", metrics.get("tramo_mas_atacado", "-")),
        ("Alta amenaza", f"{metrics.get('secuencias_alta_amenaza', 0)} secuencias"),
    ]
    cards = "".join(
        f'<div class="report-callout"><b>{html.escape(str(title))}</b><span>{html.escape(str(value))}</span></div>'
        for title, value in items
    )
    return f'<div class="report-callouts">{cards}</div>'


def _render_report_visual(match_id: int, meta: dict, payload: dict, report_text: str):
    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    riesgo = _load_table(match_id, meta["tables"].get("riesgo_resumen", "riesgo_resumen.csv"))
    ranking = _load_table(match_id, meta["tables"].get("ranking_secuencias", "ranking_secuencias.csv"))

    st.markdown('<div class="report-shell">', unsafe_allow_html=True)
    st.markdown(_report_header_html(match_id, meta), unsafe_allow_html=True)
    st.markdown('<div class="report-body">', unsafe_allow_html=True)
    st.markdown(_report_kpis_html(payload), unsafe_allow_html=True)
    st.markdown(_report_callouts_html(payload), unsafe_allow_html=True)

    st.markdown('<div class="report-section-title">Mapa principal del riesgo</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        _show_image(match_id, "matriz_ddi_ipar.png", "IDD vs IPO por secuencia")
    with c2:
        _show_image(match_id, "evolucion_temporal_tiros.png", "Evolucion temporal con tiros")

    st.markdown('<div class="report-section-title">Lectura tactica breve</div>', unsafe_allow_html=True)
    concise = _fallback_tactical_report(payload).replace("#", "").split("##")
    summary_bits = []
    for block in concise[:4]:
        clean = " ".join(line.strip("- ").strip() for line in block.splitlines() if line.strip())
        if clean:
            summary_bits.append(clean)
    summary_text = " ".join(summary_bits[:3])
    st.markdown(f'<p class="report-compact-text">{html.escape(summary_text[:1200])}</p>', unsafe_allow_html=True)

    st.markdown('<div class="report-section-title">Patrones y riesgo</div>', unsafe_allow_html=True)
    st.markdown(
        _report_table_html(
            _clusters_for_display(clusters),
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "zona_dominante", "carril_dominante"],
            {
                "tipologia": "Patron",
                "secuencias": "Sec.",
                "porcentaje": "%",
                "tiros": "Tiros",
                "tiros_puerta": "A puerta",
                "zona_dominante": "Zona",
                "carril_dominante": "Carril",
            },
            n=10,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _report_table_html(
            riesgo,
            ["cluster_trayectoria", "secuencias", "ddi_medio", "ipar_medio", "tiros", "tiros_puerta"],
            {
                "cluster_trayectoria": "Cluster",
                "secuencias": "Sec.",
                "ddi_medio": "IDD",
                "ipar_medio": "IPO",
                "tiros": "Tiros",
                "tiros_puerta": "A puerta",
            },
            n=8,
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="report-section-title">Secuencias prioritarias</div>', unsafe_allow_html=True)
    st.markdown(
        _report_table_html(
            ranking,
            [
                "secuencia_rival_id",
                "cluster_trayectoria",
                "indice_peligrosidad_accion",
                "indice_desorganizacion",
                "tipo_desorganizacion_principal",
                "tipo_finalizacion_tiro",
                "tipo_finalizacion_tiro_puerta",
            ],
            {
                "secuencia_rival_id": "Seq.",
                "cluster_trayectoria": "Cluster",
                "indice_peligrosidad_accion": "IPO",
                "indice_desorganizacion": "IDD",
                "tipo_desorganizacion_principal": "Causa",
                "tipo_finalizacion_tiro": "Tiro",
                "tipo_finalizacion_tiro_puerta": "A puerta",
            },
            n=10,
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="report-section-title">Graficos complementarios</div>', unsafe_allow_html=True)
    c3, c4 = st.columns([1, 1])
    with c3:
        _show_image(match_id, "amenaza_media_cluster.png", "Amenaza media por patron")
    with c4:
        _show_image(match_id, "causas_danio.png", "Causas del dano")
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_informe(match_id: int, meta: dict):
    _page_heading("Informe del partido")
    st.caption("Vista mas visual para cuerpo tecnico: cabecera, escudos, fecha, KPIs, graficos, tablas limpias y texto conciso.")

    report_key = f"ollama_report_{match_id}"
    source_key = f"ollama_report_source_{match_id}"
    payload_key = f"ollama_report_payload_{match_id}"

    c1, c2 = st.columns([1, 1])
    with c1:
        generate = st.button("Generar informe con Ollama", type="primary", use_container_width=True)
    with c2:
        local = st.button("Actualizar informe local", use_container_width=True)

    if generate or local:
        if not _ollama_api_key():
            st.warning("No hay clave de Ollama configurada. Se generara una version local.")
        with st.spinner("Generando informe tactico..."):
            report_text, payload, source = _generate_tactical_report(match_id, meta, use_ollama=generate)
            st.session_state[report_key] = report_text
            st.session_state[source_key] = source
            st.session_state[payload_key] = payload

    if report_key not in st.session_state:
        report_text, payload, source = _generate_tactical_report(match_id, meta, use_ollama=False)
        st.session_state[report_key] = report_text
        st.session_state[source_key] = source
        st.session_state[payload_key] = payload

    report_text = st.session_state[report_key]
    payload = st.session_state.get(payload_key) or _report_payload(match_id, meta)
    source = st.session_state.get(source_key, "ollama")
    if source == "ollama":
        st.success("Informe generado con Ollama. La vista visual usa los datos y graficos del partido.")
    else:
        st.info("Informe local generado con reglas deterministas del proyecto.")

    _render_report_visual(match_id, meta, payload, report_text)

    pdf_bytes = _markdown_to_pdf_bytes(report_text, title=f"Informe tactico partido {match_id}")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Descargar informe en PDF",
            data=pdf_bytes,
            file_name=f"informe_tactico_{match_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Descargar informe en Markdown",
            data=report_text.encode("utf-8"),
            file_name=f"informe_tactico_{match_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with st.expander("Ver redaccion completa y JSON enviado al informe"):
        st.markdown(report_text)
        st.json(payload)


def _match_report_context(match_id: int, meta: dict) -> dict:
    path = ROOT / "eventing_partidos" / f"partido_{int(match_id)}_eventing.csv"
    ctx = {"fecha": "Fecha no disponible", "competicion": "Competicion no disponible"}
    if not path.exists():
        return ctx
    try:
        sample = pd.read_csv(path, nrows=1, low_memory=False)
    except (OSError, ValueError, EmptyDataError):
        return ctx
    if sample.empty:
        return ctx
    row = sample.iloc[0]
    dt = pd.to_datetime(row.get("match_start_time"), errors="coerce")
    if pd.notna(dt):
        ctx["fecha"] = dt.strftime("%d/%m/%Y")
    for col in ["competition_name", "competition", "league_name", "league", "tournament_name"]:
        if col in sample.columns and pd.notna(row.get(col)):
            ctx["competicion"] = str(row.get(col))
            break
    else:
        season = row.get("season_id")
        sport = row.get("sport_type")
        if pd.notna(season):
            ctx["competicion"] = f"{sport or 'FOOTBALL'} | temporada {int(season)}"
    return ctx


def _logo_path(team_id) -> Path | None:
    if team_id is None or pd.isna(team_id):
        return None
    anon_path = _anonymous_badge_path(team_id)
    if ANONYMIZE_TEAM_IDENTITIES and anon_path is not None and anon_path.exists():
        return anon_path
    for suffix in ["png", "jpg", "webp"]:
        path = ROOT / "assets" / "team_logos" / f"{int(team_id)}.{suffix}"
        if path.exists():
            return path
    return None


def _pdf_add_image(fig, path: Path, box: tuple[float, float, float, float], fit: str = "contain"):
    if path is None or not Path(path).exists():
        return
    ax_img = fig.add_axes(box)
    ax_img.set_axis_off()
    try:
        img = plt.imread(path)
    except Exception:
        return
    ax_img.imshow(img)
    ax_img.set_aspect("auto" if fit == "fill" else "equal")


def _pdf_text(ax, x, y, text, size=10, weight="normal", color="#111827", width=95, line_height=0.028):
    lines = []
    for raw in str(text).splitlines():
        lines.extend(textwrap.wrap(raw, width=width) or [""])
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, fontsize=size, weight=weight, color=color, va="top")
        y -= line_height
    return y


def _pdf_page(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f8f1f2", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, transform=ax.transAxes, facecolor="#c8102e", edgecolor="none"))
    ax.add_patch(plt.Rectangle((0, 0.84), 1, 0.04, transform=ax.transAxes, facecolor="#15223b", edgecolor="none"))
    ax.text(0.035, 0.945, title, transform=ax.transAxes, fontsize=20, weight="bold", color="white", va="top")
    if subtitle:
        ax.text(0.035, 0.895, subtitle, transform=ax.transAxes, fontsize=10, color="#f7d7dc", va="top")
    return fig, ax


def _pdf_table(ax, df: pd.DataFrame, columns: list[str], labels: list[str], bbox, font_size=8, max_rows=8):
    if df is None or df.empty:
        ax.text(bbox[0], bbox[1] + bbox[3] - 0.02, "Sin datos disponibles", transform=ax.transAxes, fontsize=9, color="#6b7280")
        return
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    table = df[cols].head(max_rows).copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}" if not float(v).is_integer() else f"{int(v)}")
        else:
            table[col] = table[col].fillna("-").astype(str).str.slice(0, 32)
    shown_labels = [labels[columns.index(c)] for c in cols]
    tbl = ax.table(
        cellText=table.values,
        colLabels=shown_labels,
        cellLoc="center",
        colLoc="center",
        bbox=bbox,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d8dbe2")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#15223b")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#fbf3f4")


def _metric_card(ax, x, y, w, h, label, value, accent="#c8102e"):
    ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="#ffffff", edgecolor="#e5e7eb", lw=0.8))
    ax.add_patch(plt.Rectangle((x, y + h - 0.012), w, 0.012, transform=ax.transAxes, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.012, y + h - 0.026, str(label).upper(), transform=ax.transAxes, fontsize=7.5, color="#6b7280", weight="bold", va="top")
    ax.text(x + 0.012, y + 0.023, str(value), transform=ax.transAxes, fontsize=15, color="#111827", weight="bold", va="bottom")


def _best_pattern(payload: dict, key: str, default="-"):
    items = payload.get("tipologias", [])
    if not items:
        return default
    if key == "secuencias":
        return max(items, key=lambda x: x.get("secuencias") or 0).get("tipologia", default)
    return max(items, key=lambda x: x.get(key) or 0).get("tipologia", default)


def _report_lines(payload: dict) -> tuple[list[str], list[str]]:
    metrics = payload.get("metricas_globales", {})
    top_danger = _best_pattern(payload, "ipar_medio")
    top_ddi = _best_pattern(payload, "ddi_medio")
    causes = payload.get("causas_defensivas", [])
    cause = causes[0]["causa"] if causes else "causa defensiva no identificada"
    conclusions = [
        f"El rival genero {metrics.get('secuencias_rivales', '-')} secuencias evaluables y {metrics.get('tiros', '-')} tiros dentro de secuencia.",
        f"El patron mas peligroso fue {top_danger}; el que mas desordeno fue {top_ddi}.",
        f"La causa defensiva dominante fue {cause}.",
    ]
    improvements = [
        f"Revisar en video las secuencias criticas del patron {top_danger}.",
        f"Ajustar la proteccion de {metrics.get('zona_mas_danada', 'la zona mas danada')} y los intervalos cercanos.",
        "Reducir distancia al poseedor en progresiones y acelerar reorganizacion tras perdida.",
        "Usar IDD e IPO juntos: no revisar solo las acciones que terminan en tiro.",
    ]
    return conclusions, improvements


def _build_visual_report_pdf(match_id: int, meta: dict) -> bytes:
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    clusters_display = _clusters_for_display(clusters)
    riesgo = _load_table(match_id, meta["tables"].get("riesgo_resumen", "riesgo_resumen.csv"))
    ranking = _load_table(match_id, meta["tables"].get("ranking_secuencias", "ranking_secuencias.csv"))
    ctx = _match_report_context(match_id, meta)
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(match_id, teams["home_id"], teams["away_id"])
    metrics = payload["metricas_globales"]
    conclusions, improvements = _report_lines(payload)
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        fig, ax = _pdf_page(f"Informe tactico partido {match_id}", f"{ctx['fecha']} | {ctx['competicion']}")
        _pdf_add_image(fig, _logo_path(teams["home_id"]), (0.09, 0.67, 0.10, 0.13))
        _pdf_add_image(fig, _logo_path(teams["away_id"]), (0.81, 0.67, 0.10, 0.13))
        ax.text(0.22, 0.75, teams["home_name"], transform=ax.transAxes, fontsize=17, weight="bold", color="#111827", ha="left")
        ax.text(0.50, 0.755, score, transform=ax.transAxes, fontsize=26, weight="bold", color="#c8102e", ha="center")
        ax.text(0.78, 0.75, teams["away_name"], transform=ax.transAxes, fontsize=17, weight="bold", color="#111827", ha="right")
        ax.text(0.08, 0.62, f"Fecha: {ctx['fecha']}", transform=ax.transAxes, fontsize=11, color="#111827", weight="bold")
        ax.text(0.08, 0.585, f"Rival: {_team_name_for_id(meta, meta.get('rival_team_id'), 'Equipo Rival')}", transform=ax.transAxes, fontsize=11, color="#111827", weight="bold")
        ax.text(0.08, 0.55, f"Competicion: {ctx['competicion']}", transform=ax.transAxes, fontsize=11, color="#111827", weight="bold")

        cards = [
            ("Secuencias", metrics.get("secuencias_rivales")),
            ("Patron + repetido", _best_pattern(payload, "secuencias")),
            ("Patron + peligroso", _best_pattern(payload, "ipar_medio")),
            ("Zona + dañada", metrics.get("zona_mas_danada")),
            ("Tramo + atacado", metrics.get("tramo_mas_atacado")),
            ("IDD | IPO | PC", f"{metrics.get('ddi_medio')} | {metrics.get('ipar_medio')} | {metrics.get('pc_rival_medio')}"),
        ]
        for idx, (label, value) in enumerate(cards):
            x = 0.08 + (idx % 3) * 0.29
            y = 0.40 - (idx // 3) * 0.12
            _metric_card(ax, x, y, 0.25, 0.085, label, value)
        y = _pdf_text(ax, 0.08, 0.18, "Resumen global", size=13, weight="bold", color="#c8102e")
        _pdf_text(
            ax,
            0.08,
            y,
            " ".join(conclusions),
            size=10,
            width=118,
            line_height=0.026,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Tipologías de ataque rival", "Mapa territorial, volumen y eficacia por patrón")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("mapa_calor_notebook", "mapa_calor_clusters.png")), (0.07, 0.39, 0.86, 0.36))
        _pdf_table(
            ax,
            clusters_display,
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "duracion_media", "zona_dominante", "carril_dominante"],
            ["Tipología", "N", "%", "Tiros", "A puerta", "Dur.", "Zona", "Carril"],
            [0.05, 0.11, 0.90, 0.22],
            font_size=7.5,
            max_rows=8,
        )
        _pdf_text(ax, 0.06, 0.35, f"Lectura: el patrón más repetido fue {_best_pattern(payload, 'secuencias')} y el de mayor IPO medio fue {_best_pattern(payload, 'ipar_medio')}.", size=9.5, width=120)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Organización defensiva propia", "IDD, IPO, Pitch Control y causas de desorganización")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("matriz_ddi_ipar", "matriz_ddi_ipar.png")), (0.05, 0.43, 0.42, 0.36))
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("causas_danio", "causas_danio.png")), (0.53, 0.43, 0.42, 0.36))
        _pdf_table(
            ax,
            riesgo,
            ["cluster_trayectoria", "secuencias", "ddi_medio", "ipar_medio", "tiros", "tiros_puerta"],
            ["Tipología", "N", "IDD med.", "IPO med.", "Tiros", "A puerta"],
            [0.05, 0.13, 0.90, 0.22],
            font_size=8,
            max_rows=8,
        )
        _pdf_text(ax, 0.06, 0.38, "La tabla resume qué patrones combinan más desorganización defensiva y peligrosidad real.", size=9.5, width=120)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Analisis temporal", "Cuándo aparecen las secuencias, la peligrosidad y los tiros")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("evolucion_temporal_tiros", "evolucion_temporal_tiros.png")), (0.06, 0.37, 0.88, 0.41))
        temporal = pd.DataFrame(payload.get("analisis_temporal", []))
        _pdf_table(
            ax,
            temporal,
            ["tramo", "secuencias", "ddi_medio", "ipar_medio", "xt_max", "tiros", "tiros_puerta", "goles"],
            ["Tramo", "N", "IDD", "IPO", "xT max", "Tiros", "A puerta", "Goles"],
            [0.05, 0.10, 0.90, 0.22],
            font_size=8,
            max_rows=8,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Jugadas criticas y plan de mejora", "Secuencias prioritarias para revisar en video")
        _pdf_table(
            ax,
            ranking,
            ["secuencia_rival_id", "cluster_trayectoria", "indice_peligrosidad_accion", "indice_desorganizacion", "tipo_finalizacion_tiro", "tipo_finalizacion_tiro_puerta", "tipo_desorganizacion_principal"],
            ["ID", "Tip.", "IPO", "IDD", "Tiro", "A puerta", "Causa"],
            [0.05, 0.52, 0.90, 0.25],
            font_size=7.5,
            max_rows=8,
        )
        y = _pdf_text(ax, 0.06, 0.45, "Conclusiones", size=14, weight="bold", color="#c8102e")
        for line in conclusions:
            y = _pdf_text(ax, 0.08, y, f"- {line}", size=10, width=112, line_height=0.026)
        y -= 0.02
        y = _pdf_text(ax, 0.06, y, "Lineas a mejorar", size=14, weight="bold", color="#c8102e")
        for line in improvements:
            y = _pdf_text(ax, 0.08, y, f"- {line}", size=10, width=112, line_height=0.026)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_informe(match_id: int, meta: dict):
    _page_heading("Informe")
    _section_intro(
        "Lectura general del informe",
        "Esta pestaña genera una salida final lista para presentar. El PDF reúne marcador, resumen ejecutivo, tipologías ofensivas, estructura defensiva, momentum, secuencias prioritarias y líneas de mejora.",
    )
    pdf_key = f"visual_report_pdf_{match_id}"

    if st.button("Generar informe", type="primary", use_container_width=True):
        with st.spinner("Generando PDF del informe tactico..."):
            try:
                st.session_state[pdf_key] = _build_visual_report_pdf(match_id, meta)
            except Exception as exc:
                st.error(f"No se pudo generar el informe: {exc}")
                st.session_state.pop(pdf_key, None)

    if pdf_key in st.session_state:
        st.success("Informe generado correctamente.")
        st.download_button(
            "Descargar informe en PDF",
            data=st.session_state[pdf_key],
            file_name=f"informe_tactico_{match_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def _build_visual_report_pdf(match_id: int, meta: dict) -> bytes:
    """Generador final del informe: A4 vertical y fiel al boceto tabular."""
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        raise RuntimeError(payload["error"])

    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    clusters_display = _clusters_for_display(clusters)
    riesgo = _load_table(match_id, meta["tables"].get("riesgo_resumen", "riesgo_resumen.csv"))
    pattern_summary = pd.DataFrame(payload.get("tipologias", []))
    temporal = pd.DataFrame(payload.get("analisis_temporal", []))
    seq_sorted = seq.sort_values("score_critico", ascending=False) if "score_critico" in seq.columns else seq.copy()
    desorg_top = seq.sort_values("indice_desorganizacion", ascending=False) if "indice_desorganizacion" in seq.columns else seq_sorted
    danger_top = seq.sort_values("indice_peligrosidad_accion", ascending=False) if "indice_peligrosidad_accion" in seq.columns else seq_sorted
    ctx = _match_report_context(match_id, meta)
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(match_id, teams["home_id"], teams["away_id"])
    metrics = payload["metricas_globales"]
    conclusions, improvements = _report_lines(payload)
    buffer = io.BytesIO()
    subtitle = f"{ctx['fecha']} | {ctx['competicion']}"

    with PdfPages(buffer) as pdf:
        fig, ax = _pdf_new_page(f"Informe tactico partido {match_id}", subtitle)
        _pdf_add_image(fig, _logo_path(teams["home_id"]), (0.09, 0.735, 0.12, 0.10))
        _pdf_add_image(fig, _logo_path(teams["away_id"]), (0.79, 0.735, 0.12, 0.10))
        ax.text(0.27, 0.79, teams["home_name"], transform=ax.transAxes, fontsize=14, weight="bold", color="#111827", ha="center")
        ax.text(0.50, 0.79, score, transform=ax.transAxes, fontsize=22, weight="bold", color="#c8102e", ha="center")
        ax.text(0.73, 0.79, teams["away_name"], transform=ax.transAxes, fontsize=14, weight="bold", color="#111827", ha="center")
        ax.text(0.10, 0.70, f"Fecha: {ctx['fecha']}", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#111827")
        ax.text(0.10, 0.675, f"Rival: {_team_name_for_id(meta, meta.get('rival_team_id'), 'Equipo Rival')}", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#111827")
        ax.text(0.10, 0.650, f"Competicion: {ctx['competicion']}", transform=ax.transAxes, fontsize=9.5, weight="bold", color="#111827")
        y = _pdf_section(ax, 0.60, "Resumen global")
        cards = [
            ("Secuencias rivales", metrics.get("secuencias_rivales")),
            ("Patron + repetido", _short_pattern_name(_best_pattern(payload, "secuencias"))),
            ("Patron + peligroso", _short_pattern_name(_best_pattern(payload, "ipar_medio"))),
            ("Zona + danada", metrics.get("zona_mas_danada")),
            ("Tramo + atacado", metrics.get("tramo_mas_atacado")),
            ("IDD | IPO | PC", f"{metrics.get('ddi_medio')} | {metrics.get('ipar_medio')} | {metrics.get('pc_rival_medio')}"),
        ]
        for idx, (label, value) in enumerate(cards):
            _pdf_kpi_box(ax, 0.10 + (idx % 3) * 0.285, y - 0.075 - (idx // 3) * 0.092, 0.245, 0.058, label, value)
        _pdf_wrapped(ax, 0.10, y - 0.225, " ".join(conclusions), size=9.5, width=95, line_height=0.020)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Tipologías de ataque rival", "Mapa de calor de las tipologías y tabla resumen")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("mapa_calor_notebook", "mapa_calor_clusters.png")), (0.08, 0.56, 0.84, 0.27))
        y = _pdf_section(ax, 0.50, "Listado de tipologías")
        tipologias_txt = []
        for _, row in clusters_display.head(8).iterrows():
            name = row.get("tipologia", row.get("cluster_trayectoria", "-"))
            label = row.get("etiqueta_tactica", "")
            tipologias_txt.append(f"- Tipología {name}: {_short_pattern_name(label, 70) if label else 'nombre pendiente'}")
        _pdf_wrapped(ax, 0.10, y, "\n".join(tipologias_txt), size=8.8, width=94, line_height=0.017)
        _pdf_clean_table(
            ax,
            clusters_display,
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "duracion_media", "zona_dominante", "carril_dominante"],
            ["Tipología", "N sec.", "%", "Tiros", "A puerta", "Dur.", "Zona", "Carril"],
            [0.06, 0.10, 0.88, 0.22],
            font_size=6.6,
            max_rows=8,
        )
        _pdf_wrapped(ax, 0.08, 0.075, "Breve resumen: revisar las tipologías con más volumen y cruzarlas con tiros, tiros a puerta y zona dominante para priorizar clips.", size=8.7, width=105, line_height=0.016)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Organización defensiva propia", "IDD, IPO, xT y Pitch Control por tipología")
        _pdf_clean_table(
            ax,
            pattern_summary,
            ["tipologia", "ddi_medio", "ipar_medio", "ddi_max", "ipar_max", "xt_max", "pc_rival_medio", "pc_zona_peligrosa"],
            ["Tipología", "IDD medio", "IPO medio", "IDD max", "IPO max", "xT max", "PC rival", "PC zona pelig."],
            [0.06, 0.58, 0.88, 0.24],
            font_size=6.4,
            max_rows=8,
        )
        _pdf_clean_table(
            ax,
            riesgo,
            ["cluster_trayectoria", "secuencias", "ddi_medio", "ipar_medio", "tiros", "tiros_puerta"],
            ["Tipología", "N", "IDD med.", "IPO med.", "Tiros", "A puerta"],
            [0.06, 0.35, 0.88, 0.15],
            font_size=7.0,
            max_rows=8,
        )
        _pdf_wrapped(ax, 0.08, 0.29, "Breve resumen: la tabla superior resume el nivel defensivo por tipología. La tabla inferior ordena los patrones por combinación de desorganización, peligrosidad y finalización.", size=9.2, width=96, line_height=0.019)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Secuencias con mayor desorganización", "Top 5 según IDD")
        _pdf_clean_table(
            ax,
            desorg_top,
            ["secuencia_rival_id", "tipologia", "minuto_partido", "duracion_seg", "indice_desorganizacion", "indice_peligrosidad_accion", "tipo_finalizacion_tiro", "tipo_finalizacion_tiro_puerta", "es_gol", "tipo_desorganizacion_principal"],
            ["ID", "Tipología", "Min", "Duración", "IDD", "IPO", "Tiro", "A puerta", "Gol", "Causa"],
            [0.04, 0.58, 0.92, 0.24],
            font_size=6.3,
            max_rows=5,
        )
        _pdf_clean_table(
            ax,
            temporal,
            ["tramo", "secuencias", "ddi_medio", "ipar_medio", "xt_max", "tiros", "tiros_puerta", "goles"],
            ["Tramo", "Sec.", "IDD", "IPO", "xT max", "Tiros", "A puerta", "Goles"],
            [0.06, 0.32, 0.88, 0.17],
            font_size=7.0,
            max_rows=8,
        )
        _pdf_wrapped(ax, 0.08, 0.25, "Breve resumen de la sección: estas son las acciones donde el equipo queda más desorganizado. Prioridad de revisión: estructura de retroceso, anchura entre jugadores y control rival en zonas peligrosas.", size=9.2, width=96, line_height=0.019)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Secuencias con más amenaza/peligro", "Top 5 según IPO")
        _pdf_clean_table(
            ax,
            danger_top,
            ["secuencia_rival_id", "tipologia", "minuto_partido", "duracion_seg", "indice_peligrosidad_accion", "indice_desorganizacion", "xT_max", "tipo_finalizacion_tiro", "tipo_finalizacion_tiro_puerta", "es_gol"],
            ["ID", "Tipología", "Min", "Duración", "IPO", "IDD", "xT max", "Tiro", "A puerta", "Gol"],
            [0.04, 0.58, 0.92, 0.24],
            font_size=6.3,
            max_rows=5,
        )
        y = _pdf_section(ax, 0.48, "Jugadas críticas prioritarias")
        _pdf_clean_table(
            ax,
            seq_sorted,
            ["secuencia_rival_id", "tipologia", "minuto_partido", "score_critico", "indice_desorganizacion", "indice_peligrosidad_accion", "tipo_finalizacion_tiro", "tipo_finalizacion_tiro_puerta", "es_gol"],
            ["ID", "Tipología", "Min", "Crit.", "IDD", "IPO", "Tiro", "A puerta", "Gol"],
            [0.06, 0.25, 0.88, 0.17],
            font_size=6.8,
            max_rows=3,
        )
        _pdf_wrapped(ax, 0.08, 0.20, "Selección: las 3 jugadas cruzan desorganización y peligrosidad máxima. Sirven para analizar una a una el minuto, la tipología, el resultado de la acción y la causa principal.", size=9.1, width=96, line_height=0.018)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_new_page("Conclusiones y lineas de mejora", "Sintesis final para cuerpo tecnico")
        y = _pdf_section(ax, 0.78, "Conclusiones")
        for line in conclusions:
            y = _pdf_wrapped(ax, 0.10, y, f"- {line}", size=9.6, width=94, line_height=0.020)
        y = _pdf_section(ax, y - 0.02, "Lineas a mejorar")
        for line in improvements:
            y = _pdf_wrapped(ax, 0.10, y, f"- {line}", size=9.6, width=94, line_height=0.020)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def _coach_report_fallback(payload: dict) -> dict[str, str]:
    metrics = payload.get("metricas_globales", {})
    executive = payload.get("resumen_ejecutivo", {})
    tipologias = payload.get("tipologias", [])
    causes = payload.get("causas_defensivas", [])
    critical = payload.get("secuencias_criticas", [])
    top_freq = max(tipologias, key=lambda x: x.get("secuencias", 0), default={})
    top_danger = max(tipologias, key=lambda x: x.get("ipar_medio") or 0, default={})
    top_ddi = max(tipologias, key=lambda x: x.get("ddi_medio") or 0, default={})
    cause = causes[0].get("causa", "causa no identificada") if causes else "causa no identificada"
    momentum = payload.get("momentum_critico", {})
    crit_txt = ", ".join(str(s.get("secuencia_rival_id")) for s in critical[:5]) or "sin secuencias destacadas"
    top_seq = critical[0] if critical else {}
    top_seq_id = top_seq.get("secuencia_rival_id", executive.get("secuencia_prioritaria", "-"))
    top_seq_min = _fmt_report_value(top_seq.get("minuto_partido"), 1)
    top_seq_score = _fmt_report_value(top_seq.get("score_critico"))
    return {
        "resumen_ejecutivo": (
            f"El rival generó {metrics.get('secuencias_rivales', '-')} secuencias ofensivas, "
            f"{metrics.get('tiros', '-')} tiros y {metrics.get('tiros_puerta', '-')} remates a puerta. "
            f"La prioridad del informe es cruzar volumen, IDD e IPO para no quedarse solo con el resultado final. "
            f"La secuencia prioritaria para vídeo es la {top_seq_id}, minuto {top_seq_min}, con prioridad {top_seq_score}."
        ),
        "tipologia_destacada": (
            f"La tipología más repetida fue {_short_pattern_name(top_freq.get('tipologia', '-'), 90)} "
            f"({top_freq.get('secuencias', '-')} secuencias). La tipología con mayor amenaza media fue "
            f"{_short_pattern_name(top_danger.get('tipologia', '-'), 90)} con IPO medio {top_danger.get('ipar_medio', '-')}. "
            "Para el cuerpo técnico, esto separa lo que el rival repite de lo que realmente castiga."
        ),
        "estructura_defensiva": (
            f"El patrón que más desorganizó al bloque fue {_short_pattern_name(top_ddi.get('tipologia', '-'), 90)} "
            f"con IDD medio {top_ddi.get('ddi_medio', '-')}. La causa dominante registrada fue {cause}. "
            "La recomendación es revisar distancias entre líneas, protección de espalda y orientación de la presión "
            "en los momentos en los que el rival progresa hacia zona de remate."
        ),
        "secuencias_criticas": (
            f"Las secuencias prioritarias son {crit_txt}. Deben revisarse en vídeo porque combinan IDD, IPO, xT, "
            "Pitch Control rival o finalización. La lectura no debe limitarse a si hubo gol: una acción sin gol puede ser "
            "más valiosa para corregir si muestra una ruptura repetible de la estructura defensiva."
        ),
        "momentum_critico": (
            f"El tramo crítico detectado fue {momentum.get('tramo', metrics.get('tramo_mas_atacado', '-'))}, "
            f"con índice temporal {momentum.get('indice_temporal', '-')} y {momentum.get('secuencias', '-')} secuencias. "
            "Este tramo concentra volumen, amenaza y desorden defensivo; por eso conviene analizarlo como bloque de partido "
            "y no como acciones aisladas."
        ),
        "recomendaciones_globales": (
            f"Priorizar vídeo de {_short_pattern_name(top_danger.get('tipologia', '-'), 90)} y de las secuencias {crit_txt}. "
            f"Proteger {metrics.get('zona_mas_danada', 'la zona más dañada')} con mejores coberturas y reducir tiempo al poseedor. "
            "El plan de mejora debe centrarse en las causas repetidas, el tramo crítico y las secuencias que combinan amenaza y desorganización."
        ),
    }


def _coach_report_sections(payload: dict) -> dict[str, str]:
    fallback = _coach_report_fallback(payload)
    api_key = _ollama_api_key()
    if not api_key:
        return fallback
    prompt = (
        "Eres analista táctico profesional para un cuerpo técnico de fútbol. "
        "Redacta un informe final muy práctico, no académico, usando solo el JSON. "
        "No inventes datos ni nombres. Devuelve SOLO JSON válido con estas claves exactas: "
        "resumen_ejecutivo, tipologia_destacada, estructura_defensiva, secuencias_criticas, momentum_critico, recomendaciones_globales. "
        "Cada clave debe contener entre 3 y 5 frases breves, con lectura táctica y recomendación accionable. "
        "Prioriza lo que debe quedarse un entrenador: patrones principales, secuencias de vídeo, tramo crítico y correcciones.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        url = OLLAMA_API_URL_TEMPLATE.format(model=_ollama_model())
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "topP": 0.85, "maxOutputTokens": 4096},
        }
        response = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        text = "\n".join(
            part.get("text", "")
            for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        ).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        return {key: str(parsed.get(key) or fallback[key]) for key in fallback}
    except Exception:
        return fallback


def _coach_report_source_label() -> str:
    return "Ollama" if _ollama_api_key() else "motor local"


def _coach_card_html(title: str, value: object, detail: object = "") -> str:
    return (
        '<div class="coach-report-card">'
        f'<span>{html.escape(str(title))}</span>'
        f'<strong>{html.escape(_fmt_report_value(value))}</strong>'
        f'<small>{html.escape(str(detail or ""))}</small>'
        '</div>'
    )


def _coach_section_html(title: str, text: str) -> str:
    paragraphs = "".join(
        f"<p>{html.escape(line.strip())}</p>"
        for line in str(text).splitlines()
        if line.strip()
    )
    if not paragraphs:
        paragraphs = f"<p>{html.escape(str(text))}</p>"
    return f'<section class="coach-report-section"><h4>{html.escape(title)}</h4>{paragraphs}</section>'


def _coach_report_styles():
    st.markdown(
        """
        <style>
        .coach-report-layout {
            border: 1px solid rgba(200,16,46,0.35);
            border-top: 5px solid #c8102e;
            border-radius: 8px;
            background: rgba(31,37,50,0.94);
            padding: 18px;
            margin: 12px 0 18px 0;
        }
        .coach-report-top {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: flex-start;
            margin-bottom: 14px;
        }
        .coach-report-top h3 {
            margin: 0;
            font-size: 1.55rem;
            color: #f4f6fb;
        }
        .coach-report-top p {
            margin: 6px 0 0 0;
            color: #aab3c4;
            line-height: 1.35;
        }
        .coach-report-engine {
            border: 1px solid rgba(255,255,255,0.12);
            color: #f4f6fb;
            padding: 7px 10px;
            border-radius: 999px;
            background: rgba(200,16,46,0.18);
            font-weight: 800;
            white-space: nowrap;
        }
        .coach-report-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 16px 0;
        }
        .coach-report-card {
            min-height: 112px;
            border: 1px solid rgba(255,255,255,0.10);
            border-top: 4px solid #c8102e;
            border-radius: 8px;
            padding: 13px;
            background: #283040;
        }
        .coach-report-card span {
            display: block;
            color: #aab3c4;
            font-size: 0.76rem;
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 9px;
        }
        .coach-report-card strong {
            display: block;
            color: #ffffff;
            font-size: 1.12rem;
            line-height: 1.15;
        }
        .coach-report-card small {
            display: block;
            color: #cbd2df;
            margin-top: 8px;
            font-size: 0.84rem;
            line-height: 1.25;
        }
        .coach-report-sections {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
        }
        .coach-report-section {
            border-left: 5px solid #c8102e;
            border-radius: 8px;
            background: #30384a;
            padding: 14px 15px;
        }
        .coach-report-section h4 {
            margin: 0 0 8px 0;
            color: #ffffff;
            font-size: 1rem;
        }
        .coach-report-section p {
            color: #dbe3f3;
            line-height: 1.48;
            margin: 0;
        }
        @media (max-width: 1100px) {
            .coach-report-grid,
            .coach-report-sections {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 720px) {
            .coach-report-grid,
            .coach-report-sections {
                grid-template-columns: 1fr;
            }
            .coach-report-top {
                display: block;
            }
            .coach-report-engine {
                display: inline-block;
                margin-top: 10px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _report_records_df(payload: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(payload.get(key, []))


def _render_coach_report_preview(match_id: int, meta: dict, payload: dict, analysis: dict[str, str]):
    metrics = payload.get("metricas_globales", {})
    executive = payload.get("resumen_ejecutivo", {})
    momentum = payload.get("momentum_critico", {})
    top_seq = payload.get("secuencias_criticas", [{}])[0] if payload.get("secuencias_criticas") else {}
    top_seq_detail = (
        f"Min. {_fmt_report_value(top_seq.get('minuto_partido'), 1)} | "
        f"IDD {_fmt_report_value(top_seq.get('indice_desorganizacion'))} | "
        f"IPO {_fmt_report_value(top_seq.get('indice_peligrosidad_accion'))}"
    )
    _coach_report_styles()
    st.markdown(
        f"""
        <div class="coach-report-layout">
            <div class="coach-report-top">
                <div>
                    <h3>Informe ejecutivo para cuerpo técnico</h3>
                    <p>Resumen operativo del partido {int(match_id)}: solo aparecen los patrones, momentos y secuencias que conviene revisar primero.</p>
                </div>
                <div class="coach-report-engine">Redacción: {html.escape(_coach_report_source_label())}</div>
            </div>
            <div class="coach-report-grid">
                {_coach_card_html("Tipología más repetida", executive.get("tipologia_mas_repetida", "-"), f'{metrics.get("secuencias_rivales", "-")} secuencias rivales totales')}
                {_coach_card_html("Tipología más peligrosa", executive.get("tipologia_mas_peligrosa", "-"), f'IPO medio global {metrics.get("ipar_medio", "-")}')}
                {_coach_card_html("Secuencia prioritaria", executive.get("secuencia_prioritaria", "-"), top_seq_detail)}
                {_coach_card_html("Momentum crítico", momentum.get("tramo", executive.get("momento_critico", "-")), f'Índice temporal {momentum.get("indice_temporal", "-")}')}
            </div>
            <div class="coach-report-sections">
                {_coach_section_html("Resumen ejecutivo", analysis.get("resumen_ejecutivo", ""))}
                {_coach_section_html("Tipología destacada", analysis.get("tipologia_destacada", ""))}
                {_coach_section_html("Estructura defensiva", analysis.get("estructura_defensiva", ""))}
                {_coach_section_html("Secuencias críticas", analysis.get("secuencias_criticas", ""))}
                {_coach_section_html("Momentum crítico", analysis.get("momentum_critico", ""))}
                {_coach_section_html("Recomendaciones globales", analysis.get("recomendaciones_globales", ""))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Gráficas clave del informe")
    c1, c2 = st.columns([1, 1])
    with c1:
        _show_image(match_id, "mapa_calor_clusters.png", "Mapa de calor global de tipologías")
    with c2:
        _show_image(match_id, "evolucion_temporal_tiros.png", "Momentum, amenaza y tiros por tramo")
    c3, c4 = st.columns([1, 1])
    with c3:
        _show_image(match_id, "matriz_ddi_ipar.png", "Relación IDD/IPO por secuencia")
    with c4:
        _show_image(match_id, "ranking_secuencias.png", "Ranking de secuencias críticas")

    st.markdown("### Secuencias para revisar en vídeo")
    table_cols = [
        "secuencia_rival_id",
        "minuto_partido",
        "tipologia",
        "score_critico",
        "indice_desorganizacion",
        "indice_peligrosidad_accion",
        "xT_max",
        "causa_tactica",
        "tipo_finalizacion_tiro",
        "tipo_finalizacion_tiro_puerta",
    ]
    labels = {
        "secuencia_rival_id": "Seq.",
        "minuto_partido": "Min.",
        "tipologia": "Tipología",
        "score_critico": "Prioridad",
        "indice_desorganizacion": "IDD",
        "indice_peligrosidad_accion": "IPO",
        "xT_max": "xT",
        "causa_tactica": "Causa",
        "tipo_finalizacion_tiro": "Tiro",
        "tipo_finalizacion_tiro_puerta": "A puerta",
    }
    tabs = st.tabs(["Combinada", "IDD", "IPO"])
    with tabs[0]:
        st.markdown(_report_table_html(_report_records_df(payload, "secuencias_criticas"), table_cols, labels, n=5), unsafe_allow_html=True)
    with tabs[1]:
        st.markdown(_report_table_html(_report_records_df(payload, "secuencias_top_idd"), table_cols, labels, n=5), unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(_report_table_html(_report_records_df(payload, "secuencias_top_ipo"), table_cols, labels, n=5), unsafe_allow_html=True)


def _pdf_draw_pitch_snapshot(fig, match_id: int, meta: dict, sequence_id: object, box: tuple[float, float, float, float], title: str):
    try:
        seq_id = int(float(sequence_id))
    except (TypeError, ValueError):
        return
    traj = _load_table(match_id, meta["tables"].get("trayectorias_ligeras", "trayectorias_ligeras.csv"))
    if traj.empty or "secuencia_rival_id" not in traj.columns:
        return
    current = traj[pd.to_numeric(traj["secuencia_rival_id"], errors="coerce").eq(seq_id)].copy()
    if current.empty or not {"ball_x_m", "ball_y_m"}.issubset(current.columns):
        return
    current["ball_x_m"] = pd.to_numeric(current["ball_x_m"], errors="coerce")
    current["ball_y_m"] = pd.to_numeric(current["ball_y_m"], errors="coerce")
    current = current.dropna(subset=["ball_x_m", "ball_y_m"]).sort_values("point_order" if "point_order" in current.columns else "match_time")
    if current.empty:
        return
    ax = fig.add_axes(box)
    ax.set_facecolor("#edf2f7")
    ax.set_xlim(0, FIELD_LENGTH_M)
    ax.set_ylim(0, FIELD_WIDTH_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#c8102e")
        spine.set_linewidth(1.2)
    line_color = "#15223b"
    ax.plot([0, FIELD_LENGTH_M, FIELD_LENGTH_M, 0, 0], [0, 0, FIELD_WIDTH_M, FIELD_WIDTH_M, 0], color=line_color, lw=1.0)
    ax.plot([FIELD_LENGTH_M / 2, FIELD_LENGTH_M / 2], [0, FIELD_WIDTH_M], color=line_color, lw=1.0)
    ax.add_patch(plt.Circle((FIELD_LENGTH_M / 2, FIELD_WIDTH_M / 2), 9.15, fill=False, color=line_color, lw=1.0))
    ax.add_patch(plt.Rectangle((0, (FIELD_WIDTH_M - 40.3) / 2), 16.5, 40.3, fill=False, color=line_color, lw=1.0))
    ax.add_patch(plt.Rectangle((FIELD_LENGTH_M - 16.5, (FIELD_WIDTH_M - 40.3) / 2), 16.5, 40.3, fill=False, color=line_color, lw=1.0))
    xs = current["ball_x_m"].clip(0, FIELD_LENGTH_M)
    ys = current["ball_y_m"].clip(0, FIELD_WIDTH_M)
    ax.plot(xs, ys, color="#c8102e", lw=2.4, alpha=0.95)
    ax.scatter(xs.iloc[0], ys.iloc[0], s=28, color="#2453a6", zorder=3)
    ax.scatter(xs.iloc[-1], ys.iloc[-1], s=34, color="#f2c94c", edgecolor="#111827", linewidth=0.5, zorder=4)
    ax.set_title(title, fontsize=8, color="#111827", weight="bold", pad=4)


def _build_visual_report_pdf(match_id: int, meta: dict) -> bytes:
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    analysis = _coach_report_sections(payload)
    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    clusters_display = _clusters_for_display(clusters)
    riesgo = _load_table(match_id, meta["tables"].get("riesgo_resumen", "riesgo_resumen.csv"))
    ranking = _load_table(match_id, meta["tables"].get("ranking_secuencias", "ranking_secuencias.csv"))
    temporal = pd.DataFrame(payload.get("analisis_temporal", []))
    top_combined = _report_records_df(payload, "secuencias_criticas")
    top_idd = _report_records_df(payload, "secuencias_top_idd")
    top_ipo = _report_records_df(payload, "secuencias_top_ipo")
    ctx = _match_report_context(match_id, meta)
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(match_id, teams["home_id"], teams["away_id"])
    metrics = payload["metricas_globales"]
    executive = payload.get("resumen_ejecutivo", {})
    momentum = payload.get("momentum_critico", {})
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        fig, ax = _pdf_page(f"Informe ejecutivo partido {match_id}", f"{ctx['fecha']} | {ctx['competicion']}")
        _pdf_add_image(fig, _logo_path(teams["home_id"]), (0.075, 0.64, 0.10, 0.14))
        _pdf_add_image(fig, _logo_path(teams["away_id"]), (0.825, 0.64, 0.10, 0.14))
        ax.text(0.20, 0.73, teams["home_name"], transform=ax.transAxes, fontsize=16, weight="bold", color="#111827", ha="left")
        ax.text(0.50, 0.735, score, transform=ax.transAxes, fontsize=24, weight="bold", color="#c8102e", ha="center")
        ax.text(0.80, 0.73, teams["away_name"], transform=ax.transAxes, fontsize=16, weight="bold", color="#111827", ha="right")
        cards = [
            ("Secuencias", metrics.get("secuencias_rivales")),
            ("Tiros / puerta", f"{metrics.get('tiros')} / {metrics.get('tiros_puerta')}"),
            ("Tipologia repetida", executive.get("tipologia_mas_repetida", "-")),
            ("Tipologia peligrosa", executive.get("tipologia_mas_peligrosa", "-")),
            ("Secuencia video", executive.get("secuencia_prioritaria", "-")),
            ("Momentum critico", momentum.get("tramo", executive.get("momento_critico", "-"))),
        ]
        for idx, (label, value) in enumerate(cards):
            _metric_card(ax, 0.07 + (idx % 3) * 0.30, 0.47 - (idx // 3) * 0.13, 0.265, 0.095, label, _short_pattern_name(value, 36))
        y = _pdf_text(ax, 0.07, 0.22, "Lectura para entrenador", size=14, weight="bold", color="#c8102e")
        _pdf_text(ax, 0.07, y, analysis["resumen_ejecutivo"], size=10, width=124, line_height=0.026)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Tipologia ofensiva rival destacada", "Volumen, zonas de ataque y amenaza generada")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("mapa_calor_notebook", "mapa_calor_clusters.png")), (0.05, 0.39, 0.44, 0.37))
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("trayectorias_cluster", "trayectorias_cluster.png")), (0.53, 0.39, 0.42, 0.37))
        _pdf_table(
            ax,
            clusters_display,
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "zona_dominante", "carril_dominante"],
            ["Tipologia", "N", "%", "Tiros", "A puerta", "Zona", "Carril"],
            [0.05, 0.10, 0.90, 0.20],
            font_size=7.4,
            max_rows=8,
        )
        _pdf_text(ax, 0.06, 0.34, analysis["tipologia_destacada"], size=9.3, width=122, line_height=0.024)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Estructura defensiva", "Donde se rompe el bloque y que causas aparecen")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("matriz_ddi_ipar", "matriz_ddi_ipar.png")), (0.05, 0.42, 0.42, 0.34))
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("causas_danio", "causas_danio.png")), (0.53, 0.42, 0.42, 0.34))
        _pdf_table(
            ax,
            riesgo,
            ["cluster_trayectoria", "secuencias", "ddi_medio", "ipar_medio", "tiros", "tiros_puerta"],
            ["Tip.", "N", "IDD med.", "IPO med.", "Tiros", "A puerta"],
            [0.05, 0.11, 0.90, 0.20],
            font_size=8,
            max_rows=8,
        )
        _pdf_text(ax, 0.06, 0.35, analysis["estructura_defensiva"], size=9.3, width=122, line_height=0.024)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Momentum critico", "Tramo de partido que debe revisar primero el cuerpo tecnico")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("evolucion_temporal_tiros", "evolucion_temporal_tiros.png")), (0.06, 0.39, 0.88, 0.36))
        _pdf_table(
            ax,
            temporal,
            ["tramo", "secuencias", "ddi_medio", "ipar_medio", "xt_max", "tiros", "tiros_puerta", "goles"],
            ["Tramo", "N", "IDD", "IPO", "xT max", "Tiros", "A puerta", "Goles"],
            [0.05, 0.10, 0.90, 0.20],
            font_size=7.8,
            max_rows=8,
        )
        _pdf_text(
            ax,
            0.06,
            0.34,
            f"Tramo critico: {momentum.get('tramo', '-')} | indice temporal {momentum.get('indice_temporal', '-')} | secuencias {momentum.get('secuencias', '-')}. {analysis['momentum_critico']}",
            size=9.2,
            width=124,
            line_height=0.023,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Secuencias criticas", "Top de jugadas para ver en video")
        _pdf_add_image(fig, _asset(match_id, meta["figures"].get("ranking_secuencias", "ranking_secuencias.png")), (0.05, 0.46, 0.40, 0.28))
        if not top_combined.empty:
            ids = top_combined["secuencia_rival_id"].head(2).tolist()
            _pdf_draw_pitch_snapshot(fig, match_id, meta, ids[0], (0.52, 0.50, 0.20, 0.22), f"Secuencia {ids[0]}")
            if len(ids) > 1:
                _pdf_draw_pitch_snapshot(fig, match_id, meta, ids[1], (0.75, 0.50, 0.20, 0.22), f"Secuencia {ids[1]}")
        _pdf_table(
            ax,
            top_combined,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "score_critico", "indice_desorganizacion", "indice_peligrosidad_accion", "xT_max", "causa_tactica"],
            ["ID", "Min", "Tipologia", "Prior.", "IDD", "IPO", "xT", "Causa"],
            [0.05, 0.13, 0.90, 0.23],
            font_size=7.1,
            max_rows=5,
        )
        _pdf_text(ax, 0.06, 0.39, analysis["secuencias_criticas"], size=9.2, width=122, line_height=0.023)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Recomendaciones tacticas", "Sintesis final y prioridades de trabajo")
        _pdf_text(ax, 0.07, 0.76, "Prioridad 1 - Tipologia rival", size=13, weight="bold", color="#c8102e")
        _pdf_text(ax, 0.07, 0.71, analysis["tipologia_destacada"], size=10, width=58, line_height=0.026)
        _pdf_text(ax, 0.53, 0.76, "Prioridad 2 - Bloque defensivo", size=13, weight="bold", color="#c8102e")
        _pdf_text(ax, 0.53, 0.71, analysis["estructura_defensiva"], size=10, width=58, line_height=0.026)
        _pdf_text(ax, 0.07, 0.42, "Top IDD", size=12, weight="bold", color="#15223b")
        _pdf_table(
            ax,
            top_idd,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "indice_desorganizacion", "indice_peligrosidad_accion"],
            ["ID", "Min", "Tip.", "IDD", "IPO"],
            [0.06, 0.18, 0.39, 0.18],
            font_size=7.4,
            max_rows=5,
        )
        _pdf_text(ax, 0.53, 0.42, "Top IPO", size=12, weight="bold", color="#15223b")
        _pdf_table(
            ax,
            top_ipo,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "indice_peligrosidad_accion", "indice_desorganizacion"],
            ["ID", "Min", "Tip.", "IPO", "IDD"],
            [0.52, 0.18, 0.39, 0.18],
            font_size=7.4,
            max_rows=5,
        )
        _pdf_text(ax, 0.07, 0.12, analysis["recomendaciones_globales"], size=10.2, width=124, line_height=0.026)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_informe(match_id: int, meta: dict):
    _page_heading("Informe")
    _section_intro(
        "Lectura ejecutiva del informe",
        "Salida final pensada para entrenador: tipologia destacada, secuencias prioritarias, momentum critico, graficas clave y recomendaciones tacticas accionables.",
    )
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        st.warning(payload["error"])
        return

    report_key = f"coach_report_sections_{match_id}"
    pdf_key = f"visual_report_pdf_{match_id}"
    if report_key not in st.session_state:
        st.session_state[report_key] = _coach_report_fallback(payload)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Generar lectura con Ollama", type="primary", use_container_width=True):
            with st.spinner("Generando lectura tactica..."):
                st.session_state[report_key] = _coach_report_sections(payload)
                st.session_state.pop(pdf_key, None)
    with c2:
        if st.button("Generar PDF profesional", use_container_width=True):
            with st.spinner("Montando informe PDF con graficas y recomendaciones..."):
                try:
                    st.session_state[pdf_key] = _build_visual_report_pdf(match_id, meta)
                except Exception as exc:
                    st.error(f"No se pudo generar el informe: {exc}")
                    st.session_state.pop(pdf_key, None)

    if not _ollama_api_key():
        st.info("Ollama se usara automaticamente cuando exista OLLAMA_API_KEY en los secretos. Mientras tanto se genera una version local determinista.")

    _render_coach_report_preview(match_id, meta, payload, st.session_state[report_key])

    if pdf_key in st.session_state:
        st.success("Informe profesional generado correctamente.")
        st.download_button(
            "Descargar informe profesional en PDF",
            data=st.session_state[pdf_key],
            file_name=f"informe_profesional_{match_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def _ollama_base_url() -> str:
    secret_value = None
    try:
        secret_value = st.secrets.get("OLLAMA_BASE_URL")
    except Exception:
        secret_value = None
    return str(os.getenv("OLLAMA_BASE_URL") or secret_value or OLLAMA_BASE_URL_DEFAULT).rstrip("/")


def _ollama_model() -> str:
    secret_value = None
    try:
        secret_value = st.secrets.get("OLLAMA_MODEL")
    except Exception:
        secret_value = None
    return str(os.getenv("OLLAMA_MODEL") or secret_value or OLLAMA_MODEL_DEFAULT)


def _extract_report_json(raw_text: str) -> dict:
    text = str(raw_text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.insert(0, text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Respuesta sin JSON valido")


def _call_ollama_report(prompt_text: str) -> str:
    body = {
        "model": _ollama_model(),
        "prompt": prompt_text,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.18,
            "top_p": 0.85,
            "num_predict": 1800,
        },
    }
    response = requests.post(f"{_ollama_base_url()}/api/generate", json=body, timeout=(5, 120))
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def _coach_report_sections(payload: dict) -> dict[str, str]:
    fallback = _coach_report_fallback(payload)
    required = list(fallback.keys())
    prompt = (
        "Eres analista tactico profesional para un cuerpo tecnico de futbol. "
        "Redacta un informe final breve, profesional y accionable usando solo el JSON. "
        "No inventes datos, jugadores ni clips. Devuelve un objeto JSON puro, sin markdown ni texto alrededor, "
        "con estas claves exactas: "
        + ", ".join(required)
        + ". Cada clave debe tener entre 3 y 5 frases utiles para un entrenador. "
        "Prioriza solo lo mas importante: tipologia destacada, secuencias criticas, momentum critico, "
        "estructura defensiva y recomendaciones. No uses tablas ni introducciones generales.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        parsed = _extract_report_json(_call_ollama_report(prompt))
        sections = {key: str(parsed.get(key) or fallback[key]).strip() for key in required}
    except Exception:
        sections = fallback
    return {key: str(_clean_app_text(value) or fallback[key]) for key, value in sections.items()}


def _build_visual_report_pdf(match_id: int, meta: dict) -> bytes:
    """Build the coach PDF with Ollama text and only figures already exposed in the web app."""
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    analysis = _coach_report_sections(payload)

    seq, clusters, _, _ = _prepare_app_data(match_id, meta)
    clusters_display = _clusters_for_display(clusters)
    riesgo = _load_table(match_id, meta["tables"].get("riesgo_resumen", "riesgo_resumen.csv"))
    ranking = _load_table(match_id, meta["tables"].get("ranking_secuencias", "ranking_secuencias.csv"))
    temporal = pd.DataFrame(payload.get("analisis_temporal", []))
    top_combined = _report_records_df(payload, "secuencias_criticas")
    top_idd = _report_records_df(payload, "secuencias_top_idd")
    top_ipo = _report_records_df(payload, "secuencias_top_ipo")
    ctx = _match_report_context(match_id, meta)
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(
        match_id, teams["home_id"], teams["away_id"]
    )
    metrics = payload["metricas_globales"]
    executive = payload.get("resumen_ejecutivo", {})
    momentum = payload.get("momentum_critico", {})
    figs = meta.get("figures", {})
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        fig, ax = _pdf_page(f"Informe tactico partido {match_id}", f"{ctx['fecha']} | {ctx['competicion']}")
        _pdf_add_image(fig, _logo_path(teams["home_id"]), (0.075, 0.64, 0.10, 0.14))
        _pdf_add_image(fig, _logo_path(teams["away_id"]), (0.825, 0.64, 0.10, 0.14))
        ax.text(0.20, 0.73, teams["home_name"], transform=ax.transAxes, fontsize=16, weight="bold", color="#111827", ha="left")
        ax.text(0.50, 0.735, score, transform=ax.transAxes, fontsize=24, weight="bold", color="#c8102e", ha="center")
        ax.text(0.80, 0.73, teams["away_name"], transform=ax.transAxes, fontsize=16, weight="bold", color="#111827", ha="right")
        cards = [
            ("Secuencias", metrics.get("secuencias_rivales")),
            ("Tiros / puerta", f"{metrics.get('tiros')} / {metrics.get('tiros_puerta')}"),
            ("Tipologia repetida", executive.get("tipologia_mas_repetida", "-")),
            ("Tipologia peligrosa", executive.get("tipologia_mas_peligrosa", "-")),
            ("Secuencia video", executive.get("secuencia_prioritaria", "-")),
            ("Momentum critico", momentum.get("tramo", executive.get("momento_critico", "-"))),
        ]
        for idx, (label, value) in enumerate(cards):
            _metric_card(ax, 0.07 + (idx % 3) * 0.30, 0.47 - (idx // 3) * 0.13, 0.265, 0.095, label, _short_pattern_name(value, 36))
        y = _pdf_text(ax, 0.07, 0.22, "Resumen ejecutivo", size=14, weight="bold", color="#c8102e")
        _pdf_text(ax, 0.07, y, analysis["resumen_ejecutivo"], size=10, width=124, line_height=0.026)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Tipologia ofensiva rival", "Graficos ya disponibles en la pagina web")
        _pdf_add_image(fig, _asset(match_id, figs.get("trayectorias_cluster", "trayectorias_cluster.png")), (0.05, 0.46, 0.42, 0.29))
        _pdf_add_image(fig, _asset(match_id, figs.get("mapa_calor_notebook", "mapa_calor_clusters.png")), (0.53, 0.46, 0.42, 0.29))
        _pdf_table(
            ax,
            clusters_display,
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "zona_dominante", "carril_dominante"],
            ["Tipologia", "N", "%", "Tiros", "A puerta", "Zona", "Carril"],
            [0.05, 0.10, 0.90, 0.20],
            font_size=7.4,
            max_rows=8,
        )
        _pdf_text(ax, 0.06, 0.39, analysis["tipologia_destacada"], size=9.4, width=122, line_height=0.024)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Estructura defensiva", "IDD, IPO y causas defensivas")
        _pdf_add_image(fig, _asset(match_id, figs.get("matriz_ddi_ipar", "matriz_ddi_ipar.png")), (0.05, 0.43, 0.42, 0.32))
        _pdf_add_image(fig, _asset(match_id, figs.get("causas_danio", "causas_danio.png")), (0.53, 0.43, 0.42, 0.32))
        _pdf_table(
            ax,
            riesgo,
            ["cluster_trayectoria", "secuencias", "ddi_medio", "ipar_medio", "tiros", "tiros_puerta"],
            ["Tip.", "N", "IDD med.", "IPO med.", "Tiros", "A puerta"],
            [0.05, 0.12, 0.90, 0.18],
            font_size=8,
            max_rows=8,
        )
        _pdf_text(ax, 0.06, 0.36, analysis["estructura_defensiva"], size=9.4, width=122, line_height=0.024)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Momentum critico", "Evolucion temporal del partido")
        _pdf_add_image(fig, _asset(match_id, figs.get("evolucion_temporal_tiros", "evolucion_temporal_tiros.png")), (0.06, 0.40, 0.88, 0.35))
        _pdf_table(
            ax,
            temporal,
            ["tramo", "secuencias", "ddi_medio", "ipar_medio", "xt_max", "tiros", "tiros_puerta", "goles"],
            ["Tramo", "N", "IDD", "IPO", "xT max", "Tiros", "A puerta", "Goles"],
            [0.05, 0.10, 0.90, 0.20],
            font_size=7.8,
            max_rows=8,
        )
        _pdf_text(
            ax,
            0.06,
            0.35,
            f"Tramo critico: {momentum.get('tramo', '-')} | indice temporal {momentum.get('indice_temporal', '-')} | secuencias {momentum.get('secuencias', '-')}. {analysis['momentum_critico']}",
            size=9.2,
            width=124,
            line_height=0.023,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Secuencias criticas", "Ranking y acciones prioritarias")
        _pdf_add_image(fig, _asset(match_id, figs.get("ranking_secuencias", "ranking_secuencias.png")), (0.06, 0.47, 0.88, 0.27))
        _pdf_table(
            ax,
            top_combined,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "score_critico", "indice_desorganizacion", "indice_peligrosidad_accion", "xT_max", "causa_tactica"],
            ["ID", "Min", "Tipologia", "Prior.", "IDD", "IPO", "xT", "Causa"],
            [0.05, 0.12, 0.90, 0.22],
            font_size=7.1,
            max_rows=5,
        )
        _pdf_text(ax, 0.06, 0.39, analysis["secuencias_criticas"], size=9.2, width=122, line_height=0.023)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = _pdf_page("Recomendaciones tacticas", "Sintesis final")
        _pdf_text(ax, 0.07, 0.74, "Prioridad 1 - Tipologia rival", size=13, weight="bold", color="#c8102e")
        _pdf_text(ax, 0.07, 0.69, analysis["tipologia_destacada"], size=10, width=58, line_height=0.026)
        _pdf_text(ax, 0.53, 0.74, "Prioridad 2 - Bloque defensivo", size=13, weight="bold", color="#c8102e")
        _pdf_text(ax, 0.53, 0.69, analysis["estructura_defensiva"], size=10, width=58, line_height=0.026)
        _pdf_text(ax, 0.07, 0.42, "Top IDD", size=12, weight="bold", color="#15223b")
        _pdf_table(
            ax,
            top_idd,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "indice_desorganizacion", "indice_peligrosidad_accion"],
            ["ID", "Min", "Tip.", "IDD", "IPO"],
            [0.06, 0.18, 0.39, 0.18],
            font_size=7.4,
            max_rows=5,
        )
        _pdf_text(ax, 0.53, 0.42, "Top IPO", size=12, weight="bold", color="#15223b")
        _pdf_table(
            ax,
            top_ipo,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "indice_peligrosidad_accion", "indice_desorganizacion"],
            ["ID", "Min", "Tip.", "IPO", "IDD"],
            [0.52, 0.18, 0.39, 0.18],
            font_size=7.4,
            max_rows=5,
        )
        _pdf_text(ax, 0.07, 0.12, analysis["recomendaciones_globales"], size=10.2, width=124, line_height=0.026)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_informe(match_id: int, meta: dict):
    _page_heading("Informe")
    _section_intro(
        "Informe final con Ollama",
        "Genera un PDF profesional para entrenador con las conclusiones principales del analisis y solo con graficos ya presentes en la pagina web.",
    )
    pdf_key = f"ollama_only_report_pdf_{match_id}"

    if st.button("Generar informe", type="primary", use_container_width=True):
        st.session_state.pop(pdf_key, None)
        with st.spinner("Ollama esta redactando y montando el informe..."):
            try:
                st.session_state[pdf_key] = _build_visual_report_pdf(match_id, meta)
            except Exception as exc:
                st.error(f"No se pudo generar el informe con Ollama: {exc}")

    if pdf_key in st.session_state:
        st.success("Informe generado correctamente.")
        st.download_button(
            "Descargar informe en PDF",
            data=st.session_state[pdf_key],
            file_name=f"informe_ollama_{match_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


PDF_RED = "#c8102e"
PDF_NAVY = "#15223b"
PDF_INK = "#111827"
PDF_MUTED = "#667085"
PDF_PAGE_BG = "#f4f6fa"
PDF_PANEL = "#ffffff"


def _elite_pdf_clean(value, default: str = "-") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    if "\u00c3" in text or "\u00c2" in text:
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    return text or default


def _elite_pdf_short(value, max_len: int = 38) -> str:
    text = _elite_pdf_clean(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip(" -.,") + "."


def _elite_pdf_page(title: str, subtitle: str = "", page: int | None = None, total: int | None = None):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#ffffff", constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.04, 0.04), 0.92, 0.92, transform=ax.transAxes, facecolor=PDF_PAGE_BG, edgecolor="#e5e7eb", linewidth=0.8))
    ax.add_patch(plt.Rectangle((0.04, 0.86), 0.92, 0.10, transform=ax.transAxes, facecolor=PDF_RED, edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.04, 0.82), 0.92, 0.04, transform=ax.transAxes, facecolor=PDF_NAVY, edgecolor="none"))
    ax.text(0.07, 0.925, title, transform=ax.transAxes, fontsize=21, weight="bold", color="white", va="top")
    if subtitle:
        ax.text(0.07, 0.875, subtitle, transform=ax.transAxes, fontsize=10.5, color="#fbe8eb", va="top")
    if page is not None and total is not None:
        ax.text(0.92, 0.055, f"{page}/{total}", transform=ax.transAxes, fontsize=8.5, color=PDF_MUTED, ha="right", va="bottom")
    return fig, ax


def _elite_pdf_text(
    ax,
    x: float,
    y: float,
    text,
    size: float = 9.5,
    weight: str = "normal",
    color: str = PDF_INK,
    width: int = 95,
    line_height: float | None = None,
    max_lines: int | None = None,
) -> float:
    clean = _elite_pdf_clean(text, "")
    lines: list[str] = []
    for raw in clean.splitlines() or [""]:
        lines.extend(textwrap.wrap(raw, width=width) or [""])
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,:;") + "..."
    line_height = line_height if line_height is not None else max(0.017, size / 370)
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, fontsize=size, weight=weight, color=color, va="top")
        y -= line_height
    return y


def _elite_pdf_panel(ax, x: float, y: float, w: float, h: float, face: str = PDF_PANEL, edge: str = "#e1e5ec", lw: float = 0.8):
    ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=lw))


def _elite_pdf_text_panel(ax, x: float, y: float, w: float, h: float, title: str, body: str, max_lines: int = 7):
    _elite_pdf_panel(ax, x, y, w, h)
    ax.add_patch(plt.Rectangle((x, y + h - 0.012), w, 0.012, transform=ax.transAxes, facecolor=PDF_RED, edgecolor="none"))
    ax.text(x + 0.018, y + h - 0.035, title, transform=ax.transAxes, fontsize=12.5, weight="bold", color=PDF_RED, va="top")
    _elite_pdf_text(ax, x + 0.018, y + h - 0.075, body, size=9.2, width=max(45, int(w * 122)), line_height=0.022, max_lines=max_lines)


def _elite_pdf_add_image(fig, ax, path: Path | None, box: tuple[float, float, float, float], title: str | None = None, pad: float = 0.012):
    x, y, w, h = box
    _elite_pdf_panel(ax, x, y, w, h)
    title_h = 0.036 if title else 0.0
    if title:
        ax.text(x + 0.014, y + h - 0.016, title, transform=ax.transAxes, fontsize=10.5, weight="bold", color=PDF_NAVY, va="top")
    if path is None or not Path(path).exists():
        ax.text(x + w / 2, y + h / 2, "Grafico no disponible", transform=ax.transAxes, fontsize=9, color=PDF_MUTED, ha="center", va="center")
        return
    try:
        img = plt.imread(path)
    except Exception:
        ax.text(x + w / 2, y + h / 2, "No se pudo cargar el grafico", transform=ax.transAxes, fontsize=9, color=PDF_MUTED, ha="center", va="center")
        return
    inner = [x + pad, y + pad, max(0.01, w - pad * 2), max(0.01, h - pad * 2 - title_h)]
    if title:
        inner[1] = y + pad
    ax_img = fig.add_axes(inner)
    ax_img.set_axis_off()
    ax_img.imshow(img)
    ax_img.set_aspect("equal")


def _elite_pdf_logo(fig, path: Path | None, box: tuple[float, float, float, float]):
    if path is None or not Path(path).exists():
        return
    try:
        img = plt.imread(path)
    except Exception:
        return
    ax_img = fig.add_axes(box)
    ax_img.set_axis_off()
    ax_img.imshow(img)
    ax_img.set_aspect("equal")


def _elite_metric_card(ax, x: float, y: float, w: float, h: float, label: str, value, accent: str = PDF_RED):
    _elite_pdf_panel(ax, x, y, w, h, face="#ffffff", edge="#d9dee8")
    ax.add_patch(plt.Rectangle((x, y + h - 0.012), w, 0.012, transform=ax.transAxes, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.014, y + h - 0.029, str(label).upper(), transform=ax.transAxes, fontsize=7.9, weight="bold", color=PDF_MUTED, va="top")
    shown = _elite_pdf_short(value, 44)
    wrap_width = max(9, int(w * 70))
    lines = textwrap.wrap(shown, width=wrap_width) or [shown]
    lines = lines[:2]
    font_size = 15 if len(lines) == 1 and len(shown) <= 12 else 11 if len(lines) == 1 else 9.4
    line_y = y + h - 0.053
    for line in lines:
        ax.text(x + 0.014, line_y, line, transform=ax.transAxes, fontsize=font_size, weight="bold", color=PDF_INK, va="top")
        line_y -= 0.027


def _elite_pdf_table(ax, df: pd.DataFrame, columns: list[str], labels: list[str], bbox, font_size: float = 7.4, max_rows: int = 5):
    if df is None or df.empty:
        _elite_pdf_panel(ax, bbox[0], bbox[1], bbox[2], bbox[3])
        ax.text(bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, "Sin datos disponibles", transform=ax.transAxes, fontsize=9, color=PDF_MUTED, ha="center", va="center")
        return
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return
    _elite_pdf_panel(ax, bbox[0], bbox[1], bbox[2], bbox[3], face="#ffffff", edge="#d7dce5")
    table = df[cols].head(max_rows).copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.2f}" if abs(float(v)) < 10 else f"{float(v):.1f}")
        else:
            table[col] = table[col].fillna("-").astype(str).map(lambda v: _elite_pdf_short(v, 26))
    shown_labels = [labels[columns.index(col)] for col in cols]
    tbl = ax.table(
        cellText=table.values,
        colLabels=shown_labels,
        cellLoc="center",
        colLoc="center",
        bbox=bbox,
    )
    tbl.set_zorder(8)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    for (row, _), cell in tbl.get_celld().items():
        cell.set_zorder(8)
        cell.get_text().set_zorder(9)
        cell.set_edgecolor("#d7dce5")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor(PDF_NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#ffffff")
            cell.get_text().set_color(PDF_INK)


def _elite_fig_path(match_id: int, figures: dict, *keys: str, fallback: str) -> Path:
    for key in keys:
        filename = figures.get(key)
        if filename:
            path = _asset(match_id, filename)
            if path.exists():
                return path
    return _asset(match_id, fallback)


def _build_visual_report_pdf(match_id: int, meta: dict) -> bytes:
    """Build a fixed-layout coach dossier with Ollama text and web-app figures only."""
    payload = _report_payload(match_id, meta)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    analysis = _coach_report_sections(payload)

    _, clusters, _, _ = _prepare_app_data(match_id, meta)
    clusters_display = _clusters_for_display(clusters)
    temporal = pd.DataFrame(payload.get("analisis_temporal", []))
    top_combined = _report_records_df(payload, "secuencias_criticas")
    top_idd = _report_records_df(payload, "secuencias_top_idd")
    top_ipo = _report_records_df(payload, "secuencias_top_ipo")
    causes = pd.DataFrame(payload.get("causas_defensivas", []))
    ctx = _match_report_context(match_id, meta)
    teams = _presentation_teams(match_id, meta)
    score = _score_text(meta.get("resultado")) or _score_text(meta.get("score")) or _infer_score(
        match_id, teams["home_id"], teams["away_id"]
    )
    metrics = payload.get("metricas_globales", {})
    executive = payload.get("resumen_ejecutivo", {})
    momentum = payload.get("momentum_critico", {})
    figs = meta.get("figures", {})
    buffer = io.BytesIO()
    total_pages = 6

    with PdfPages(buffer) as pdf:
        fig, ax = _elite_pdf_page(f"Informe tactico | Partido {match_id}", f"{ctx['fecha']} | {ctx['competicion']}", 1, total_pages)
        _elite_pdf_panel(ax, 0.07, 0.62, 0.86, 0.155, face="#ffffff")
        _elite_pdf_logo(fig, _logo_path(teams["home_id"]), (0.095, 0.646, 0.07, 0.09))
        _elite_pdf_logo(fig, _logo_path(teams["away_id"]), (0.835, 0.646, 0.07, 0.09))
        ax.text(0.18, 0.704, _elite_pdf_short(teams["home_name"], 26), transform=ax.transAxes, fontsize=15, weight="bold", color=PDF_INK, ha="left", va="center")
        ax.text(0.82, 0.704, _elite_pdf_short(teams["away_name"], 26), transform=ax.transAxes, fontsize=15, weight="bold", color=PDF_INK, ha="right", va="center")
        ax.add_patch(plt.Rectangle((0.455, 0.675), 0.09, 0.06, transform=ax.transAxes, facecolor=PDF_RED, edgecolor="none"))
        ax.text(0.50, 0.705, _elite_pdf_clean(score), transform=ax.transAxes, fontsize=20, weight="bold", color="white", ha="center", va="center")
        cards = [
            ("Secuencias", metrics.get("secuencias_rivales")),
            ("Tiros / puerta", f"{metrics.get('tiros', '-')} / {metrics.get('tiros_puerta', '-')}"),
            ("Tipologia repetida", executive.get("tipologia_mas_repetida", "-")),
            ("Tipologia peligrosa", executive.get("tipologia_mas_peligrosa", "-")),
            ("Secuencia prioritaria", executive.get("secuencia_prioritaria", "-")),
            ("Momentum critico", momentum.get("tramo", executive.get("momento_critico", "-"))),
        ]
        for idx, (label, value) in enumerate(cards):
            _elite_metric_card(ax, 0.07 + (idx % 3) * 0.30, 0.465 - (idx // 3) * 0.112, 0.26, 0.084, label, value)
        _elite_pdf_text_panel(ax, 0.07, 0.105, 0.86, 0.175, "Resumen ejecutivo", analysis["resumen_ejecutivo"], max_lines=6)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _elite_pdf_page("Tipologia ofensiva rival", "Trayectorias y mapas de calor ya presentes en la pagina web", 2, total_pages)
        _elite_pdf_add_image(
            fig,
            ax,
            _elite_fig_path(match_id, figs, "trayectorias_cluster", fallback="trayectorias_cluster.png"),
            (0.065, 0.465, 0.41, 0.29),
            "Trayectorias por tipologia",
        )
        _elite_pdf_add_image(
            fig,
            ax,
            _elite_fig_path(match_id, figs, "mapa_calor_notebook", "mapa_calor_clusters", fallback="mapa_calor_clusters.png"),
            (0.525, 0.465, 0.41, 0.29),
            "Mapa de calor por tipologia",
        )
        _elite_pdf_table(
            ax,
            clusters_display,
            ["tipologia", "secuencias", "porcentaje", "tiros", "tiros_puerta", "zona_dominante"],
            ["Tipologia", "N", "%", "Tiros", "A puerta", "Zona"],
            [0.07, 0.285, 0.86, 0.125],
            font_size=7.1,
            max_rows=4,
        )
        _elite_pdf_text_panel(ax, 0.07, 0.095, 0.86, 0.155, "Lectura tactica", analysis["tipologia_destacada"], max_lines=5)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _elite_pdf_page("Estructura defensiva", "IDD, IPO y causas defensivas prioritarias", 3, total_pages)
        defensive_cards = [
            ("IDD medio", metrics.get("ddi_medio", "-")),
            ("IDD maximo", metrics.get("ddi_max", "-")),
            ("IPO medio", metrics.get("ipar_medio", "-")),
            ("IPO maximo", metrics.get("ipar_max", "-")),
            ("Pitch control rival", metrics.get("pc_rival_medio", "-")),
            ("Zona mas dañada", metrics.get("zona_mas_danada", "-")),
        ]
        for idx, (label, value) in enumerate(defensive_cards):
            _elite_metric_card(ax, 0.07 + (idx % 3) * 0.30, 0.67 - (idx // 3) * 0.11, 0.26, 0.082, label, value)
        _elite_pdf_table(
            ax,
            causes,
            ["causa", "secuencias"],
            ["Causa", "Secuencias"],
            [0.07, 0.37, 0.40, 0.13],
            font_size=7.2,
            max_rows=5,
        )
        _elite_pdf_table(
            ax,
            clusters_display,
            ["tipologia", "ddi_medio", "ipar_medio", "tiros", "zona_dominante"],
            ["Tipologia", "IDD", "IPO", "Tiros", "Zona"],
            [0.53, 0.37, 0.40, 0.13],
            font_size=7.0,
            max_rows=4,
        )
        _elite_pdf_text_panel(ax, 0.07, 0.115, 0.86, 0.20, "Diagnostico defensivo", analysis["estructura_defensiva"], max_lines=7)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _elite_pdf_page("Momentum critico", "Tramo temporal que concentra mayor prioridad defensiva", 4, total_pages)
        momentum_cards = [
            ("Tramo", momentum.get("tramo", "-")),
            ("Indice temporal", momentum.get("indice_temporal", "-")),
            ("Secuencias", momentum.get("secuencias", "-")),
            ("Tiros", momentum.get("tiros", "-")),
            ("IDD medio", momentum.get("idd_medio", "-")),
            ("IPO medio", momentum.get("ipo_medio", "-")),
        ]
        for idx, (label, value) in enumerate(momentum_cards):
            _elite_metric_card(ax, 0.07 + (idx % 3) * 0.30, 0.67 - (idx // 3) * 0.11, 0.26, 0.082, label, value, accent="#f2c94c" if idx == 1 else PDF_RED)
        _elite_pdf_table(
            ax,
            temporal,
            ["tramo", "secuencias", "ddi_medio", "ipar_medio", "xt_max", "tiros", "tiros_puerta", "goles"],
            ["Tramo", "N", "IDD", "IPO", "xT", "Tiros", "A puerta", "Goles"],
            [0.07, 0.37, 0.86, 0.14],
            font_size=7.1,
            max_rows=7,
        )
        momentum_text = f"Tramo critico: {_elite_pdf_clean(momentum.get('tramo'))}. {analysis['momentum_critico']}"
        _elite_pdf_text_panel(ax, 0.07, 0.12, 0.86, 0.19, "Lectura del momentum", momentum_text, max_lines=7)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _elite_pdf_page("Secuencias criticas", "Top de acciones que debe revisar primero el cuerpo tecnico", 5, total_pages)
        _elite_pdf_table(
            ax,
            top_combined,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "score_critico", "indice_desorganizacion", "indice_peligrosidad_accion", "xT_max"],
            ["ID", "Min", "Tipologia", "Prior.", "IDD", "IPO", "xT"],
            [0.07, 0.61, 0.86, 0.15],
            font_size=7.1,
            max_rows=5,
        )
        _elite_pdf_table(
            ax,
            top_idd,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "indice_desorganizacion", "indice_peligrosidad_accion"],
            ["ID", "Min", "Tipologia", "IDD", "IPO"],
            [0.07, 0.405, 0.40, 0.13],
            font_size=6.8,
            max_rows=5,
        )
        _elite_pdf_table(
            ax,
            top_ipo,
            ["secuencia_rival_id", "minuto_partido", "tipologia", "indice_peligrosidad_accion", "indice_desorganizacion"],
            ["ID", "Min", "Tipologia", "IPO", "IDD"],
            [0.53, 0.405, 0.40, 0.13],
            font_size=6.8,
            max_rows=5,
        )
        ax.text(0.07, 0.555, "Prioridad combinada", transform=ax.transAxes, fontsize=11, weight="bold", color=PDF_NAVY, va="top")
        ax.text(0.07, 0.355, "Top IDD", transform=ax.transAxes, fontsize=11, weight="bold", color=PDF_NAVY, va="top")
        ax.text(0.53, 0.355, "Top IPO", transform=ax.transAxes, fontsize=11, weight="bold", color=PDF_NAVY, va="top")
        _elite_pdf_text_panel(ax, 0.07, 0.10, 0.86, 0.18, "Conclusiones sobre secuencias", analysis["secuencias_criticas"], max_lines=6)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _elite_pdf_page("Recomendaciones tacticas", "Sintesis final accionable", 6, total_pages)
        _elite_pdf_text_panel(ax, 0.07, 0.52, 0.40, 0.24, "Prioridad 1 - Tipologia rival", analysis["tipologia_destacada"], max_lines=7)
        _elite_pdf_text_panel(ax, 0.53, 0.52, 0.40, 0.24, "Prioridad 2 - Bloque defensivo", analysis["estructura_defensiva"], max_lines=7)
        _elite_pdf_text_panel(ax, 0.07, 0.265, 0.40, 0.20, "Prioridad 3 - Secuencias criticas", analysis["secuencias_criticas"], max_lines=5)
        _elite_pdf_text_panel(ax, 0.53, 0.265, 0.40, 0.20, "Prioridad 4 - Momentum", analysis["momentum_critico"], max_lines=5)
        _elite_pdf_text_panel(ax, 0.07, 0.075, 0.86, 0.14, "Plan de trabajo global", analysis["recomendaciones_globales"], max_lines=4)
        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_informe(match_id: int, meta: dict):
    _page_heading("Informe")
    _section_intro(
        "Informe final con Ollama",
        "Genera un PDF profesional para entrenador con conclusiones principales, KPIs, graficos ya presentes en la pagina web y recomendaciones tacticas.",
    )
    pdf_key = f"ollama_report_pdf_{match_id}"

    if st.button("Generar informe", type="primary", use_container_width=True):
        st.session_state.pop(pdf_key, None)
        with st.spinner("Ollama esta redactando y montando el informe..."):
            try:
                st.session_state[pdf_key] = _build_visual_report_pdf(match_id, meta)
            except Exception as exc:
                st.error(f"No se pudo generar el informe con Ollama: {exc}")

    if pdf_key in st.session_state:
        st.success("Informe generado correctamente.")
        st.download_button(
            "Descargar informe en PDF",
            data=st.session_state[pdf_key],
            file_name=f"informe_ollama_{match_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )


def main():
    inject_style()
    inject_selectbox_dropdown_fix()
    inject_sidebar_toggle_v2()
    if not render_login():
        st.stop()
    if "app_view" not in st.session_state:
        st.session_state["app_view"] = "portal"
    render_global_sidebar()

    view = str(st.session_state.get("app_view", "portal"))
    if view == "portal":
        render_topbar()
        render_portal_home()
    elif view == "admin_users":
        if not _is_admin():
            st.session_state["app_view"] = "portal"
            st.rerun()
        render_topbar()
        render_admin_users()
    elif view == "club":
        team = _team_by_id(st.session_state.get("selected_team_id"))
        if not _can_access_team(int(team["team_id"])):
            st.warning("Tu usuario no tiene permiso para acceder a este club.")
            st.session_state["app_view"] = "portal"
            st.rerun()
        render_topbar(team)
        team_name = _team_display_name(team)
        st.markdown(
            f'<div class="app-breadcrumb">Menú principal / <b>{html.escape(team_name)}</b></div>',
            unsafe_allow_html=True,
        )
        render_club_home(team)
    elif view == "analysis":
        team = _team_by_id(st.session_state.get("selected_team_id"))
        if not _can_access_team(int(team["team_id"])):
            st.warning("Tu usuario no tiene permiso para acceder a este analisis.")
            st.session_state["app_view"] = "portal"
            st.rerun()
        match_id = st.session_state.get("selected_match_id")
        matches = _team_matches(int(team["team_id"]))
        if match_id is None and not matches.empty:
            match_id = int(matches.iloc[0]["match_id"])
            st.session_state["selected_match_id"] = match_id
        if match_id is None:
            st.session_state["app_view"] = "club"
            st.rerun()
        render_analysis_workspace(team, int(match_id))
    else:
        st.session_state["app_view"] = "portal"
        st.rerun()

    st.markdown('<div class="bottom-safe-space"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

