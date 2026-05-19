from __future__ import annotations

import streamlit as st

try:
    import pyrebase
except ImportError:  # pragma: no cover - optional dependency in local development
    pyrebase = None

try:
    from google.cloud import firestore
except ImportError:  # pragma: no cover - optional dependency in local development
    firestore = None


def secret_value(*names: str) -> str:
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


def firebase_config() -> dict:
    return {
        "apiKey": secret_value("FIREBASE_API_KEY", "api_key"),
        "authDomain": secret_value("FIREBASE_AUTH_DOMAIN", "auth_domain"),
        "projectId": secret_value("FIREBASE_PROJECT_ID", "project_id"),
        "storageBucket": secret_value("FIREBASE_STORAGE_BUCKET", "storage_bucket"),
        "messagingSenderId": secret_value("FIREBASE_MESSAGING_SENDER_ID", "messaging_sender_id"),
        "appId": secret_value("FIREBASE_APP_ID", "app_id"),
    }


def firebase_enabled() -> bool:
    cfg = firebase_config()
    return bool(cfg["apiKey"] and cfg["projectId"])


@st.cache_resource(show_spinner=False)
def firebase_auth_client():
    if pyrebase is None or not firebase_enabled():
        return None
    cfg = {key: value for key, value in firebase_config().items() if value}
    try:
        return pyrebase.initialize_app(cfg).auth()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def firestore_client():
    if firestore is None:
        return None
    try:
        return firestore.Client(project=secret_value("FIREBASE_PROJECT_ID", "project_id"))
    except Exception:
        return None
