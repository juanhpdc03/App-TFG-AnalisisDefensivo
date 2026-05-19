from __future__ import annotations

from datetime import datetime, timezone

import requests

from tfg_analysis.firebase_config import admin_emails, firebase_auth_client, firebase_config, firebase_enabled

USER_ROLES = ("invitado", "analista", "admin")
USERS_COLLECTION = "usuarios"
LOGS_COLLECTION = "logs"
TEAMS_COLLECTION = "equipos"


class FirebaseAuthError(RuntimeError):
    pass


def _auth_url(action: str) -> str:
    return f"https://identitytoolkit.googleapis.com/v1/accounts:{action}?key={firebase_config()['apiKey']}"


def _firestore_url(path: str) -> str:
    project_id = firebase_config()["projectId"]
    clean_path = path.strip("/")
    return f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{clean_path}"


def firebase_error(response: requests.Response) -> str:
    try:
        message = response.json().get("error", {}).get("message", "")
    except ValueError:
        message = response.text
    friendly = {
        "EMAIL_EXISTS": "Ese correo ya esta registrado.",
        "EMAIL_NOT_FOUND": "No existe ninguna cuenta con ese correo.",
        "INVALID_LOGIN_CREDENTIALS": "Correo electronico o contrasena incorrectos.",
        "INVALID_PASSWORD": "Correo electronico o contrasena incorrectos.",
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


def normalize_role(role: str | None) -> str:
    clean_role = str(role or "invitado").strip().lower()
    if clean_role == "guest":
        return "invitado"
    return clean_role if clean_role in USER_ROLES else "invitado"


def normalize_profile(profile: dict) -> dict:
    out = dict(profile)
    out["rol"] = normalize_role(out.get("rol"))
    if out["rol"] != "analista":
        out.pop("equipo", None)
    return out


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


def _firestore_headers(id_token: str) -> dict:
    return {"Authorization": f"Bearer {id_token}", "Content-Type": "application/json"}


def _firebase_auth_rest(action: str, email: str, password: str) -> dict:
    response = requests.post(
        _auth_url(action),
        json={"email": email, "password": password, "returnSecureToken": True},
        timeout=15,
    )
    if not response.ok:
        raise FirebaseAuthError(firebase_error(response))
    return response.json()


def _normalize_pyrebase_user(user: dict) -> dict:
    return {
        "email": user.get("email", ""),
        "idToken": user.get("idToken", ""),
        "localId": user.get("localId") or user.get("localId".lower(), ""),
        "refreshToken": user.get("refreshToken", ""),
    }


def firebase_sign_in(email: str, password: str) -> dict:
    auth = firebase_auth_client()
    if auth is not None:
        try:
            return _normalize_pyrebase_user(auth.sign_in_with_email_and_password(email, password))
        except Exception as exc:
            raise FirebaseAuthError("Correo electronico o contrasena incorrectos.") from exc
    return _firebase_auth_rest("signInWithPassword", email, password)


def firebase_sign_up(email: str, password: str) -> dict:
    auth = firebase_auth_client()
    if auth is not None:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            return _normalize_pyrebase_user(user)
        except Exception as exc:
            raise FirebaseAuthError("No se ha podido registrar el correo en Firebase.") from exc
    return _firebase_auth_rest("signUp", email, password)


def save_user_profile(uid: str, id_token: str, profile: dict):
    response = requests.patch(
        _firestore_url(f"{USERS_COLLECTION}/{uid}"),
        headers=_firestore_headers(id_token),
        json={"fields": _firestore_fields(profile)},
        timeout=15,
    )
    if not response.ok:
        raise FirebaseAuthError(firebase_error(response))


def log_action(id_token: str, email: str, action: str, role: str = "", detail: str = "") -> bool:
    if not firebase_enabled() or not id_token:
        return False
    payload = {
        "usuario": str(email or "").strip().lower(),
        "accion": str(action),
        "rol": normalize_role(role) if role else "",
        "detalle": str(detail),
        "fecha": datetime.now(timezone.utc).isoformat(),
    }
    response = requests.post(
        _firestore_url(LOGS_COLLECTION),
        headers=_firestore_headers(id_token),
        json={"fields": _firestore_fields(payload)},
        timeout=10,
    )
    return response.ok


def default_profile_for_email(email: str) -> dict:
    clean_email = email.strip().lower()
    role = "admin" if clean_email in admin_emails() else "invitado"
    return {"email": clean_email, "rol": role}


def load_user_profile(uid: str, id_token: str, email: str) -> dict:
    response = requests.get(
        _firestore_url(f"{USERS_COLLECTION}/{uid}"),
        headers=_firestore_headers(id_token),
        timeout=15,
    )
    if response.ok:
        profile = _parse_firestore_doc(response.json())
        if profile.get("email"):
            return normalize_profile(profile)
    if response.status_code not in (403, 404):
        raise FirebaseAuthError(firebase_error(response))

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
                        save_user_profile(
                            uid,
                            id_token,
                            normalize_profile({k: v for k, v in profile.items() if not k.startswith("_")}),
                        )
                    except FirebaseAuthError:
                        pass
                return normalize_profile(profile)
    profile = default_profile_for_email(email)
    try:
        save_user_profile(uid, id_token, profile)
    except FirebaseAuthError as exc:
        raise FirebaseAuthError(
            "La cuenta existe en Authentication, pero no se ha podido crear su perfil en Firestore. "
            "Revisa que las reglas permitan crear usuarios/{uid} al propio usuario autenticado."
        ) from exc
    return profile


def register_guest(email: str, password: str) -> tuple[dict, dict]:
    if not firebase_enabled():
        raise FirebaseAuthError("Firebase no esta configurado.")
    auth_data = firebase_sign_up(email, password)
    profile = default_profile_for_email(email)
    if profile["rol"] != "admin":
        profile["rol"] = "invitado"
    save_user_profile(str(auth_data["localId"]), str(auth_data["idToken"]), profile)
    log_action(str(auth_data["idToken"]), email, "registro", profile.get("rol", ""), "Alta inicial en la plataforma")
    return auth_data, profile


def login(email: str, password: str) -> tuple[dict, dict]:
    if not firebase_enabled():
        raise FirebaseAuthError("Firebase no esta configurado.")
    auth_data = firebase_sign_in(email, password)
    profile = load_user_profile(str(auth_data["localId"]), str(auth_data["idToken"]), email)
    log_action(str(auth_data["idToken"]), email, "login", profile.get("rol", ""), "Inicio de sesion correcto")
    return auth_data, profile


def list_users(id_token: str) -> list[dict]:
    if not firebase_enabled() or not id_token:
        return []
    response = requests.get(
        _firestore_url(USERS_COLLECTION),
        headers=_firestore_headers(id_token),
        timeout=15,
    )
    if response.status_code == 404:
        return []
    if not response.ok:
        raise FirebaseAuthError(firebase_error(response))
    users = [_parse_firestore_doc(doc) for doc in response.json().get("documents", [])]
    return sorted(users, key=lambda item: str(item.get("email", "")).lower())


def list_teams(id_token: str) -> list[dict]:
    if not firebase_enabled() or not id_token:
        return []
    response = requests.get(
        _firestore_url(TEAMS_COLLECTION),
        headers=_firestore_headers(id_token),
        timeout=15,
    )
    if response.status_code == 404:
        return []
    if not response.ok:
        return []
    teams = [_parse_firestore_doc(doc) for doc in response.json().get("documents", [])]
    return sorted(teams, key=lambda item: str(item.get("nombre", item.get("name", ""))).lower())


def sync_teams(id_token: str, teams: list[dict]) -> bool:
    if not firebase_enabled() or not id_token:
        return False
    ok = True
    for team in teams:
        try:
            team_id = int(team.get("team_id"))
        except (TypeError, ValueError):
            continue
        name = str(team.get("name") or team.get("short_name") or f"Equipo {team_id}").strip()
        payload = {
            "team_id": team_id,
            "nombre": name,
            "activo": True,
        }
        response = requests.patch(
            _firestore_url(f"{TEAMS_COLLECTION}/{team_id}"),
            headers=_firestore_headers(id_token),
            json={"fields": _firestore_fields(payload)},
            timeout=15,
        )
        ok = ok and response.ok
    return ok


def update_user(id_token: str, doc_id: str, email: str, role: str, team: str):
    if not firebase_enabled() or not id_token:
        raise FirebaseAuthError("Conecta Firebase para editar usuarios desde la app.")
    clean_role = normalize_role(role)
    payload = {"email": email, "rol": clean_role}
    if clean_role == "analista":
        payload["equipo"] = team
    response = requests.patch(
        _firestore_url(f"{USERS_COLLECTION}/{doc_id}"),
        headers=_firestore_headers(id_token),
        json={"fields": _firestore_fields(payload)},
        params=[
            ("updateMask.fieldPaths", "email"),
            ("updateMask.fieldPaths", "rol"),
            ("updateMask.fieldPaths", "equipo"),
        ],
        timeout=15,
    )
    if not response.ok:
        raise FirebaseAuthError(firebase_error(response))
    log_action(id_token, email, "actualizar_usuario", clean_role, f"Rol={clean_role}; equipo={team if clean_role == 'analista' else ''}")
