from __future__ import annotations

import re


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def is_valid_email(email: str) -> bool:
    return EMAIL_PATTERN.match(normalize_email(email)) is not None


def validate_login_credentials(email: str, password: str) -> str | None:
    if not normalize_email(email) or not password:
        return "Introduce correo electronico y contrasena."
    if not is_valid_email(email):
        return "Introduce un correo electronico valido."
    return None


def validate_registration_credentials(email: str, password: str, password_confirm: str) -> str | None:
    base_error = validate_login_credentials(email, password)
    if base_error:
        return base_error
    if len(password) < 6:
        return "La contrasena debe tener al menos 6 caracteres."
    if password != password_confirm:
        return "Las contrasenas no coinciden."
    return None
