"""Password hashing and account configuration helpers for API authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from typing import Any

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32


def hash_password(password: str) -> str:
    """Create a versioned scrypt password hash suitable for an env secret."""
    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("Password must be a string of at least 12 characters.")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_BYTES,
    )
    encode = lambda value: base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    return f"scrypt-v1${encode(salt)}${encode(digest)}"


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify only the supported scrypt encoding; reject malformed hashes safely."""
    if not isinstance(password, str) or not isinstance(encoded_hash, str):
        return False
    try:
        scheme, salt_part, digest_part = encoded_hash.split("$", 2)
        if scheme != "scrypt-v1":
            return False

        def decode(value: str) -> bytes:
            return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

        salt = decode(salt_part)
        expected = decode(digest_part)
        if len(salt) != _SALT_BYTES or len(expected) != _KEY_BYTES:
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=_KEY_BYTES,
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeError):
        return False


def load_users(raw_json: str | None) -> dict[str, dict[str, Any]]:
    """Parse and validate the configured JSON account registry."""
    if not raw_json or not raw_json.strip():
        return {}
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise ValueError("URBANGUARD_USERS_JSON must contain valid JSON.") from error
    if not isinstance(payload, list):
        raise ValueError("URBANGUARD_USERS_JSON must be a JSON array of user objects.")

    users: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each configured user must be a JSON object.")
        username = item.get("username")
        password_hash = item.get("password_hash")
        role = item.get("role")
        if not isinstance(username, str) or not username.strip() or len(username) > 64:
            raise ValueError("Each configured user needs a username of 1–64 characters.")
        username = username.strip()
        if username in users:
            raise ValueError(f"Duplicate configured username: {username}.")
        if not _is_scrypt_hash(password_hash):
            raise ValueError(f"User {username!r} needs a valid scrypt-v1 password_hash.")
        if not isinstance(role, str) or role not in {"admin", "surveyor"}:
            raise ValueError(f"User {username!r} role must be 'admin' or 'surveyor'.")

        area_bbox = item.get("area_bbox")
        if role == "surveyor":
            if (
                not isinstance(area_bbox, list)
                or len(area_bbox) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in area_bbox)
            ):
                raise ValueError(f"Surveyor {username!r} needs area_bbox [min_lon,min_lat,max_lon,max_lat].")
            min_lon, min_lat, max_lon, max_lat = map(float, area_bbox)
            if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
                raise ValueError(f"Surveyor {username!r} has an invalid area_bbox.")
            area_bbox = [min_lon, min_lat, max_lon, max_lat]
        elif area_bbox is not None:
            raise ValueError(f"Admin {username!r} must use area_bbox=null.")

        users[username] = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "display_name": str(item.get("display_name") or username)[:100],
            "area_name": str(item.get("area_name") or ("All Hyderabad" if role == "admin" else "Assigned area"))[:120],
            "area_bbox": area_bbox,
        }
    return users


def _is_scrypt_hash(encoded_hash: Any) -> bool:
    if not isinstance(encoded_hash, str):
        return False
    try:
        scheme, salt_part, digest_part = encoded_hash.split("$", 2)
        if scheme != "scrypt-v1":
            return False
        decode = lambda value: base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        return len(decode(salt_part)) == _SALT_BYTES and len(decode(digest_part)) == _KEY_BYTES
    except (ValueError, TypeError):
        return False


def main() -> None:
    """Prompt for a password and print its hash without echoing the password."""
    from getpass import getpass

    password = getpass("Password (at least 12 characters): ")
    confirmation = getpass("Confirm password: ")
    if not hmac.compare_digest(password, confirmation):
        raise SystemExit("Passwords did not match.")
    try:
        print(hash_password(password))
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
