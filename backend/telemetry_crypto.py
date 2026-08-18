"""Seal/unseal for client-buffered challenge telemetry."""

from __future__ import annotations

import base64
import json
import secrets
from typing import Any, Dict, List

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes, hmac

MAX_SEALED_EVENTS = 50_000
MAX_PLAINTEXT_BYTES = 8_000_000


def new_telemetry_key() -> bytes:
    return secrets.token_bytes(32)


def telemetry_key_to_b64(key: bytes) -> str:
    return base64.urlsafe_b64encode(key).decode("ascii").rstrip("=")


def telemetry_key_from_b64(raw: str) -> bytes:
    return _b64decode(raw)


def _b64decode(raw: str) -> bytes:
    text = (raw or "").strip()
    pad = "=" * ((4 - len(text) % 4) % 4)
    text = text.replace("-", "+").replace("_", "/") + pad
    return base64.b64decode(text)


def _decrypt_ctr_hmac(key: bytes, iv: bytes, ciphertext: bytes, mac: bytes) -> bytes:
    if len(iv) != 16 or len(mac) != 32:
        raise ValueError("Invalid sealed telemetry nonce or mac.")
    check = hmac.HMAC(key, hashes.SHA256())
    check.update(iv)
    check.update(ciphertext)
    try:
        check.verify(mac)
    except Exception as exc:
        raise ValueError("Could not decrypt local telemetry.") from exc
    decryptor = Cipher(algorithms.AES(key), modes.CTR(iv)).decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def unseal_events(key: bytes, sealed: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(sealed, dict):
        raise ValueError("Sealed telemetry must be an object.")
    alg = str(sealed.get("alg") or "AES-GCM").upper().replace("_", "-")
    try:
        iv = _b64decode(str(sealed.get("iv") or ""))
        ciphertext = _b64decode(str(sealed.get("ciphertext") or ""))
    except Exception as exc:
        raise ValueError("Invalid sealed telemetry encoding.") from exc
    if not ciphertext:
        raise ValueError("Empty sealed telemetry.")

    if alg in ("AES-GCM", "AESGCM"):
        if len(iv) not in (12, 16):
            raise ValueError("Invalid sealed telemetry nonce.")
        try:
            plaintext = AESGCM(key).decrypt(iv, ciphertext, None)
        except Exception as exc:
            raise ValueError("Could not decrypt local telemetry.") from exc
    elif alg in ("AES-256-CTR-HMAC", "AES256CTRHMAC"):
        try:
            mac = _b64decode(str(sealed.get("mac") or ""))
        except Exception as exc:
            raise ValueError("Invalid sealed telemetry encoding.") from exc
        plaintext = _decrypt_ctr_hmac(key, iv, ciphertext, mac)
    else:
        raise ValueError("Unsupported telemetry cipher.")

    if len(plaintext) > MAX_PLAINTEXT_BYTES:
        raise ValueError("Sealed telemetry is too large.")
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Decrypted telemetry is not valid JSON.") from exc
    events = payload.get("events") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("Decrypted telemetry has no events list.")
    if len(events) > MAX_SEALED_EVENTS:
        raise ValueError("Sealed telemetry has too many events.")
    return events
