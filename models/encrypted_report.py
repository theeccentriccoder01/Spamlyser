"""Encrypted multi-format report export for Spamlyser Pro.

Provides AES-256-GCM encryption wrappers around CSV, JSON, and PDF export
data so sensitive classification results can be shared securely.
"""

import io
import json
import logging
import os
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


def _derive_key(master_password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a 256-bit AES key from *master_password* using a random salt.

    Returns ``(key, salt)``.  The salt **must** be persisted alongside the
    ciphertext so the key can be re-derived during decryption.
    """
    import hashlib

    if salt is None:
        salt = os.urandom(16)
    raw = hashlib.pbkdf2_hmac(
        "sha256",
        master_password.encode("utf-8"),
        salt,
        iterations=600_000,
        dklen=32,
    )
    return raw, salt


def encrypt_bytes(plaintext: bytes, password: str) -> bytes:
    """Encrypt *plaintext* with AES-256-GCM and return the packed ciphertext.

    The returned byte-string has the layout::

        salt (16) || nonce (12) || ciphertext (variable) || tag (16)

    The tag is part of AESGCM output (appended to ciphertext).
    """
    key, salt = _derive_key(password)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, None)
    return salt + nonce + ct_and_tag


def decrypt_bytes(ciphertext: bytes, password: str) -> bytes:
    """Reverse :func:`encrypt_bytes`.

    Raises ``cryptography.exceptions.InvalidTag`` if the password is wrong
    or the data has been tampered with.
    """
    salt = ciphertext[:16]
    nonce = ciphertext[16:28]
    ct_and_tag = ciphertext[28:]
    key, _ = _derive_key(password, salt=salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct_and_tag, None)


class ReportEncryptor:
    """High-level helper for encrypting export payloads in Streamlit."""

    def __init__(self, password: str | None = None):
        self._password = password

    @property
    def is_ready(self) -> bool:
        return bool(self._password)

    def encrypt_csv(self, csv_text: str) -> bytes:
        return encrypt_bytes(csv_text.encode("utf-8"), self._password)

    def encrypt_json(self, json_text: str) -> bytes:
        return encrypt_bytes(json_text.encode("utf-8"), self._password)

    def encrypt_pdf(self, pdf_bytes: bytes) -> bytes:
        return encrypt_bytes(pdf_bytes, self._password)

    def export_encrypted(
        self,
        history: list[dict[str, Any]],
        fmt: str = "CSV",
    ) -> tuple[bytes, str]:
        """Produce an encrypted payload and a suitable file name.

        Returns ``(encrypted_bytes, suggested_filename)``.
        """
        import pandas as pd

        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(history)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(str)

        if fmt == "CSV":
            plain = df.to_csv(index=False).encode("utf-8")
            fname = f"spamlyser_encrypted_{ts}.csv.enc"
        elif fmt == "JSON":
            plain = json.dumps(
                history, indent=2, ensure_ascii=False, default=str
            ).encode("utf-8")
            fname = f"spamlyser_encrypted_{ts}.json.enc"
        else:
            from .export_feature import dataframe_to_pdf

            plain = dataframe_to_pdf(df).read()
            fname = f"spamlyser_encrypted_{ts}.pdf.enc"

        return self.encrypt_bytes(plain), fname

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """Low-level encrypt; delegates to :func:`encrypt_bytes`."""
        return encrypt_bytes(plaintext, self._password)
