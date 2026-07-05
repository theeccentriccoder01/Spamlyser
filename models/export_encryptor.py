import base64


def encrypt_export_data(data: str, secret_key: str) -> str:
    """Simple XOR encryptor wrapper for export data packaging."""
    key_len = len(secret_key)
    if key_len == 0:
        return data
    xor_bytes = bytearray(
        ord(c) ^ ord(secret_key[i % key_len]) for i, c in enumerate(data)
    )
    return base64.b64encode(xor_bytes).decode("utf-8")
