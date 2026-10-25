"""Copy the existing CKKS keypair into separated client/server locations."""

import filecmp
import os
import shutil
from pathlib import Path

import numpy as np
import tenseal as ts

from common.config import CKKS
from client.encryption.ckks_client import CKKSClient


LEGACY_SECRET = Path("keys/ckks_secret_context.bin")
LEGACY_PUBLIC = Path("keys/ckks_public_context.bin")


def _copy_once(source: Path, destination: Path, mode: int | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required key artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not filecmp.cmp(source, destination, shallow=False):
            raise FileExistsError(f"Refusing to overwrite a different key artifact: {destination}")
    else:
        shutil.copyfile(source, destination)
    if mode is not None:
        os.chmod(destination, mode)


def _verify_public_context(client: CKKSClient, public_bytes: bytes) -> None:
    public_context = ts.context_from(public_bytes)
    if public_context.has_secret_key():
        raise RuntimeError("Refusing to use a context that contains a secret key.")
    sample = np.asarray([1.0, 0.25, -0.5], dtype=np.float64)
    encrypted = client.encrypt(sample).serialize()
    server_vector = ts.ckks_vector_from(public_context, encrypted)
    encrypted_dot = server_vector.dot(server_vector).serialize()
    decrypted_dot = ts.ckks_vector_from(client._context, encrypted_dot).decrypt()[0]
    if abs(decrypted_dot - float(np.dot(sample, sample))) > 1e-4:
        raise RuntimeError("Public context is not compatible with the client secret context.")


def main() -> None:
    secret_destination = Path(CKKS.secret_context_path).expanduser()
    public_destination = Path(CKKS.context_path)
    if LEGACY_SECRET.is_file():
        _copy_once(LEGACY_SECRET, secret_destination, mode=0o600)
    elif not secret_destination.is_file():
        raise FileNotFoundError("Neither the legacy nor client-local secret context exists.")
    else:
        os.chmod(secret_destination, 0o600)

    client = CKKSClient()
    public_bytes = client.export_public_context_bytes()
    _verify_public_context(client, public_bytes)

    public_destination.parent.mkdir(parents=True, exist_ok=True)
    if public_destination.exists():
        existing = public_destination.read_bytes()
        _verify_public_context(client, existing)
    else:
        with public_destination.open("xb") as target:
            target.write(public_bytes)
        os.chmod(public_destination, 0o644)

    print("Existing client secret context is stored outside the repository.")
    print("Server context verified public-only and stored under server/config/.")
    print("Original key artifacts were retained as backups; no key bytes were printed.")


if __name__ == "__main__":
    main()
