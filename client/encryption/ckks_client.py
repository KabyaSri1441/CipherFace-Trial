"""Client-owned CKKS encryption and decryption operations."""

import os
from pathlib import Path

import numpy as np
import tenseal as ts

from common.config import CKKS
from common.utils import get_logger

logger = get_logger(__name__)


class CKKSClient:
    """Own the secret context; export only a compatible public context."""

    def __init__(self):
        secret_path = Path(CKKS.secret_context_path).expanduser().resolve()
        workspace = Path.cwd().resolve()
        if secret_path.is_relative_to(workspace):
            raise RuntimeError("Refusing to load a CKKS secret context from the project workspace.")

        secret_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(secret_path.parent, 0o700)
        if secret_path.exists():
            os.chmod(secret_path, 0o600)
            self._context = self._load_context(str(secret_path))
            if not self._context.has_secret_key():
                raise RuntimeError("Client context does not contain the required CKKS secret key.")
            return

        context = self._create_context()
        public_path = Path(CKKS.context_path)
        if public_path.exists():
            public_context = ts.context_from(public_path.read_bytes())
            if public_context.has_secret_key() or not self._contexts_compatible(context, public_context):
                raise RuntimeError(
                    "Existing server public context does not match the new client key; refusing to replace it."
                )
        else:
            public_path.parent.mkdir(parents=True, exist_ok=True)
            with public_path.open("xb") as target:
                target.write(context.serialize(save_secret_key=False))
            os.chmod(public_path, 0o644)

        self._context = context
        self._save_context(self._context, str(secret_path), save_secret_key=True)
        os.chmod(secret_path, 0o600)
        logger.info("Generated new CKKS keypair.")

    @staticmethod
    def _contexts_compatible(secret_context: ts.Context, public_context: ts.Context) -> bool:
        try:
            sample = np.asarray([1.0, 0.25, -0.5], dtype=np.float64)
            encrypted = ts.ckks_vector(secret_context, sample.tolist()).serialize()
            server_vector = ts.ckks_vector_from(public_context, encrypted)
            encrypted_dot = server_vector.dot(server_vector).serialize()
            decrypted = ts.ckks_vector_from(secret_context, encrypted_dot).decrypt()[0]
            return abs(decrypted - float(np.dot(sample, sample))) < 1e-4
        except (ValueError, RuntimeError, TypeError):
            return False

    @staticmethod
    def _create_context() -> ts.Context:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=CKKS.poly_modulus_degree,
            coeff_mod_bit_sizes=list(CKKS.coeff_mod_bit_sizes),
        )
        context.generate_galois_keys()
        context.global_scale = 2 ** CKKS.global_scale_bits
        return context

    @staticmethod
    def _save_context(context: ts.Context, path: str, save_secret_key: bool) -> None:
        with open(path, "xb") as target:
            target.write(context.serialize(save_secret_key=save_secret_key))

    @staticmethod
    def _load_context(path: str) -> ts.Context:
        with open(path, "rb") as source:
            return ts.context_from(source.read())

    def encrypt(self, embedding: np.ndarray) -> ts.CKKSVector:
        """Encrypt a 1D float embedding into a CKKS ciphertext vector."""
        return ts.ckks_vector(self._context, embedding.tolist())

    def decrypt_scalar(self, encrypted_result: ts.CKKSVector) -> float:
        """Decrypt the single similarity score returned by the server."""
        return encrypted_result.decrypt()[0]

    def export_public_context_bytes(self) -> bytes:
        """Bytes to transmit to the server (no secret key included)."""
        serialized = self._context.serialize(save_secret_key=False)
        public_context = ts.context_from(serialized)
        if public_context.has_secret_key():
            raise RuntimeError("Refusing to export a server context containing the secret key.")
        return serialized
