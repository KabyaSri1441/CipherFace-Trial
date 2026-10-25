"""
server/encryption/ckks_server.py

The server loads ONLY the public CKKS context (no secret key), so it
is mathematically incapable of decrypting anything. It can perform
homomorphic operations (addition, multiplication, dot product) on
ciphertexts, but the result stays encrypted and is sent back to the
client for decryption.
"""

import tenseal as ts

from common.utils import get_logger

logger = get_logger(__name__)


class CKKSServerContext:
    """
    Purpose:
        Hold the public CKKS context received from the client, and
        deserialize incoming encrypted embeddings/templates.
    """

    def __init__(self, public_context_bytes: bytes):
        self._context = ts.context_from(public_context_bytes)

    def load_encrypted_vector(self, serialized_vector: bytes) -> ts.CKKSVector:
        return ts.ckks_vector_from(self._context, serialized_vector)

    @property
    def context(self) -> ts.Context:
        return self._context
