"""Initialize client-owned CKKS keys and export a public-only server context."""

from pathlib import Path

import tenseal as ts
import numpy as np

from client.encryption.ckks_client import CKKSClient
from common.config import CKKS


def main() -> None:
    client = CKKSClient()
    public_bytes = client.export_public_context_bytes()
    public_context = ts.context_from(public_bytes)
    if public_context.has_secret_key():
        raise RuntimeError("Refusing to write a server context that contains a secret key.")

    public_path = Path(CKKS.context_path)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    if public_path.exists():
        existing = ts.context_from(public_path.read_bytes())
        if existing.has_secret_key():
            raise RuntimeError("Existing server context unexpectedly contains a secret key.")
        sample = np.asarray([1.0, 0.25, -0.5], dtype=np.float64)
        encrypted = client.encrypt(sample).serialize()
        server_vector = ts.ckks_vector_from(existing, encrypted)
        encrypted_dot = server_vector.dot(server_vector).serialize()
        decrypted_dot = ts.ckks_vector_from(client._context, encrypted_dot).decrypt()[0]
        if abs(decrypted_dot - float(np.dot(sample, sample))) > 1e-4:
            raise RuntimeError("Existing server context does not match the client keypair.")
    else:
        with public_path.open("xb") as target:
            target.write(public_bytes)

    print("Client CKKS context is available in user-local key storage.")
    print("Server context is public-only; no key material was printed.")


if __name__ == "__main__":
    main()
