"""
common/utils.py

Small stateless helper functions shared by client and server code.
Keeping these here avoids duplicating normalization/logging logic
in five different files.
"""

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with consistent formatting."""
    return logging.getLogger(name)


def l2_normalize(vector: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    L2-normalize a 1-D numpy vector.

    Input:  vector -> np.ndarray, shape (D,)
    Output: unit-norm np.ndarray, shape (D,)

    Used identically for FaceNet embeddings, ArcFace embeddings, and
    the fused embedding, so it lives in one place.
    """
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Plaintext cosine similarity, used only for local testing/debugging
    (e.g. sanity-checking that the encrypted result matches the
    plaintext result). Never used on the server in production, since
    the server must never see plaintext embeddings.
    """
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))
