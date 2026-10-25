"""
client/fusion.py

NEW module — this is the project's research contribution.

Combines the FaceNet embedding and ArcFace embedding into a single
fused embedding using weighted feature-level concatenation.

Why feature-level (not score-level) fusion:
    Score-level fusion would require encrypting TWO embeddings and
    running TWO homomorphic similarity computations on the server
    (once per backbone), doubling CKKS cost and server complexity.
    Feature-level fusion produces ONE vector, so the rest of the
    pipeline (normalization -> CKKS encryption -> homomorphic match)
    stays exactly as it already works today.

Math:
    f_fn, f_af  : L2-normalized 512-d embeddings
    f_cat       = [ alpha * f_fn ; beta * f_af ]        (1024-d)
    f_fused     = f_cat / ||f_cat||_2

alpha/beta are fixed hyperparameters (default 0.5/0.5). They are NOT
learned — they can be tuned by a small grid search over a validation
set (see scripts/tune_fusion_weights.py) without training any model.
"""

import numpy as np

from common.config import FUSION
from common.utils import l2_normalize, get_logger

logger = get_logger(__name__)


class FeatureFusion:
    """
    Purpose:
        Fuse a FaceNet embedding and an ArcFace embedding into one
        1024-d embedding for encryption and matching.

    Input:
        facenet_embedding: np.ndarray, shape (512,), L2-normalized
        arcface_embedding: np.ndarray, shape (512,), L2-normalized

    Output:
        np.ndarray, shape (1024,), dtype float32, L2-normalized.
    """

    def __init__(self, alpha: float = FUSION.alpha, beta: float = FUSION.beta):
        self.alpha = alpha
        self.beta = beta

    def fuse(self, facenet_embedding: np.ndarray, arcface_embedding: np.ndarray) -> np.ndarray:
        if not np.isclose(self.alpha + self.beta, 1.0):
            raise ValueError("Fusion weights must sum to 1.")
        if facenet_embedding.shape != arcface_embedding.shape:
            raise ValueError("FaceNet and ArcFace embeddings must have matching shapes.")

        weighted = (
            self.alpha * l2_normalize(facenet_embedding)
            + self.beta * l2_normalize(arcface_embedding)
        ).astype(np.float32)
        fused = l2_normalize(weighted)

        if fused.shape[0] != FUSION.fused_dim:
            raise ValueError(
                f"Fused embedding dim {fused.shape[0]} != expected {FUSION.fused_dim}"
            )
        return fused

    @staticmethod
    def concatenate(
        facenet_embedding: np.ndarray,
        arcface_embedding: np.ndarray,
    ) -> np.ndarray:
        if facenet_embedding.shape != arcface_embedding.shape:
            raise ValueError("FaceNet and ArcFace embeddings must have matching shapes.")
        concatenated = np.concatenate((
            l2_normalize(facenet_embedding),
            l2_normalize(arcface_embedding),
        )).astype(np.float32)
        return l2_normalize(concatenated)
