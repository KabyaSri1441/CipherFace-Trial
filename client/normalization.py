"""
client/normalization.py

Explicit "Embedding Normalization" pipeline stage (kept as its own
file to mirror the block diagram exactly). It is a thin wrapper
around common.utils.l2_normalize, applied here as the LAST step
before encryption — i.e. it re-normalizes the fused embedding
(fusion already normalizes internally, but this stage exists so the
pipeline stays correct even if fusion is bypassed for a single-model
ablation run).
"""

import numpy as np
from common.utils import l2_normalize


class EmbeddingNormalizer:
    """
    Purpose:
        Guarantee the vector handed to CKKS encryption is unit-norm,
        regardless of which upstream stage produced it.

    Input:  np.ndarray, shape (D,)
    Output: np.ndarray, shape (D,), unit L2 norm.
    """

    @staticmethod
    def normalize(embedding: np.ndarray) -> np.ndarray:
        return l2_normalize(embedding.astype(np.float32))
