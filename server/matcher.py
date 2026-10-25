"""
server/matcher.py

What the server computes: a homomorphic dot product between the
incoming encrypted probe embedding and a stored encrypted enrollment
template. Because both vectors are L2-normalized before encryption,
their dot product IS their cosine similarity — so this single
homomorphic multiply-and-sum gives the similarity score, still
encrypted, with no decryption anywhere on the server.
"""

import tenseal as ts

from common.utils import get_logger

logger = get_logger(__name__)


class EncryptedMatcher:
    """
    Purpose:
        Compute encrypted cosine similarity between a probe embedding
        and an enrolled template, both already CKKS-encrypted.

    Input:
        probe: ts.CKKSVector      (from client, this verification attempt)
        template: ts.CKKSVector   (stored at enrollment time, same user)

    Output:
        ts.CKKSVector containing a single encrypted scalar: the
        cosine similarity. Server cannot read this value.
    """

    @staticmethod
    def compute_encrypted_similarity(probe: ts.CKKSVector, template: ts.CKKSVector) -> ts.CKKSVector:
        # CKKSVector.dot() performs the homomorphic multiply + sum.
        return probe.dot(template)
