"""
client/verification.py

Accept/Reject decision. This runs on the CLIENT after it decrypts the
similarity score returned by the server, because the client is the
only party allowed to see plaintext similarity values in the
privacy-preserving design. See README "Trust model" section for the
simplified alternative used by many capstone demos.
"""

from common.config import VERIFICATION


class VerificationDecision:
    """
    Purpose:
        Apply the accept/reject threshold to a decrypted similarity
        score.

    Input:  similarity -> float in [-1, 1]
    Output: dict {"similarity": float, "decision": "ACCEPT"|"REJECT"}
    """

    @staticmethod
    def decide(similarity: float) -> dict:
        decision = "ACCEPT" if similarity >= VERIFICATION.similarity_threshold else "REJECT"
        return {"similarity": similarity, "decision": decision}
