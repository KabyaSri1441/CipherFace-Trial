"""
scripts/verify_demo.py

Run this to verify a probe image against an already-enrolled user.

Usage:
    python -m scripts.verify_demo --user alice --image path/to/probe.jpg

Flow (mirrors the client-server diagram exactly, just in one process
for demo simplicity — client and server are still separate objects
that only exchange bytes, so splitting them into two real processes
later is a networking change, not an ML change):

CLIENT: capture -> detect -> align -> facenet -> arcface -> fuse
        -> normalize -> encrypt -> send ciphertext
SERVER: receive -> load template -> homomorphic dot product
        -> send encrypted similarity back
CLIENT: decrypt -> threshold -> ACCEPT/REJECT
"""

import argparse
from PIL import Image

from client.pipeline import ClientPipeline
from client.verification import VerificationDecision
from server.pipeline import ServerService
from common.config import CKKS
from common.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Claimed user id")
    parser.add_argument("--image", required=True, help="Path to probe photo")
    args = parser.parse_args()

    client = ClientPipeline()
    image = Image.open(args.image).convert("RGB")

    encrypted_probe = client.image_to_encrypted_embedding(image)
    if encrypted_probe is None:
        logger.error("No face detected in probe image. Aborting.")
        return

    with open(CKKS.context_path, "rb") as f:
        public_context_bytes = f.read()

    server = ServerService(public_context_bytes)
    encrypted_result = server.verify(args.user, encrypted_probe)

    similarity = client.decrypt_similarity(encrypted_result)
    result = VerificationDecision.decide(similarity)

    logger.info(f"Similarity: {result['similarity']:.4f} -> {result['decision']}")


if __name__ == "__main__":
    main()
