"""
enrollment/enroll.py

Run this ONCE per user to register them.

Usage:
    python -m enrollment.enroll --user alice --image path/to/alice.jpg

Flow:
    1. Client pipeline turns the image into an encrypted embedding.
    2. Client's public CKKS context is written to disk (simulating
       "send once to server").
    3. Server pipeline stores the encrypted embedding as that user's
       enrollment template.

In a real deployment, steps 2-3 happen over HTTP/gRPC. Here they are
simulated with local files under keys/ and templates/ so the whole
project runs without standing up network infrastructure — swap
`LocalChannel` for an HTTP client/server later without touching the
ML code.
"""

import argparse
from PIL import Image

from client.pipeline import ClientPipeline
from server.pipeline import ServerService
from common.config import CKKS
from common.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Unique user id")
    parser.add_argument("--image", required=True, help="Path to enrollment photo")
    args = parser.parse_args()

    client = ClientPipeline()
    image = Image.open(args.image).convert("RGB")

    encrypted_bytes = client.image_to_encrypted_embedding(image)
    if encrypted_bytes is None:
        logger.error("No face detected in enrollment image. Aborting.")
        return

    # "Send" the public context to the server (simulated as a file read
    # from the path CKKSClient already wrote it to).
    with open(CKKS.context_path, "rb") as f:
        public_context_bytes = f.read()

    server = ServerService(public_context_bytes)
    server.enroll(args.user, encrypted_bytes)

    logger.info(f"User '{args.user}' enrolled successfully.")


if __name__ == "__main__":
    main()
