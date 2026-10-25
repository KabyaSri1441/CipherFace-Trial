from pathlib import Path
from PIL import Image
import tenseal as ts

from client.pipeline import ClientPipeline
from server.matcher import EncryptedMatcher
from common.config import VERIFICATION, CKKS


DATABASE_DIR = Path("database/encrypted_templates_v2")
PUBLIC_CONTEXT = Path(CKKS.context_path)

THRESHOLD = VERIFICATION.similarity_threshold


def main():

    # -----------------------------
    # GET PROBE IMAGE
    # -----------------------------
    probe_path = input("Enter probe image path: ").strip()
    probe = Path(probe_path)

    if not probe.exists():
        raise FileNotFoundError(f"Probe image not found: {probe}")

    print("\nLoading client pipeline...")
    client = ClientPipeline()

    # -----------------------------
    # PROCESS PROBE
    # -----------------------------
    print("Processing probe image...")

    image = Image.open(probe).convert("RGB")

    encrypted_probe_bytes = client.image_to_encrypted_embedding(image)

    if encrypted_probe_bytes is None:
        print("No face detected.")
        return

    print("Probe encrypted successfully.")

    # -----------------------------
    # SERVER SIDE
    # -----------------------------
    print("Loading public CKKS context...")

    with open(PUBLIC_CONTEXT, "rb") as f:
        context = ts.context_from(f.read())

    encrypted_probe = ts.ckks_vector_from(
        context,
        encrypted_probe_bytes
    )

    matcher = EncryptedMatcher()

    results = []

    template_files = sorted(DATABASE_DIR.glob("*.bin"))

    print(f"\nSearching {len(template_files)} encrypted templates...\n")

    for template_path in template_files:

        person = template_path.stem

        print(f"Matching against: {person}")

        with open(template_path, "rb") as f:
            template_bytes = f.read()

        encrypted_template = ts.ckks_vector_from(
            context,
            template_bytes
        )

        encrypted_similarity = matcher.compute_encrypted_similarity(
            encrypted_probe,
            encrypted_template
        )

        # Send encrypted result back to client for decryption
        similarity = client.decrypt_similarity(
            encrypted_similarity.serialize()
        )

        results.append((person, similarity))

    # -----------------------------
    # SORT RESULTS
    # -----------------------------
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    print("\n" + "=" * 60)
    print("MATCHING RESULTS")
    print("=" * 60)

    for person, similarity in results:
        print(f"{person:<40} {similarity:.4f}")

    # -----------------------------
    # BEST MATCH
    # -----------------------------
    best_person, best_similarity = results[0]

    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)

    print(f"Best match: {best_person}")
    print(f"Similarity: {best_similarity:.4f}")
    print(f"Threshold:  {THRESHOLD:.2f}")

    if best_similarity >= THRESHOLD:
        print("Result:     ACCEPT")
    else:
        print("Result:     REJECT")

    print("=" * 60)


if __name__ == "__main__":
    main()