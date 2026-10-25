from pathlib import Path
from PIL import Image
import tenseal as ts

from client.pipeline import ClientPipeline
from server.matcher import EncryptedMatcher
from common.config import VERIFICATION


DATABASE_DIR = Path("database/encrypted_templates_v2")
PUBLIC_CONTEXT = Path("keys/ckks_public_context.bin")

THRESHOLD = VERIFICATION.similarity_threshold


def get_second_image(person_dir):
    images = sorted(person_dir.glob("*.jpg"))

    if len(images) >= 2:
        return images[1]

    return None


def main():

    print("\n" + "=" * 60)
    print("       FACE VERIFICATION SYSTEM EVALUATION")
    print("=" * 60)

    # -----------------------------------------
    # LOAD CLIENT + SERVER
    # -----------------------------------------

    print("\nLoading client pipeline...")

    client = ClientPipeline()
    matcher = EncryptedMatcher()

    with open(PUBLIC_CONTEXT, "rb") as f:
        context = ts.context_from(f.read())

    # -----------------------------------------
    # GET DATABASE USERS
    # -----------------------------------------

    template_files = sorted(
        DATABASE_DIR.glob("*.bin")
    )

    print(
        f"Database templates: "
        f"{len(template_files)}"
    )

    # -----------------------------------------
    # TEST RESULTS
    # -----------------------------------------

    genuine_results = []
    impostor_results = []

    # Test only people who have multiple images
    test_people = []

    for template_file in template_files:

        person = template_file.stem

        person_dir = (
            Path(
                "data/lfw/lfw-deepfunneled/"
                "lfw-deepfunneled"
            )
            / person
        )

        probe = get_second_image(person_dir)

        if probe is not None:
            test_people.append(
                (person, template_file, probe)
            )

    print(
        f"People with a second test image: "
        f"{len(test_people)}"
    )

    # -----------------------------------------
    # GENUINE MATCH TESTS
    # -----------------------------------------

    print("\n" + "-" * 60)
    print("GENUINE MATCH TESTS")
    print("-" * 60)

    for person, template_file, probe in test_people:

        print(f"\nTesting: {person}")

        image = Image.open(probe).convert("RGB")

        encrypted_probe_bytes = (
            client.image_to_encrypted_embedding(
                image
            )
        )

        if encrypted_probe_bytes is None:
            print("No face detected. Skipping.")
            continue

        encrypted_probe = ts.ckks_vector_from(
            context,
            encrypted_probe_bytes
        )

        with open(template_file, "rb") as f:
            template_bytes = f.read()

        encrypted_template = ts.ckks_vector_from(
            context,
            template_bytes
        )

        encrypted_similarity = (
            matcher.compute_encrypted_similarity(
                encrypted_probe,
                encrypted_template
            )
        )

        similarity = client.decrypt_similarity(
            encrypted_similarity.serialize()
        )

        result = similarity >= THRESHOLD

        genuine_results.append(result)

        print(
            f"Similarity: {similarity:.4f} | "
            f"{'ACCEPT' if result else 'REJECT'}"
        )

    # -----------------------------------------
    # IMPOSTOR TESTS
    # -----------------------------------------

    print("\n" + "-" * 60)
    print("IMPOSTOR MATCH TESTS")
    print("-" * 60)

    for i in range(
        min(len(test_people), 10)
    ):

        person, _, probe = test_people[i]

        # Choose another person's template
        other_index = (
            (i + 1) % len(test_people)
        )

        other_person, other_template, _ = (
            test_people[other_index]
        )

        print(
            f"\nTesting: {person} "
            f"→ {other_person}"
        )

        image = Image.open(
            probe
        ).convert("RGB")

        encrypted_probe_bytes = (
            client.image_to_encrypted_embedding(
                image
            )
        )

        if encrypted_probe_bytes is None:
            print("No face detected. Skipping.")
            continue

        encrypted_probe = ts.ckks_vector_from(
            context,
            encrypted_probe_bytes
        )

        with open(
            other_template,
            "rb"
        ) as f:
            template_bytes = f.read()

        encrypted_template = ts.ckks_vector_from(
            context,
            template_bytes
        )

        encrypted_similarity = (
            matcher.compute_encrypted_similarity(
                encrypted_probe,
                encrypted_template
            )
        )

        similarity = client.decrypt_similarity(
            encrypted_similarity.serialize()
        )

        result = similarity >= THRESHOLD

        # For impostor test, REJECT is correct
        impostor_results.append(not result)

        print(
            f"Similarity: {similarity:.4f} | "
            f"{'ACCEPT' if result else 'REJECT'}"
        )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    genuine_correct = sum(
        genuine_results
    )

    impostor_correct = sum(
        impostor_results
    )

    total_tests = (
        len(genuine_results)
        + len(impostor_results)
    )

    total_correct = (
        genuine_correct
        + impostor_correct
    )

    accuracy = (
        total_correct / total_tests * 100
        if total_tests > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("             EVALUATION SUMMARY")
    print("=" * 60)

    print(
        f"Genuine tests:       "
        f"{len(genuine_results)}"
    )

    print(
        f"Genuine accepted:    "
        f"{genuine_correct}"
    )

    print(
        f"Impostor tests:      "
        f"{len(impostor_results)}"
    )

    print(
        f"Impostors rejected:  "
        f"{impostor_correct}"
    )

    print(
        f"Total tests:         "
        f"{total_tests}"
    )

    print(
        f"Correct decisions:   "
        f"{total_correct}"
    )

    print(
        f"Accuracy:            "
        f"{accuracy:.2f}%"
    )

    print(
        f"Threshold:           "
        f"{THRESHOLD}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()