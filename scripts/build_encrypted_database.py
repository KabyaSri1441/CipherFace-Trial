from pathlib import Path
from PIL import Image

from client.pipeline import ClientPipeline
from database.database_manager import EncryptedDatabase


# LFW dataset location
LFW_DIR = Path(
    "data/lfw/lfw-deepfunneled/lfw-deepfunneled"
)

# Encrypted database location
DATABASE_DIR = Path(
    "database/encrypted_templates_v2"
)

# Number of people to enroll
NUM_PEOPLE = 10

# Minimum images required per person
IMAGES_PER_PERSON = 3


def main():

    # Make sure directories exist
    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not LFW_DIR.exists():
        raise FileNotFoundError(
            f"LFW dataset not found: {LFW_DIR}"
        )

    # -----------------------------------------
    # FIND PEOPLE WITH ENOUGH IMAGES
    # -----------------------------------------

    people = []

    for person_dir in sorted(LFW_DIR.iterdir()):

        if not person_dir.is_dir():
            continue

        images = sorted(
            person_dir.glob("*.jpg")
        )

        if len(images) >= IMAGES_PER_PERSON:

            people.append(
                (person_dir.name, images)
            )

        if len(people) >= NUM_PEOPLE:
            break

    print(
        f"Selected {len(people)} people."
    )

    if len(people) < NUM_PEOPLE:
        raise RuntimeError(
            "Not enough people with required images."
        )

    # -----------------------------------------
    # LOAD CLIENT PIPELINE
    # -----------------------------------------

    print(
        "\nLoading FaceNet + ArcFace + CKKS..."
    )

    client = ClientPipeline()

    # -----------------------------------------
    # LOAD DATABASE MANAGER
    # -----------------------------------------

    db = EncryptedDatabase(
        DATABASE_DIR
    )

    # -----------------------------------------
    # ENROLL PEOPLE
    # -----------------------------------------

    for person_name, images in people:

        print("\n" + "=" * 60)
        print(f"PERSON: {person_name}")
        print("=" * 60)

        # Use first image as enrollment image
        enrollment_image = images[0]

        print(
            f"Enrollment image: "
            f"{enrollment_image.name}"
        )

        # Open image
        image = Image.open(
            enrollment_image
        ).convert("RGB")

        # -----------------------------------------
        # FACE → EMBEDDING → ENCRYPTION
        # -----------------------------------------

        encrypted_embedding = (
            client.image_to_encrypted_embedding(
                image
            )
        )

        if encrypted_embedding is None:

            print(
                "WARNING: No face detected. "
                "Skipping."
            )

            continue

        # -----------------------------------------
        # STORE THROUGH DATABASE MANAGER
        # -----------------------------------------

        output_file = db.add_template(
            person_name,
            encrypted_embedding
        )

        print(
            "Encrypted template stored:"
        )

        print(
            f"  {output_file}"
        )

    # -----------------------------------------
    # DATABASE SUMMARY
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("DATABASE CREATION COMPLETE")
    print("=" * 60)

    users = db.list_users()

    print(
        f"Encrypted templates stored: "
        f"{len(users)}"
    )

    print("\nEnrolled users:")

    for user in users:
        print(
            f"  - {user}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()