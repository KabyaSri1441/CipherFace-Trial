from pathlib import Path
from PIL import Image

from client.pipeline import ClientPipeline
from database.database_manager import EncryptedDatabase
from database.metadata_manager import MetadataDatabase


def main():

    print("\n" + "=" * 60)
    print("       PRIVACY-PRESERVING FACE ENROLLMENT")
    print("=" * 60)

    # -----------------------------------------
    # USER INPUT
    # -----------------------------------------

    person_name = input(
        "\nEnter person name: "
    ).strip()

    if not person_name:
        print("ERROR: Person name cannot be empty.")
        return

    image_path = input(
        "Enter image path: "
    ).strip()

    image = Path(image_path)

    if not image.exists():
        print(f"ERROR: Image not found: {image}")
        return

    # -----------------------------------------
    # DATABASES
    # -----------------------------------------

    encrypted_db = EncryptedDatabase()
    metadata_db = MetadataDatabase()

    if encrypted_db.user_exists(person_name):

        choice = input(
            f"\n{person_name} already exists. "
            "Overwrite? (y/n): "
        ).strip().lower()

        if choice != "y":
            print("Enrollment cancelled.")
            return

    # -----------------------------------------
    # LOAD PIPELINE
    # -----------------------------------------

    print("\nLoading FaceNet + ArcFace + CKKS...")

    client = ClientPipeline()

    # -----------------------------------------
    # PROCESS IMAGE
    # -----------------------------------------

    print("Processing face image...")

    try:
        img = Image.open(image).convert("RGB")
    except Exception as e:
        print(f"ERROR: Could not open image: {e}")
        return

    encrypted_embedding = (
        client.image_to_encrypted_embedding(img)
    )

    if encrypted_embedding is None:
        print("\nERROR: No face detected.")
        return

    print(
        "Face processed and embedding encrypted."
    )

    # -----------------------------------------
    # STORE ENCRYPTED TEMPLATE
    # -----------------------------------------

    output_file = encrypted_db.add_template(
        person_name,
        encrypted_embedding
    )

    # -----------------------------------------
    # STORE METADATA
    # -----------------------------------------

    metadata_db.add_user(
        person_name=person_name,
        template_file=str(output_file),
        status="ENCRYPTED"
    )

    # -----------------------------------------
    # SUCCESS
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("ENROLLMENT SUCCESSFUL")
    print("=" * 60)

    print(f"Person:       {person_name}")
    print(f"Template:     {output_file}")
    print("Status:       ENCRYPTED")
    print("Metadata DB:  database/metadata.db")
    print("=" * 60)

    print(
        f"\nTotal enrolled users: "
        f"{metadata_db.count_users()}"
    )


if __name__ == "__main__":
    main()