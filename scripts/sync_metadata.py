from pathlib import Path

from database.database_manager import EncryptedDatabase
from database.metadata_manager import MetadataDatabase


def main():

    encrypted_db = EncryptedDatabase()
    metadata_db = MetadataDatabase()

    users = encrypted_db.list_users()

    print("\n" + "=" * 60)
    print("SYNCING ENCRYPTED DATABASE → METADATA DATABASE")
    print("=" * 60)

    for person_name in users:

        template_file = (
            encrypted_db.database_dir
            / f"{person_name}.bin"
        )

        metadata_db.add_user(
            person_name=person_name,
            template_file=str(template_file),
            status="ENCRYPTED"
        )

        print(f"Synced: {person_name}")

    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)

    print(
        f"Total users in metadata DB: "
        f"{metadata_db.count_users()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()