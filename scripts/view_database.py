from database.metadata_manager import MetadataDatabase


def main():

    db = MetadataDatabase()

    users = db.list_users()

    print("\n" + "=" * 80)
    print("                 SERVER DATABASE")
    print("=" * 80)

    print(f"\nTotal enrolled users: {len(users)}\n")

    print(
        f"{'ID':<5}"
        f"{'PERSON':<40}"
        f"{'STATUS':<15}"
        f"TEMPLATE"
    )

    print("-" * 80)

    for user_id, person, template, status, created_at in users:

        print(
            f"{user_id:<5}"
            f"{person:<40}"
            f"{status:<15}"
            f"{template}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()