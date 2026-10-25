from database.database_manager import EncryptedDatabase


def main():
    db = EncryptedDatabase()

    users = db.list_users()

    print("\n" + "=" * 50)
    print("ENCRYPTED DATABASE")
    print("=" * 50)

    print(f"Total enrolled users: {len(users)}\n")

    for i, user in enumerate(users, 1):
        print(f"{i}. {user}")

    print("=" * 50)


if __name__ == "__main__":
    main()