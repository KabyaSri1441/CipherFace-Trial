from pathlib import Path


class EncryptedDatabase:

    def __init__(self, database_dir="database/encrypted_templates_v2"):
        self.database_dir = Path(database_dir)
        self.database_dir.mkdir(parents=True, exist_ok=True)

    def add_template(self, person_name, encrypted_template):
        file_path = self.database_dir / f"{person_name}.bin"

        with open(file_path, "wb") as f:
            f.write(encrypted_template)

        return file_path

    def get_template(self, person_name):
        file_path = self.database_dir / f"{person_name}.bin"

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def list_users(self):
        return sorted(
            file.stem
            for file in self.database_dir.glob("*.bin")
        )

    def delete_user(self, person_name):
        file_path = self.database_dir / f"{person_name}.bin"

        if file_path.exists():
            file_path.unlink()
            return True

        return False

    def user_exists(self, person_name):
        return (
            self.database_dir / f"{person_name}.bin"
        ).exists()