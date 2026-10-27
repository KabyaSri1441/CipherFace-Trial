import sqlite3
from pathlib import Path


class MetadataDatabase:

    def __init__(self, db_path="database/metadata_v2.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        self._create_table()
        self._create_verification_logs_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT UNIQUE NOT NULL,
                    template_file TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def add_user(
        self,
        person_name,
        template_file,
        status="ENCRYPTED"
    ):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO users
                (person_name, template_file, status)
                VALUES (?, ?, ?)
            """, (
                person_name,
                template_file,
                status
            ))

            conn.commit()

    def get_user(self, person_name):
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT id, person_name,
                       template_file, status,
                       created_at
                FROM users
                WHERE person_name = ?
            """, (person_name,))

            return cursor.fetchone()

    def list_users(self):
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT id, person_name,
                       template_file, status,
                       created_at
                FROM users
                ORDER BY id
            """)

            return cursor.fetchall()

    def delete_user(self, person_name):
        with self._connect() as conn:
            cursor = conn.execute("""
                DELETE FROM users
                WHERE person_name = ?
            """, (person_name,))

            conn.commit()

            return cursor.rowcount > 0

    def count_users(self):
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM users
            """)

            return cursor.fetchone()[0]

    def _create_verification_logs_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    decision TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    threshold REAL NOT NULL,
                    ciphertext_sha256 TEXT,
                    ciphertext_bytes INTEGER,
                    request_id TEXT
                )
            """)
            conn.commit()

    def add_verification_log(
        self,
        user_id,
        decision,
        similarity,
        threshold,
        ciphertext_sha256=None,
        ciphertext_bytes=None,
        request_id=None,
    ):
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO verification_logs (
                    user_id,
                    decision,
                    similarity,
                    threshold,
                    ciphertext_sha256,
                    ciphertext_bytes,
                    request_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    decision,
                    similarity,
                    threshold,
                    ciphertext_sha256,
                    ciphertext_bytes,
                    request_id,
                ),
            )
            conn.commit()
            return cursor.lastrowid