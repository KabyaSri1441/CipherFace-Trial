"""
server/pipeline.py

Server-side service for privacy-preserving face verification.

The server:
1. Receives an encrypted probe embedding.
2. Loads encrypted templates from the database.
3. Performs homomorphic similarity computation.
4. Returns encrypted similarity scores.

The server never decrypts biometric embeddings or similarity scores.
"""

from pathlib import Path
import os

import tenseal as ts

from database.database_manager import EncryptedDatabase
from database.metadata_manager import MetadataDatabase
from server.encryption.ckks_server import CKKSServerContext
from server.matcher import EncryptedMatcher
from common.config import FUSION
from common.utils import get_logger


logger = get_logger(__name__)


class ServerService:

    def __init__(self, public_context_bytes: bytes):

        self.ctx = CKKSServerContext(
            public_context_bytes
        )

        self.encrypted_db = EncryptedDatabase(
            os.environ.get("FACE_TEMPLATE_DATABASE", "database/encrypted_templates_v2")
        )
        self.metadata_db = MetadataDatabase(
            os.environ.get("FACE_METADATA_DATABASE", "database/metadata_v2.db")
        )

        logger.info(
            "Server service initialized."
        )

    def enroll(
        self,
        user_id: str,
        encrypted_embedding_bytes: bytes
    ) -> None:
        encrypted_vector = self.ctx.load_encrypted_vector(encrypted_embedding_bytes)
        if encrypted_vector.size() != FUSION.fused_dim:
            raise ValueError(f"Encrypted template must contain {FUSION.fused_dim} values.")

        output_file = self.encrypted_db.add_template(
            user_id,
            encrypted_embedding_bytes
        )

        self.metadata_db.add_user(
            person_name=user_id,
            template_file=str(output_file),
            status="ENCRYPTED"
        )

        logger.info("Encrypted template stored.")

    def verify(
        self,
        user_id: str,
        encrypted_probe_bytes: bytes
    ) -> bytes:

        template_bytes = (
            self.encrypted_db.get_template(user_id)
        )

        if template_bytes is None:
            raise FileNotFoundError(
                "No enrolled template exists for this identity."
            )

        probe_vector = self.ctx.load_encrypted_vector(encrypted_probe_bytes)
        template_vector = self.ctx.load_encrypted_vector(template_bytes)
        if probe_vector.size() != FUSION.fused_dim:
            raise ValueError(f"Encrypted query must contain {FUSION.fused_dim} values.")
        if template_vector.size() != FUSION.fused_dim:
            raise ValueError(f"Encrypted template must contain {FUSION.fused_dim} values.")

        encrypted_similarity = (
            EncryptedMatcher.compute_encrypted_similarity(
                probe_vector,
                template_vector
            )
        )

        logger.info("Encrypted verification completed.")

        return encrypted_similarity.serialize()

    def list_users(self):

        return self.encrypted_db.list_users()

    def user_exists(self, user_id: str):

        return self.encrypted_db.user_exists(
            user_id
        )

    def delete_user(self, user_id: str) -> bool:
        deleted_template = self.encrypted_db.delete_user(user_id)
        deleted_metadata = self.metadata_db.delete_user(user_id)
        return deleted_template or deleted_metadata