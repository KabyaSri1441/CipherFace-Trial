import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tenseal as ts
from PIL import Image

from client.detector import FaceDetectorAligner
from client.embeddings.arcface_extractor import ArcFaceExtractor
from client.embeddings.facenet_extractor import FaceNetExtractor
from client.encryption.ckks_client import CKKSClient
from client.fusion import FeatureFusion
from client.normalization import EmbeddingNormalizer
from client.verification import VerificationDecision
from common.utils import cosine_similarity
from server.encryption.ckks_server import CKKSServerContext
from server.pipeline import ServerService


class ModelAndCkksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = CKKSClient()
        cls.detector = FaceDetectorAligner()
        cls.facenet = FaceNetExtractor()
        cls.arcface = ArcFaceExtractor()
        cls.fusion = FeatureFusion()
        cls.image_path = Path("data/enrollment/user1_test.jpg")
        with Image.open(cls.image_path) as source:
            detection = cls.detector.detect_and_align(source.convert("RGB"))
        if detection is None:
            raise RuntimeError("The checked-in enrollment sample has no detectable face.")
        cls.detection = detection
        cls.facenet_embedding = cls.facenet.extract(detection.aligned_face)
        cls.arcface_embedding = cls.arcface.extract(detection.aligned_face)
        cls.embedding = EmbeddingNormalizer.normalize(
            cls.fusion.fuse(cls.facenet_embedding, cls.arcface_embedding)
        )

    def test_face_models_and_fusion_dimensions_and_norms(self):
        self.assertEqual(self.facenet_embedding.shape, (512,))
        self.assertEqual(self.arcface_embedding.shape, (512,))
        self.assertEqual(self.embedding.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(self.facenet_embedding)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(self.arcface_embedding)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(self.embedding)), 1.0, places=5)

    def test_ckks_dot_product_matches_plaintext_and_has_public_only_server_context(self):
        server_context = CKKSServerContext(self.client.export_public_context_bytes())
        self.assertFalse(server_context.context.has_secret_key())
        encrypted_a = self.client.encrypt(self.embedding).serialize()
        encrypted_b = self.client.encrypt(self.embedding).serialize()
        encrypted_result = server_context.load_encrypted_vector(encrypted_a).dot(
            server_context.load_encrypted_vector(encrypted_b)
        ).serialize()
        decrypted = ts.ckks_vector_from(self.client._context, encrypted_result).decrypt()[0]
        plaintext = cosine_similarity(self.embedding, self.embedding)
        self.assertAlmostEqual(decrypted, plaintext, delta=1e-4)

    def test_threshold_decision_and_server_enrollment_verification(self):
        self.assertEqual(VerificationDecision.decide(0.9)["decision"], "ACCEPT")
        self.assertEqual(VerificationDecision.decide(-0.1)["decision"], "REJECT")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with patch.dict(os.environ, {
                "FACE_TEMPLATE_DATABASE": str(root / "templates"),
                "FACE_METADATA_DATABASE": str(root / "metadata.db"),
            }):
                server = ServerService(self.client.export_public_context_bytes())
                encrypted_template = self.client.encrypt(self.embedding).serialize()
                server.enroll("Test_Identity", encrypted_template)
                encrypted_score = server.verify(
                    "Test_Identity",
                    self.client.encrypt(self.embedding).serialize(),
                )
                score = self.client.decrypt_scalar(
                    ts.ckks_vector_from(self.client._context, encrypted_score)
                )
                self.assertAlmostEqual(score, 1.0, delta=1e-4)
                self.assertEqual(server.list_users(), ["Test_Identity"])
                self.assertTrue(server.delete_user("Test_Identity"))
                self.assertFalse(server.user_exists("Test_Identity"))
                self.assertFalse(server.metadata_db.get_user("Test_Identity"))


if __name__ == "__main__":
    unittest.main()
