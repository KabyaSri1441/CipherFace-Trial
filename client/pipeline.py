"""
client/pipeline.py

Orchestrates the entire CLIENT-side flow:

Capture
    -> Detect
    -> Get coordinates + landmarks
    -> Align
    -> FaceNet
    -> ArcFace
    -> Fuse
    -> Normalize
    -> CKKS Encrypt
    -> serialized ciphertext

The spatial information (bounding box and landmarks) is kept on the
client side. Only the final normalized embedding is encrypted and
sent to the server.
"""

from typing import Optional
import numpy as np

from client.detector import FaceDetectorAligner
from client.embeddings.facenet_extractor import FaceNetExtractor
from client.embeddings.arcface_extractor import ArcFaceExtractor
from client.fusion import FeatureFusion
from client.normalization import EmbeddingNormalizer
from client.encryption.ckks_client import CKKSClient
from client.face_data import FaceDetectionResult

from common.utils import get_logger


logger = get_logger(__name__)


class ClientPipeline:
    """
    Purpose:
        Turn a raw captured image into a serialized CKKS ciphertext.

    The pipeline also keeps access to the face detection information:

        - bounding box
        - facial landmarks
        - detection confidence
        - image dimensions
        - aligned face

    Only the final normalized embedding is passed to CKKS.
    """

    def __init__(self):

        self.detector = FaceDetectorAligner()

        self.facenet = FaceNetExtractor()

        self.arcface = ArcFaceExtractor()

        self.fusion = FeatureFusion()

        self.normalizer = EmbeddingNormalizer()

        self.ckks = CKKSClient()

    # ============================================================
    # DETECTION ONLY
    # ============================================================

    def detect_face(self, raw_image) -> Optional[FaceDetectionResult]:
        """
        Run MTCNN detection and return the complete face information.

        This is useful when we want to inspect:

            - bounding box
            - landmarks
            - confidence
            - aligned face

        Returns:
            FaceDetectionResult
            or None if no face was detected.
        """

        return self.detector.detect_and_align(raw_image)

    # ============================================================
    # FULL CLIENT PIPELINE
    # ============================================================

    def image_to_encrypted_embedding(
        self,
        raw_image
    ) -> Optional[bytes]:
        """
        Input:
            raw_image -> PIL.Image or np.ndarray

        Output:
            bytes -> serialized CKKS ciphertext

        Returns:
            None if no face was detected.

        Pipeline:

            Image
              ↓
            MTCNN
              ↓
            Bounding Box + Landmarks
              ↓
            Aligned Face
              ↓
            FaceNet
              ↓
            ArcFace
              ↓
            Feature Fusion
              ↓
            L2 Normalization
              ↓
            CKKS Encryption
        """

        # --------------------------------------------------------
        # 1. FACE DETECTION + ALIGNMENT
        # --------------------------------------------------------

        detection = self.detector.detect_and_align(
            raw_image
        )

        if detection is None:
            return None

        # --------------------------------------------------------
        # 2. GET ALIGNED FACE
        # --------------------------------------------------------
        #
        # The detector now returns a FaceDetectionResult instead
        # of returning the aligned face directly.
        #
        # The embedding models still receive exactly what they
        # received before:
        #
        #       np.ndarray (160, 160, 3)
        #
        # --------------------------------------------------------

        aligned_face = detection.aligned_face

        # --------------------------------------------------------
        # 4. FACENET
        # --------------------------------------------------------

        fn_embedding = self.facenet.extract(
            aligned_face
        )

        logger.info(
            f"FaceNet embedding dimension: "
            f"{fn_embedding.shape}"
        )

        # --------------------------------------------------------
        # 5. ARCFACE
        # --------------------------------------------------------

        af_embedding = self.arcface.extract(
            aligned_face
        )

        logger.info(
            f"ArcFace embedding dimension: "
            f"{af_embedding.shape}"
        )

        # --------------------------------------------------------
        # 6. FEATURE FUSION
        # --------------------------------------------------------

        fused = self.fusion.fuse(
            fn_embedding,
            af_embedding
        )

        logger.info(
            f"Fused embedding dimension: "
            f"{fused.shape}"
        )

        # --------------------------------------------------------
        # 7. FINAL L2 NORMALIZATION
        # --------------------------------------------------------

        fused = self.normalizer.normalize(
            fused
        )

        logger.info(
            f"Normalized embedding dimension: "
            f"{fused.shape}"
        )

        logger.info(
            f"Normalized embedding L2 norm: "
            f"{np.linalg.norm(fused):.6f}"
        )

        # --------------------------------------------------------
        # 8. CKKS ENCRYPTION
        # --------------------------------------------------------

        encrypted = self.ckks.encrypt(
            fused
        )

        logger.info(
            "Embedding encrypted successfully."
        )

        # --------------------------------------------------------
        # 9. SERIALIZE CIPHERTEXT
        # --------------------------------------------------------

        return encrypted.serialize()

    # ============================================================
    # DECRYPT SERVER SIMILARITY
    # ============================================================

    def decrypt_similarity(
        self,
        encrypted_result_bytes: bytes
    ) -> float:
        """
        Input:
            serialized CKKS ciphertext returned by server

        Output:
            plaintext similarity score
        """

        import tenseal as ts

        vector = ts.ckks_vector_from(
            self.ckks._context,
            encrypted_result_bytes
        )

        return self.ckks.decrypt_scalar(
            vector
        )

    # ============================================================
    # PUBLIC CKKS CONTEXT
    # ============================================================

    def public_context_bytes(self) -> bytes:
        """
        Context to send to the server once.

        Secret key is not included.
        """

        return self.ckks.export_public_context_bytes()