"""
common/config.py

Single source of truth for every tunable constant used across the
client and server pipelines. Nothing in this project should hard-code
a magic number outside this file.
"""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """Config for MTCNN face detection + alignment."""
    image_size: int = 160          # FaceNet expects 160x160 aligned crops
    margin: int = 20               # pixel margin around detected box before crop
    min_face_size: int = 40        # ignore faces smaller than this (px)
    thresholds: tuple = (0.6, 0.7, 0.7)   # MTCNN P/R/O-net thresholds
    device: str = "cpu"            # "cuda" if a GPU is available


@dataclass(frozen=True)
class EmbeddingConfig:
    """Config for the two embedding backbones."""
    facenet_pretrained: str = "vggface2"   # facenet-pytorch pretrained weights
    facenet_dim: int = 512

    arcface_model_name: str = "buffalo_l"  # insightface model pack
    arcface_dim: int = 512

    device: str = "cpu"


@dataclass(frozen=True)
class FusionConfig:
    """Validation-selected weighted embedding fusion configuration."""
    method: str = "weighted_embedding"
    alpha: float = 0.4     # validation-selected weight on FaceNet embedding
    beta: float = 0.6      # validation-selected weight on ArcFace embedding
    fused_dim: int = 512
    concatenated_dim: int = 1024


@dataclass(frozen=True)
class CKKSConfig:
    """
    CKKS encryption parameters (TenSEAL).

    poly_modulus_degree=8192 gives 4096 usable slots per ciphertext,
    comfortably more than the 1024-d fused embedding, while keeping
    encryption/decryption fast enough for real-time verification.
    """
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple = (60, 40, 40, 60)
    global_scale_bits: int = 40
    context_path: str = os.environ.get(
        "FACE_SERVER_PUBLIC_CONTEXT",
        "server/config/ckks_public_context.bin",
    )
    secret_context_path: str = os.environ.get(
        "FACE_CLIENT_SECRET_CONTEXT",
        str(Path.home() / ".local/share/cipherface/client/ckks_secret_context.bin"),
    )


@dataclass(frozen=True)
class VerificationConfig:
    """Decision threshold for accept/reject on cosine similarity in [-1, 1]."""
    similarity_threshold: float = 0.22415309377886594

DETECTION = DetectionConfig()
EMBEDDING = EmbeddingConfig()
FUSION = FusionConfig()
CKKS = CKKSConfig()
VERIFICATION = VerificationConfig()
