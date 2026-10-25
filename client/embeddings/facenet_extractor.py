"""
client/embeddings/facenet_extractor.py

Wraps your EXISTING FaceNet embedding step.

Library: facenet-pytorch (InceptionResnetV1)
Pretrained model: InceptionResnetV1(pretrained="vggface2")
Output: 512-dimensional embedding.
"""

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from common.config import EMBEDDING
from common.utils import l2_normalize, get_logger

logger = get_logger(__name__)


class FaceNetExtractor:
    """
    Purpose:
        Convert an aligned 160x160 face crop into a 512-d FaceNet
        embedding.

    Input:
        np.ndarray, shape (160, 160, 3), dtype uint8 (from FaceDetectorAligner).

    Output:
        np.ndarray, shape (512,), dtype float32, L2-normalized.
    """

    def __init__(self):
        self._model = InceptionResnetV1(
            pretrained=EMBEDDING.facenet_pretrained
        ).eval().to(EMBEDDING.device)

    @torch.no_grad()
    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through FaceNet.

        aligned_face: uint8 array (160,160,3) in RGB.
        Returns: L2-normalized float32 embedding, shape (512,).
        """
        tensor = torch.from_numpy(aligned_face).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0          # FaceNet's expected input scaling
        tensor = tensor.unsqueeze(0).to(EMBEDDING.device)

        embedding = self._model(tensor)            # shape (1, 512)
        embedding = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        return l2_normalize(embedding)
