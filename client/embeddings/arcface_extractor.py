"""
client/embeddings/arcface_extractor.py

REWRITTEN — no FaceAnalysis dependency at all.

Why the previous fix still failed:
    insightface.app.FaceAnalysis is a fixed pipeline object. Internally
    it hard-requires a detection model to be present in self.models
    (assert 'detection' in self.models), regardless of what you pass
    into allowed_modules. There is no supported way to make
    FaceAnalysis detection-free. So FaceAnalysis cannot be used here
    at all, not even just to hold the recognition model.

Fix:
    Skip FaceAnalysis entirely. Load the ArcFace recognition ONNX
    model (w600k_r50.onnx from the buffalo_l pack) directly through
    insightface.model_zoo.get_model(), which builds a plain
    ArcFaceONNX object with a get_feat() method — a pure
    image-in/embedding-out forward pass, no detector involved
    anywhere in this file.

Library: insightface (model_zoo submodule only)
Pretrained model: w600k_r50.onnx (ArcFace recognition net, buffalo_l pack)
"""

import os
import zipfile
import urllib.request

import cv2
import numpy as np
from insightface.model_zoo import model_zoo

from common.config import EMBEDDING
from common.utils import l2_normalize, get_logger

logger = get_logger(__name__)

# Official InsightFace release containing the buffalo_l model pack
# (same pack FaceAnalysis(name="buffalo_l") would have downloaded).
_BUFFALO_L_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
_RECOGNITION_FILENAME = "w600k_r50.onnx"


class ArcFaceExtractor:
    """
    Purpose:
        Convert an aligned face crop into a 512-d ArcFace embedding
        using ONLY the recognition ONNX model — no detector is ever
        loaded or invoked by this class.

    Input:
        np.ndarray, shape (160, 160, 3), dtype uint8, RGB
        (the aligned crop MTCNN already produced).

    Output:
        np.ndarray, shape (512,), dtype float32, L2-normalized.

    API (unchanged from before): extract(aligned_face) -> np.ndarray
    """

    def __init__(self):
        model_path = self._ensure_recognition_model()
        self._rec_model = model_zoo.get_model(model_path)
        self._rec_model.prepare(ctx_id=-1)  # CPU; use 0 for first GPU
        self._input_size = self._rec_model.input_size  # (112, 112)
        logger.info("ArcFace recognition model loaded.")

    @staticmethod
    def _ensure_recognition_model() -> str:
        """
        Locate w600k_r50.onnx on disk; download+extract the buffalo_l
        pack only if it isn't already present (e.g. from an earlier
        run of this project, or a prior FaceAnalysis initialization).
        Returns the absolute path to the .onnx file.
        """
        model_dir = os.path.join(
            os.path.expanduser("~"), ".insightface", "models", EMBEDDING.arcface_model_name
        )
        rec_path = os.path.join(model_dir, _RECOGNITION_FILENAME)

        if os.path.isfile(rec_path):
            return rec_path

        logger.info("ArcFace recognition model not found locally — downloading buffalo_l pack...")
        os.makedirs(model_dir, exist_ok=True)
        zip_path = model_dir + ".zip"

        urllib.request.urlretrieve(_BUFFALO_L_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir)
        os.remove(zip_path)

        if not os.path.isfile(rec_path):
            raise FileNotFoundError(
                f"Download completed but {_RECOGNITION_FILENAME} was not found under "
                f"{model_dir}. If this persists, manually download the buffalo_l pack "
                f"from {_BUFFALO_L_ZIP_URL} and extract it into that folder."
            )
        return rec_path

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        aligned_face: uint8 RGB array (160,160,3), MTCNN's crop.
        Returns: L2-normalized float32 embedding, shape (512,).

        Pure forward pass: BGR convert -> resize to 112x112 -> get_feat().
        No detection happens anywhere in this method.
        """
        bgr_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr_face, self._input_size)

        raw_embedding = self._rec_model.get_feat(resized)  # shape (1, 512)
        embedding = np.asarray(raw_embedding).reshape(-1).astype(np.float32)

        return l2_normalize(embedding)
