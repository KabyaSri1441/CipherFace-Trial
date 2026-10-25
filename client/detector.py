"""
client/detector.py

MTCNN face detection + alignment.

Returns a FaceDetectionResult containing:
- aligned 160x160 face
- bounding box coordinates
- detection confidence
- five facial landmarks
- original image dimensions
"""

from typing import Optional

import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

from common.config import DETECTION
from common.utils import get_logger
from client.face_data import (
    BoundingBox,
    FacialLandmarks,
    FaceDetectionResult,
)

logger = get_logger(__name__)

_REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceDetectorAligner:
    """Detect faces and align the largest one using MTCNN's five landmarks.

    When multiple faces are detected, the largest bounding-box area wins;
    ties retain MTCNN's first returned detection.
    """

    def __init__(self):
        self._mtcnn = MTCNN(
            image_size=DETECTION.image_size,
            margin=DETECTION.margin,
            min_face_size=DETECTION.min_face_size,
            thresholds=list(DETECTION.thresholds),
            device=DETECTION.device,
            post_process=False,
            select_largest=True,
        )

    def detect_and_align(
        self,
        image: "Image.Image | np.ndarray"
    ) -> Optional[FaceDetectionResult]:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            image = Image.fromarray(image)

        image = image.convert("RGB")
        image_width, image_height = image.size
        boxes, probabilities, landmarks = self._mtcnn.detect(image, landmarks=True)

        if (
            boxes is None
            or probabilities is None
            or landmarks is None
            or len(boxes) == 0
            or len(probabilities) != len(boxes)
            or len(landmarks) != len(boxes)
        ):
            logger.warning("No usable face detected in input image.")
            return None

        areas = [
            max(0.0, float(box[2] - box[0]))
            * max(0.0, float(box[3] - box[1]))
            for box in boxes
        ]
        index = int(np.argmax(areas))
        x1, y1, x2, y2 = [float(value) for value in boxes[index]]
        probability = float(probabilities[index])
        landmark = np.asarray(landmarks[index], dtype=np.float32)

        if x2 <= x1 or y2 <= y1 or landmark.shape != (5, 2):
            logger.warning("Invalid face box or landmark coordinates.")
            return None

        facial_landmarks = FacialLandmarks(
            left_eye=(float(landmark[0][0]), float(landmark[0][1])),
            right_eye=(float(landmark[1][0]), float(landmark[1][1])),
            nose=(float(landmark[2][0]), float(landmark[2][1])),
            left_mouth=(float(landmark[3][0]), float(landmark[3][1])),
            right_mouth=(float(landmark[4][0]), float(landmark[4][1])),
        )
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

        target_landmarks = _REFERENCE_LANDMARKS_112 * (
            DETECTION.image_size / 112.0
        )
        transform, _ = cv2.estimateAffinePartial2D(
            landmark,
            target_landmarks,
            method=cv2.LMEDS,
        )
        if transform is None:
            logger.warning("Could not estimate face alignment transform.")
            return None

        aligned_face = cv2.warpAffine(
            np.asarray(image),
            transform,
            (DETECTION.image_size, DETECTION.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(np.uint8, copy=False)

        return FaceDetectionResult(
            bbox=bbox,
            landmarks=facial_landmarks,
            confidence=probability,
            image_width=image_width,
            image_height=image_height,
            aligned_face=aligned_face,
            face_count=len(boxes),
        )