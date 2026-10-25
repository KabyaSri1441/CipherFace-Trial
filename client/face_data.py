from dataclasses import dataclass
import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def as_array(self):
        return np.array(
            [self.x1, self.y1, self.x2, self.y2],
            dtype=np.float32
        )


@dataclass
class FacialLandmarks:
    left_eye: tuple
    right_eye: tuple
    nose: tuple
    left_mouth: tuple
    right_mouth: tuple

    def as_array(self):
        return np.array(
            [
                self.left_eye,
                self.right_eye,
                self.nose,
                self.left_mouth,
                self.right_mouth
            ],
            dtype=np.float32
        )


@dataclass
class FaceDetectionResult:
    bbox: BoundingBox
    landmarks: FacialLandmarks
    confidence: float

    image_width: int
    image_height: int

    aligned_face: np.ndarray
    face_count: int = 1