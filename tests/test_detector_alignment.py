import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from client.detector import FaceDetectorAligner


class FaceDetectorAlignmentTests(unittest.TestCase):
    @staticmethod
    def _landmarks(offset_x=20, offset_y=20, scale=1.0):
        template = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        return template * scale + np.array([offset_x, offset_y], dtype=np.float32)

    @staticmethod
    def _detector(detections):
        detector = FaceDetectorAligner.__new__(FaceDetectorAligner)
        detector._mtcnn = Mock()
        detector._mtcnn.detect.return_value = detections
        return detector

    def test_no_face_returns_none(self):
        detector = self._detector((None, None, None))

        result = detector.detect_and_align(Image.new("RGB", (200, 200)))

        self.assertIsNone(result)

    def test_one_face_produces_aligned_rgb_face(self):
        image = np.zeros((220, 220, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(220, dtype=np.uint8)[None, :]
        detections = (
            np.array([[20, 20, 180, 190]], dtype=np.float32),
            np.array([0.99], dtype=np.float32),
            np.array([self._landmarks()], dtype=np.float32),
        )

        result = self._detector(detections).detect_and_align(image)

        self.assertIsNotNone(result)
        self.assertEqual(result.aligned_face.shape, (160, 160, 3))
        self.assertEqual(result.aligned_face.dtype, np.uint8)
        self.assertEqual(result.image_width, 220)
        self.assertEqual(result.landmarks.as_array().shape, (5, 2))
        self.assertGreater(int(result.aligned_face[:, :, 0].max()), 0)

    def test_multiple_faces_selects_largest_area(self):
        detections = (
            np.array([[10, 10, 70, 70], [35, 25, 190, 195]], dtype=np.float32),
            np.array([0.98, 0.91], dtype=np.float32),
            np.array([
                self._landmarks(offset_x=10, offset_y=10, scale=0.6),
                self._landmarks(offset_x=35, offset_y=25, scale=1.0),
            ], dtype=np.float32),
        )

        result = self._detector(detections).detect_and_align(
            Image.new("RGB", (230, 230), (80, 120, 160))
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.bbox.as_array().tolist(), [35.0, 25.0, 190.0, 195.0])
        self.assertAlmostEqual(result.confidence, 0.91)


if __name__ == "__main__":
    unittest.main()