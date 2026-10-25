import unittest

import numpy as np

from evaluation.scoring import (
    concatenation_score,
    cosine_score,
    score_level_fusion,
    weighted_embedding_score,
)


class EvaluationScoringTests(unittest.TestCase):
    def setUp(self):
        self.facenet_a = np.array([1.0, 0.0], dtype=np.float32)
        self.facenet_b = np.array([0.8, 0.6], dtype=np.float32)
        self.arcface_a = np.array([0.0, 1.0], dtype=np.float32)
        self.arcface_b = np.array([0.6, 0.8], dtype=np.float32)

    def test_standalone_cosine_normalizes_inputs(self):
        self.assertAlmostEqual(cosine_score(self.facenet_a, self.facenet_b), 0.8)

    def test_concatenation_matches_equal_score_fusion_for_unit_vectors(self):
        facenet_score = cosine_score(self.facenet_a, self.facenet_b)
        arcface_score = cosine_score(self.arcface_a, self.arcface_b)
        expected = score_level_fusion(facenet_score, arcface_score, 0.5)
        actual = concatenation_score(
            self.facenet_a,
            self.arcface_a,
            self.facenet_b,
            self.arcface_b,
        )
        self.assertAlmostEqual(actual, expected)

    def test_weighted_embedding_fusion_alpha_endpoints_select_one_model(self):
        self.assertAlmostEqual(
            weighted_embedding_score(
                self.facenet_a, self.arcface_a, self.facenet_b, self.arcface_b, 1.0
            ),
            cosine_score(self.facenet_a, self.facenet_b),
        )
        self.assertAlmostEqual(
            weighted_embedding_score(
                self.facenet_a, self.arcface_a, self.facenet_b, self.arcface_b, 0.0
            ),
            cosine_score(self.arcface_a, self.arcface_b),
        )

    def test_fusion_rejects_out_of_range_alpha(self):
        with self.assertRaises(ValueError):
            score_level_fusion(0.2, 0.3, 1.1)
        with self.assertRaises(ValueError):
            weighted_embedding_score(
                self.facenet_a, self.arcface_a, self.facenet_b, self.arcface_b, -0.1
            )


if __name__ == "__main__":
    unittest.main()