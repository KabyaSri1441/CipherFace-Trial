"""Scoring primitives for standalone and fused face embeddings."""

import numpy as np


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Embedding must have a finite, non-zero L2 norm.")
    return values / norm


def cosine_score(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    first = _unit_vector(embedding_a)
    second = _unit_vector(embedding_b)
    if first.shape != second.shape:
        raise ValueError("Embeddings must have the same dimension.")
    return float(np.dot(first, second))


def concatenation_score(
    facenet_a: np.ndarray,
    arcface_a: np.ndarray,
    facenet_b: np.ndarray,
    arcface_b: np.ndarray,
) -> float:
    first = np.concatenate((_unit_vector(facenet_a), _unit_vector(arcface_a)))
    second = np.concatenate((_unit_vector(facenet_b), _unit_vector(arcface_b)))
    return cosine_score(first, second)


def weighted_embedding_score(
    facenet_a: np.ndarray,
    arcface_a: np.ndarray,
    facenet_b: np.ndarray,
    arcface_b: np.ndarray,
    alpha: float,
) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Fusion alpha must be between 0 and 1.")
    first = _unit_vector(alpha * _unit_vector(facenet_a) + (1.0 - alpha) * _unit_vector(arcface_a))
    second = _unit_vector(alpha * _unit_vector(facenet_b) + (1.0 - alpha) * _unit_vector(arcface_b))
    return float(np.dot(first, second))


def score_level_fusion(facenet_score: float, arcface_score: float, alpha: float) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Fusion alpha must be between 0 and 1.")
    return float(alpha * facenet_score + (1.0 - alpha) * arcface_score)