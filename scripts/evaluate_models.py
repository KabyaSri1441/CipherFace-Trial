"""Generate shared-pair raw scores for FaceNet, ArcFace, and fusion candidates."""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from client.detector import FaceDetectorAligner
from client.embeddings.arcface_extractor import ArcFaceExtractor
from client.embeddings.facenet_extractor import FaceNetExtractor
from evaluation.protocol import FacePair
from evaluation.scoring import (
    concatenation_score,
    cosine_score,
    score_level_fusion,
    weighted_embedding_score,
)


PAIR_DIR = Path("results/protocol")
RESULTS_DIR = Path("results")
ALPHAS = tuple(round(index / 10, 1) for index in range(11))


def _read_pairs(path: Path) -> list[FacePair]:
    with path.open(newline="", encoding="utf-8") as source:
        return [
            FacePair(
                pair_id=row["pair_id"],
                split=row["split"],
                label=int(row["label"]),
                image_a=row["image_a"],
                image_b=row["image_b"],
                identity_a=row["identity_a"],
                identity_b=row["identity_b"],
            )
            for row in csv.DictReader(source)
        ]


def _write_scores(path: Path, rows: list[tuple[str, int, float]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(("pair_id", "label", "score"))
        writer.writerows(rows)


def run(split: str) -> None:
    pair_path = PAIR_DIR / f"{split}_pairs.csv"
    if not pair_path.is_file():
        raise FileNotFoundError(f"Fixed {split} pair list is missing: {pair_path}")
    pairs = _read_pairs(pair_path)
    if not pairs:
        raise ValueError(f"Fixed {split} pair list is empty.")

    logging.getLogger("client.detector").setLevel(logging.WARNING)
    detector = FaceDetectorAligner()
    facenet = FaceNetExtractor()
    arcface = ArcFaceExtractor()
    embedding_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def get_embeddings(image_path: str) -> tuple[np.ndarray, np.ndarray] | None:
        if image_path not in embedding_cache:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            detection = detector.detect_and_align(image)
            if detection is None:
                embedding_cache[image_path] = None
            else:
                embedding_cache[image_path] = (
                    facenet.extract(detection.aligned_face),
                    arcface.extract(detection.aligned_face),
                )
        return embedding_cache[image_path]

    facenet_rows: list[tuple[str, int, float]] = []
    arcface_rows: list[tuple[str, int, float]] = []
    concat_rows: list[tuple[str, int, float]] = []
    candidate_rows: list[tuple[str, int, str, float, float]] = []
    scored_pairs: list[FacePair] = []
    failed_pairs: list[tuple[str, int, str, str, str]] = []

    for pair in pairs:
        embeddings_a = get_embeddings(pair.image_a)
        embeddings_b = get_embeddings(pair.image_b)
        if embeddings_a is None or embeddings_b is None:
            undetected = [
                image_path
                for image_path, embeddings in (
                    (pair.image_a, embeddings_a),
                    (pair.image_b, embeddings_b),
                )
                if embeddings is None
            ]
            failed_pairs.append((
                pair.pair_id,
                pair.label,
                pair.image_a,
                pair.image_b,
                "|".join(undetected),
            ))
            continue
        scored_pairs.append(pair)
        facenet_a, arcface_a = embeddings_a
        facenet_b, arcface_b = embeddings_b
        facenet_score = cosine_score(facenet_a, facenet_b)
        arcface_score = cosine_score(arcface_a, arcface_b)
        facenet_rows.append((pair.pair_id, pair.label, facenet_score))
        arcface_rows.append((pair.pair_id, pair.label, arcface_score))
        concat_rows.append((
            pair.pair_id,
            pair.label,
            concatenation_score(facenet_a, arcface_a, facenet_b, arcface_b),
        ))

        for alpha in ALPHAS:
            candidate_rows.append((
                pair.pair_id,
                pair.label,
                "weighted_embedding",
                alpha,
                weighted_embedding_score(
                    facenet_a, arcface_a, facenet_b, arcface_b, alpha
                ),
            ))
            candidate_rows.append((
                pair.pair_id,
                pair.label,
                "score_level",
                alpha,
                score_level_fusion(facenet_score, arcface_score, alpha),
            ))

    model_dir = RESULTS_DIR / "facenet"
    arcface_dir = RESULTS_DIR / "arcface"
    fusion_dir = RESULTS_DIR / "fusion"
    model_dir.mkdir(parents=True, exist_ok=True)
    arcface_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)

    with (PAIR_DIR / f"{split}_scored_pairs.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(FacePair.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(pair.__dict__ for pair in scored_pairs)
    with (PAIR_DIR / f"{split}_detection_failures.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("pair_id", "label", "image_a", "image_b", "undetected_images"))
        writer.writerows(failed_pairs)

    _write_scores(model_dir / f"{split}_scores.csv", facenet_rows)
    _write_scores(arcface_dir / f"{split}_scores.csv", arcface_rows)
    _write_scores(fusion_dir / f"{split}_concatenation_scores.csv", concat_rows)
    with (fusion_dir / f"{split}_candidate_scores.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("pair_id", "label", "method", "alpha", "score"))
        writer.writerows(candidate_rows)

    metadata = {
        "split": split,
        "pair_list": str(pair_path),
        "pair_count": len(scored_pairs),
        "source_pair_count": len(pairs),
        "detection_failed_pair_count": len(failed_pairs),
        "unique_images_processed": len(embedding_cache),
        "unique_images_with_no_face": sum(value is None for value in embedding_cache.values()),
        "facenet": "facenet-pytorch InceptionResnetV1 vggface2, 512-D",
        "arcface": "InsightFace buffalo_l/w600k_r50.onnx, 512-D",
        "preprocessing": "MTCNN five-landmark similarity alignment, RGB 160x160 output",
        "candidate_methods": ["weighted_embedding", "score_level"],
        "candidate_alphas": list(ALPHAS),
        "embedding_cache": "in-memory only; not written to disk",
        "threshold": None,
    }
    with (fusion_dir / f"{split}_candidate_grid.json").open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2)
        target.write("\n")

    print(json.dumps({
        "split": split,
        "pairs": len(pairs),
        "scored_pairs": len(scored_pairs),
        "detection_failed_pairs": len(failed_pairs),
        "unique_images": len(embedding_cache),
        "facenet_scores": len(facenet_rows),
        "arcface_scores": len(arcface_rows),
        "concatenation_scores": len(concat_rows),
        "fusion_candidate_scores": len(candidate_rows),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    run(args.split)


if __name__ == "__main__":
    main()
