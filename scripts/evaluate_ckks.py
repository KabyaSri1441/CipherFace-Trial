"""Compare plaintext and CKKS verification on the same fixed LFW test pairs."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
import tenseal as ts
from PIL import Image

from client.detector import FaceDetectorAligner
from client.embeddings.arcface_extractor import ArcFaceExtractor
from client.embeddings.facenet_extractor import FaceNetExtractor
from client.encryption.ckks_client import CKKSClient
from common.config import CKKS
from common.utils import l2_normalize
from evaluation.metrics import analyze_scores
from evaluation.protocol import FacePair
from server.encryption.ckks_server import CKKSServerContext
from server.matcher import EncryptedMatcher


PAIR_PATH = Path("results/protocol/test_scored_pairs.csv")
RESULT_DIR = Path("results/encrypted")
PLAINTEXT_DIR = Path("results/plaintext")
CONFIG_PATH = Path("results/final/selected_config_v3.json")


def _read_pairs() -> list[FacePair]:
    with PAIR_PATH.open(newline="", encoding="utf-8") as source:
        return [
            FacePair(
                pair_id=row["pair_id"], split=row["split"], label=int(row["label"]),
                image_a=row["image_a"], image_b=row["image_b"],
                identity_a=row["identity_a"], identity_b=row["identity_b"],
            )
            for row in csv.DictReader(source)
        ]


def _write_csv(path: Path, headers: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(headers)
        writer.writerows(rows)


def run() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    alpha = float(config["fusion"]["alpha"])
    threshold = float(config["thresholds"]["fusion"])
    pairs = _read_pairs()
    detector = FaceDetectorAligner()
    facenet = FaceNetExtractor()
    arcface = ArcFaceExtractor()
    ckks_client = CKKSClient()
    server_context = CKKSServerContext(ckks_client.export_public_context_bytes())
    embedding_cache: dict[str, np.ndarray] = {}
    model_seconds = 0.0
    plaintext_seconds = 0.0
    encryption_seconds = 0.0
    homomorphic_seconds = 0.0
    decryption_seconds = 0.0
    start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    run_started = time.perf_counter()
    rows: list[tuple[str, int, float, float, float, float, int, int, int, int]] = []

    def get_embedding(image_path: str) -> np.ndarray:
        nonlocal model_seconds
        if image_path not in embedding_cache:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            model_started = time.perf_counter()
            detection = detector.detect_and_align(image)
            if detection is None:
                raise RuntimeError(f"Fixed test pair image has no face: {image_path}")
            facenet_embedding = facenet.extract(detection.aligned_face)
            arcface_embedding = arcface.extract(detection.aligned_face)
            fused = l2_normalize(
                alpha * facenet_embedding + (1.0 - alpha) * arcface_embedding
            )
            model_seconds += time.perf_counter() - model_started
            embedding_cache[image_path] = fused.astype(np.float32)
        return embedding_cache[image_path]

    for pair in pairs:
        embedding_a = get_embedding(pair.image_a)
        embedding_b = get_embedding(pair.image_b)

        stage_started = time.perf_counter()
        plaintext_score = float(np.dot(embedding_a, embedding_b))
        plaintext_seconds += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        encrypted_a_bytes = ckks_client.encrypt(embedding_a).serialize()
        encrypted_b_bytes = ckks_client.encrypt(embedding_b).serialize()
        encryption_seconds += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        encrypted_a = server_context.load_encrypted_vector(encrypted_a_bytes)
        encrypted_b = server_context.load_encrypted_vector(encrypted_b_bytes)
        encrypted_score_bytes = EncryptedMatcher.compute_encrypted_similarity(
            encrypted_a, encrypted_b
        ).serialize()
        homomorphic_seconds += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        encrypted_score = ckks_client.decrypt_scalar(
            ts.ckks_vector_from(ckks_client._context, encrypted_score_bytes)
        )
        decryption_seconds += time.perf_counter() - stage_started

        absolute_error = abs(plaintext_score - encrypted_score)
        relative_error = absolute_error / max(abs(plaintext_score), 1e-12)
        rows.append((
            pair.pair_id, pair.label, plaintext_score, encrypted_score,
            absolute_error, relative_error, len(encrypted_a_bytes),
            len(encrypted_b_bytes), len(encrypted_score_bytes),
            int(plaintext_score >= threshold),
        ))

    total_seconds = time.perf_counter() - run_started
    end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLAINTEXT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        RESULT_DIR / "test_similarity_comparison.csv",
        (
            "pair_id", "label", "plaintext_score", "encrypted_score",
            "absolute_error", "relative_error", "query_ciphertext_bytes",
            "template_ciphertext_bytes", "result_ciphertext_bytes",
            "plaintext_prediction",
        ),
        rows,
    )

    pair_ids = [row[0] for row in rows]
    labels = np.asarray([row[1] for row in rows], dtype=np.int32)
    plaintext_scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    encrypted_scores = np.asarray([row[3] for row in rows], dtype=np.float64)
    plaintext_metrics = analyze_scores(labels, plaintext_scores, threshold)
    encrypted_metrics = analyze_scores(labels, encrypted_scores, threshold)

    for directory, stem, scores, metrics in (
        (PLAINTEXT_DIR, "test_selected_fusion", plaintext_scores, plaintext_metrics),
        (RESULT_DIR, "test_selected_fusion_ckks", encrypted_scores, encrypted_metrics),
    ):
        with (directory / f"{stem}_metrics.json").open("x", encoding="utf-8") as target:
            json.dump({key: value for key, value in metrics.items() if key not in {"predictions", "roc"}}, target, indent=2)
            target.write("\n")
        _write_csv(
            directory / f"{stem}_scores.csv",
            ("pair_id", "label", "score", "threshold", "prediction"),
            [
                (pair_id, int(label), float(score), threshold, int(prediction))
                for pair_id, label, score, prediction in zip(pair_ids, labels, scores, metrics["predictions"])
            ],
        )
        _write_csv(
            directory / f"{stem}_roc.csv",
            ("false_accept_rate", "true_accept_rate", "false_reject_rate", "threshold"),
            [
                (point["false_accept_rate"], point["true_accept_rate"], point["false_reject_rate"], point["threshold"])
                for point in metrics["roc"]
            ],
        )

    errors = np.asarray([row[4] for row in rows], dtype=np.float64)
    relative_errors = np.asarray([row[5] for row in rows], dtype=np.float64)
    report = {
        "protocol": "Official LFW DevTest scored-pair subset shared with plaintext models",
        "pair_list": str(PAIR_PATH),
        "pair_count": len(rows),
        "alpha": alpha,
        "threshold": threshold,
        "ckks_parameters": {
            "poly_modulus_degree": CKKS.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(CKKS.coeff_mod_bit_sizes),
            "global_scale_bits": CKKS.global_scale_bits,
        },
        "plaintext_metrics": {key: value for key, value in plaintext_metrics.items() if key not in {"predictions", "roc"}},
        "encrypted_metrics": {key: value for key, value in encrypted_metrics.items() if key not in {"predictions", "roc"}},
        "similarity_error": {
            "absolute_mean": float(np.mean(errors)),
            "absolute_median": float(np.median(errors)),
            "absolute_max": float(np.max(errors)),
            "relative_mean": float(np.mean(relative_errors)),
            "relative_median": float(np.median(relative_errors)),
            "relative_max": float(np.max(relative_errors)),
            "relative_error_definition": "absolute_error / max(abs(plaintext_score), 1e-12)",
        },
        "timing_seconds": {
            "face_detection_and_embedding_total": model_seconds,
            "plaintext_similarity_total": plaintext_seconds,
            "encryption_total": encryption_seconds,
            "homomorphic_deserialize_dot_serialize_total": homomorphic_seconds,
            "decryption_total": decryption_seconds,
            "end_to_end_total": total_seconds,
            "per_pair_encryption": encryption_seconds / len(rows),
            "per_pair_homomorphic": homomorphic_seconds / len(rows),
            "per_pair_decryption": decryption_seconds / len(rows),
        },
        "ciphertext_bytes": {
            "query_mean": float(np.mean([row[6] for row in rows])),
            "template_mean": float(np.mean([row[7] for row in rows])),
            "similarity_result_mean": float(np.mean([row[8] for row in rows])),
        },
        "peak_rss_native_units": int(end_rss),
        "peak_rss_delta_native_units": int(max(0, end_rss - start_rss)),
        "memory_unit_note": "resource.ru_maxrss is reported in bytes on macOS and KiB on Linux.",
        "server_context_secret_key": False,
    }
    with (RESULT_DIR / "test_ckks_correctness_metrics.json").open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")

    with Path("results/final/privacy_comparison.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("privacy_configuration", "accuracy", "far", "frr", "eer_estimate", "auc", "encryption_s", "homomorphic_s", "decryption_s", "total_s"),
        )
        writer.writeheader()
        writer.writerow({
            "privacy_configuration": "plaintext",
            "accuracy": plaintext_metrics["accuracy"],
            "far": plaintext_metrics["far"],
            "frr": plaintext_metrics["frr"],
            "eer_estimate": plaintext_metrics["eer_estimate"],
            "auc": plaintext_metrics["auc"],
            "encryption_s": 0.0,
            "homomorphic_s": 0.0,
            "decryption_s": 0.0,
            "total_s": model_seconds + plaintext_seconds,
        })
        writer.writerow({
            "privacy_configuration": "ckks",
            "accuracy": encrypted_metrics["accuracy"],
            "far": encrypted_metrics["far"],
            "frr": encrypted_metrics["frr"],
            "eer_estimate": encrypted_metrics["eer_estimate"],
            "auc": encrypted_metrics["auc"],
            "encryption_s": encryption_seconds,
            "homomorphic_s": homomorphic_seconds,
            "decryption_s": decryption_seconds,
            "total_s": total_seconds,
        })

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
