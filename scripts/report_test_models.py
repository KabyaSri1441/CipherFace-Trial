"""Report final-test metrics using thresholds frozen on validation data."""

import csv
import json
from pathlib import Path

import numpy as np

from scripts.select_validation_config import _read_scores, _write_experiment


RESULTS_DIR = Path("results")
CONFIG_PATH = RESULTS_DIR / "final/selected_config_v3.json"


def run() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = config["thresholds"]
    models = (
        ("facenet", RESULTS_DIR / "facenet/test_scores.csv", thresholds["facenet"]),
        ("arcface", RESULTS_DIR / "arcface/test_scores.csv", thresholds["arcface"]),
        (
            "concatenation",
            RESULTS_DIR / "fusion/test_concatenation_scores.csv",
            thresholds["concatenation"],
        ),
    )
    common_pair_ids: list[str] | None = None
    common_labels: np.ndarray | None = None
    summary = []

    for name, path, threshold in models:
        pair_ids, labels, scores = _read_scores(path)
        if common_pair_ids is None:
            common_pair_ids, common_labels = pair_ids, labels
        elif pair_ids != common_pair_ids or not np.array_equal(labels, common_labels):
            raise ValueError("Final model score files do not use identical test pairs.")
        if threshold is not None:
            metrics = _write_experiment(
                RESULTS_DIR / ("fusion" if name == "concatenation" else name),
                "test",
                name,
                pair_ids,
                labels,
                scores,
                name,
                decision_threshold=threshold,
            )
            summary.append(metrics)

    selected_method = config["fusion"]["method"]
    selected_alpha = config["fusion"]["alpha"]
    if selected_method == "concatenation":
        pair_ids, labels, scores = _read_scores(
            RESULTS_DIR / "fusion/test_concatenation_scores.csv"
        )
    elif selected_method in {"weighted_embedding", "score_level"}:
        candidate_path = RESULTS_DIR / "fusion/test_candidate_scores.csv"
        with candidate_path.open(newline="", encoding="utf-8") as source:
            candidates = [
                row for row in csv.DictReader(source)
                if row["method"] == selected_method
                and np.isclose(float(row["alpha"]), selected_alpha)
            ]
        if not candidates:
            raise ValueError("Selected fusion candidate is missing from test scores.")
        if [row["pair_id"] for row in candidates] != common_pair_ids:
            raise ValueError("Selected fusion candidate test pairs are misaligned.")
        pair_ids = common_pair_ids
        labels = np.asarray([int(row["label"]) for row in candidates], dtype=np.int32)
        scores = np.asarray([float(row["score"]) for row in candidates], dtype=np.float64)
    else:
        raise ValueError(f"Unknown selected fusion method: {selected_method}")

    selected_metrics = _write_experiment(
        RESULTS_DIR / "fusion",
        "test",
        "selected_fusion",
        pair_ids,
        labels,
        scores,
        selected_method,
        selected_alpha,
        thresholds["fusion"],
    )
    summary.append(selected_metrics)

    comparison_path = RESULTS_DIR / "final/test_model_comparison.csv"
    with comparison_path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("model", "method", "alpha", "threshold", "accuracy", "far", "frr", "eer_estimate", "auc"),
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in summary)

    print(json.dumps({
        "test_pairs": len(common_pair_ids),
        "selected_fusion": config["fusion"],
        "models": summary,
    }, indent=2))


if __name__ == "__main__":
    run()
