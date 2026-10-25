"""Select fusion and EER thresholds using validation scores only."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import analyze_scores


RESULTS_DIR = Path("results")


def _read_scores(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return (
        [row["pair_id"] for row in rows],
        np.asarray([int(row["label"]) for row in rows], dtype=np.int32),
        np.asarray([float(row["score"]) for row in rows], dtype=np.float64),
    )


def _write_experiment(
    directory: Path,
    split: str,
    name: str,
    pair_ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    method: str,
    alpha: float | None = None,
    decision_threshold: float | None = None,
) -> dict:
    analysis = analyze_scores(labels, scores, decision_threshold)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{name}"

    metrics = {
        key: value
        for key, value in analysis.items()
        if key not in {"predictions", "roc"}
    }
    metrics.update({"model": name, "method": method, "alpha": alpha})
    with (directory / f"{stem}_metrics.json").open("x", encoding="utf-8") as target:
        json.dump(metrics, target, indent=2)
        target.write("\n")

    with (directory / f"{stem}_predictions.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("pair_id", "label", "score", "threshold", "prediction"))
        writer.writerows(
            (pair_id, int(label), float(score), analysis["threshold"], int(prediction))
            for pair_id, label, score, prediction in zip(
                pair_ids, labels, scores, analysis["predictions"]
            )
        )

    with (directory / f"{stem}_confusion_matrix.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("actual", "predicted_impostor", "predicted_genuine"))
        writer.writerow(("impostor", *analysis["confusion_matrix"]["matrix"][0]))
        writer.writerow(("genuine", *analysis["confusion_matrix"]["matrix"][1]))

    with (directory / f"{stem}_roc.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=list(analysis["roc"][0]))
        writer.writeheader()
        writer.writerows(analysis["roc"])

    figure, axis = plt.subplots(figsize=(6, 5))
    roc = analysis["roc"]
    axis.plot(
        [row["false_accept_rate"] for row in roc],
        [row["true_accept_rate"] for row in roc],
        label=f"AUC {analysis['auc']:.4f}",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set(xlabel="False acceptance rate", ylabel="True acceptance rate", title=f"{split.title()}: {name}")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(directory / f"{stem}_roc.png", dpi=150)
    plt.close(figure)
    return metrics


def run() -> None:
    split = "validation"
    model_scores = {}
    pair_ids: list[str] | None = None
    labels: np.ndarray | None = None

    for name, path in (
        ("facenet", RESULTS_DIR / "facenet/validation_scores.csv"),
        ("arcface", RESULTS_DIR / "arcface/validation_scores.csv"),
        ("concatenation", RESULTS_DIR / "fusion/validation_concatenation_scores.csv"),
    ):
        current_ids, current_labels, scores = _read_scores(path)
        if pair_ids is None:
            pair_ids, labels = current_ids, current_labels
        elif current_ids != pair_ids or not np.array_equal(current_labels, labels):
            raise ValueError("Model score files do not use identical validation pairs.")
        model_scores[name] = scores

    candidate_path = RESULTS_DIR / "fusion/validation_candidate_scores.csv"
    with candidate_path.open(newline="", encoding="utf-8") as source:
        candidate_rows = list(csv.DictReader(source))
    candidate_groups: dict[tuple[str, float], dict[str, tuple[int, float]]] = {}
    for row in candidate_rows:
        key = (row["method"], float(row["alpha"]))
        candidate_groups.setdefault(key, {})[row["pair_id"]] = (
            int(row["label"]),
            float(row["score"]),
        )

    candidates = []
    for (method, alpha), scores_by_id in candidate_groups.items():
        if set(scores_by_id) != set(pair_ids):
            raise ValueError(f"Fusion candidate {method}/{alpha} has different pair IDs.")
        candidate_labels = np.asarray([scores_by_id[pair_id][0] for pair_id in pair_ids])
        if not np.array_equal(candidate_labels, labels):
            raise ValueError(f"Fusion candidate {method}/{alpha} has different labels.")
        scores = np.asarray([scores_by_id[pair_id][1] for pair_id in pair_ids])
        metrics = analyze_scores(labels, scores)
        candidates.append({
            "method": method,
            "alpha": alpha,
            "scores": scores,
            "auc": metrics["auc"],
            "eer_estimate": metrics["eer_estimate"],
            "threshold": metrics["threshold"],
        })

    concatenation_metrics = analyze_scores(labels, model_scores["concatenation"])
    candidates.append({
        "method": "concatenation",
        "alpha": None,
        "scores": model_scores["concatenation"],
        "auc": concatenation_metrics["auc"],
        "eer_estimate": concatenation_metrics["eer_estimate"],
        "threshold": concatenation_metrics["threshold"],
    })

    for name, scores in model_scores.items():
        directory = RESULTS_DIR / ("fusion" if name == "concatenation" else name)
        _write_experiment(directory, split, f"{name}_v3", pair_ids, labels, scores, name)

    candidate_summary = sorted(
        candidates,
        key=lambda row: (-row["auc"], row["eer_estimate"], row["method"], row["alpha"]),
    )
    selected = candidate_summary[0]
    selected_scores = selected["scores"]
    selected_name = (
        f"{selected['method']}_v3"
        if selected["alpha"] is None
        else f"{selected['method']}_alpha_{selected['alpha']:.1f}_v3"
    )
    fusion_metrics = _write_experiment(
        RESULTS_DIR / "fusion",
        split,
        selected_name,
        pair_ids,
        labels,
        selected_scores,
        selected["method"],
        selected["alpha"],
    )

    with (RESULTS_DIR / "fusion/validation_grid_metrics_v3.csv").open(
        "x", newline="", encoding="utf-8"
    ) as target:
        writer = csv.writer(target)
        writer.writerow(("method", "alpha", "auc", "eer_estimate", "threshold"))
        writer.writerows(
            (row["method"], row["alpha"], row["auc"], row["eer_estimate"], row["threshold"])
            for row in candidate_summary
        )

    config = {
        "config_version": 3,
        "selected_on": "validation only",
        "pair_list": "results/protocol/validation_scored_pairs.csv",
        "pair_count": len(pair_ids),
        "threshold_rule": "ROC point minimizing absolute FAR-FRR difference",
        "thresholds": {
            "facenet": model_scores and analyze_scores(labels, model_scores["facenet"])["threshold"],
            "arcface": analyze_scores(labels, model_scores["arcface"])["threshold"],
            "concatenation": analyze_scores(labels, model_scores["concatenation"])["threshold"],
            "fusion": fusion_metrics["threshold"],
        },
        "fusion": {
            "method": selected["method"],
            "alpha": selected["alpha"],
            "selection_rule": "highest validation AUC, then lowest validation EER estimate, then method and alpha order",
            "validation_auc": selected["auc"],
            "validation_eer_estimate": selected["eer_estimate"],
        },
        "model_validation_metrics": {
            name: {
                "auc": float(analyze_scores(labels, scores)["auc"]),
                "eer_estimate": float(analyze_scores(labels, scores)["eer_estimate"]),
            }
            for name, scores in model_scores.items()
        },
        "source_threshold": "No prior threshold was used for selection.",
    }
    final_dir = RESULTS_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    with (final_dir / "selected_config_v3.json").open("x", encoding="utf-8") as target:
        json.dump(config, target, indent=2)
        target.write("\n")

    print(json.dumps({
        "validation_pairs": len(pair_ids),
        "selected_fusion": config["fusion"],
        "thresholds": config["thresholds"],
        "model_validation_metrics": config["model_validation_metrics"],
    }, indent=2))


if __name__ == "__main__":
    run()