"""Load and persist the supplied identity-disjoint LFW pair protocol."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FacePair:
    pair_id: str
    split: str
    label: int
    image_a: str
    image_b: str
    identity_a: str
    identity_b: str


def _image_path(image_root: Path, identity: str, image_number: str) -> Path:
    try:
        number = int(image_number)
    except ValueError as exc:
        raise ValueError(f"Invalid image number {image_number!r} for {identity!r}.") from exc
    return image_root / identity / f"{identity}_{number:04d}.jpg"


def _read_pairs(data_dir: Path, image_root: Path, split: str) -> list[FacePair]:
    pairs: list[FacePair] = []
    pair_number = 0

    same_path = data_dir / f"matchpairs{split}.csv"
    with same_path.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(source)
        next(rows, None)
        for row in rows:
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns in {same_path}: {row!r}")
            identity, image_a_number, image_b_number = row
            image_a = _image_path(image_root, identity, image_a_number)
            image_b = _image_path(image_root, identity, image_b_number)
            pair_number += 1
            pairs.append(FacePair(
                pair_id=f"{split.lower()}-{pair_number:05d}",
                split=split.lower(),
                label=1,
                image_a=str(image_a),
                image_b=str(image_b),
                identity_a=identity,
                identity_b=identity,
            ))

    different_path = data_dir / f"mismatchpairs{split}.csv"
    with different_path.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(source)
        next(rows, None)
        for row in rows:
            if len(row) != 4:
                raise ValueError(f"Expected 4 columns in {different_path}: {row!r}")
            identity_a, image_a_number, identity_b, image_b_number = row
            image_a = _image_path(image_root, identity_a, image_a_number)
            image_b = _image_path(image_root, identity_b, image_b_number)
            pair_number += 1
            pairs.append(FacePair(
                pair_id=f"{split.lower()}-{pair_number:05d}",
                split=split.lower(),
                label=0,
                image_a=str(image_a),
                image_b=str(image_b),
                identity_a=identity_a,
                identity_b=identity_b,
            ))

    missing = [path for pair in pairs for path in (Path(pair.image_a), Path(pair.image_b)) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LFW pair references missing image: {missing[0]}")
    return pairs


def load_lfw_pair_lists(data_dir: Path, image_root: Path) -> dict[str, list[FacePair]]:
    """Use official DevTrain pairs for validation and DevTest pairs for test."""
    data_dir = Path(data_dir)
    image_root = Path(image_root)
    validation = _read_pairs(data_dir, image_root, "DevTrain")
    test = _read_pairs(data_dir, image_root, "DevTest")

    validation_identities = {
        identity
        for pair in validation
        for identity in (pair.identity_a, pair.identity_b)
    }
    test_identities = {
        identity
        for pair in test
        for identity in (pair.identity_a, pair.identity_b)
    }
    overlap = validation_identities & test_identities
    if overlap:
        raise ValueError(f"LFW validation/test identities overlap: {sorted(overlap)[:5]}")

    return {"validation": validation, "test": test}


def save_lfw_pair_lists(
    pair_lists: dict[str, Iterable[FacePair]],
    output_dir: Path,
) -> Path:
    """Write immutable CSV pair lists and protocol metadata; never overwrite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    for split, pairs in pair_lists.items():
        destination = output_dir / f"{split}_pairs.csv"
        with destination.open("x", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(FacePair.__dataclass_fields__))
            writer.writeheader()
            count = 0
            for pair in pairs:
                writer.writerow(asdict(pair))
                count += 1
        written[split] = count

    metadata = {
        "protocol": "LFW official pair files; DevTrain used for validation and DevTest for final test",
        "identity_disjoint": True,
        "random_seed": None,
        "pair_generation": "Pairs are copied from the supplied official CSV lists; no resampling is performed.",
        "counts": written,
        "sources": [
            "matchpairsDevTrain.csv",
            "mismatchpairsDevTrain.csv",
            "matchpairsDevTest.csv",
            "mismatchpairsDevTest.csv",
        ],
    }
    with (output_dir / "protocol.json").open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2)
        target.write("\n")
    return output_dir