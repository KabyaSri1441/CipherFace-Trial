"""Materialize the supplied LFW validation and test pair lists."""

import json
from pathlib import Path

from evaluation.protocol import load_lfw_pair_lists, save_lfw_pair_lists


DATA_DIR = Path("data/lfw")
IMAGE_ROOT = Path("data/lfw/lfw-deepfunneled/lfw-deepfunneled")
OUTPUT_DIR = Path("results/protocol")


def main() -> None:
    pair_lists = load_lfw_pair_lists(DATA_DIR, IMAGE_ROOT)
    save_lfw_pair_lists(pair_lists, OUTPUT_DIR)
    print(json.dumps({split: len(pairs) for split, pairs in pair_lists.items()}, indent=2))
    print(f"Fixed pair lists written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()