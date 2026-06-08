"""02: Reproducible train/val/test split of FFHQ-1024.

Reads the full FFHQ on disk (official NVlabs layout: images1024x1024/XXXXX/XXXXX.png,
70k images), draws a seeded subset, and splits it 90/5/5 into train/val/test.

Defaults (subset 40000):
    train 36,000 | val 2,000 | test 2,000

The remaining images stay on disk, just unreferenced — bumping to 70k (or down to
30k) later is a one-flag change (--subset-size). The split is a plain seeded random
split; FairFace-stratification happens later in 04_caption_ffhq.py, as designed.

Each split's job:
    train -> LoRA training
    val   -> training-stability monitoring ONLY (val-MSE as a divergence canary,
             never for checkpoint selection -- that's the Part B lesson)
    test  -> identity gallery (03) + FID/KID reference set (09)

Inputs:  data/images1024x1024/**/*.png
Outputs: data/ffhq_train.jsonl, data/ffhq_val.jsonl, data/ffhq_test.jsonl
         Each line: {"image_id": "00000", "image_path": "data/images1024x1024/00000/00000.png"}
         (image_path is relative to the repo root for portability.)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dcfaces.paths import (
    FFHQ_RAW,
    FFHQ_TEST,
    FFHQ_TRAIN,
    FFHQ_VAL,
    PROJECT_ROOT,
    ensure_dirs,
)


def write_manifest(path: Path, images: list[Path]) -> None:
    """Write one jsonl line per image: {image_id, image_path(relative to repo root)}."""
    with open(path, "w") as f:
        for p in images:
            rec = {"image_id": p.stem, "image_path": p.relative_to(PROJECT_ROOT).as_posix()}
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=40_000,
        help="How many images to use (sampled from all on disk). Default 40000.",
    )
    parser.add_argument("--val-frac", type=float, default=0.05, help="Val fraction (default 0.05).")
    parser.add_argument("--test-frac", type=float, default=0.05, help="Test fraction (default 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42).")
    args = parser.parse_args()

    ensure_dirs()

    images = sorted(FFHQ_RAW.rglob("*.png"))  # deterministic base order
    n_disk = len(images)
    if n_disk == 0:
        raise SystemExit(
            f"No images found under {FFHQ_RAW}. Download FFHQ first (expected "
            f"images1024x1024/XXXXX/XXXXX.png)."
        )

    rng = random.Random(args.seed)
    rng.shuffle(images)

    subset_size = min(args.subset_size, n_disk)
    if subset_size < args.subset_size:
        print(f"WARNING: only {n_disk:,} images on disk; using all of them.")
    subset = images[:subset_size]

    n = len(subset)
    n_val = round(n * args.val_frac)
    n_test = round(n * args.test_frac)
    val = subset[:n_val]
    test = subset[n_val : n_val + n_test]
    train = subset[n_val + n_test :]

    write_manifest(FFHQ_TRAIN, train)
    write_manifest(FFHQ_VAL, val)
    write_manifest(FFHQ_TEST, test)

    print(f"FFHQ on disk: {n_disk:,} images | subset: {n:,} (seed {args.seed})")
    print(f"  train: {len(train):,}  -> {FFHQ_TRAIN}")
    print(f"  val:   {len(val):,}  -> {FFHQ_VAL}")
    print(f"  test:  {len(test):,}  -> {FFHQ_TEST}")

    # Sanity: splits are disjoint and cover the subset exactly.
    ids = [p.stem for p in subset]
    assert len(set(ids)) == n, "duplicate image stems detected"
    assert len(train) + len(val) + len(test) == n, "splits do not sum to subset size"


if __name__ == "__main__":
    main()
