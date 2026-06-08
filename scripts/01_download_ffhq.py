"""01: Download FFHQ-1024 into data/images1024x1024/.

Source: the pravsels/FFHQ_1024 HF mirror, which stores the 70k images as 7 zip
shards of 10,000 PNGs each. Canonical origin is NVlabs FFHQ (Karras et al. 2019,
non-commercial research use); the mirror redistributes the same images under their
original 5-digit filenames and avoids the rate-limited official Google Drive.

Size is set by FFHQ_SUBSET_SIZE (env) or --subset-size (default 40000), rounded up
to whole 10k shards. Resumable: a shard with a .done marker is skipped on re-run,
and its zip is removed after extraction.

Inputs:  none
Outputs: data/images1024x1024/<index>.png
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from dcfaces.paths import FFHQ_RAW, ensure_dirs

REPO_ID = "pravsels/FFHQ_1024"
TOTAL_SHARDS = 7
IMAGES_PER_SHARD = 10_000
SHARD_TEMPLATE = "shard_{i}_of_7.zip"


def n_shards_for(subset_size: int) -> int:
    return max(1, min(TOTAL_SHARDS, math.ceil(subset_size / IMAGES_PER_SHARD)))


def extract_shard(zip_path: Path, out_dir: Path) -> int:
    """Extract PNGs (flat) into out_dir, skipping any already present."""
    written = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in (m for m in z.namelist() if m.lower().endswith(".png")):
            dest = out_dir / Path(name).name
            if dest.exists():
                continue
            with z.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subset-size", type=int, default=int(os.environ.get("FFHQ_SUBSET_SIZE", 40_000)),
        help="Images to fetch; rounded up to whole 10k shards (default 40000).",
    )
    parser.add_argument("--repo-id", default=REPO_ID, help="HF dataset repo id.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep shard zips after extraction.")
    args = parser.parse_args()

    ensure_dirs()
    FFHQ_RAW.mkdir(parents=True, exist_ok=True)
    shard_dir = FFHQ_RAW / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    n = n_shards_for(args.subset_size)
    print(f"Downloading {n}/{TOTAL_SHARDS} shards of {args.repo_id} (~{n * IMAGES_PER_SHARD:,} images) -> {FFHQ_RAW}")

    total = 0
    for i in range(1, n + 1):
        fname = SHARD_TEMPLATE.format(i=i)
        done = shard_dir / f"{fname}.done"
        if done.exists():
            print(f"[{i}/{n}] {fname}: already done")
            continue
        print(f"[{i}/{n}] {fname}: downloading")
        zip_path = Path(hf_hub_download(repo_id=args.repo_id, filename=fname, repo_type="dataset", local_dir=str(shard_dir)))
        print(f"[{i}/{n}] {fname}: extracting")
        total += extract_shard(zip_path, FFHQ_RAW)
        done.write_text(fname + "\n")
        if not args.keep_zips:
            zip_path.unlink(missing_ok=True)

    n_png = sum(1 for _ in FFHQ_RAW.glob("*.png"))
    print(f"Done. +{total:,} new this run; {n_png:,} PNGs in {FFHQ_RAW}.")


if __name__ == "__main__":
    main()
