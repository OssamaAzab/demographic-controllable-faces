"""02: Stratified train/val/test split of FFHQ-1024 (27k/1.5k/1.5k).

Inputs:  data/ffhq_1024/*.png
Outputs: data/ffhq_train.jsonl, data/ffhq_val.jsonl, data/ffhq_test.jsonl

Each jsonl line: {"image_path": "...", "image_id": "..."}
Stratification by FairFace race + age bucket runs in 04_caption_ffhq.py;
this script just does a random reproducible split.

TODO:
- List all images in FFHQ_RAW
- Shuffle with fixed seed
- Split 27k/1.5k/1.5k (or proportional if FFHQ_SUBSET_SIZE != 30000)
- Write three jsonl manifests
"""

from dcfaces.paths import FFHQ_RAW, FFHQ_TRAIN, FFHQ_VAL, FFHQ_TEST, ensure_dirs


def main() -> None:
    ensure_dirs()
    print(f"TODO: split FFHQ → {FFHQ_TRAIN}, {FFHQ_VAL}, {FFHQ_TEST}")


if __name__ == "__main__":
    main()
