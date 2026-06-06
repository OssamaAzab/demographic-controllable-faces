"""01: Download FFHQ-1024 dataset (or a 30k subset).

Inputs:  none (env: FFHQ_SUBSET_SIZE override, default 30000)
Outputs: data/ffhq_1024/*.png  (30k images by default, ~40 GB)

TODO:
- Use `huggingface_hub.snapshot_download` from `nuwandavek/ffhq-1024` or equivalent
- Verify image count, file integrity
- Support resume (skip files already on disk)
"""

from dcfaces.paths import FFHQ_RAW, ensure_dirs


def main() -> None:
    ensure_dirs()
    print(f"TODO: implement FFHQ download → {FFHQ_RAW}")


if __name__ == "__main__":
    main()
