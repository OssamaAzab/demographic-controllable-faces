"""Build a complete browsable gallery: for every (method, identity, prompt) cell,
pick the best of the 5 seeds and write a downscaled JPEG preview. 8 x 10 x 21 = 1680
images. Full-resolution PNGs (~3 GB) stay out of git; these 512 px previews (~90 MB)
let a reader scan how each method handles each prompt for each identity. For
full-resolution detail see the curated set in results/samples/.

"Best seed" is objective and reproducible: the seed with a detected face and the
highest AdaFace identity score (falling back to ArcFace, then to seed 42). AdaFace is
the de-circularized identity metric, so this selects the seed where the person is
most clearly rendered, and discards no-face/garbage seeds. It does not optimize for
control, so control failures stay visible.

Run after scripts/09 (needs the committed scores). Idempotent.
"""

from __future__ import annotations

import json
from collections import defaultdict

from PIL import Image

from dcfaces.paths import BENCHMARK_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "gallery"
PREVIEW = 512


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    manifest = load_jsonl(BENCHMARK_DIR / "manifest.jsonl")
    base = {r["path"]: r for r in load_jsonl(BENCHMARK_DIR / "scores.jsonl")}
    extra = {r["path"]: r for r in load_jsonl(BENCHMARK_DIR / "scores_extra.jsonl")}

    def rank(path):
        b = base.get(path, {})
        ada = extra.get(path, {}).get("adaface_cos")
        arc = b.get("identity_cos")
        ident = ada if ada is not None else (arc if arc is not None else -1.0)
        return (1 if b.get("face_detected") else 0, ident)

    cells = defaultdict(list)
    for r in manifest:
        cells[(r["method"], r["id"], r["prompt_key"])].append((r["seed"], r["path"]))

    written = 0
    low_sat = 0
    for (method, idn, pkey), seeds in cells.items():
        # best by (face_detected, identity); stable tie-break on seed
        best = max(seeds, key=lambda sp: rank(sp[1]) + (sp[0] == 42,))
        src = BENCHMARK_DIR / best[1]
        if not src.exists():
            continue
        dst = OUT / method / idn / f"{pkey}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(src).convert("RGB").resize((PREVIEW, PREVIEW), Image.LANCZOS)
        im.save(dst, "JPEG", quality=90)
        written += 1
        # cheap grayscale guard on a subsample
        if written % 50 == 0:
            import numpy as np
            if np.asarray(im.convert("HSV"))[..., 1].mean() < 12:
                low_sat += 1

    (OUT / "README.md").write_text(
        "# Full gallery\n\n"
        f"One preview per (method, identity, prompt): {written} images, 512 px JPEG.\n"
        "Layout is `<method>/<identity>/<prompt_key>.jpg`. The best of 5 seeds is shown,\n"
        "chosen by detected-face + highest AdaFace identity (see\n"
        "`scripts/make_full_gallery.py`); selection does not optimize for control, so\n"
        "control successes and failures are both visible. Full-resolution examples are in\n"
        "`results/samples/`. All on SDXL 1.0 + fp16-fix VAE (HyperLoRA on RealVisXL v4.0).\n"
    )

    print(f"wrote {written} previews to {OUT}")
    if low_sat:
        print(f"WARNING: {low_sat} sampled previews look near-grayscale")
    else:
        print("FULL_GALLERY_OK")


if __name__ == "__main__":
    main()
