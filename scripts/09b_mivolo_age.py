"""09b: MiVOLO continuous-age scoring for the age-control metric.

MiVOLO depends on timm 0.8, which conflicts with the main environment's timm 1.0,
so it runs in an isolated virtualenv with its own pinned deps:

    python -m venv .venv-mivolo
    .venv-mivolo/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
    .venv-mivolo/bin/pip install timm==0.8.13.dev0 ultralytics==8.1.0 lapx opencv-python-headless omegaconf huggingface_hub
    PYTHONPATH=external/MiVOLO .venv-mivolo/bin/python scripts/09b_mivolo_age.py

Outputs results/benchmark/mivolo_ages.json (path -> predicted age). Script 09's
aggregation prefers this continuous age for the age-MAE metric, falling back to the
FairFace age-bucket midpoint where MiVOLO finds no face. Resumable.

Run from the repo root (uses repo-relative paths; does NOT import dcfaces, since the
isolated venv has only MiVOLO's deps).
"""

import json
import os
import sys

import cv2

sys.path.insert(0, "external/MiVOLO")
from mivolo.model.mi_volo import MiVOLO  # noqa: E402
from mivolo.model.yolo_detector import Detector  # noqa: E402

BENCH = "results/benchmark"
OUT = f"{BENCH}/mivolo_ages.json"
DET = "models/mivolo/detector/yolov8x_person_face.pt"
CKPT = "models/mivolo/mivolo_imdb.pth.tar"


def main():
    manifest = [json.loads(line) for line in open(f"{BENCH}/manifest.jsonl") if line.strip()]
    age_rows = [r for r in manifest if r["category"] == "age"]
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [r for r in age_rows if r["path"] not in done]
    print(f"{len(age_rows)} age images, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    det = Detector(DET, "cpu", verbose=False)
    age_model = MiVOLO(CKPT, "cpu", half=False, use_persons=True, disable_faces=False, verbose=False)
    for i, r in enumerate(todo):
        img = cv2.imread(f"{BENCH}/{r['path']}")
        if img is None:
            continue
        detected = det.predict(img)
        age_model.predict(img, detected)
        ages = [a for a in detected.ages if a is not None]
        if ages:
            done[r["path"]] = float(sum(ages) / len(ages))
        if (i + 1) % 100 == 0:
            json.dump(done, open(OUT, "w"))
            print(f"  {i + 1}/{len(todo)}", flush=True)
    json.dump(done, open(OUT, "w"))
    print(f"MIVOLO_AGES_DONE {len(done)}/{len(age_rows)}")


if __name__ == "__main__":
    main()
