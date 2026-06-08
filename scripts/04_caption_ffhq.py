"""04: Hybrid BLIP-2 + FairFace captioning of the FFHQ train + val splits.

Caption template (the contribution):
    "a photo of a {age_bucket} year old {race} {gender}, {blip2_description}"
    e.g. "a photo of a 30-39 year old East Asian woman, with long black hair,
          wearing a white blouse"

Why hybrid: pure BLIP-2 -> no demographic anchor; pure FairFace tags -> template
memorization. Hybrid -> demographic responsiveness + visual variety (Lesson 4).

Resumable: writes one jsonl line per image and skips image_ids already present,
so a crash/interrupt during the multi-hour run just resumes. --recompute starts
fresh. Batched: FairFace + BLIP-2 run together per batch on the GPU.

Inputs:  data/ffhq_train.jsonl, data/ffhq_val.jsonl
Outputs: data/ffhq_metadata.jsonl
         line: {image_id, image_path, split, caption, fairface_age,
                fairface_race, fairface_gender, blip2_desc, blip2_raw}
"""

from __future__ import annotations

import argparse
import json

from PIL import Image
from tqdm import tqdm

from dcfaces.paths import FFHQ_METADATA, FFHQ_TRAIN, FFHQ_VAL, PROJECT_ROOT, ensure_dirs


def build_caption(age_bucket: str, race: str, gender: str, desc: str) -> str:
    head = f"a photo of a {age_bucket} year old {race} {gender}"
    return f"{head}, {desc}" if desc else head


def load_records(splits: list[str]) -> list[dict]:
    manifests = {"train": FFHQ_TRAIN, "val": FFHQ_VAL}
    recs = []
    for split in splits:
        for line in open(manifests[split]):
            r = json.loads(line)
            r["split"] = split
            recs.append(r)
    return recs


def already_done(path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        try:
            done.add(json.loads(line)["image_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Images per batch (default 8).")
    parser.add_argument("--num-beams", type=int, default=3, help="BLIP-2 beams (default 3).")
    parser.add_argument("--max-new-tokens", type=int, default=30, help="BLIP-2 max new tokens.")
    parser.add_argument("--splits", default="train,val", help="Comma list (default train,val).")
    parser.add_argument("--limit", type=int, default=0, help="Cap #images (0=all); for testing.")
    parser.add_argument("--recompute", action="store_true", help="Ignore existing metadata, redo all.")
    args = parser.parse_args()

    ensure_dirs()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    recs = load_records(splits)

    done = set() if args.recompute else already_done(FFHQ_METADATA)
    todo = [r for r in recs if r["image_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(recs):,} images in {splits}; {len(done):,} already captioned; {len(todo):,} to do.")
    if not todo:
        print("Nothing to do.")
        return

    # Import heavy deps only after the cheap early-exit above.
    from dcfaces.captioning.blip2 import Blip2Captioner
    from dcfaces.demographics import FairFaceClassifier

    clf = FairFaceClassifier()
    captioner = Blip2Captioner(num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)

    mode = "w" if args.recompute else "a"
    written = 0
    with open(FFHQ_METADATA, mode) as out:
        for start in tqdm(range(0, len(todo), args.batch_size), desc="captioning"):
            chunk = todo[start : start + args.batch_size]
            images = [Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB") for r in chunk]
            ff = clf.classify_batch(images)
            desc = captioner.describe_batch(images)
            for r, fr, (clean, raw) in zip(chunk, ff, desc):
                rec = {
                    "image_id": r["image_id"],
                    "image_path": r["image_path"],
                    "split": r["split"],
                    "caption": build_caption(fr.age_bucket, fr.race, fr.gender, clean),
                    "fairface_age": fr.age_bucket,
                    "fairface_race": fr.race,
                    "fairface_gender": fr.gender,
                    "blip2_desc": clean,
                    "blip2_raw": raw,
                    "blip2_filtered": bool(raw) and not clean,
                }
                out.write(json.dumps(rec) + "\n")
                written += 1
            out.flush()

    print(f"Done. Wrote {written:,} captions -> {FFHQ_METADATA}")


if __name__ == "__main__":
    main()
