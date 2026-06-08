"""03: Curate 10 benchmark identities from the FFHQ test split.

Goal: 10 single-reference identities covering FairFace's 7 race classes x both
genders x a spread of ages, each a clean/frontal/single-face image (reference
quality matters -- Lesson 3). Picked from the TEST split only, so the LoRA never
trains on a benchmark identity (no leakage).

Pipeline:
  1. For every test image: FairFace demographics (race/gender/age) + insightface
     detection score & head pose. Cached to _candidates.jsonl (reused on re-run;
     --recompute to force).
  2. A candidate "passes" if exactly 1 face, det_score >= --min-det, and
     |yaw|,|pitch| <= --max-pose (frontal).
  3. Fill 10 target (race, gender, age) slots, ranking by frontality. Graceful
     relaxation when a slot is scarce in FFHQ: exact -> drop age -> drop pose
     filter -> drop gender -> drop race. Each pick records which tier was used.
  4. Manual override: configs/identity_gallery_overrides.yaml may pin
     {id_00N: "<ffhq_image_id>"} to force a specific image into a slot.

Deviation from stub: pose/neutrality uses insightface (already a project dep,
gives yaw/pitch/roll directly) instead of MediaPipe. FairFace uses ViT heads
(see dcfaces.demographics) since the official ResNet34 weights are GDrive-only.

Inputs:  data/ffhq_test.jsonl  [+ optional configs/identity_gallery_overrides.yaml]
Outputs: data/identity_gallery/id_{001..010}/ref.jpg
         data/identity_gallery/id_{001..010}/metadata.json
         data/identity_gallery/_candidates.jsonl   (analysis cache)
         data/identity_gallery/gallery_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import yaml
from PIL import Image
from tqdm import tqdm

from dcfaces.paths import CONFIG_DIR, FFHQ_TEST, IDENTITY_GALLERY, PROJECT_ROOT, ensure_dirs

# 10 target slots: all 7 FairFace races covered, genders balanced 5/5, ages spread.
TARGET_SLOTS = [
    ("white", "man", "60-69"),
    ("Black", "woman", "20-29"),
    ("East Asian", "man", "30-39"),
    ("Southeast Asian", "woman", "40-49"),
    ("South Asian", "man", "20-29"),
    ("Middle Eastern", "woman", "30-39"),
    ("Latino", "man", "50-59"),
    ("white", "woman", "20-29"),
    ("Black", "man", "40-49"),
    ("East Asian", "woman", "60-69"),
]


@dataclass
class Candidate:
    image_id: str
    image_path: str
    race: str
    gender: str
    age_bucket: str
    race_raw: str
    race_score: float
    gender_score: float
    age_score: float
    det_score: float
    yaw: float
    pitch: float
    roll: float
    n_faces: int

    @property
    def frontality(self) -> float:
        return -(abs(self.yaw) + abs(self.pitch) + 0.5 * abs(self.roll))

    def passes(self, min_det: float, max_pose: float) -> bool:
        return (
            self.n_faces == 1
            and self.det_score >= min_det
            and abs(self.yaw) <= max_pose
            and abs(self.pitch) <= max_pose
        )


def analyze_test_set(batch_size: int) -> list[Candidate]:
    """Run FairFace + insightface over every test image."""
    from dcfaces.demographics import FairFaceClassifier
    from dcfaces.faces import FaceAnalyzer

    recs = [json.loads(line) for line in open(FFHQ_TEST)]
    clf = FairFaceClassifier()
    analyzer = FaceAnalyzer()

    cands: list[Candidate] = []
    for start in tqdm(range(0, len(recs), batch_size), desc="analyzing test set"):
        chunk = recs[start : start + batch_size]
        images = [Image.open(PROJECT_ROOT / r["image_path"]) for r in chunk]
        ff = clf.classify_batch(images)
        for r, img, fr in zip(chunk, images, ff):
            face = analyzer.analyze(img)
            if face is None:
                continue  # no detectable face -> not a usable reference
            cands.append(
                Candidate(
                    image_id=r["image_id"],
                    image_path=r["image_path"],
                    race=fr.race,
                    gender=fr.gender,
                    age_bucket=fr.age_bucket,
                    race_raw=fr.race_raw,
                    race_score=fr.race_score,
                    gender_score=fr.gender_score,
                    age_score=fr.age_score,
                    det_score=face.det_score,
                    yaw=face.yaw,
                    pitch=face.pitch,
                    roll=face.roll,
                    n_faces=face.n_faces,
                )
            )
    return cands


def load_or_build_cache(cache_path, recompute: bool, batch_size: int) -> list[Candidate]:
    if cache_path.exists() and not recompute:
        cands = [Candidate(**json.loads(line)) for line in open(cache_path)]
        print(f"Loaded {len(cands)} cached candidates from {cache_path}")
        return cands
    cands = analyze_test_set(batch_size)
    with open(cache_path, "w") as f:
        for c in cands:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"Analyzed {len(cands)} candidates -> cached to {cache_path}")
    return cands


def pick_for_slot(slot, pool, used, min_det, max_pose):
    """Return (best Candidate, selection_note) using graceful relaxation tiers."""
    race, gender, age = slot
    avail = [c for c in pool if c.image_id not in used]

    def f(rs=None, gs=None, ag=None, require_pass=True):
        out = avail
        if rs:
            out = [c for c in out if c.race == rs]
        if gs:
            out = [c for c in out if c.gender == gs]
        if ag:
            out = [c for c in out if c.age_bucket == ag]
        if require_pass:
            out = [c for c in out if c.passes(min_det, max_pose)]
        return out

    tiers = [
        (f(race, gender, age, True), "exact match"),
        (f(race, gender, None, True), "age relaxed"),
        (f(race, gender, None, False), "age + pose-filter relaxed"),
        (f(race, None, None, True), "gender relaxed"),
        (f(race, None, None, False), "gender + pose-filter relaxed"),
        (f(None, gender, None, True), "race relaxed (last resort)"),
    ]
    for group, note in tiers:
        if group:
            # deterministic: best frontality, then det_score, then image_id
            best = max(group, key=lambda c: (c.frontality, c.det_score, -int(c.image_id)))
            return best, note
    return None, "NO CANDIDATE"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recompute", action="store_true", help="Re-run analysis, ignore cache.")
    parser.add_argument("--min-det", type=float, default=0.6, help="Min detection score (default 0.6).")
    parser.add_argument("--max-pose", type=float, default=20.0, help="Max |yaw|,|pitch| deg (default 20).")
    parser.add_argument("--batch-size", type=int, default=32, help="FairFace batch size (default 32).")
    args = parser.parse_args()

    ensure_dirs()
    cache_path = IDENTITY_GALLERY / "_candidates.jsonl"
    by_id = {c.image_id: c for c in load_or_build_cache(cache_path, args.recompute, args.batch_size)}
    pool = list(by_id.values())

    overrides_path = CONFIG_DIR / "identity_gallery_overrides.yaml"
    overrides = {}
    if overrides_path.exists():
        overrides = (yaml.safe_load(overrides_path.read_text()) or {}).get("overrides", {}) or {}
        print(f"Applying {len(overrides)} override(s) from {overrides_path}")

    used: set[str] = set()
    summary = []
    for i, slot in enumerate(TARGET_SLOTS, start=1):
        slot_id = f"id_{i:03d}"
        if slot_id in overrides:
            forced = str(overrides[slot_id])
            cand = by_id.get(forced)
            if cand is None:
                print(f"  WARNING: override {slot_id}={forced} not in test set; auto-selecting.")
                cand, note = pick_for_slot(slot, pool, used, args.min_det, args.max_pose)
            else:
                note = "manual override"
        else:
            cand, note = pick_for_slot(slot, pool, used, args.min_det, args.max_pose)

        if cand is None:
            print(f"  {slot_id}: NO CANDIDATE for {slot} — skipped")
            continue
        used.add(cand.image_id)

        out_dir = IDENTITY_GALLERY / slot_id
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.open(PROJECT_ROOT / cand.image_path).convert("RGB").save(out_dir / "ref.jpg", quality=95)

        meta = {
            "id": slot_id,
            "ffhq_image_id": cand.image_id,
            "source_image_path": cand.image_path,
            "target_slot": {"race": slot[0], "gender": slot[1], "age_bucket": slot[2]},
            "fairface": {
                "race": cand.race,
                "race_raw": cand.race_raw,
                "gender": cand.gender,
                "age_bucket": cand.age_bucket,
                "scores": {
                    "race": round(cand.race_score, 4),
                    "gender": round(cand.gender_score, 4),
                    "age": round(cand.age_score, 4),
                },
            },
            "pose": {"yaw": round(cand.yaw, 2), "pitch": round(cand.pitch, 2), "roll": round(cand.roll, 2)},
            "det_score": round(cand.det_score, 4),
            "selection_notes": note,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        summary.append(meta)
        tgt = f"{slot[0]}/{slot[1]}/{slot[2]}"
        got = f"{cand.race}/{cand.gender}/{cand.age_bucket}"
        print(f"  {slot_id}: target {tgt:34} -> {cand.image_id} ({got}) [{note}]")

    (IDENTITY_GALLERY / "gallery_summary.json").write_text(json.dumps(summary, indent=2))
    races = {m["fairface"]["race"] for m in summary}
    genders = [m["fairface"]["gender"] for m in summary]
    print(
        f"\nGallery: {len(summary)}/10 identities -> {IDENTITY_GALLERY}\n"
        f"  races covered: {sorted(races)} ({len(races)}/7)\n"
        f"  gender balance: {genders.count('man')} men / {genders.count('woman')} women"
    )


if __name__ == "__main__":
    main()
