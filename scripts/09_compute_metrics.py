"""09: Score the benchmark generations into the metric suite + Pareto frontier.

Per-image metrics (one row per image, resumable):
    Identity:      ArcFace cosine, AdaFace cosine (the non-circular cross-check)
    Alignment:     CLIP-Score
    Aesthetic:     HPSv2, PickScore
    Perceptual:    DreamSim distance to the reference
    Race control:  FairFace accuracy on race_* prompts
    Age control:   FairFace age-bucket MAE on age_* prompts
    Accessory:     CLIP zero-shot presence on accessory_* prompts

Per-method distribution metrics (face-cropped):
    CLIP-FID, KID  vs the FFHQ test set

Not included: MiVOLO (continuous age) is blocked by a hard timm 0.8-vs-1.0
conflict with the rest of the stack; the FairFace age-bucket MAE is used instead.

The base per-image metrics live in scores.jsonl; the heavier extras live in
scores_extra.jsonl and are merged on `path` at aggregation, so each set can be
(re)computed independently. Both are resumable.

    python scripts/09_compute_metrics.py                  # score everything missing, then aggregate
    python scripts/09_compute_metrics.py --aggregate-only # re-draw tables/plot only
"""

from __future__ import annotations

import argparse
import json

from PIL import Image

from dcfaces.paths import (
    BENCHMARK_DIR,
    FFHQ_TEST,
    FIGURES_DIR,
    IDENTITY_GALLERY,
    PROJECT_ROOT,
    TABLES_DIR,
    ensure_dirs,
)

SCORES = BENCHMARK_DIR / "scores.jsonl"
SCORES_EXTRA = BENCHMARK_DIR / "scores_extra.jsonl"
MANIFEST = BENCHMARK_DIR / "manifest.jsonl"
CROP_ROOT = BENCHMARK_DIR / "facecrops"

REQUESTED_RACE = {
    "race_white": "white", "race_black": "Black", "race_east_asian": "East Asian",
    "race_southeast_asian": "Southeast Asian", "race_south_asian": "South Asian",
    "race_middle_eastern": "Middle Eastern", "race_latino": "Latino",
}
ACCESSORY_TEXT = {
    "accessory_0": ("a person wearing glasses", "a person not wearing glasses"),
    "accessory_1": ("a person with a full beard", "a person without a beard"),
    "accessory_2": ("a smiling person", "a person with a neutral expression"),
    "accessory_3": ("a person wearing a baseball cap", "a person not wearing a hat"),
    "accessory_4": ("a person with red hair", "a person without red hair"),
    "accessory_5": ("a person wearing heavy eye makeup", "a person without eye makeup"),
}
AGE_MIDPOINTS = {
    "0-2": 1, "3-9": 6, "10-19": 15, "20-29": 25, "30-39": 35,
    "40-49": 45, "50-59": 55, "60-69": 65, "70+": 75, "more than 70": 75,
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_all() -> None:
    """Base per-image metrics: ArcFace identity, CLIP-Score, FairFace race/age, CLIP accessory."""
    ensure_dirs()
    from dcfaces.demographics import FairFaceClassifier
    from dcfaces.metrics.clip_score import CLIPScorer
    from dcfaces.metrics.identity import IdentityScorer

    rows = load_jsonl(MANIFEST)
    done = {r["path"] for r in load_jsonl(SCORES)} if SCORES.exists() else set()
    todo = [r for r in rows if r["path"] not in done]
    print(f"[base] {len(rows)} images, {len(done)} scored, {len(todo)} to do")
    if not todo:
        return
    ident = IdentityScorer()
    clip = CLIPScorer()
    fair = FairFaceClassifier()
    ref_emb = {
        d.name: ident.embed(Image.open(d / "ref.jpg"))
        for d in sorted(IDENTITY_GALLERY.glob("id_*")) if (d / "ref.jpg").exists()
    }
    with SCORES.open("a") as out:
        for i, r in enumerate(todo):
            img = Image.open(BENCHMARK_DIR / r["path"]).convert("RGB")
            emb = ident.embed(img)
            rec = {
                "path": r["path"], "method": r["method"], "id": r["id"], "gender": r["gender"],
                "category": r["category"], "prompt_key": r["prompt_key"], "seed": r["seed"],
                "face_detected": emb is not None,
                "identity_cos": ident.cosine(emb, ref_emb.get(r["id"])),
                "clip_score": clip.score(img, r["prompt"]),
            }
            if r["category"] == "race":
                req = REQUESTED_RACE[r["prompt_key"]]
                pred = fair.classify(img).race
                rec.update(race_requested=req, race_pred=pred, race_match=int(pred == req))
            elif r["category"] == "age":
                req = int(r["prompt_key"].split("_")[1])
                pred = AGE_MIDPOINTS.get(fair.classify(img).age_bucket, float("nan"))
                rec.update(age_requested=req, age_pred=pred, age_abs_err=abs(pred - req))
            elif r["category"] == "accessory":
                pos, neg = ACCESSORY_TEXT[r["prompt_key"]]
                rec["accessory_prob"] = clip.zero_shot(img, [pos, neg])[0]
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 200 == 0:
                print(f"  [base] {i + 1}/{len(todo)}")
    print(f"[base] scored -> {SCORES}")


def score_extra() -> None:
    """Heavier per-image metrics: AdaFace, HPSv2, PickScore, DreamSim."""
    ensure_dirs()
    from dcfaces.metrics.adaface import AdaFaceScorer
    from dcfaces.metrics.aesthetic import HPSScorer, PickScorer
    from dcfaces.metrics.identity import IdentityScorer
    from dcfaces.metrics.perceptual import DreamSimScorer

    rows = load_jsonl(MANIFEST)
    done = {r["path"] for r in load_jsonl(SCORES_EXTRA)} if SCORES_EXTRA.exists() else set()
    todo = [r for r in rows if r["path"] not in done]
    print(f"[extra] {len(rows)} images, {len(done)} scored, {len(todo)} to do")
    if not todo:
        return
    ident = IdentityScorer()
    ada = AdaFaceScorer(device="cuda")
    hps = HPSScorer()
    pick = PickScorer(device="cuda")
    dsim = DreamSimScorer(device="cuda")

    ref_ada, ref_img = {}, {}
    for d in sorted(IDENTITY_GALLERY.glob("id_*")):
        if (d / "ref.jpg").exists():
            det = ident.detect(Image.open(d / "ref.jpg"))
            ref_ada[d.name] = ada.embed(det[1]) if det else None
            ref_img[d.name] = Image.open(d / "ref.jpg").convert("RGB")

    with SCORES_EXTRA.open("a") as out:
        for i, r in enumerate(todo):
            img = Image.open(BENCHMARK_DIR / r["path"]).convert("RGB")
            det = ident.detect(img)
            ada_cos = ada.cosine(ada.embed(det[1]), ref_ada.get(r["id"])) if det else float("nan")
            rec = {
                "path": r["path"], "adaface_cos": ada_cos,
                "hpsv2": hps.score(img, r["prompt"]), "pickscore": pick.score(img, r["prompt"]),
                "dreamsim": dsim.distance(img, ref_img[r["id"]]),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 200 == 0:
                print(f"  [extra] {i + 1}/{len(todo)}")
    print(f"[extra] scored -> {SCORES_EXTRA}")


def compute_distribution() -> None:
    """Per-method CLIP-FID + KID on face crops vs the FFHQ test set."""
    import pandas as pd

    from dcfaces.metrics.distribution import FaceCropper, compute_fid_kid

    cropper = FaceCropper()
    ref_dir = CROP_ROOT / "_ffhq_ref"
    if not ref_dir.exists() or not any(ref_dir.glob("*.png")):
        ref_dir.mkdir(parents=True, exist_ok=True)
        for i, rr in enumerate(load_jsonl(FFHQ_TEST)):
            c = cropper.crop(Image.open(PROJECT_ROOT / rr["image_path"]))
            if c is not None:
                c.save(ref_dir / f"{i:05d}.png")
        print(f"[dist] cropped {len(list(ref_dir.glob('*.png')))} FFHQ reference faces")

    gen_root = BENCHMARK_DIR / "generations"
    out_rows = []
    for mdir in sorted(p for p in gen_root.glob("*") if p.is_dir()):
        method = mdir.name
        cdir = CROP_ROOT / method
        gens = sorted(mdir.rglob("*.png"))
        if not cdir.exists() or len(list(cdir.glob("*.png"))) < len(gens) * 0.8:
            cdir.mkdir(parents=True, exist_ok=True)
            for j, png in enumerate(gens):
                c = cropper.crop(Image.open(png))
                if c is not None:
                    c.save(cdir / f"{j:05d}.png")
            print(f"[dist] {method}: cropped {len(list(cdir.glob('*.png')))}/{len(gens)} faces")
        clip_fid, kid = compute_fid_kid(cdir, ref_dir)
        out_rows.append({"method": method, "clip_fid": clip_fid, "kid": kid})
        print(f"[dist] {method}: CLIP-FID {clip_fid:.2f}  KID {kid:.2f}")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).set_index("method").to_csv(TABLES_DIR / "distribution.csv")


def aggregate():
    import pandas as pd

    df = pd.read_json(SCORES, lines=True)
    if SCORES_EXTRA.exists():
        df = df.merge(pd.read_json(SCORES_EXTRA, lines=True), on="path", how="left")

    g = df.groupby("method")
    table = pd.DataFrame({
        "n": g.size(),
        "face_rate": g["face_detected"].mean(),
        "arcface": g["identity_cos"].mean(),
        "clip_score": g["clip_score"].mean(),
    })
    for col in ("adaface_cos", "hpsv2", "pickscore", "dreamsim"):
        if col in df:
            table[col.replace("_cos", "")] = g[col].mean()
    table["race_acc"] = df[df.category == "race"].groupby("method")["race_match"].mean()
    # age control: prefer MiVOLO continuous age (scripts/09b), fall back to the
    # FairFace age-bucket midpoint where MiVOLO found no face.
    age_df = df[df.category == "age"].copy()
    mivolo_path = BENCHMARK_DIR / "mivolo_ages.json"
    if mivolo_path.exists():
        mv = json.loads(mivolo_path.read_text())
        age_df["mivolo_err"] = (age_df["path"].map(mv) - age_df["age_requested"]).abs()
        age_df["age_err"] = age_df["mivolo_err"].fillna(age_df["age_abs_err"])
        table["age_mae"] = age_df.groupby("method")["age_err"].mean()
        table["age_mae_fairface"] = age_df.groupby("method")["age_abs_err"].mean()
    else:
        table["age_mae"] = age_df.groupby("method")["age_abs_err"].mean()
    table["accessory_prob"] = df[df.category == "accessory"].groupby("method")["accessory_prob"].mean()
    table["age_control"] = 1 - (table["age_mae"] / 30).clip(0, 1)
    table["controllability"] = table[["race_acc", "accessory_prob", "age_control"]].mean(axis=1)

    dist_path = TABLES_DIR / "distribution.csv"
    if dist_path.exists():
        table = table.join(pd.read_csv(dist_path, index_col="method"))
    div_path = TABLES_DIR / "diversity.csv"
    if div_path.exists():
        table = table.join(pd.read_csv(div_path, index_col="method"))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "headline.csv")
    by_race = df[df.category == "race"].groupby(["method", "race_requested"])["race_match"].mean().unstack()
    by_race.to_csv(TABLES_DIR / "race_by_class.csv")

    print("\n=== headline (per method) ===")
    print(table.round(3).to_string())
    print("\n=== race accuracy by requested class ===")
    print(by_race.round(2).to_string())
    return table


def pareto_plot(table) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # use the de-circularized AdaFace identity when available, else ArcFace
    xcol = "adaface" if "adaface" in table else "arcface"
    x, y = table[xcol], table["controllability"]
    nd = [m for m in table.index if not ((x > x[m]) & (y > y[m])).any()]
    order = sorted(nd, key=lambda m: x[m])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y, s=70, zorder=3)
    ax.plot(x[order], y[order], "--", color="crimson", zorder=2, label="Pareto frontier")
    for m in table.index:
        ax.annotate(m, (x[m], y[m]), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel(f"Identity preservation  ({xcol} cosine →)")
    ax.set_ylabel("Demographic controllability  (composite →)")
    ax.set_title("Identity vs controllability — Pareto frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "identity_vs_control_pareto.png", dpi=130)
    print(f"\nPareto frontier ({', '.join(order)}) -> {FIGURES_DIR / 'identity_vs_control_pareto.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate-only", action="store_true", help="Skip scoring; re-draw tables/plot.")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-extra", action="store_true")
    parser.add_argument("--skip-dist", action="store_true")
    args = parser.parse_args()
    if not args.aggregate_only:
        if not args.skip_base:
            score_all()
        if not args.skip_extra:
            score_extra()
        if not args.skip_dist:
            compute_distribution()
    pareto_plot(aggregate())


if __name__ == "__main__":
    main()
