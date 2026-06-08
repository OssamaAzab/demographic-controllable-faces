"""Separate model failure from FairFace measurement error on race control.

The winner LoRA scores low on Latino (0.44) and Southeast Asian (0.03) race
control. Both are FairFace's weakest classes, so the low scores might come from
the classifier, not the model. Two checks:

  1. Classifier ceiling. Run FairFace on the FairFace validation split (human
     labels, independent of our captions) and report per-class accuracy plus the
     confusion. This is the most the metric can score on real faces of each class.
     The caption labels are not used as truth -- they came from FairFace, so that
     would be circular.
  2. Visual. Save a contact sheet of the winner's generations per race to eyeball.

A class whose ceiling is itself low (and whose generations look right) points to a
measurement artifact; a high ceiling with low generation accuracy points to the model.

Inputs:  results/per_race_breakdown.json, the winning checkpoint, HuggingFaceM4/FairFace (val)
Outputs: results/race_ceiling.csv, results/figures/sample_<race>.png
"""

from __future__ import annotations

import argparse
import collections
import csv
import json

from dcfaces.paths import DEMO_LORA_CKPTS, FIGURES_DIR, RESULTS_DIR, ensure_dirs

RACES = ["white", "Black", "East Asian", "Southeast Asian", "South Asian", "Middle Eastern", "Latino"]


def classifier_ceiling(limit: int):
    """Per-class accuracy and confusion of FairFace on the human-labeled val split."""
    from datasets import load_dataset

    from dcfaces.demographics import RACE_CANON, FairFaceClassifier

    ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="validation")
    if limit:
        ds = ds.select(range(limit))
    names = ds.features["race"].names
    clf = FairFaceClassifier()

    correct, total = collections.Counter(), collections.Counter()
    confusion = collections.defaultdict(collections.Counter)
    batch = 64
    for start in range(0, len(ds), batch):
        chunk = ds[start : start + batch]
        preds = clf.classify_batch(chunk["image"])
        for true_idx, fr in zip(chunk["race"], preds):
            true = RACE_CANON.get(names[true_idx], names[true_idx])
            total[true] += 1
            correct[true] += int(fr.race == true)
            confusion[true][fr.race] += 1
        print(f"  ceiling {min(start + batch, len(ds))}/{len(ds)}", end="\r", flush=True)
    print()
    ceiling = {r: correct[r] / total[r] for r in RACES if total[r]}
    return ceiling, total, confusion


def contact_sheets(ages, seed):
    """Save one generated contact sheet per race for visual inspection."""
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    from PIL import Image, ImageDraw

    winner = next(
        int(r["step"]) for r in csv.DictReader(open(RESULTS_DIR / "checkpoint_selection.csv"))
        if r["winner"] == "True"
    )
    ckpt = DEMO_LORA_CKPTS / f"step_{winner}"
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.load_lora_weights(str(ckpt))

    tmpl = "a photo of a {age} year old {race} {gender}, professional portrait, studio lighting, neutral expression"
    genders = ["man", "woman"]
    cell = 256
    for race in RACES:
        cells = []
        for gender in genders:
            for age in ages:
                g = torch.Generator(device="cuda").manual_seed(seed)
                img = pipe(
                    prompt=tmpl.format(age=age, race=race, gender=gender),
                    num_inference_steps=30, guidance_scale=6.0, height=1024, width=1024, generator=g,
                ).images[0].resize((cell, cell))
                ImageDraw.Draw(img).text((4, 4), f"{age} {gender}", fill="yellow")
                cells.append(img)
        cols, rows = len(ages), len(genders)
        sheet = Image.new("RGB", (cols * cell, rows * cell + 20), "black")
        ImageDraw.Draw(sheet).text((4, 4), f"prompted: {race}", fill="white")
        for i, img in enumerate(cells):
            sheet.paste(img, ((i % cols) * cell, 20 + (i // cols) * cell))
        out = FIGURES_DIR / f"sample_{race.replace(' ', '_')}.png"
        sheet.save(out)
        print(f"  {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="Cap val images for the ceiling (0=all).")
    parser.add_argument("--no-sheets", action="store_true", help="Skip the generated contact sheets.")
    args = parser.parse_args()
    ensure_dirs()

    gen = {k: v["acc"] for k, v in json.load(open(RESULTS_DIR / "per_race_breakdown.json")).items()}

    print("Classifier ceiling on the FairFace validation split:")
    ceiling, total, confusion = classifier_ceiling(args.limit)

    if not args.no_sheets:
        print("\nContact sheets (winner generations):")
        contact_sheets(ages=[25, 50, 75], seed=42)

    rows = []
    print(f"\n{'race':16}{'gen_acc':>9}{'ceiling':>9}{'n_real':>8}  top real-face confusion / reading")
    for race in RACES:
        c = ceiling.get(race, float("nan"))
        g = gen.get(race, float("nan"))
        wrong = [(p, n) for p, n in confusion[race].most_common() if p != race]
        top = f"{wrong[0][0]} {wrong[0][1] / total[race] * 100:.0f}%" if wrong else "-"
        # compare generation to the real-face ceiling, not the ceiling alone
        gap = g - c
        if gap >= -0.1:
            reading = "generation at/above ceiling - control good"
        elif gap <= -0.3:
            reading = "generation far below ceiling - model failure"
        elif c < 0.65:
            reading = "below ceiling, ceiling itself low - mostly measurement"
        else:
            reading = "below ceiling - mixed"
        print(f"{race:16}{g:9.2f}{c:9.2f}{total[race]:8}  {top:14} {reading}")
        rows.append({
            "race": race, "generation_acc": round(g, 3), "classifier_ceiling": round(c, 3),
            "n_real_val": total[race], "top_real_confusion": top, "reading": reading,
        })

    out = RESULTS_DIR / "race_ceiling.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{out}")


if __name__ == "__main__":
    main()
