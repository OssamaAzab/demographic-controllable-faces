"""Build the qualitative delayed-injection rescue figure for the README. The example
cell is chosen from the scored data, not hardcoded: among race prompts, the
(identity, race) cell with the largest race-control gain from the rescue where PuLID
and PuLID+demo both miss the target, PuLID+delayed hits it, and AdaFace identity stays
high across all three. The shown seed is the representative shared seed (clean rescue
pattern, closest to the cell means). Writes results/figures/qualitative_rescue.png.
"""

from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

from dcfaces.paths import BENCHMARK_DIR, FIGURES_DIR, IDENTITY_GALLERY

METHODS = ["pulid_only", "pulid_with_demo_lora", "pulid_delayed"]
LABELS = {"pulid_only": "PuLID", "pulid_with_demo_lora": "PuLID + demo LoRA",
          "pulid_delayed": "PuLID + delayed injection"}
GEN = BENCHMARK_DIR / "generations"
PANEL, GAP, LABEL_H, SCORE_H, MARGIN = 512, 10, 44, 60, 12
GRAY, RED, GREEN = (85, 85, 85), (158, 59, 59), (47, 107, 58)


def load():
    base = [json.loads(line) for line in (BENCHMARK_DIR / "scores.jsonl").read_text().splitlines()]
    ada = {}
    for line in (BENCHMARK_DIR / "scores_extra.jsonl").read_text().splitlines():
        r = json.loads(line)
        ada[r["path"]] = r.get("adaface_cos")
    return base, ada


def select_cell(base, ada):
    cells = defaultdict(lambda: defaultdict(dict))
    for r in base:
        if r["category"] == "race" and r["method"] in METHODS:
            cells[(r["id"], r["prompt_key"], r["race_requested"])][r["method"]][r["seed"]] = {
                "match": r["race_match"], "pred": r["race_pred"], "ada": ada.get(r["path"]),
            }
    best = None
    for cell, md in cells.items():
        if not all(m in md for m in METHODS):
            continue
        race = {m: mean(v["match"] for v in md[m].values()) for m in METHODS}
        idn = {m: mean(v["ada"] for v in md[m].values() if v["ada"]) for m in METHODS}
        gain = race["pulid_delayed"] - max(race["pulid_only"], race["pulid_with_demo_lora"])
        clean = [s for s in md["pulid_only"]
                 if md["pulid_only"][s]["match"] == 0
                 and md["pulid_with_demo_lora"][s]["match"] == 0
                 and md["pulid_delayed"][s]["match"] == 1]
        if not (race["pulid_only"] <= 0.4 and race["pulid_with_demo_lora"] <= 0.4
                and race["pulid_delayed"] >= 0.6 and min(idn.values()) >= 0.5 and clean):
            continue
        # representative shared seed: clean pattern, closest to per-method mean identity
        seed = min(clean, key=lambda s: sum(abs(md[m][s]["ada"] - idn[m]) for m in METHODS))
        key = (gain, min(idn.values()))
        if best is None or key > best[0]:
            best = (key, cell, seed, md)
    return best[1], best[2], best[3]


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def centered(draw, cx, y, text, fnt, fill):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


def main():
    base, ada = load()
    (idn, pkey, target), seed, md = select_cell(base, ada)

    print(f"selected cell: {idn} / {pkey}  target={target}  seed={seed}")
    for m in METHODS:
        v = md[m][seed]
        print(f"  {m:22s} race_pred={v['pred']:14s} match={v['match']}  adaface={v['ada']:.3f}")

    panels = [{"name": "Reference", "img": IDENTITY_GALLERY / idn / "ref.jpg",
               "band": GRAY, "l1": "target race", "l2": target}]
    for m in METHODS:
        v = md[m][seed]
        hit = v["match"] == 1
        panels.append({"name": LABELS[m], "img": GEN / m / idn / pkey / f"seed_{seed}.png",
                       "band": GREEN if hit else RED,
                       "l1": f"{v['pred']} ({'hit' if hit else 'miss'})",
                       "l2": f"identity {v['ada']:.2f}"})

    n = len(panels)
    W = MARGIN * 2 + n * PANEL + (n - 1) * GAP
    H = MARGIN * 2 + LABEL_H + PANEL + SCORE_H
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f_lab, f_sc = font(24), font(22)

    for i, p in enumerate(panels):
        x = MARGIN + i * (PANEL + GAP)
        cx = x + PANEL / 2
        draw.rectangle([x, MARGIN, x + PANEL, MARGIN + LABEL_H], fill=p["band"])
        centered(draw, cx, MARGIN + 9, p["name"], f_lab, "white")
        img = Image.open(p["img"]).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)
        canvas.paste(img, (x, MARGIN + LABEL_H))
        y = MARGIN + LABEL_H + PANEL + 8
        centered(draw, cx, y, p["l1"], f_sc, "black")
        centered(draw, cx, y + 26, p["l2"], f_sc, (90, 90, 90))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "qualitative_rescue.png"
    canvas.save(out)
    print(f"wrote {out}  ({W}x{H})")


if __name__ == "__main__":
    main()
