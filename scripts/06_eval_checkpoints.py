"""06: Pareto-frontier checkpoint selection for the demographic LoRA.

For each saved checkpoint (EMA LoRA from 05): generate the eval grid
(4 ages x 7 races x 2 genders x 4 seeds = 224 imgs), score age MAE + race acc
(FairFace) and CLIP-Score, then pick the checkpoint on the **Pareto frontier** of
(age_mae down, race_acc up) closest to the ideal corner. Copy it to the canonical
models/demo_lora.safetensors. Pareto (not a composite) keeps the identity-vs-
control trade-off transparent (PROJECT_PLAN sec.8).

Metric note: age uses the FairFace age head (MiVOLO stand-in; GDrive-only). HPSv2
is deferred (not a Pareto axis). CLIP-Score is reported for context.

Config:  configs/checkpoint_selection.yaml
Inputs:  models/demo_lora_checkpoints/step_*/pytorch_lora_weights.safetensors
Outputs: models/demo_lora.safetensors        (winning checkpoint)
         results/checkpoint_selection.csv
         results/figures/checkpoint_pareto.png

Smoke-test (few prompts, 2 checkpoints):
    python scripts/06_eval_checkpoints.py --max-prompts 4 --max-checkpoints 2
"""

from __future__ import annotations

import argparse
import csv
import shutil

import yaml

from dcfaces.paths import (
    CONFIG_DIR,
    DEMO_LORA,
    DEMO_LORA_CKPTS,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
)

RACES_CANON = ["white", "Black", "East Asian", "Southeast Asian", "South Asian", "Middle Eastern", "Latino"]


def build_prompt_grid(cfg) -> list[dict]:
    ep = cfg["eval_prompts"]
    grid = []
    for age in ep["ages"]:
        for race in ep["races"]:
            for gender in ep["genders"]:
                prompt = ep["template"].format(age=age, race=race, gender=gender)
                for seed in ep["seeds"]:
                    grid.append({"prompt": prompt, "age": age, "race": race, "gender": gender, "seed": seed})
    return grid


def pareto_frontier(rows: list[dict]) -> list[int]:
    """Indices of non-dominated rows: minimize age_mae, maximize race_acc."""
    front = []
    for i, a in enumerate(rows):
        dominated = any(
            b is not a
            and b["age_mae"] <= a["age_mae"]
            and b["race_acc"] >= a["race_acc"]
            and (b["age_mae"] < a["age_mae"] or b["race_acc"] > a["race_acc"])
            for b in rows
        )
        if not dominated:
            front.append(i)
    return front


def pick_winner(rows, frontier_idx) -> int:
    """On the frontier, pick closest to ideal (min age_mae_norm, max race_acc_norm)."""
    maes = [r["age_mae"] for r in rows]
    accs = [r["race_acc"] for r in rows]
    mae_lo, mae_hi = min(maes), max(maes)
    acc_lo, acc_hi = min(accs), max(accs)

    def norm(x, lo, hi):
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    best_i, best_d = frontier_idx[0], float("inf")
    for i in frontier_idx:
        mae_n = norm(rows[i]["age_mae"], mae_lo, mae_hi)  # ideal 0
        acc_n = norm(rows[i]["race_acc"], acc_lo, acc_hi)  # ideal 1
        d = (mae_n**2 + (1 - acc_n) ** 2) ** 0.5
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-prompts", type=int, default=0, help="Cap generations/ckpt (0=all); smoke.")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="Cap #checkpoints (0=all); smoke.")
    args = parser.parse_args()

    ensure_dirs()
    cfg = yaml.safe_load((CONFIG_DIR / "checkpoint_selection.yaml").read_text())

    checkpoints = sorted(DEMO_LORA_CKPTS.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    checkpoints = [c for c in checkpoints if (c / "pytorch_lora_weights.safetensors").exists()]
    if not checkpoints:
        raise SystemExit(f"No checkpoints in {DEMO_LORA_CKPTS} — run 05 first.")
    if args.max_checkpoints:
        checkpoints = checkpoints[: args.max_checkpoints]

    grid = build_prompt_grid(cfg)
    if args.max_prompts:
        grid = grid[: args.max_prompts]
    print(f"{len(checkpoints)} checkpoints x {len(grid)} generations each")

    # ---- heavy imports / models ----
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    from dcfaces.demographics import FairFaceClassifier
    from dcfaces.metrics.clip_score import CLIPScorer
    from dcfaces.training.eval import AGE_MIDPOINTS

    base = cfg.get("base", "stabilityai/stable-diffusion-xl-base-1.0")
    vae_id = cfg.get("vae", "madebyollin/sdxl-vae-fp16-fix")
    steps = cfg["inference"]["steps"]
    guidance = cfg["inference"]["guidance_scale"]
    res = cfg["inference"]["resolution"]

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    classifier = FairFaceClassifier()
    clip = CLIPScorer()

    rows = []
    for ckpt in checkpoints:
        step = int(ckpt.name.split("_")[1])
        pipe.load_lora_weights(str(ckpt))
        age_errs, race_hits, clips = [], 0, []
        for item in grid:
            g = torch.Generator(device="cuda").manual_seed(item["seed"])
            image = pipe(
                prompt=item["prompt"], num_inference_steps=steps,
                guidance_scale=guidance, height=res, width=res, generator=g,
            ).images[0]
            fr = classifier.classify(image)
            if fr.age_bucket in AGE_MIDPOINTS:
                age_errs.append(abs(AGE_MIDPOINTS[fr.age_bucket] - item["age"]))
            race_hits += int(fr.race == item["race"])
            clips.append(clip.score(image, item["prompt"]))
        pipe.unload_lora_weights()
        row = {
            "step": step,
            "age_mae": sum(age_errs) / len(age_errs) if age_errs else float("nan"),
            "race_acc": race_hits / len(grid),
            "clip_score": sum(clips) / len(clips),
            "n": len(grid),
        }
        rows.append(row)
        print(f"  step {step}: age_mae={row['age_mae']:.2f} race_acc={row['race_acc']:.3f} clip={row['clip_score']:.3f}")

    # ---- Pareto select ----
    frontier = pareto_frontier(rows)
    winner_i = pick_winner(rows, frontier)
    winner = rows[winner_i]
    for i, r in enumerate(rows):
        r["frontier"] = i in frontier
        r["winner"] = i == winner_i

    # copy winner to canonical path
    win_ckpt = DEMO_LORA_CKPTS / f"step_{winner['step']}" / "pytorch_lora_weights.safetensors"
    shutil.copy(win_ckpt, DEMO_LORA)
    print(f"\nWINNER: step {winner['step']} "
          f"(age_mae={winner['age_mae']:.2f}, race_acc={winner['race_acc']:.3f}) -> {DEMO_LORA}")

    # ---- CSV ----
    csv_path = RESULTS_DIR / "checkpoint_selection.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "age_mae", "race_acc", "clip_score", "n", "frontier", "winner"])
        w.writeheader()
        w.writerows(rows)
    print(f"metrics -> {csv_path}")

    # ---- Pareto plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([r["age_mae"] for r in rows], [r["race_acc"] for r in rows], c="lightgray", label="checkpoints")
        fr = [rows[i] for i in sorted(frontier, key=lambda i: rows[i]["age_mae"])]
        ax.plot([r["age_mae"] for r in fr], [r["race_acc"] for r in fr], "o-", color="tab:blue", label="Pareto frontier")
        ax.scatter([winner["age_mae"]], [winner["race_acc"]], marker="*", s=320, color="tab:red", zorder=5, label=f"winner (step {winner['step']})")
        for r in rows:
            ax.annotate(str(r["step"]), (r["age_mae"], r["race_acc"]), fontsize=7, alpha=0.7)
        ax.set_xlabel("age MAE (lower better)")
        ax.set_ylabel("race accuracy (higher better)")
        ax.set_title("Demographic LoRA — checkpoint Pareto frontier")
        ax.legend()
        fig.tight_layout()
        fig_path = FIGURES_DIR / "checkpoint_pareto.png"
        fig.savefig(fig_path, dpi=150)
        print(f"figure -> {fig_path}")
    except Exception as e:  # noqa: BLE001
        print(f"plot skipped ({e})")


if __name__ == "__main__":
    main()
