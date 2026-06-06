"""06: Pareto-frontier checkpoint selection for the demographic LoRA.

Inputs:  models/demo_lora_checkpoints/step_*/
         configs/checkpoint_selection.yaml
Outputs: models/demo_lora.safetensors  (winning checkpoint, copied to canonical path)
         results/checkpoint_selection.csv
         results/figures/checkpoint_pareto.png

Procedure:
    For each of the 10 saved checkpoints:
        1. Load EMA weights
        2. Generate 224 images (4 ages × 7 races × 2 genders × 4 seeds)
        3. Score: MiVOLO MAE, FairFace acc, CLIP-Score, HPSv2
    Plot Pareto frontier on (mivolo_mae, fairface_acc).
    Pick the checkpoint on the frontier closest to ideal (low MAE, high acc).
    Copy that checkpoint to models/demo_lora.safetensors.

Why Pareto (not composite score): transparent about the tradeoff. The writeup
can explicitly say "we picked the checkpoint that gives X age MAE for Y race
accuracy", rather than hiding the choice in a weighted composite.

Expected runtime: ~4 hours on 20 GB GPU.
"""

from dcfaces.paths import (
    CONFIG_DIR,
    DEMO_LORA,
    DEMO_LORA_CKPTS,
    FIGURES_DIR,
    RESULTS_DIR,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    config_path = CONFIG_DIR / "checkpoint_selection.yaml"
    print(
        f"TODO: Pareto-eval checkpoints — config={config_path}, "
        f"ckpts={DEMO_LORA_CKPTS}, csv={RESULTS_DIR}/checkpoint_selection.csv, "
        f"figure={FIGURES_DIR}/checkpoint_pareto.png, winner={DEMO_LORA}"
    )


if __name__ == "__main__":
    main()
