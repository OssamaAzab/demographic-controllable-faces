"""05: Train the demographic LoRA on FFHQ + hybrid captions.

Inputs:  data/ffhq_train.jsonl, data/ffhq_val.jsonl, data/ffhq_metadata.jsonl
         configs/demo_lora.yaml
Outputs: models/demo_lora_checkpoints/step_{1000, 2000, ..., 10000}/
         models/demo_lora_checkpoints/step_*/unet_ema/ (EMA weights)

Config reference: configs/demo_lora.yaml

Training loop:
- Standard SDXL LoRA training (PEFT) at rank 32, BF16, gradient checkpointing,
  AdamW8bit, xformers, EMA decay 0.9999
- Saves checkpoint + EMA weights every 1000 steps (10 total)
- Mid-training task eval every 1000 steps (8 prompts × 2 seeds, MiVOLO + FairFace)
  → logged to wandb as the real validation signal (not val-MSE)
- val-MSE logged only for training-stability monitoring

Expected runtime: ~7 hours on 20 GB GPU for 10k steps.
"""

from dcfaces.paths import (
    CONFIG_DIR,
    DEMO_LORA_CKPTS,
    FFHQ_METADATA,
    FFHQ_TRAIN,
    FFHQ_VAL,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    config_path = CONFIG_DIR / "demo_lora.yaml"
    print(
        f"TODO: train demo LoRA — config={config_path}, "
        f"train={FFHQ_TRAIN}, val={FFHQ_VAL}, captions={FFHQ_METADATA}, "
        f"out={DEMO_LORA_CKPTS}"
    )


if __name__ == "__main__":
    main()
