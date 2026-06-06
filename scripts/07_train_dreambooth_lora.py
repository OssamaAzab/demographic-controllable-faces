"""07: Train one per-identity DreamBooth-LoRA for each benchmark identity.

Inputs:  data/identity_gallery/id_*/ref.jpg
         configs/dreambooth_lora.yaml
Outputs: models/dreambooth_loras/id_*/dreambooth.safetensors

Comparison baseline only — these are the "old school" per-identity LoRAs from
Part C. PuLID is the primary method.

For each identity:
    1. Generate ~100 class images of "a photo of a person" (prior preservation)
    2. Train 1000 steps with instance_prompt="a photo of sks person"
    3. Save LoRA weights to models/dreambooth_loras/{id}/dreambooth.safetensors

Expected runtime: ~25 min per identity × 10 = ~4 hours on 20 GB GPU.
"""

from dcfaces.paths import (
    CONFIG_DIR,
    DREAMBOOTH_LORAS,
    IDENTITY_GALLERY,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    config_path = CONFIG_DIR / "dreambooth_lora.yaml"
    print(
        f"TODO: train per-id DreamBooth LoRAs — config={config_path}, "
        f"gallery={IDENTITY_GALLERY}, out={DREAMBOOTH_LORAS}"
    )


if __name__ == "__main__":
    main()
