"""08: Run the full benchmark — 4 methods × 10 identities × ~60 prompts × 10 seeds.

Inputs:  models/demo_lora.safetensors
         models/dreambooth_loras/id_*/
         data/identity_gallery/id_*/
         configs/benchmark.yaml
Outputs: results/benchmark/generations/{method}/{id}/{prompt_key}/seed_*.png
         results/benchmark/manifest.jsonl  (every (method, id, prompt, seed) row)

Methods evaluated (configs/benchmark.yaml):
    - pulid_with_demo_lora   (the contribution)
    - pulid_only             (ablation: PuLID without the demo LoRA)
    - photomaker
    - ip_adapter_faceid
    - dreambooth_lora        (old-school baseline)

Total: ~24k images @ 6 sec each = ~40 hours on 20 GB GPU.
Run in 4-hour chunks (10 chunks) to avoid thermal throttling on the RTX 4000 Ada.
"""

from dcfaces.paths import (
    BENCHMARK_DIR,
    CONFIG_DIR,
    DEMO_LORA,
    DREAMBOOTH_LORAS,
    IDENTITY_GALLERY,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    config_path = CONFIG_DIR / "benchmark.yaml"
    print(
        f"TODO: run benchmark — config={config_path}, "
        f"lora={DEMO_LORA}, db_loras={DREAMBOOTH_LORAS}, "
        f"gallery={IDENTITY_GALLERY}, out={BENCHMARK_DIR}"
    )


if __name__ == "__main__":
    main()
