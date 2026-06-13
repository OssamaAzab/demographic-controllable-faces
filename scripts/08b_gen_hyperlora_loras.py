"""08b: Pre-generate HyperLoRA per-identity ID-LoRAs for the benchmark.

HyperLoRA's hypernetwork predicts an identity LoRA from each reference image in
one forward pass (zero-shot, no training). Run this once before the benchmark;
it writes models/hyperlora_loras/id_*.safetensors, which the `hyperlora` method
in script 08 loads onto RealVisXL.

Requires the gitignored external/ComfyUI-HyperLoRA clone and the assembled
models/hyperlora_root (CLIP-ViT-L + antelopev2 + hyper_lora weights).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dcfaces.paths import IDENTITY_GALLERY, MODELS_DIR, PROJECT_ROOT, ensure_dirs

HL_REPO = PROJECT_ROOT / "external" / "ComfyUI-HyperLoRA"
MODELS_ROOT = MODELS_DIR / "hyperlora_root"
OUT = MODELS_DIR / "hyperlora_loras"


def main() -> None:
    ensure_dirs()
    gen = HL_REPO / "standalone" / "generate_loras.py"
    if not gen.exists():
        raise SystemExit(f"HyperLoRA clone not found at {HL_REPO}")
    if not MODELS_ROOT.exists():
        raise SystemExit(f"HyperLoRA models_dir not assembled at {MODELS_ROOT}")
    ids = sorted(d for d in IDENTITY_GALLERY.glob("id_*") if (d / "ref.jpg").exists())
    if not ids:
        raise SystemExit(f"No identities in {IDENTITY_GALLERY} — run 03_build_identity_gallery.py first.")
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for d in ids:
            shutil.copy(d / "ref.jpg", staging / f"{d.name}.jpg")
        env = {**os.environ, "PYTHONPATH": str(HL_REPO)}
        subprocess.run(
            [sys.executable, str(gen),
             "--models_dir", str(MODELS_ROOT), "--indir", str(staging), "--outdir", str(OUT),
             "--model", "sdxl_hyper_id_lora_v1_fidelity", "--dtype", "fp16", "--device", "cuda"],
            check=True, env=env,
        )
    print(f"done: {len(ids)} identities -> {OUT}")


if __name__ == "__main__":
    main()
