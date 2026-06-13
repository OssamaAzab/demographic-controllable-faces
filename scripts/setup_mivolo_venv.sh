#!/usr/bin/env bash
# MiVOLO (the continuous-age metric) depends on timm 0.8, which conflicts with the
# main environment's timm 1.0, so it lives in its own CPU-only virtualenv. Created
# once; scripts/09b_mivolo_age.py runs in it as a subprocess. Pins match MiVOLO's
# upstream requirements; torch is the CPU build (age scoring is not GPU-bound and
# this keeps the isolated env off the main CUDA stack).
#
# Run from the repo root after scripts/setup_external.sh (it imports from
# external/MiVOLO via PYTHONPATH).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m venv .venv-mivolo
.venv-mivolo/bin/pip install --quiet --upgrade pip
.venv-mivolo/bin/pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cpu
.venv-mivolo/bin/pip install \
  timm==0.8.13.dev0 ultralytics==8.1.0 lapx opencv-python-headless omegaconf huggingface_hub

echo "MiVOLO venv ready: .venv-mivolo"
echo "run with: .venv-mivolo/bin/python scripts/09b_mivolo_age.py"
