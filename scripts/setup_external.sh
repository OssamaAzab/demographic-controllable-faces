#!/usr/bin/env bash
# Clone the external identity/metric method repos at the commits this benchmark was
# built against, apply the one local patch they need, and install two small
# compatibility shims into the active virtualenv. Idempotent: safe to re-run.
#
# Run from the repo root with the main .venv active:
#     source .venv/bin/activate
#     scripts/setup_external.sh
set -euo pipefail
cd "$(dirname "$0")/.."

clone() {  # name url commit
  local name=$1 url=$2 commit=$3 dst="external/$1"
  if [ -d "$dst/.git" ]; then
    echo "external/$name present, fetching"
    git -C "$dst" fetch --quiet origin || true
  else
    git clone --quiet "$url" "$dst"
  fi
  git -C "$dst" checkout --quiet "$commit"
  echo "external/$name @ $commit"
}

mkdir -p external
clone PuLID             https://github.com/ToTheBeginning/PuLID.git        1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa
clone PhotoMaker        https://github.com/TencentARC/PhotoMaker.git       060b4fcb10b76a4554edf565d6106b7e36c968f0
clone IP-Adapter        https://github.com/tencent-ailab/IP-Adapter.git    62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2
clone ComfyUI-HyperLoRA https://github.com/bytedance/ComfyUI-HyperLoRA.git 108d4c32eb6bb77d386a6fb1a3d05d6826df8bcd
clone AdaFace           https://github.com/mk-minchul/AdaFace.git          c60eaa786a42c03444f3df7096dbaf9d57ae010d
clone MiVOLO            https://github.com/WildChlamydia/MiVOLO.git        37475e3f8818b5f22448003feec3e64b01bfb188

# PhotoMaker passes resume_download to a diffusers helper that no longer accepts it.
PATCH=patches/photomaker-pipeline-resume-download.patch
if patch -p1 -R --dry-run -s -d external/PhotoMaker < "$PATCH" >/dev/null 2>&1; then
  echo "PhotoMaker patch already applied"
else
  patch -p1 -d external/PhotoMaker < "$PATCH"
  echo "applied $PATCH"
fi

# basicsr 1.4.2 imports torchvision.transforms.functional_tensor, removed in tv 0.17.
TV_DIR=$(python -c "import torchvision.transforms as t, pathlib; print(pathlib.Path(t.__file__).parent)")
SHIM="$TV_DIR/functional_tensor.py"
if [ -f "$SHIM" ]; then
  echo "torchvision functional_tensor shim present"
else
  cat > "$SHIM" <<'PY'
"""Compatibility shim: torchvision removed this module in 0.17, but basicsr 1.4.2
still imports rgb_to_grayscale from here. Re-export it from its new home."""

from torchvision.transforms.functional import rgb_to_grayscale

__all__ = ["rgb_to_grayscale"]
PY
  echo "wrote torchvision functional_tensor shim"
fi

# hpsv2 vendors open_clip but ships without its BPE vocab; copy it from open_clip.
python - <<'PY'
import pathlib
import shutil

import hpsv2
import open_clip

dst = pathlib.Path(hpsv2.__file__).parent / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
if dst.exists():
    print("hpsv2 open_clip vocab present")
else:
    src = pathlib.Path(open_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print("copied hpsv2 open_clip vocab")
PY

echo "external setup done"
