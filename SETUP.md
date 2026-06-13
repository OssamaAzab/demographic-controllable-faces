# Setup and reproduction

The repo tracks code, configs, and results (CSVs + figures). It does not track the
external method repos, the model weights, or the datasets — those are fetched by the
steps below. This file is the full path from a fresh clone to a runnable benchmark.

## Prerequisites

- Linux, CUDA 12, Python 3.12
- One GPU with 20 GB+ VRAM (developed on an RTX 4000 Ada)
- ~320 GB free disk (FFHQ, weights, and the ~7k generated images)
- `git` and the `patch` utility

All data/model/output paths auto-detect from the repo root via
`src/dcfaces/paths.py`; HuggingFace caches stay in-repo under `.cache/`. Nothing is
written outside the clone.

## 1. Main environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements.txt
```

The metric stack (insightface, clean-fid, basicsr, hpsv2, dreamsim) is version-sensitive
— newer numpy/timm break onnxruntime and the FairFace head. Install those packages
against the pinned constraints so the resolver cannot drift:

```bash
pip install dreamsim hpsv2 basicsr -c .metric_constraints.txt
```

## 2. External method repos

PuLID, PhotoMaker, IP-Adapter, HyperLoRA, AdaFace, and MiVOLO are imported from
`external/`, not vendored. Clone them at the pinned commits and apply the one patch
they need:

```bash
scripts/setup_external.sh
```

This clones each repo at the exact commit the benchmark was built against, applies
`patches/photomaker-pipeline-resume-download.patch` (PhotoMaker passes a kwarg that
current diffusers rejected), and installs two compatibility shims into the venv (the
`torchvision.transforms.functional_tensor` re-export basicsr still imports, and the
open_clip BPE vocab hpsv2 ships without).

## 3. Model weights

```bash
python scripts/00_fetch_weights.py
```

Downloads everything available as a clean HuggingFace pull: IP-Adapter FaceID Plus v2
(+ image encoder), PhotoMaker v2, PickScore, HyperLoRA (and its CLIP/insightface
layout), and RealVisXL v4.0. SDXL 1.0 + the fp16-fix VAE download automatically on
first use.

Three weights need a manual download (license click-through or a non-HF host); the
script prints the sources and target paths when it finishes:

- AdaFace IR-50 (MS1MV2) -> `models/adaface/adaface_ir50_ms1mv2.ckpt`
- MiVOLO age model + person/face detector -> `models/mivolo/`
- insightface antelopev2 -> `external/PuLID/models/antelopev2/` (PuLID also fetches this on first run)

The demographic LoRA and the per-identity DreamBooth LoRAs are trained, not
downloaded (scripts 05 and 07).

## 4. MiVOLO virtualenv (age metric)

MiVOLO needs `timm 0.8`, which conflicts with the main stack's `timm 1.0`, so it runs
in its own CPU venv and is called as a subprocess by `scripts/09b_mivolo_age.py`:

```bash
scripts/setup_mivolo_venv.sh
```

## 5. Run the pipeline

With the environment in place, follow the numbered scripts in the README
("Running the pipeline"). The benchmark run is resumable and the metric scoring caches
per-image results, so partial runs pick up where they left off.

## Troubleshooting

- `from pkg_resources import packaging` ImportError from the `clip` package on newer
  setuptools: `pip install "setuptools<81"`.
- onnxruntime / insightface failing after a fresh install usually means numpy 2.x got
  pulled in — reinstall the metric packages with `-c .metric_constraints.txt`.
- MiVOLO loading a YOLO checkpoint with `weights_only` errors: the venv must use
  `torch==2.4.1` (newer torch defaults to `weights_only=True` and rejects the pickle).
