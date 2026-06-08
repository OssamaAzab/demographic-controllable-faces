# Demographic-Controllable Identity-Preserving Face Generation

A reproducible benchmark of identity-preserving face generation methods with controllable demographic attributes (age, ethnicity) and accessory attributes (glasses, expression, hairstyle), built on **SDXL + PuLID + a custom demographic LoRA trained on FFHQ**.

> *Hero figure will be added after the benchmark run.*

## TL;DR results

| Method | Identity (AdaFace ↑) | Age control (MiVOLO MAE ↓) | Race control (FairFace acc ↑) | Quality (HPSv2 ↑) |
|---|---|---|---|---|
| PuLID + demographic LoRA | *tbd* | *tbd* | *tbd* | *tbd* |
| PuLID (no LoRA) | *tbd* | *tbd* | *tbd* | *tbd* |
| PhotoMaker v2 | *tbd* | *tbd* | *tbd* | *tbd* |
| IP-Adapter FaceID v2 | *tbd* | *tbd* | *tbd* | *tbd* |
| DreamBooth-LoRA (per ID) | *tbd* | *tbd* | *tbd* | *tbd* |

Full results in [WRITEUP.md](WRITEUP.md). Live demo: *to be added*.

---

## What this project does

Given a **reference photo of a target person** and a **text prompt with demographic / accessory attributes**, generate an image of that specific person rendered with the requested attributes:

```
ref:  [photo of Alice]
prompt: "a photo of a 70 year old East Asian woman wearing round glasses, smiling"

→ output: Alice, rendered as a 70-year-old East Asian woman with round glasses,
  smiling, in studio-portrait quality.
```

The contribution: a **demographic LoRA fine-tuned on FFHQ + hybrid BLIP-2 / FairFace captions** that makes SDXL precisely responsive to demographic tokens, used alongside **PuLID** for identity preservation. Evaluated against three other identity methods on a 10-axis metric suite.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full design doc.

---

## Demographic LoRA — results

The demographic LoRA is trained and selected (the method benchmark is still to
run). It is a rank-32 SDXL LoRA on 36k FFHQ images with hybrid FairFace + BLIP-2
captions. Checkpoints are chosen on a task-aware mid-training eval — the
(age MAE, race accuracy) Pareto frontier — not validation MSE. The model overfits
past ~4k steps, so selection kept step 4000 over the later, worse checkpoints
([results/figures/checkpoint_pareto.png](results/figures/checkpoint_pareto.png)).

Per-race control on the selected checkpoint:

| Race | white | Black | East Asian | South Asian | Middle Eastern | Latino | Southeast Asian |
|---|---|---|---|---|---|---|---|
| Race accuracy | 1.00 | 1.00 | 1.00 | 0.94 | 0.78 | 0.44 | 0.03 |

Two classes score low for different reasons. **Southeast Asian is a model
limitation**: the token collapses to the East Asian prior (generations score 0.03
against a 0.65 FairFace ceiling on real faces). **Latino is mostly a measurement
artifact**: FairFace reaches only ~0.60 on real Latino faces, and the generations
sit just below that. The disambiguation method and evidence are in
[WRITEUP.md](WRITEUP.md) (`results/race_ceiling.csv`,
`results/figures/sample_*.png`, `results/per_race_breakdown.json`).

---

## Setup

Tested on Linux + CUDA 12 + Python 3.10+, single GPU with **20 GB+ VRAM** (developed on an RTX 4000 Ada).

```bash
git clone https://github.com/USERNAME/demographic-controllable-faces.git
cd demographic-controllable-faces

python -m venv .venv
source .venv/bin/activate

make install              # pip install -e . + pip install -r requirements.txt
cp .env.example .env      # edit if needed (HF_TOKEN, GPU device)
```

**No path configuration required.** All data, model, and output paths auto-detect from the repo root via `src/dcfaces/paths.py`.

---

## Running the pipeline

End-to-end (one command, ~60 hours total wall-clock):

```bash
make all
```

Or stage-by-stage:

```bash
make setup     # download FFHQ, build splits, curate identity gallery (~1h)
make caption   # BLIP-2 + FairFace captioning of 30k FFHQ images (~10h)
make train     # demographic LoRA + per-identity LoRAs + checkpoint selection (~9h)
make bench     # 24k generations across 4 methods × 10 identities (~40h)
```

Or individual scripts in order:

```bash
python scripts/01_download_ffhq.py
python scripts/02_split_ffhq.py
python scripts/03_build_identity_gallery.py
python scripts/04_caption_ffhq.py
python scripts/05_train_demo_lora.py
python scripts/06_eval_checkpoints.py
python scripts/07_train_dreambooth_lora.py
python scripts/08_run_benchmark.py
python scripts/09_compute_metrics.py
```

---

## Repository structure

```
demographic-controllable-faces/
├── README.md                       # this file
├── PROJECT_PLAN.md                 # full design doc with every decision explained
├── WRITEUP.md                      # narrative + findings + limitations + ethics
├── LICENSE                         # MIT
├── pyproject.toml                  # makes src/dcfaces/ pip-installable
├── requirements.txt                # pinned versions
├── Makefile                        # pipeline entry points
├── .env.example                    # template for environment variables
├── .gitignore                      # excludes data/, models/, results/, .cache/
├── configs/                        # all hyperparameters as YAML
├── src/dcfaces/                    # core importable package
│   ├── paths.py                    # single source of truth for I/O paths
│   ├── data/                       # FFHQ + identity gallery loaders
│   ├── captioning/                 # BLIP-2 + FairFace
│   ├── training/                   # LoRA training loops
│   ├── inference/                  # PuLID / PhotoMaker / IP-Adapter pipelines
│   ├── metrics/                    # identity, demographic, quality, alignment
│   └── utils/
├── scripts/                        # numbered runnables (execute in order)
├── notebooks/                      # exploration + analysis
├── tests/                          # sanity tests for portability
├── data/                           # gitignored — regenerated by pipeline
├── models/                         # gitignored — regenerated by training
├── results/                        # gitignored — regenerated by benchmark
└── demo/app.py                     # Gradio demo
```

---

## Methods compared

| Method | Type | Reference |
|---|---|---|
| **PuLID-SDXL** (primary) | Encoder-based, no per-ID training | [arXiv:2404.16022](https://arxiv.org/abs/2404.16022) |
| **PhotoMaker v2** | Encoder-based | [arXiv:2312.04461](https://arxiv.org/abs/2312.04461) |
| **IP-Adapter FaceID v2** | Adapter + face embedding | [arXiv:2308.06721](https://arxiv.org/abs/2308.06721) |
| **DreamBooth-LoRA** | Per-identity LoRA training | [arXiv:2208.12242](https://arxiv.org/abs/2208.12242) |

---

## Metrics

| Axis | Metrics |
|---|---|
| Identity preservation | AdaFace, ArcFace cosine similarity |
| Age control | MiVOLO v2 mean absolute error |
| Race control | FairFace classifier accuracy |
| Accessory control | CelebA attribute classifier, CLIP zero-shot |
| Prompt alignment | CLIP-Score |
| Quality (no-reference) | HPSv2, PickScore |
| Distribution match | CLIP-FID, KID |
| Perceptual sim | DreamSim |

---

## Hardware requirements

- **VRAM**: 20 GB+ (full pipeline tested on RTX 4000 Ada)
- **Disk**: ~150 GB (datasets, models, generations)
- **Total compute**: ~60 GPU-hours over 2–3 weeks of calendar time

If running on lower-VRAM hardware:
- Drop benchmark to 5 identities × 3 methods → ~12 GPU-hours
- Use rank-16 LoRA instead of rank-32 → fits in 14 GB

---

## Ethics

Face-generation systems have obvious dual-use risks (deepfakes, identity misuse) and demographic-classification systems carry well-documented bias concerns. This project:

- Uses only public-domain identities (see `data/identity_gallery/*/metadata.json` for provenance)
- Reports demographic groupings as a **proxy for visual phenotype**, not as a biological or cultural claim
- Documents FFHQ's known demographic skew and its effect on results
- Notes that the FairFace classifier used for evaluation has its own confusions (Southeast/East Asian, Latino/white), so per-class race accuracy reflects agreement with an imperfect proxy, not ground truth
- Discusses mitigations (watermarking, attribution) that would be needed for any deployment

Full ethics discussion in [WRITEUP.md](WRITEUP.md#ethics).

---

## License

[MIT](LICENSE). If you use this code or benchmark, please cite (BibTeX TBA).

---

## Acknowledgments

Built on the shoulders of: FFHQ (NVlabs), SDXL (Stability AI), PuLID (ByteDance), PhotoMaker (ByteDance), IP-Adapter (Tencent), FairFace (UCLA), MiVOLO (SberAI), HPSv2.
