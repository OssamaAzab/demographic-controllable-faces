# Demographic-Controllable Identity-Preserving Face Generation

Design document for a benchmark of identity-preserving face-generation methods
that also control demographic attributes (age, ethnicity, gender) and accessory
attributes (glasses, expression, hair). The contribution is a *demographic LoRA*
for SDXL trained on FFHQ with hybrid FairFace + BLIP-2 captions, evaluated
alongside three established identity methods on a multi-metric suite, with an
explicit Pareto analysis of the identity-vs-control trade-off.

Target hardware: a single 20 GB GPU (RTX 4000 Ada). Roughly 50–60 GPU-hours end
to end.

## 1. Goal

Given a single reference photo of a person and a prompt specifying demographic
and accessory attributes, generate that person rendered with the requested
attributes — and measure both identity preservation and attribute control
against state-of-the-art baselines.

## 2. Compute budget

| Workload | VRAM | Wall-time |
|---|---|---|
| BLIP-2 + FairFace captioning, 38k images | ~12 GB | ~1.3 h |
| FairFace + insightface gallery curation, 2k images | ~2 GB | ~7 min |
| SDXL LoRA training, 15k steps | ~17 GB | ~16–18 h |
| Inference (1024², 30 steps) | ~10 GB | ~3–4 s/image |
| Full benchmark | ~12 GB | ~40 h |

SDXL inference fits in ~10 GB; rank-32 LoRA training fits in ~17 GB with bf16 +
gradient checkpointing + 8-bit Adam + xformers. A full SDXL fine-tune (~28 GB) or
FLUX (24 GB inference, 40 GB+ training) would not fit.

## 3. Models

### 3.1 Base: SDXL 1.0
- `stabilityai/stable-diffusion-xl-base-1.0`
- VAE swap to `madebyollin/sdxl-vae-fp16-fix`: the default SDXL VAE overflows
  fp16 and produces NaN/black images; the fixed VAE is a drop-in with identical
  quality.
- SDXL over SD 1.5 (dated, 512²) and FLUX (too large for 20 GB): native 1024²,
  mature adapter ecosystem, fits the GPU.

### 3.2 Identity-preservation methods (compared)
- **PuLID-SDXL** (primary) — encoder-based, no per-identity training.
- **PhotoMaker v2** — encoder-based.
- **IP-Adapter FaceID Plus v2** — adapter + face embedding.
- **DreamBooth-LoRA** — per-identity LoRA; the classical baseline.

A four-method comparison with a Pareto analysis tells a story about which method
wins on which axis, rather than reporting a single method in isolation.

### 3.3 Captioner
- **BLIP-2** (`Salesforce/blip2-opt-2.7b`) for the natural-language half of the
  caption. Richer than BLIP-1; lighter than LLaVA-1.5 (which would not leave room
  for the demographic models on a 20 GB card).
- BLIP-2 recognizes some public figures present in FFHQ and emits names/news text
  instead of descriptions; those captions are detected by a marker filter and
  reduced to the demographic head, so no named entities enter training data.

### 3.4 Demographic labelers
- **Race / gender / age**: the FairFace taxonomy (7 race classes, 2 genders,
  9 age buckets), via transformers-native ViT classifiers
  (`dima806/fairface_age_image_detection`, `dima806/fairface_gender_image_detection`,
  `NikhilJaddu/fairface-race-vit`). The original FairFace ResNet-34 weights are
  distributed only via Google Drive with no Hub mirror; these ViT models
  reproduce its taxonomy and training data while remaining reliably installable.
- **Age estimation** for checkpoint selection uses the FairFace age head
  (bucket midpoint). MiVOLO is planned as a more precise age estimator for the
  final benchmark metrics.
- **Face detection / head pose** (gallery curation): insightface (buffalo_l).

FairFace is chosen for its balance-aware training; reporting race requires a
bias-aware model and an explicit statement of its limitations (see Ethics).

## 4. Dataset

### 4.1 Training data: FFHQ-1024
- Source: the official NVlabs Flickr-Faces-HQ release (Karras et al. 2019), the
  full 70k images at 1024², for non-commercial research use.
- v1 trains on a **40k subset**; the remaining images stay on disk for a possible
  scale-up. Rank-32 LoRA largely saturates by 20–30k samples, so 40k gives
  headroom without the cost of captioning all 70k.
- FFHQ over CelebA-HQ (celebrity-skewed, narrower) and VGGFace2 (noisy): high
  quality, diverse, Creative-Commons, and standard in the literature. Single
  dataset for both training and benchmark identities — no second-dataset
  confounds.

### 4.2 Benchmark identities
- 10 single-reference images curated from the **test split** (never seen in
  training, so no leakage).
- Chosen for demographic coverage: all 7 FairFace race classes, gender-balanced,
  spread across ages. Curation ranks candidates by FairFace demographics and
  insightface frontality (single face, low yaw/pitch, high detection score) and
  fills target slots, relaxing constraints when a demographic is scarce in FFHQ.
- Each identity stores one reference image plus metadata (FairFace tags, pose,
  provenance).
- The reference-count ablation is dropped: every identity has a single reference,
  matching how PuLID/PhotoMaker/IP-Adapter are used in practice.

## 5. Captioning — hybrid BLIP-2 + FairFace

Template:

```
"a photo of a {age_bucket} year old {race} {gender}, {blip2_description}"
e.g. "a photo of a 30-39 year old East Asian woman, with long black hair, wearing a white blouse"
```

Pure BLIP-2 gives no demographic anchor; pure FairFace tags give no visual
variation and invite template memorization. The hybrid form supplies demographic
tokens to respond to *and* natural attribute variety to generalize over. For each
image: run FairFace for the demographic tags, BLIP-2 for the description, then
combine. Output is one JSONL line per image.

## 6. Splits

Seeded random split of the 40k subset (90/5/5). Stratification by demographic
is deferred to captioning, since FairFace labels do not exist before the split.

| Split | Size | Purpose |
|---|---|---|
| Train | 36,000 | LoRA training |
| Val | 2,000 | training-stability monitoring only |
| Test | 2,000 | benchmark identities + FID/KID reference |

## 7. LoRA training

PEFT LoRA on the SDXL UNet attention projections; text encoders frozen.

| Setting | Value | Rationale |
|---|---|---|
| rank / alpha | 32 / 16 | capacity for demographic + accessory diversity; α/r = 0.5 keeps updates conservative |
| target modules | `to_q,to_k,to_v,to_out.0` | attention is where text conditioning enters the UNet |
| learning rate | 1e-4, cosine, 200 warmup | standard LoRA LR; warmup avoids early instability |
| steps | 15,000 (~1.7 epochs) | ceiling; the best checkpoint is selected afterward |
| batch | 1 × grad-accum 4 | effective batch 4 within the VRAM budget |
| precision | bf16 weights, fp32 LoRA params (autocast) | bf16 avoids fp16 NaNs; fp32 params for stable Adam |
| memory | gradient checkpointing, AdamW8bit, xformers | fits ~17 GB on a 20 GB card |
| EMA | decay 0.9999 | smoother weights; the EMA copy is what gets saved |

This trains ~46 M LoRA parameters on top of SDXL's ~2.6 B frozen weights, so the
base model's general capabilities are preserved while it specializes for
demographic responsiveness. A full fine-tune was avoided: it does not fit the
GPU and tends to degrade general knowledge.

Per-identity DreamBooth-LoRA baselines (rank 16, ~1000 steps each, token-based
binding) are trained separately for the comparison.

## 8. Checkpoint selection

Validation reconstruction loss (MSE) measures denoising accuracy, not whether the
model performs the target task; selecting on it can pick a checkpoint that every
downstream metric disagrees with. Selection is therefore task-aware:

- **During training** (every 1000 steps): generate a small fixed prompt set and
  score age MAE (FairFace) and race accuracy (FairFace). This gives a plottable
  "attribute responsiveness over training" curve. Validation MSE is logged for
  stability monitoring only, never for selection.
- **After training**: evaluate every saved checkpoint on a larger grid
  (4 ages × 7 races × 2 genders × 4 seeds) and pick the checkpoint on the
  **Pareto frontier** of (age MAE ↓, race accuracy ↑) closest to the ideal
  corner. Pareto keeps the trade-off explicit rather than hiding it in a weighted
  composite.

## 9. Inference stack

```
reference photo ─► PuLID encoder ─► identity embedding
                                          │
                  prompt ─► SDXL base + demographic LoRA + PuLID adapter ─► 1024² image
```

SDXL provides image quality, the demographic LoRA interprets attribute tokens,
and the identity method enforces likeness — each component with a separable job.

## 10. Metrics

| Axis | Metrics |
|---|---|
| Identity preservation | AdaFace, ArcFace cosine similarity |
| Age control | MiVOLO age MAE |
| Race control | FairFace accuracy |
| Accessory control | CelebA attribute classifier, CLIP zero-shot |
| Prompt alignment | CLIP-Score |
| Quality (no reference) | HPSv2, PickScore |
| Distribution match | CLIP-FID, KID (face-cropped) |
| Perceptual similarity | DreamSim |

Inception FID without face cropping is avoided: aesthetic mismatch with FFHQ
would dominate it. FID/KID use face-cropped, aligned 224² inputs.

## 11. Experiments

| # | Experiment | Shows |
|---|---|---|
| 1 | Method comparison (PuLID / PhotoMaker / IP-Adapter / DreamBooth) | which identity method wins overall |
| 2 | Age control at {20, 40, 60, 80} | age MAE per method, identity retention |
| 3 | Race control across the 7 classes | race accuracy, identity retention |
| 4 | Accessory control (glasses, hat, beard, smile, makeup) | attribute classifier + CLIP |
| 5 | Identity-vs-control Pareto frontier | the central trade-off figure |
| 6 | demo-LoRA ablation (with vs without) | does the LoRA help? |
| 7 | Memorization audit | generations resemble references, not training images |
| 8 | Failure analysis | worst-quartile examples per method |

Experiment 5 (the Pareto plot) is the headline figure. Benchmark scope is the
five method configurations × 10 identities × the prompt set × multiple seeds,
defined exactly in `configs/benchmark.yaml`.

## 12. Repository layout

```
demographic-controllable-faces/
├── README.md / WRITEUP.md / PROJECT_PLAN.md
├── pyproject.toml / requirements.txt / Makefile / LICENSE
├── configs/            # all hyperparameters as YAML
├── src/dcfaces/        # importable package (paths, data, captioning, training, inference, metrics)
├── scripts/            # numbered runnables, 01–09, executed in order
├── notebooks/          # exploration + analysis
├── tests/              # path-portability checks
├── data/ models/ results/   # gitignored, regenerated by the pipeline
└── demo/app.py         # Gradio demo
```

Scripts are reproducible runnables; notebooks are exploratory. Hyperparameters
live in `configs/` so the exact training setup is auditable.

## 13. Ethics

To be documented in `WRITEUP.md`:

1. Race classification is a coarse visual proxy, not a biological or cultural
   claim; the FairFace categories and their limitations are stated explicitly.
2. Identities are drawn from FFHQ (Creative-Commons), with provenance recorded.
3. FFHQ's demographic skew (younger, Western-leaning) is documented along with
   its effect on results.
4. Controllable face generation carries deepfake/identity-misuse risk;
   deployment mitigations (watermarking, attribution) are discussed.
5. Compute footprint (GPU-hours, estimated energy) is reported.

## 14. Reproducibility

- Pinned `requirements.txt`.
- Seeded splits and training (`--seed 42`).
- Configs committed as YAML.
- `Makefile` documents the pipeline stages.
- Eval prompt sets versioned.
- Model artifacts uploaded to the Hub where size permits, otherwise noted.

## 15. Limitations and risks

| Risk | Mitigation |
|---|---|
| LoRA underperforms on rare demographics | scale to 70k / higher rank |
| BLIP-2 captions repetitive or name-leaking | marker filter; LLaVA fallback if needed |
| Identity method weak on some ethnicities | document the gap rather than hide it |
| Long runs thermally throttle | chunk into shorter batches, monitor |
| FID flatters LoRA models for the wrong reason | CLIP-FID + KID with face crops |
