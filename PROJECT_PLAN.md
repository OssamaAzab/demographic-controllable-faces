# Demographic-Controllable Identity-Preserving Face Generation

A step-by-step project plan for v2 of the AML face-generation work. Every choice has a "why" attached. Designed to fit a single RTX 4000 Ada (20 GB VRAM), 320 GB scratch disk, ~2–3 weeks of calendar time, ~50–60 hours of GPU time.

---

## 0. Thesis (one sentence)

**Build a system that generates faces of a target identity while precisely controlling demographic attributes (age, ethnicity, gender) and accessory attributes (glasses, expression, hairstyle), and rigorously measure both axes against state-of-the-art baselines.**

---

## 1. Honest resume assessment — read this FIRST

You asked me to be honest. Here is the honest read.

### What this project is GOOD for on a resume
- ✅ **ML Engineer / Applied Scientist roles** — shows you can take a vague problem, design a benchmark, train models, evaluate them quantitatively, and ship a demo. That's the daily job.
- ✅ **Generative AI / Computer Vision startups** — modern stack (SDXL 2024 + PuLID 2024), end-to-end pipeline. Directly transferable.
- ✅ **Research engineer roles** — shows the research workflow: thesis → experiments → metrics → analysis → writeup.
- ✅ **PhD / Master's research applications** — signals you can structure a research-flavored project independently.
- ✅ **Mid-level interview signal** — the project demonstrates: dataset preparation, training pipelines, multi-metric evaluation, Pareto analysis, ethics awareness, reproducibility. All things interviewers probe.

### What this project is NOT
- ⚠️ **Not novel research.** Demographic / attribute control via LoRA already exists (Concept Sliders, ConceptBed, attribute editing papers). You are *building and benchmarking*, not inventing. That's fine — most portfolio pieces are not novel — but don't pitch it as "I invented X." Pitch it as "I built a rigorous benchmark of identity + attribute control methods."
- ⚠️ **Not a viral-demo project.** It's a methodical evaluation project. The demo is nice but not the headliner.
- ⚠️ **Won't get you into FAIR / OpenAI / Anthropic research roles by itself.** Those want first-author papers. This is "engineering competence at research-quality" — strong for industry, mid for top-tier research labs.

### What WOULD elevate it
- ✅ **Open-source it as a usable library** — `pip install face-controllable` with documented APIs. Now it's not just a project; it's a tool people can use.
- ✅ **Write a Medium / blog post** — explains the design choices, the Part B lesson, the Pareto tradeoff. Recruiters and engineers read these.
- ✅ **Get the HF Spaces demo to ~100 likes** — visible traction signal.
- ✅ **Workshop submission** — even a CVPR workshop paper on the benchmark would convert this from "portfolio" to "publication."
- ✅ **Include the lessons-learned narrative** from your Part A/B/C work. "I tried full fine-tuning and learned X, so v2 uses LoRA because Y" is a story that hiring managers love. It shows growth.

### Net honest verdict

**Solid mid-tier portfolio project. Definitely worth doing.** Will help you stand out vs. the median ML applicant who has only "trained a ResNet on CIFAR." Will NOT by itself be a moat against top candidates with publications. Pair it with one or two other strong signals (publications, open-source traction, real impact at a previous role) and you have a competitive senior-ML-engineer profile.

**Worth your 2–3 weeks? Yes — *if* you write the writeup and ship the demo.** Without those, it's a folder of code on GitHub that nobody reads. With them, it's a story you can walk an interviewer through.

---

## 2. Hardware budget

**Target GPU**: RTX 4000 Ada Generation, 20 GB VRAM, 130W TDP.
**Disk**: 320 GB scratch (plenty of room).
**Total compute**: ~50–60 GPU-hours over the project.

| Workload | VRAM | Wall-time |
|---|---|---|
| BLIP-2 captioning, 30k images | 12 GB | ~10 hours |
| FairFace labeling, 30k images | 4 GB | ~50 min |
| SDXL LoRA training (10k steps) | 17 GB | ~7 hours |
| Inference (1024², 30 steps) | 10 GB | ~6 sec/image |
| Full benchmark (~25k generations) | 12 GB | ~40 hours |

**Why this GPU is fine**: SDXL inference is ~10 GB; SDXL LoRA training with rank 32 + BF16 + gradient checkpointing + AdamW8bit fits in ~17 GB. FLUX would not fit. Full SDXL fine-tune would not fit either.

---

## 3. Models — what we use and why

### 3.1 Base model: SDXL 1.0
- **Model**: `stabilityai/stable-diffusion-xl-base-1.0`
- **VAE swap**: `madebyollin/sdxl-vae-fp16-fix` — the default SDXL VAE has FP16 NaN issues at inference; this fix is mandatory.
- **Why SDXL** (not SD 1.5, not FLUX):
  - **SD 1.5** is dated (2022). Looks weak on a 2026 resume.
  - **FLUX** needs 24+ GB just for inference and 80 GB for training. Won't fit your GPU.
  - **SDXL 1.0** is the sweet spot: native 1024×1024, mature ecosystem (all major identity adapters have SDXL versions), fits in 20 GB.

### 3.2 Identity-preservation methods (compared)
- 🥇 **PuLID-SDXL** (primary) — encoder-based, no per-identity training, SOTA identity scores in 2024 benchmarks
- 🥈 **PhotoMaker v2** — ByteDance, strong fallback, slightly easier to install
- 🥉 **IP-Adapter FaceID Plus v2** — older but well-known, useful as a "well-understood baseline"
- 🥉 **DreamBooth-LoRA per identity** — comparison baseline; the "old school" approach you used in Part C

**Why include all four**: a single-method paper looks weak. A four-method comparison with Pareto analysis tells a *story* about which method wins on which axis. That's what makes the project look like research, not a tutorial.

### 3.3 Captioning model (offline, training prep)
- **BLIP-2** (`Salesforce/blip2-opt-2.7b`) — generates the natural-language part of the hybrid caption
- **Why BLIP-2 not BLIP-1**: BLIP-1 captions are short and generic ("a woman smiling"). BLIP-2 produces richer descriptions ("a woman with long brown hair smiling at the camera, wearing a black blazer") that give the demographic LoRA more attribute supervision.
- **Why not LLaVA-1.5-7B**: it'd be slightly better but uses 14 GB VRAM. BLIP-2 uses 12 GB — tighter fit, still good.

### 3.4 Demographic labelers (offline, eval + caption prep)
- **Age**: **MiVOLO v2** (`iitolstykh/mivolo_v2`) — 3-year MAE, SOTA on age estimation
- **Race + gender**: **FairFace** classifier (`dchen236/FairFace`) — 7 race classes (White, Black, East Asian, Southeast Asian, Indian, Middle Eastern, Latino), trained on a balanced corpus to reduce bias
- **Why these specifically**: both are research-grade, not commercial APIs. Show methodological rigor. FairFace's balance-aware training is the most important: any portfolio piece touching race classification in 2026 should explicitly use a bias-aware model.

---

## 4. Dataset — what we use and why

### 4.1 Training data: FFHQ-1024 (30k subset)
- **Source**: NVlabs/FFHQ, 70k images at 1024×1024
- **Used subset**: 30k for first iteration, possibly 70k for second iteration if needed
- **Splits** (stratified by FairFace race + age bucket):
  - Train: 27,000
  - Val: 1,500
  - Test: 1,500 (also serves as FID/KID reference set)

**Why FFHQ**:
- Highest-quality consumer face dataset at 1024×1024
- Diverse (wider age + ethnicity range than CelebA-HQ)
- Creative Commons licensed
- Standard in face-generation papers — recruiters/reviewers recognize the name immediately
- Modern aesthetic (Flickr 2010s, looks contemporary in 2026)

**Why not CelebA-HQ for training**: celebrity-skewed, narrower age range, smaller (30k total).
**Why not VGGFace2**: messy, quality varies, harder to curate.
**Why not the full 70k upfront**: rank-32 LoRA saturates by ~20–30k samples. Iteration speed matters more than marginal quality for v1. We can upgrade to 70k + rank 64 if v1 shows demographic blind spots.

### 4.2 Benchmark identities: single-ref FFHQ-curated gallery
- **Source**: 10 images hand-picked from the **FFHQ test split** (never seen during training, no leakage risk)
- **Curated**: 10 identities chosen for **demographic diversity** covering FairFace's 7 race classes × both genders × age diversity
- **Per identity**: 1 clean reference image + a `metadata.json` with FairFace tags + provenance

**Why FFHQ (not CelebA-HQ)**: keeps the project to a **single dataset**, eliminates celebrity-dataset bias confounds, and avoids the consent/ethics complications of celebrity face data. FFHQ has no identity labels but we don't need them — PuLID works from a single reference image, and modern personalization papers measure identity preservation as `AdaFace(generation, reference)`, not against held-out same-identity reals.

**Why curate 10 identities (not 5 or 100)**:
- 5 = too few for statistical claims, demographic coverage gaps
- 100 = too much compute (each adds ~4 hours of benchmark generation)
- 10 = covers FairFace's 7 race classes, both genders, multiple ages

**Trade-off accepted**: we drop the "# of reference images" ablation (1 vs 3 vs 5 vs 10) — each identity has only 1 ref. Cleaner narrative is worth losing that one experiment.

---

## 5. Captioning pipeline — hybrid BLIP-2 + FairFace template

### 5.1 The template

```
"a photo of a {fairface_age_bucket} year old {fairface_race} {fairface_gender}, {blip2_description}"

# Example:
# "a photo of a 30-39 year old East Asian woman, with long black hair, 
#  wearing a white blouse, neutral expression, studio lighting"
```

### 5.2 Why hybrid (not pure BLIP-2, not pure FairFace tags)

- **Pure BLIP-2 alone**: generic captions, no demographic specificity → LoRA never learns to respond to "age 60" or "East Asian"
- **Pure FairFace tags alone**: structured but no visual variation → LoRA memorizes the template, loses generalization
- **Hybrid**: structured demographic anchors **+** natural-language variation → LoRA learns to respond to demographic tokens AND to natural attribute descriptions ("with glasses", "smiling", etc.)

This is the **same lesson from Part B's attribute-caption fix**, scaled up. There, you fixed the "debiasing" issue by adding explicit attribute tags. Here, you're doing the same but with demographic tags + richer NL descriptions.

### 5.3 Pipeline

For each FFHQ training image:
1. Load image at 1024×1024
2. Run FairFace classifier → `{age_bucket, race, gender}`
3. Run BLIP-2 → `{visual_description}`
4. Combine via template → final caption
5. Append to `data/ffhq_metadata.jsonl`

**Compute**: 30k images × 1.2 sec/image (BLIP-2 dominant) = **~10 hours**. Run overnight.

---

## 6. Train / Val / Test splits

| Split | Size (30k) | Size (70k) | Purpose |
|---|---|---|---|
| Train | 27,000 | 63,000 | LoRA training |
| Val | 1,500 | 3,500 | Training-stability loss tracking |
| Test | 1,500 | 3,500 | FID/KID reference + final metrics |

### 6.1 How splits are made
- **Random** seeded for reproducibility
- **Stratified** by FairFace race + age bucket (so all classes appear in val and test in proportion)
- Done **before captioning** — test images never see the captioner's outputs in training

### 6.2 Why stratify
Vanilla random splits can leave rare demographic groups (Middle Eastern, Latino) underrepresented in val/test, making metrics noisy on those groups. Stratification guarantees ≥50 examples of every class in the test set.

---

## 7. LoRA fine-tuning

### 7.1 Method
- **PEFT-LoRA** via `diffusers` `train_text_to_image_lora_sdxl.py` (or our extended version with EMA + mid-training task eval)
- **What it does**: adds rank-32 low-rank adapters to SDXL's UNet attention layers. ~25M trainable parameters on top of SDXL's ~2.6B frozen parameters.

### 7.2 Why LoRA (not full fine-tune)
- **Full fine-tune of SDXL** needs 28+ GB → does not fit your 20 GB GPU
- **Full fine-tune** also has Part B's lesson: hurts general knowledge while improving on the specific dataset
- **LoRA** is the standard 2024–2026 approach for personalization and attribute steering. Concept Sliders, PuLID, PhotoMaker — all LoRA-based.
- **LoRA** preserves SDXL's world knowledge while specializing for demographic responsiveness
- **LoRA fits** in 17 GB on your GPU

### 7.3 Config

```python
# Demographic LoRA (one-time training, ~7 hours)
pretrained_model            = "stabilityai/stable-diffusion-xl-base-1.0"
vae                         = "madebyollin/sdxl-vae-fp16-fix"
resolution                  = 1024
train_batch_size            = 1
gradient_accumulation_steps = 4              # effective BS = 4
lora_rank                   = 32
lora_alpha                  = 16
learning_rate               = 1e-4
lr_scheduler                = "cosine"
max_train_steps             = 10_000         # ~1.5 epochs over 27k images
mixed_precision             = "bf16"
gradient_checkpointing      = True
optimizer                   = "AdamW8bit"
enable_xformers             = True
use_ema                     = True
ema_decay                   = 0.9999
checkpointing_steps         = 1000           # 10 checkpoints saved
```

### 7.4 Why each setting

| Setting | Value | Why |
|---|---|---|
| `lora_rank` | 32 | Sweet spot. 16 = too narrow for demographic + accessory diversity; 64 = doesn't fit on 20 GB |
| `learning_rate` | 1e-4 | Standard LoRA LR; higher diverges, lower undertrains |
| `lr_scheduler` | cosine | Cleaner convergence than linear; standard in diffusion |
| `max_train_steps` | 10,000 | 1.5 epochs over 27k samples; enough signal without overfitting |
| `mixed_precision` | bf16 | Ada GPUs have strong BF16; FP16 has NaN risk |
| `gradient_checkpointing` | True | Saves ~7 GB at the cost of ~30% slowdown — essential on 20 GB |
| `optimizer` | AdamW8bit | bitsandbytes 8-bit Adam saves ~3 GB vs FP32 Adam |
| `enable_xformers` | True | Memory-efficient attention, saves another ~2 GB |
| `use_ema` | True | EMA weights are smoother and typically score better than any single step's weights |

### 7.5 Per-identity DreamBooth-LoRA (comparison baseline)
- Trained per identity, ~1000 steps, rank 16, LR 1e-4
- ~25 min per identity × 10 identities = ~4 hours total
- Token-based binding: `"a photo of sks person"`

**Why include this baseline**: shows the "old school" approach from Part C as a reference point, so the writeup can quantitatively argue "encoder-based methods (PuLID) are better than per-identity LoRA on these metrics."

---

## 8. Checkpoint selection — the Part B lesson applied

### 8.1 What we do NOT do
- ❌ Pick the best checkpoint based on validation MSE. **Part B taught us this is wrong.** Val-MSE measures denoising accuracy, not task performance.

### 8.2 What we DO do — task-aware mid-training eval

Every 1000 training steps, alongside the checkpoint save:

```python
TINY_VAL_PROMPTS = [
    "a 25 year old white woman, portrait",
    "a 65 year old white woman, portrait",
    "a 25 year old East Asian man, portrait",
    "a 65 year old East Asian man, portrait",
    "a 45 year old Black woman, portrait",
    "a 45 year old South Asian man, portrait",
    "a 80 year old Latino man, portrait",
    "a 35 year old Middle Eastern woman, portrait",
]  # 8 prompts × 2 seeds = 16 generations, ~2 min on your GPU

tiny_age_mae   = MiVOLO(generations).mae_vs_prompted_age()
tiny_race_acc  = FairFace(generations).accuracy_vs_prompted_race()
log({'step': step, 'tiny_age_mae': tiny_age_mae, 'tiny_race_acc': tiny_race_acc})
```

### 8.3 Why this works (and val-MSE doesn't)
- Directly measures the **task** the model is trying to learn
- Detects task-overfitting (when the curve plateaus or reverses)
- Gives a plottable "attribute responsiveness over training" curve for the writeup
- Adds only ~20 min total across the 7-hour training run

### 8.4 Final checkpoint selection: Pareto frontier

After training, run a larger eval on the 10 saved checkpoints:
- 56 prompts (4 ages × 7 races × 2 genders)
- 4 seeds per prompt
- = 224 generations per checkpoint × ~6 sec = ~22 min per checkpoint
- × 10 checkpoints = ~4 hours of post-training eval

Score each checkpoint on:
| Metric | Why |
|---|---|
| MiVOLO age MAE | Primary task: did age control work? |
| FairFace race acc | Primary task: did race control work? |
| CLIP-Score | Did the image match the prompt overall? |
| HPSv2 | Is the image quality good? |

**Plot Pareto frontier** of (age MAE, race acc). Pick a checkpoint on the frontier matching your priorities (likely balanced).

### 8.5 Why Pareto (not composite score)
- Pareto is **transparent about the tradeoff**: "we picked the checkpoint that gives X age MAE for Y race acc"
- Composite-score weighting is opaque and arbitrary
- Pareto plots are paper-quality figures
- Reviewers and recruiters prefer explicit tradeoffs to hidden weights

---

## 9. Inference pipeline

```
Reference photo(s) of target identity
              │
              ▼
        PuLID encoder ──► identity embedding
                                │
                                ▼
                    ┌───────────────────────┐
                    │ SDXL base             │
                    │ + demographic LoRA    │  ◄── prompt: "a 60 year old
                    │ + PuLID adapter       │      East Asian woman with
                    │                       │      glasses, smiling"
                    └───────────────────────┘
                                │
                                ▼
                       1024×1024 image
```

### Why this stack
- **SDXL base** (frozen) — provides general image quality
- **Demographic LoRA** — interprets attribute words ("60 year old", "East Asian", "glasses")
- **PuLID** — enforces identity from reference photo
- Each component has a single, decouplable job

---

## 10. Metrics suite

| # | Metric | Library | Measures | Why |
|---|---|---|---|---|
| 1 | **AdaFace cossim** | `mk-minchul/AdaFace` | Identity preservation | Primary identity metric; 2022 SOTA, more discriminative than ArcFace |
| 2 | **ArcFace cossim** | `insightface` | Identity preservation | Secondary; comparability with prior literature |
| 3 | **MiVOLO age MAE** | `iitolstykh/mivolo_v2` | Age controllability | "Age 60" → 60? |
| 4 | **FairFace race acc** | `dchen236/FairFace` | Race controllability | "East Asian" → East Asian? |
| 5 | **CelebA attribute classifier** | pretrained CelebA-HQ-attribute classifier (model only, no dataset dependency) | Accessory control (glasses, hat, beard) | "Glasses" → glasses? |
| 6 | **CLIP zero-shot accessory** | CLIP cosine | Accessory control (any attribute) | Cheap, no extra model |
| 7 | **CLIP-Score** | `openai/clip-vit-large` | Prompt alignment | Generic text-image agreement |
| 8 | **HPSv2** | `tgxs002/HPSv2` | Learned human preference | Quality without reference dataset confound |
| 9 | **PickScore** | `yuvalkirstain/PickScore_v1` | Learned human preference | Second human-pref signal |
| 10 | **CLIP-FID** | `cleanfid` | Distribution match (CLIP features) | Less aesthetic-bound than Inception FID |
| 11 | **KID** | `cleanfid` | Distribution match, sample-efficient | More stable on small N than FID |
| 12 | **DreamSim** | `ssundaram21/dreamsim` | Perceptual similarity to reference | Face-aware perceptual sim |

### 10.1 Why this many metrics

Each measures a different axis. The benchmark table tells a story only if multiple axes are covered:

| Axis | Metrics |
|---|---|
| Identity preservation | AdaFace, ArcFace |
| Demographic control | MiVOLO, FairFace |
| Accessory control | CelebA classifier, CLIP zero-shot |
| Prompt alignment | CLIP-Score |
| Generic quality | HPSv2, PickScore |
| Distribution match | CLIP-FID, KID |
| Perceptual sim | DreamSim |

### 10.2 What we DON'T report
- ❌ **val-MSE** as a quality metric (Part B's lesson)
- ❌ **Vanilla Inception FID without face cropping** — aesthetic mismatch with FFHQ would dominate; we use CLIP-FID + KID with face crops instead

### 10.3 Preprocessing for FID/KID
Both generations and reference are **face-cropped to 224×224 aligned** before feature extraction. Removes background/framing confounds. This is the practice in modern personalization papers.

---

## 11. Experiments — the resume table

| # | Experiment | What it shows |
|---|---|---|
| 1 | **Method comparison**: PuLID vs PhotoMaker vs IP-Adapter FaceID vs DreamBooth-LoRA | Which identity method wins overall |
| 2 | **Demographic control — age**: target ID at age {20, 40, 60, 80} | MiVOLO MAE per method, identity retention |
| 3 | **Demographic control — race**: target ID across 7 FairFace classes | FairFace acc, AdaFace retention |
| 4 | **Accessory control**: glasses, hat, beard, smile, makeup | CelebA classifier + CLIP zero-shot |
| 5 | **Identity-vs-control Pareto frontier** | The central tradeoff figure |
| ~~6~~ | ~~Reference ablation~~ | **DROPPED**: single-ref FFHQ identities (no multi-ref available). May add in v2 with curated multi-photo identities. |
| 7 | **demo-LoRA ablation**: with vs without | Does the LoRA actually help? |
| 8 | **Memorization audit**: ArcFace(gen vs train) ≤ ArcFace(gen vs ref) | Same bar as Part B |
| 9 | **Failure analysis**: worst-quartile examples per method | Qualitative honesty |

### 11.1 The headline figure
**Experiment 5** — the Pareto plot of identity retention vs attribute control strength. This is the most publishable-looking figure in the project. It directly visualizes the central tradeoff in personalized generation. Get this right; everything else is supporting.

### 11.2 Total benchmark scope
- 10 identities × 4 methods × ~60 prompts × 10 seeds = **24,000 generations**
- @ 6 sec/gen = **~40 hours of GPU time**

---

## 12. Two-week milestone plan

| Week | Day | Task | Output |
|---|---|---|---|
| W1 | 1 | Env setup, FFHQ download, identity gallery curation | Working pipeline, FFHQ on disk |
| W1 | 2 | `scripts/03_build_identity_gallery.py` — curate 10 FFHQ-test-split identities | `data/identity_gallery/` |
| W1 | 3 | `scripts/split_ffhq.py` — stratified train/val/test | 3 jsonl manifests |
| W1 | 4 | `scripts/caption_ffhq.py` — BLIP-2 + FairFace pipeline (overnight) | `data/ffhq_metadata.jsonl` |
| W1 | 5 | Demographic LoRA training (overnight + day) | `models/demo_lora.safetensors` + 10 checkpoints |
| W1 | 6 | `scripts/eval_checkpoints.py` — Pareto on 10 checkpoints | Best checkpoint selected |
| W1 | 7 | DreamBooth-LoRA per identity (5–10 identities) | 10 LoRA files |
| W2 | 8–10 | Full benchmark: 24k generations across 4 methods | `results.csv` |
| W2 | 11 | Analysis: plots, tables, failure analysis | `figures/`, `tables/` |
| W2 | 12 | `WRITEUP.md` + README | Project narrative |
| W2 | 13 | Gradio demo + HuggingFace Spaces deploy | Live demo URL |
| W2 | 14 | Polish, push to GitHub | Public repo |

---

## 13. Repository layout

```
demographic-controllable-faces/
├── README.md                    # hero figure + results table + demo link
├── WRITEUP.md                   # full narrative — thesis + findings + limitations
├── PROJECT_PLAN.md              # this document
├── requirements.txt             # pinned versions
├── Makefile                     # `make caption`, `make train`, `make bench`
├── LICENSE                      # MIT or Apache-2.0
├── configs/
│   ├── demo_lora.yaml
│   ├── dreambooth_lora.yaml
│   └── checkpoint_selection.yaml
├── data/
│   ├── ffhq_metadata.jsonl
│   ├── ffhq_train.jsonl
│   ├── ffhq_val.jsonl
│   ├── ffhq_test.jsonl
│   └── identity_gallery/
│       ├── id_001/{refs/, holdout/, metadata.json}
│       └── ... (id_002 through id_010)
├── scripts/
│   ├── build_identity_gallery.py
│   ├── split_ffhq.py
│   ├── caption_ffhq.py
│   ├── train_demo_lora.py
│   ├── train_dreambooth_lora.py
│   ├── eval_checkpoints.py
│   ├── benchmark.py
│   └── compute_metrics.py
├── notebooks/
│   ├── 01_explore_ffhq.ipynb
│   ├── 02_caption_audit.ipynb
│   ├── 03_analyze_results.ipynb
│   └── 04_failure_cases.ipynb
├── models/
│   ├── demo_lora.safetensors
│   └── dreambooth_lora/
├── results/
│   ├── results.csv
│   ├── figures/
│   └── tables/
└── demo/
    └── app.py                   # Gradio
```

### Why this layout
- `scripts/` and `notebooks/` separated: scripts are reproducible runnables, notebooks are exploratory.
- `configs/` makes hyperparameters first-class — reviewers can audit the exact training config.
- `Makefile` documents the pipeline (`make caption && make train && make bench`).
- `WRITEUP.md` separate from README: README is the elevator pitch, WRITEUP is the narrative.

---

## 14. Ethics section (mandatory in 2026 for face-gen projects)

Include in `WRITEUP.md`:

1. **Race classification limitations** — FairFace's 7 categories are coarse and reflect a social/visual construct, not biological reality.
2. **Identity consent** — only public-domain identities, sources cited, no private faces.
3. **Training data bias** — FFHQ skews younger / Western; document this and its effect on results.
4. **Dual-use risk** — controllable face generation has obvious misuse potential (deepfakes, identity fraud). Discuss what mitigations would be needed for deployment (watermarking, attribution).
5. **Compute footprint** — log GPU hours, estimate kWh and CO2. Show awareness.

**Why mandatory**: In 2026, any face-gen project missing an ethics section reads as careless. Recruiters at AI safety teams, responsible-AI teams, and any major AI lab actively look for this.

---

## 15. Reproducibility checklist

- ✅ Pinned `requirements.txt` (every package + version)
- ✅ Seed-controlled splits (`split_ffhq.py` accepts `--seed 42`)
- ✅ Configs in YAML, committed
- ✅ `Makefile` documents the full pipeline
- ✅ Random seeds documented in `WRITEUP.md`
- ✅ HF Hub model artifacts uploaded (or noted as too-large)
- ✅ Eval prompts versioned (`eval_prompts_v1.jsonl`)
- ✅ Wandb / Tensorboard logs preserved

---

## 16. Risks and failure modes

Document these honestly in the writeup. Recruiters can tell when projects are sanitized vs honest.

| Risk | Mitigation |
|---|---|
| Demographic LoRA underperforms on rare classes | Upgrade to 70k FFHQ + rank 64 |
| BLIP-2 captions are too repetitive | Add LLaVA-1.5 as fallback captioner |
| PuLID fails on certain ethnicities | Document the gap in writeup; don't hide |
| GPU thermal-throttles during long runs | `nvidia-smi` monitoring; chunk benchmark into 2-hour batches |
| Memorization on common celebrity identities | The Part B memorization audit catches this |
| FFHQ download is slow / interrupted | Resumable download, cache to scratch |
| FID metrics flatter LoRA-tuned models for wrong reasons | Use CLIP-FID + KID, face-crop preprocessing, report multiple references |

---

## 17. Locked-in decisions

After all the discussion, these are committed:

- ✅ **Base**: SDXL 1.0 + `sdxl-vae-fp16-fix`
- ✅ **Identity methods**: PuLID (primary) + PhotoMaker + IP-Adapter FaceID v2 + DreamBooth-LoRA (baseline)
- ✅ **Training data**: FFHQ-1024, 30k subset for v1
- ✅ **Single-dataset rule**: FFHQ for training + benchmark identities (NO CelebA-HQ dependency)
- ✅ **Benchmark identities**: 10 single-ref identities curated from the FFHQ test split, demographically diverse
- ✅ **Splits**: 27k / 1.5k / 1.5k, stratified by FairFace
- ✅ **Captioning**: hybrid BLIP-2 + FairFace template
- ✅ **LoRA config**: rank 32, BF16, grad-ckpt, AdamW8bit, EMA, 10k steps
- ✅ **Checkpoint selection**: mid-training task eval + post-training Pareto frontier (NOT val-MSE)
- ✅ **Metrics**: 10-metric suite, face-cropped FID/KID
- ✅ **Benchmark**: 10 identities × 4 methods × ~60 prompts × 10 seeds = 24k generations
- ✅ **Project path**: `/scratch/demographic-controllable-faces/`
- ✅ **GitHub repo name**: `demographic-controllable-faces`
- ✅ **Ethics section**: mandatory
- ✅ **Demo**: Gradio on HF Spaces

---

## 18. What's NEXT (concrete day-1 actions)

If you commit to this plan, here's the day-1 to-do:

1. ✅ Project location locked at `/scratch/demographic-controllable-faces/`
2. Write `scripts/02_split_ffhq.py` — stratified train/val/test split
3. Write `scripts/03_build_identity_gallery.py` — curate 10 FFHQ-test-split identities
4. Start FFHQ download (parallel to script writing)
5. By end of day-1: `data/identity_gallery/` populated, splits ready, FFHQ downloading overnight

---

## 19. Honest closing note

**Will doing this project change your life?** No.

**Will it look meaningfully better on your resume than "fine-tuned BERT for sentiment classification"?** Yes, by a lot.

**Will it be a unique signal vs other ML candidates with portfolio projects?** Only if you:
1. Actually ship the demo
2. Write the writeup with the lessons-learned narrative
3. Get it on GitHub publicly with clean code
4. Mention it in interviews as the place you learned the val-MSE / Pareto / Part B lessons

The technical work is half. The presentation is the other half. Don't skip the writeup.

If you do all of it: you have a solid Generative AI / Computer Vision portfolio piece that demonstrates research thinking + engineering competence + ethics awareness. That's a strong signal.

If you just train the LoRA and don't write it up: it's just another folder of code on GitHub.

The choice is yours. Commit if you're going to commit.
