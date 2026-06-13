# Writeup

This benchmark started from a simple contribution — a demographic LoRA to add
age/race control to identity-preserving face generation — and turned into a more
interesting result: that control **cannot be stacked onto a strong identity method**,
and the thing that *does* recover it is scheduling, not weights. What follows is the
full arc: the trade-off the benchmark measures, the demographic LoRA on its own, why
stacking it fails, the rescue that works, and the confounds we had to control for.

## The benchmark and the trade-off

8 method configurations are evaluated on 10 single-reference FFHQ identities, 21
prompts (identity, age, race, accessory, compositional), and 5 seeds, on SDXL 1.0 +
the fp16-fix VAE (HyperLoRA on RealVisXL — see confounds). Ten metrics span identity
(ArcFace, AdaFace), demographic control (MiVOLO age, FairFace race, CLIP-zero-shot
accessory), prompt alignment (CLIP-Score), aesthetics (HPSv2, PickScore),
distribution (face-cropped CLIP-FID, KID), and perceptual distance (DreamSim).

The central result is a steep **identity vs controllability trade-off**
(`results/figures/identity_vs_control_pareto.png`). The Pareto frontier runs from
PuLID (identity 0.73 AdaFace, controllability 0.36) through the DreamBooth variants
and PuLID-delayed to HyperLoRA (identity 0.56, controllability 0.58). The headline
numbers are in the README. The qualitative companion
(`results/figures/method_control_grid.png`) makes it visible: PuLID renders the same
locked face down every control column; HyperLoRA follows the prompt but drifts off
the reference.

## The demographic LoRA (standalone)

### Training and checkpoint selection

A rank-32 LoRA on SDXL UNet attention, trained for 15k steps over 36k FFHQ images
with hybrid FairFace + BLIP-2 captions (`configs/demo_lora.yaml`).

Checkpoints are chosen on task performance, not validation loss. Every 1000 steps the
model generates a fixed demographic prompt set and is scored on age MAE and race
accuracy; after training, all checkpoints are compared on the (age MAE, race
accuracy) Pareto frontier and the point closest to the ideal corner is selected. The
model overfits: race accuracy on the 224-image selection grid peaks near step 3-4k
and declines to step 15k (0.77 -> 0.58), so the later checkpoints a "train longest"
rule would keep are the worst. Selection picked **step 4000**. Validation
reconstruction loss was deliberately not used — it tracks denoising accuracy, not
whether the demographic tokens take effect.

- Frontier figure: `results/figures/checkpoint_pareto.png`

### Per-race control (selected checkpoint, step 4000)

4 ages x 7 races x 2 genders x 4 seeds, each scored with FairFace:

| Race | white | Black | East Asian | South Asian | Middle Eastern | Latino | Southeast Asian |
|---|---|---|---|---|---|---|---|
| Race accuracy | 1.00 | 1.00 | 1.00 | 0.94 | 0.78 | 0.44 | 0.03 |

The aggregate (0.74) understates the model: it is pulled down by one genuine failure
(Southeast Asian) and one measurement-limited class (Latino); the other five,
including the rarest training class (Middle Eastern, 1.6% of images), are controlled
reliably. So **on its own the demographic LoRA works.**

## Why stacking the demographic LoRA fails

The contribution was to stack this LoRA on a strong identity method (PuLID) for
*controllable, identity-preserving* generation. It does not work, and we established
that three independent ways:

1. **Scale sweep.** Sweeping `demo_scale` from 0 to 1 on PuLID never produces the
   demographic shift — at every scale the identity-injected demographics (the
   reference person's own age/race) dominate. `PuLID + demo LoRA` is Pareto-dominated
   by PuLID alone (identity 0.68 vs 0.73, controllability 0.35 vs 0.36).
2. **`id_scale x demo_scale` grid.** The only cell where demographics shift is when
   PuLID's identity is turned nearly off (`id_scale=0.3, demo_scale=0`) — i.e. the
   prompt wins because the embedding is weak, and the identity is lost. Adding the
   demo LoRA at any identity strength does not produce the shift; it pulls back toward
   the original demographics. This pinpointed the **identity embedding**, not the
   LoRA, as what controls the demographics.
3. **ArcFace-embedding editing.** Shifting the injected ArcFace embedding along
   learned age/race directions nudges predicted age by one bucket (65->55, then
   plateaus) and moves race not at all — because PuLID also conditions on EVA-CLIP
   visual features that re-impose the demographics.

We then tested whether this is specific to PuLID's attention-*injection* mechanism by
stacking the demo LoRA on **HyperLoRA** (whose identity is itself a LoRA) on SDXL 1.0.
It also fails: race accuracy is flat at 0.33 across `demo_scale` 0/0.5/0.8. So the
obstruction is not the injection mechanism — **any strong identity signal, injected or
LoRA, overrides the demographic LoRA when the two are combined as parallel weights.**

What the demo LoRA *does* add when stacked is **color and FFHQ-realism**: it lowers
FID/KID and DreamSim on both PuLID and DreamBooth (it was trained on colorful FFHQ).
On DreamBooth it also raises identity (0.58->0.66) while trading away controllability.
So its honest role as an add-on is a realism/regularization effect, not control.

## The rescue: delayed identity injection

The grid finding — that the identity *embedding* locks demographics — points at a
different lever: *when* identity is applied, not how strongly. Coarse structure (age,
face shape, broad coloring) is decided in the early, high-noise diffusion steps; fine
identity in the later steps. If identity injection is **held off for the first N of 30
steps**, the prompt establishes the demographic first, and identity then refines the
person's features on top.

Implemented by gating PuLID's per-step identity injection (the positive-branch ID
embedding is zeroed until step N; `src/dcfaces/benchmark/methods.py`, `delay_start`).
A sweep over N across 5 identities (East Asian prompt) gives a clean trade-off:

| start_step | race accuracy | identity (AdaFace) |
|---|---|---|
| 0 (baseline) | 0.40 | 0.72 |
| 13 | 0.60 | 0.69 |
| 16 | 0.60 | 0.62 |
| 19 | 0.80 | 0.47 |

`start=13` is the sweet spot: **0.60 race accuracy at 0.69 identity** — vs PuLID's
0.34 race accuracy on the same prompt at 0.72 identity. As a full benchmark method
(`pulid_delayed`, `delay_start=13`) it lands on the Pareto frontier and **dominates
`pulid_with_demo_lora`** (same identity 0.68, controllability 0.41 vs 0.35). It is
training-free.

### The rescue is attribute-dependent

Delayed injection recovers **race** (a surface attribute — coloring, set early and
preserved) but **not age** (a structural attribute — bone structure and wrinkles,
which the late identity steps re-impose). In the start-step sweep, age only flips to
"20" at `start=22`, where identity has collapsed to 0.16. MiVOLO confirms it: the
`pulid_delayed` "age 20" generations still read ~58 years old. So the clean statement
is: *scheduling identity injection rescues surface demographic control but cannot move
structural attributes without losing identity.*

## Confounds and methodology

- **Grayscale -> FID confound (fixed).** PuLID and IP-Adapter on vanilla SDXL produced
  near-grayscale outputs (mean HSV saturation ~2-5 vs ~40-95 for the others), which
  unfairly inflated their CLIP-FID/KID and depressed aesthetics. An anti-grayscale
  negative prompt restores natural color (saturation ~2 -> ~70), and the affected
  methods were regenerated. The grayscale specifically hit the plain-SDXL/no-demo
  methods; the demo-LoRA and RealVisXL variants were already colorful. After the fix,
  PuLID's FID improved only modestly (40 -> 38, KID 66 -> 54), so the demo LoRA's
  distribution edge is partly real, not purely a color artifact.
- **ArcFace circularity (checked).** Methods that inject an ArcFace embedding can
  score an inflated ArcFace identity. AdaFace is the independent cross-check; here the
  two agree closely (PuLID 0.727 ArcFace vs 0.725 AdaFace), so PuLID's identity lead is
  genuine, not circular.
- **Race measurement (decomposed).** FairFace is both the demographic head and the
  metric, so its labels cannot be ground truth. When prompted "Southeast Asian",
  outputs are labeled East Asian ~30% of the time and Southeast Asian only ~7% — so the
  near-zero Southeast Asian score is largely the SE/East-Asian conflation (counting
  "any Asian" gives ~37%, in line with other classes), with a real model residual.
  Latino is similarly FairFace's weakest class (~0.60 on real faces), confused with
  white.
- **HyperLoRA base confound (quantified).** HyperLoRA is incompatible with vanilla
  SDXL 1.0 and runs on RealVisXL v4.0. A side-test of its ID-LoRA on both bases gives
  identical race accuracy (0.33) and only a +0.09 identity gain on RealVisXL — so its
  controllability lead is the method, while its identity number is modestly
  base-boosted.
- **MiVOLO isolation.** MiVOLO (continuous age) depends on `timm 0.8`, incompatible
  with the main stack's `timm 1.0`. It runs in a separate `.venv-mivolo` and is called
  as a subprocess (`scripts/09b_mivolo_age.py`); the main aggregation merges its ages
  and falls back to the FairFace age bucket where MiVOLO finds no face.

## Rendering diversity

For a fixed (method, identity, prompt) the 5 seeds should give varied renderings of
the *same* person (pose, expression, framing), not identical clones. Mean pairwise
DreamSim across seeds on the identity-only prompts (`scripts/09c_diversity.py`), read
alongside identity:

| Method | Seed diversity | Identity (AdaFace) |
|---|---|---|
| PhotoMaker v2 | 0.32 | 0.50 |
| IP-Adapter FaceID v2 | 0.28 | 0.52 |
| PuLID + delayed injection | 0.22 | 0.68 |
| PuLID | 0.21 | 0.73 |
| DreamBooth-LoRA | 0.20 | 0.58 |
| DreamBooth + demographic LoRA | 0.12 | 0.66 |
| PuLID + demographic LoRA | 0.11 | 0.68 |

Two readings. First, the demographic LoRA **collapses rendering diversity**: stacking
it roughly halves seed-to-seed variation at unchanged identity (PuLID 0.21 -> 0.11,
DreamBooth 0.20 -> 0.12), pushing outputs toward canonical, near-clone faces. That is a
third cost of the stack (after lost identity and no control gain) and another margin by
which `pulid_delayed` (0.22) beats `pulid_with_demo` (0.11). Second, the high diversity
of PhotoMaker / IP-Adapter is partly identity *drift* (they also have the lowest
identity), so diversity is read with the identity column, not as a standalone virtue.

Note this is *rendering* diversity (varied presentation of one person), not identity
diversity (which would contradict the goal and is captured by the identity metric).
Per-image memorization (nearest-neighbour to training data) does not apply here: the
generations are identity-conditioned on held-out test references, so distance to
training measures coincidental phenotype overlap, not copying — a clean memorization
test would need reference-free demographic-LoRA samples (future work).

## Known limitations

- **No SOTA-winning contribution.** The headline is an honest negative (stacking
  fails) plus a modest training-free rescue (surface attributes only), not a new
  method that beats the field.
- **Southeast Asian collapse.** The token largely defaults to East Asian; partly
  measurement, partly model. A targeted oversampling re-run of the demo LoRA is the
  natural fix.
- **Single GPU, single dataset.** 20 GB VRAM, FFHQ only, 10 identities — a focused
  study, not large-scale.
- **Reference-free memorization not tested.** Rendering diversity is measured (above),
  but a clean nearest-neighbour-to-training memorization check would require
  reference-free demographic-LoRA samples, which this benchmark does not generate.

## Measurement and ethics note

The seven FairFace classes are a coarse visual proxy, not a biological or cultural
category, and the classifier carries its own confusions (notably Southeast/East Asian
and Latino/white). Race accuracy here measures agreement with that proxy, not ground
truth about a person. The project uses only public-domain FFHQ identities; any
deployment of identity-preserving generation would require watermarking and
attribution safeguards.
