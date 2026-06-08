# Writeup

Findings as the project progresses. This currently covers the demographic LoRA;
the method benchmark is added once it is run.

## Demographic LoRA

### Training and checkpoint selection

A rank-32 LoRA on SDXL UNet attention, trained for 15k steps over 36k FFHQ images
with hybrid FairFace + BLIP-2 captions (`configs/demo_lora.yaml`).

Checkpoints are chosen on task performance, not validation loss. Every 1000 steps
the model generates a fixed demographic prompt set and is scored on age MAE
(FairFace age head) and race accuracy; after training, all checkpoints are
compared on the (age MAE, race accuracy) Pareto frontier and the point closest to
the ideal corner is selected.

The model overfits: race accuracy on the 224-image selection grid peaks near
step 3-4k and declines to step 15k (0.77 -> 0.58), so the later checkpoints that a
"train longest" rule would keep are the worst. Selection picked **step 4000**.
Validation reconstruction loss (MSE) was deliberately not used — it tracks
denoising accuracy, not whether the demographic tokens take effect.

- Frontier figure: `results/figures/checkpoint_pareto.png`
- Per-checkpoint scores: `results/checkpoint_selection.csv`

### Per-race control (selected checkpoint, step 4000)

4 ages x 7 races x 2 genders x 4 seeds, each generation scored with FairFace
(`results/per_race_breakdown.json`):

| Race | Race accuracy |
|---|---|
| white | 1.00 |
| Black | 1.00 |
| East Asian | 1.00 |
| South Asian | 0.94 |
| Middle Eastern | 0.78 |
| Latino | 0.44 |
| Southeast Asian | 0.03 |

The aggregate (0.74) understates the model. It is pulled down by one genuine
failure (Southeast Asian) and one measurement-limited class (Latino); the other
five, including the rarest class in the training data (Middle Eastern, 1.6% of
images), are controlled reliably.

### Model failure vs measurement artifact

The two low scores are different in kind. FairFace is also the metric, so its own
labels cannot serve as ground truth (that would be circular). Instead the
classifier ceiling was measured on the FairFace validation split — 10,954
human-labeled faces — giving the most the metric can score on real faces of each
class (`scripts/diagnose_race_control.py`, `results/race_ceiling.csv`,
`results/figures/sample_*.png`).

- **Southeast Asian is a model limitation.** FairFace classifies real Southeast
  Asian faces at 0.65, but the Southeast Asian generations score 0.03 — far below
  what the metric allows. The generations read as East Asian (FairFace assigns
  31/32 to East Asian), and at matched seeds the Southeast Asian token barely
  shifts the output away from the East Asian one. The model collapses the
  Southeast Asian token into its East Asian prior.
- **Latino is mostly a measurement artifact.** FairFace reaches only ~0.60 on
  real Latino faces — its weakest class, confused 14% with white. The generations
  (0.44) sit just below that ceiling and look like plausible Latino faces, so most
  of the gap is the metric, not the model.
- On the easy classes the generations are classified more accurately than real
  faces (white 1.00 vs 0.79 ceiling). This indicates the model produces
  prototypical, canonical faces; intra-class diversity is a separate property,
  measured later (planned: pairwise identity similarity, nearest-neighbour
  distance to training images).

### Known limitations

- **Southeast Asian collapse.** The Southeast Asian token is effectively ignored
  and the model defaults to East Asian. Planned fix: targeted Southeast Asian
  oversampling and reduced East Asian dominance during training.
- **Measurement ceiling.** Race control is scored with a FairFace-taxonomy
  classifier that itself reaches only 0.60-0.87 on real faces; Southeast/East
  Asian and Latino/white are its weakest boundaries. Per-class numbers should be
  read with this ceiling in mind, and the aggregate is unreliable for the fuzzy
  classes.
- **Prototypical generation.** High control accuracy on common classes coincides
  with canonical faces; within-class diversity is not yet quantified.

### Measurement and ethics note

The seven FairFace classes are a coarse visual proxy, not a biological or cultural
category, and the classifier carries its own confusions (notably Southeast/East
Asian and Latino/white). Race accuracy here measures agreement with that proxy,
not ground truth about a person.
