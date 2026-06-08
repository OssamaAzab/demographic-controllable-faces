"""Demographic labelling via FairFace-taxonomy classifiers.

Shared by 03 (identity gallery), 04 (captioning), and 09 (FairFace race-accuracy
metric), so it lives in the package rather than in any one script.

Backend note (flagged deviation from the locked "dchen236/FairFace"):
    The original FairFace is a ResNet34 multi-task net whose weights are
    Google-Drive-only (no HF mirror -- the same quota wall that blocks FFHQ) and
    need bespoke architecture code. We instead use transformers-native ViT
    classifiers that reproduce FairFace's exact taxonomy (7 race / 2 gender /
    9 age buckets), trained on FairFace data. They are reliable, cached in-repo,
    and the backend is isolated here so swapping in the official weights later is
    a one-file change. Cite FairFace (Karkkainen & Joo, WACV 2021) for the
    taxonomy and training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

AGE_MODEL = "dima806/fairface_age_image_detection"
GENDER_MODEL = "dima806/fairface_gender_image_detection"
RACE_MODEL = "NikhilJaddu/fairface-race-vit"

# Map raw model labels -> the project's canonical vocabulary (configs/*.yaml use
# "South Asian"/"Latino"/"white", and prompts use "man"/"woman").
RACE_CANON = {
    "White": "white",
    "Black": "Black",
    "East Asian": "East Asian",
    "Southeast Asian": "Southeast Asian",
    "Indian": "South Asian",
    "Middle Eastern": "Middle Eastern",
    "Latino_Hispanic": "Latino",
}
GENDER_CANON = {"Female": "woman", "Male": "man"}


@dataclass
class FairFaceResult:
    age_bucket: str  # e.g. "30-39" ("more than 70" -> "70+")
    race: str  # canonical, e.g. "South Asian"
    gender: str  # "man" / "woman"
    age_score: float
    race_score: float
    gender_score: float
    race_raw: str  # raw model label, e.g. "Indian"


def _clean_age(label: str) -> str:
    return "70+" if label.lower().startswith("more than") else label


class FairFaceClassifier:
    """Three ViT heads (age, gender, race) reproducing the FairFace taxonomy.

    Faces are expected to be roughly centered (FFHQ aligned images qualify), so
    we feed the full image to each model's own processor (224, ImageNet norm).
    """

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._heads = {}
        for key, repo in (("age", AGE_MODEL), ("gender", GENDER_MODEL), ("race", RACE_MODEL)):
            proc = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForImageClassification.from_pretrained(repo).to(self.device).eval()
            self._heads[key] = (proc, model)

    @torch.no_grad()
    def _predict(self, key: str, images: list[Image.Image]) -> list[tuple[str, float]]:
        proc, model = self._heads[key]
        inputs = proc(images=images, return_tensors="pt").to(self.device)
        logits = model(**inputs).logits
        probs = logits.softmax(dim=-1)
        scores, idx = probs.max(dim=-1)
        id2label = model.config.id2label
        return [(id2label[int(i)], float(s)) for i, s in zip(idx.tolist(), scores.tolist())]

    def classify_batch(self, images: list[Image.Image]) -> list[FairFaceResult]:
        imgs = [im.convert("RGB") for im in images]
        age = self._predict("age", imgs)
        gender = self._predict("gender", imgs)
        race = self._predict("race", imgs)
        out = []
        for (a, asc), (g, gsc), (r, rsc) in zip(age, gender, race):
            out.append(
                FairFaceResult(
                    age_bucket=_clean_age(a),
                    race=RACE_CANON.get(r, r),
                    gender=GENDER_CANON.get(g, g),
                    age_score=asc,
                    race_score=rsc,
                    gender_score=gsc,
                    race_raw=r,
                )
            )
        return out

    def classify(self, image: Image.Image) -> FairFaceResult:
        return self.classify_batch([image])[0]
