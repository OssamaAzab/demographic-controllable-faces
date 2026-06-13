"""Human-preference scorers: HPSv2 and PickScore (higher = more preferred / on-prompt)."""

from __future__ import annotations

import torch
from PIL import Image

from dcfaces.paths import MODELS_DIR


class HPSScorer:
    def __init__(self, version: str = "v2.1"):
        import hpsv2

        self._hps = hpsv2
        self.version = version

    def score(self, image: Image.Image, prompt: str) -> float:
        return float(self._hps.score([image], prompt, hps_version=self.version)[0])


class PickScorer:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained(str(MODELS_DIR / "pickscore")).to(device).eval()

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        ii = {k: v.to(self.device) for k, v in self.proc(images=image, return_tensors="pt").items()}
        ti = {k: v.to(self.device) for k, v in
              self.proc(text=prompt, return_tensors="pt", padding=True, truncation=True).items()}
        ie = self.model.get_image_features(**ii)
        te = self.model.get_text_features(**ti)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        return float((self.model.logit_scale.exp() * (ie * te).sum(-1))[0])
