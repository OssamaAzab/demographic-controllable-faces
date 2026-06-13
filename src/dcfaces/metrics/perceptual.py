"""DreamSim perceptual distance between a generation and its reference (0 = identical)."""

from __future__ import annotations

import torch
from PIL import Image

from dcfaces.paths import CACHE_DIR


class DreamSimScorer:
    def __init__(self, device: str = "cuda"):
        from dreamsim import dreamsim

        self.device = device
        self.model, self.preprocess = dreamsim(
            pretrained=True, cache_dir=str(CACHE_DIR / "dreamsim"), device=device
        )

    @torch.no_grad()
    def distance(self, a: Image.Image, b: Image.Image) -> float:
        ta = self.preprocess(a.convert("RGB")).to(self.device)
        tb = self.preprocess(b.convert("RGB")).to(self.device)
        return float(self.model(ta, tb)[0])
