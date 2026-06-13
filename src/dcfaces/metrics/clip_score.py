"""CLIP-Score: text-image alignment via CLIP cosine similarity.

Used by 06 (checkpoint selection) and 09 (benchmark metrics). Reports the raw
image-text cosine similarity (higher = better prompt alignment). Cached in-repo.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPScorer:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(
            text=[text], images=[image.convert("RGB")],
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        out = self.model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((img * txt).sum(dim=-1).item())

    @torch.no_grad()
    def zero_shot(self, image: Image.Image, texts: list[str]) -> list[float]:
        """Softmax CLIP probabilities over `texts` for one image (zero-shot).

        Used for accessory control: a [positive, negative] pair gives the
        probability that the requested accessory is present.
        """
        inputs = self.processor(
            text=texts, images=[image.convert("RGB")],
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        probs = self.model(**inputs).logits_per_image.softmax(dim=-1)[0]
        return [float(p) for p in probs]
