"""FFHQ + hybrid-caption dataset for SDXL LoRA training.

Joins an image manifest (data/ffhq_train.jsonl) with the captions produced by
04 (data/ffhq_metadata.jsonl) on image_id. Yields the pieces SDXL needs:
pixel_values in [-1, 1] plus the micro-conditioning sizes (original/target/crop)
that go into SDXL's added time-ids. FFHQ images are already 1024x1024 aligned,
so resize+center-crop is effectively a no-op but kept for robustness.
"""

from __future__ import annotations

import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dcfaces.paths import PROJECT_ROOT


class FFHQCaptionDataset(Dataset):
    def __init__(self, manifest_path, metadata_path, resolution: int = 1024):
        captions = {}
        for line in open(metadata_path):
            r = json.loads(line)
            captions[r["image_id"]] = r["caption"]

        self.items: list[tuple[str, str]] = []
        missing = 0
        for line in open(manifest_path):
            r = json.loads(line)
            cap = captions.get(r["image_id"])
            if cap is None:
                missing += 1
                continue
            self.items.append((r["image_path"], cap))
        if not self.items:
            raise RuntimeError(
                f"No (image, caption) pairs. Did 04 finish? "
                f"manifest={manifest_path}, metadata={metadata_path}"
            )
        self.missing = missing
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        image_path, caption = self.items[idx]
        img = Image.open(PROJECT_ROOT / image_path).convert("RGB")
        pixel_values = self.transform(img)
        return {
            "pixel_values": pixel_values,
            "caption": caption,
            # SDXL micro-conditioning (no random crop: FFHQ is pre-aligned 1024).
            "original_size": (self.resolution, self.resolution),
            "crop_top_left": (0, 0),
            "target_size": (self.resolution, self.resolution),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]).contiguous(),
        "captions": [b["caption"] for b in batch],
        "original_sizes": [b["original_size"] for b in batch],
        "crop_top_lefts": [b["crop_top_left"] for b in batch],
        "target_sizes": [b["target_size"] for b in batch],
    }
