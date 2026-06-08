"""BLIP-2 captioner producing the natural-language half of the hybrid caption.

The demographic half ("a photo of a 30-39 year old East Asian woman") comes from
FairFace (dcfaces.demographics); BLIP-2 supplies the rest ("with long black hair,
wearing a white blouse, ..."). Two cleanups are applied to BLIP-2's raw output:

  * strip_subject: BLIP-2 restates the subject ("a woman with ..."), which would
    double up the demographic half, so the leading subject boilerplate is removed.

  * is_clean_description: BLIP-2 RECOGNISES famous faces (FFHQ has public figures)
    and emits names / news / photo-credits instead of a visual description
    (e.g. "former secretary of state hillary clinton", "... says she's not running
    for president in 2016"). Prompting doesn't fix this. We detect those via
    high-precision markers (years, titles, news verbs, photo-credit sites) and
    drop the description -> the caption falls back to the demographic head only.
    This avoids injecting named entities into training captions (also relevant to
    the memorization audit). The drop is recorded as blip2_filtered in 04's output.

Cached in-repo (.cache/hf).
"""

from __future__ import annotations

import re

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# Leading "a/an/the [close-up photo of] [a] [young/little/...] <subject>" boilerplate.
_SUBJECT_RE = re.compile(
    r"^(?:a|an|the)\s+"
    r"(?:(?:close[\s-]?up|cropped|small|large)\s+)?"
    r"(?:(?:photo|picture|image|portrait|headshot|shot|view)\s+of\s+)?"
    r"(?:a\s+|an\s+)?"
    r"(?:(?:young|old|elderly|little|middle[\s-]?aged|smiling|happy|beautiful|handsome"
    r"|cute|serious|bald|attractive)\s+)*"
    r"(?:man|woman|person|people|boy|girl|guy|lady|male|female|gentleman|child|baby|kid"
    r"|adult|teenager|teen)\b"
    r"[\s,'’-]*",
    re.IGNORECASE,
)

# High-precision markers of a name/news/photo-credit caption (not a visual description).
_BAD_MARKER_RE = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\b(?:says?|said|president|senator|secretary|minister|governor|mayor|congressman"
    r"|congresswoman|actor|actress|singer|rapper|celebrity|politician|ceo|chairman"
    r"|getty|reuters|copyright|instagram)\b"
    r"|stock photo|photo by",
    re.IGNORECASE,
)


def strip_subject(caption: str) -> str:
    """'a woman with long hair' -> 'with long hair'; tidy dangling verbs."""
    s = caption.strip().rstrip(".").strip()
    out = _SUBJECT_RE.sub("", s, count=1).strip()
    if not out:
        out = s
    out = re.sub(r"^(?:has|have)\s+", "with ", out, count=1, flags=re.IGNORECASE)
    out = re.sub(r"^(?:is|are|was|were)\s+", "", out, count=1, flags=re.IGNORECASE)
    return out.strip()


def is_clean_description(desc: str) -> bool:
    """False if the description looks like a name/news/photo-credit, not appearance."""
    return bool(desc) and _BAD_MARKER_RE.search(desc) is None


def clean_caption(raw: str) -> str:
    """Strip subject boilerplate and drop non-visual (name/news) captions to ''."""
    desc = strip_subject(raw)
    return desc if is_clean_description(desc) else ""


class Blip2Captioner:
    def __init__(
        self,
        model_id: str = "Salesforce/blip2-opt-2.7b",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        num_beams: int = 3,
        max_new_tokens: int = 30,
        min_new_tokens: int = 5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        # use_fast=False: BLIP-2's tokenizer.json is a newer `tokenizers` format than
        # the pinned 0.19 can parse; the slow GPT2 tokenizer reads vocab/merges.
        self.processor = Blip2Processor.from_pretrained(model_id, use_fast=False)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def describe_batch(self, images: list[Image.Image]) -> list[tuple[str, str]]:
        """Return (clean_description, raw_caption) per image. clean='' if filtered."""
        imgs = [im.convert("RGB") for im in images]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device, self.dtype)
        out = self.model.generate(
            **inputs,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
        )
        raw = [c.strip() for c in self.processor.batch_decode(out, skip_special_tokens=True)]
        return [(clean_caption(c), c) for c in raw]
