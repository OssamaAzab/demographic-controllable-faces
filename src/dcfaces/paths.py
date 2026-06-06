"""Project-relative path constants. The single source of truth for I/O locations.

PROJECT_ROOT is auto-detected from this file's location, so the project works
regardless of where the user clones the repo. No script in this codebase
hardcodes an absolute path.

The HuggingFace cache is forced to live inside the repo (.cache/hf/) so that
model downloads don't pollute the user's home directory or get re-downloaded
when the repo is moved.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"
HF_CACHE_DIR: Path = CACHE_DIR / "hf"

FFHQ_RAW: Path = DATA_DIR / "ffhq_1024"
FFHQ_METADATA: Path = DATA_DIR / "ffhq_metadata.jsonl"
FFHQ_TRAIN: Path = DATA_DIR / "ffhq_train.jsonl"
FFHQ_VAL: Path = DATA_DIR / "ffhq_val.jsonl"
FFHQ_TEST: Path = DATA_DIR / "ffhq_test.jsonl"
IDENTITY_GALLERY: Path = DATA_DIR / "identity_gallery"

DEMO_LORA: Path = MODELS_DIR / "demo_lora.safetensors"
DEMO_LORA_CKPTS: Path = MODELS_DIR / "demo_lora_checkpoints"
DREAMBOOTH_LORAS: Path = MODELS_DIR / "dreambooth_loras"

BENCHMARK_DIR: Path = RESULTS_DIR / "benchmark"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "hub"))


def ensure_dirs() -> None:
    """Create all write-target directories if missing. Safe to call anytime."""
    for d in (
        DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        CACHE_DIR,
        HF_CACHE_DIR,
        IDENTITY_GALLERY,
        DEMO_LORA_CKPTS,
        DREAMBOOTH_LORAS,
        BENCHMARK_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
