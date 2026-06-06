"""Portability sanity tests: every path should resolve to a location inside
the repo, regardless of where the user clones it. This is the test we run to
catch any accidentally-introduced hardcoded absolute paths.
"""

from pathlib import Path

from dcfaces.paths import (
    BENCHMARK_DIR,
    CACHE_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DEMO_LORA,
    DEMO_LORA_CKPTS,
    DREAMBOOTH_LORAS,
    FFHQ_METADATA,
    FFHQ_RAW,
    FFHQ_TEST,
    FFHQ_TRAIN,
    FFHQ_VAL,
    FIGURES_DIR,
    HF_CACHE_DIR,
    IDENTITY_GALLERY,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TABLES_DIR,
)

ALL_PATHS = [
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    CACHE_DIR,
    HF_CACHE_DIR,
    CONFIG_DIR,
    FFHQ_RAW,
    FFHQ_METADATA,
    FFHQ_TRAIN,
    FFHQ_VAL,
    FFHQ_TEST,
    IDENTITY_GALLERY,
    DEMO_LORA,
    DEMO_LORA_CKPTS,
    DREAMBOOTH_LORAS,
    BENCHMARK_DIR,
    FIGURES_DIR,
    TABLES_DIR,
]


def test_project_root_is_repo_root():
    """PROJECT_ROOT should be the repo root, identified by README.md presence."""
    assert (PROJECT_ROOT / "README.md").exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert (PROJECT_ROOT / "src" / "dcfaces" / "paths.py").exists()


def test_no_hardcoded_absolute_paths():
    """Every path constant should be relative to PROJECT_ROOT."""
    for p in ALL_PATHS:
        assert isinstance(p, Path)
        assert PROJECT_ROOT in p.parents or p == PROJECT_ROOT, (
            f"Path {p} is not inside PROJECT_ROOT={PROJECT_ROOT}"
        )


def test_no_scratch_in_paths():
    """Catch the most common foot-gun: a hardcoded /scratch/ path."""
    for p in ALL_PATHS:
        s = str(p)
        if "/scratch/" in str(PROJECT_ROOT):
            continue  # repo legitimately lives under /scratch on dev machine
        assert "/scratch/" not in s, f"Hardcoded /scratch/ in path: {p}"


def test_ensure_dirs_is_idempotent():
    """ensure_dirs() should be safe to call multiple times."""
    from dcfaces.paths import ensure_dirs

    ensure_dirs()
    ensure_dirs()  # should not raise


def test_hf_cache_inside_repo():
    """HF cache must live inside the repo, not in user's home directory."""
    assert PROJECT_ROOT in HF_CACHE_DIR.parents
