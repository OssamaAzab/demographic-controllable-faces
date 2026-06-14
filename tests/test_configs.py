"""Smoke tests: every config in configs/ parses, and benchmark.yaml carries the
sections the pipeline reads. No weights, GPU, or external repos required.
"""

import yaml

from dcfaces.paths import CONFIG_DIR


def _load(name: str) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
    assert isinstance(cfg, dict) and cfg, f"{name} did not parse to a non-empty dict"
    return cfg


def test_all_configs_parse():
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert files, "no config files found"
    for f in files:
        cfg = yaml.safe_load(f.read_text())
        assert isinstance(cfg, dict) and cfg, f"{f.name} did not parse to a non-empty dict"


def test_benchmark_config_has_expected_sections():
    cfg = _load("benchmark.yaml")
    for key in ("model", "methods", "prompts", "seeds"):
        assert key in cfg, f"benchmark.yaml missing '{key}'"
    assert isinstance(cfg["seeds"], list) and cfg["seeds"], "seeds should be a non-empty list"
