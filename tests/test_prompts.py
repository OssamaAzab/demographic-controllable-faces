"""Smoke test for benchmark prompt expansion. prompts.py is loaded in isolation so
the test does not pull torch in through the benchmark package __init__; it needs no
weights, GPU, or external repos.
"""

import importlib.util

import yaml

from dcfaces.paths import CONFIG_DIR, PROJECT_ROOT


def _expand_prompts():
    path = PROJECT_ROOT / "src" / "dcfaces" / "benchmark" / "prompts.py"
    spec = importlib.util.spec_from_file_location("dcfaces_prompts_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expand_prompts


def _prompts_cfg() -> dict:
    return yaml.safe_load((CONFIG_DIR / "benchmark.yaml").read_text())["prompts"]


def test_expand_prompts_matches_config():
    pcfg = _prompts_cfg()
    prompts = _expand_prompts()(pcfg)

    # one entry per configured prompt across the five categories (21 for the shipped config)
    expected = (
        len(pcfg["identity_only"])
        + len(pcfg["age"]["ages"])
        + len(pcfg["race"]["races"])
        + len(pcfg["accessory"])
        + len(pcfg["compositional"])
    )
    assert len(prompts) == expected

    assert all({"key", "category", "template"} <= p.keys() for p in prompts)
    keys = [p["key"] for p in prompts]
    assert len(keys) == len(set(keys)), "prompt keys are not unique"
    assert {p["category"] for p in prompts} == {
        "identity", "age", "race", "accessory", "compositional",
    }


def test_placeholders_are_resolved():
    for p in _expand_prompts()(_prompts_cfg()):
        assert "{age}" not in p["template"]
        assert "{race}" not in p["template"]
