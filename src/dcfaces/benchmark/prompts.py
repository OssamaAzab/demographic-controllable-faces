"""Flatten the benchmark prompt config into a keyed, method-agnostic list.

Each entry keeps the literal ``{identity}`` placeholder. The per-method identity
phrase ("sks man" for DreamBooth, "man" for the reference-image methods) is
substituted at generation time, so the same prompt set is reused across methods.
"""

from __future__ import annotations


def expand_prompts(pcfg: dict) -> list[dict]:
    """Return one entry per concrete prompt: {key, category, template}.

    ``key`` is a filesystem-safe identifier used in the output path and manifest.
    ``template`` still contains ``{identity}``; everything else is resolved.
    """
    prompts: list[dict] = []

    for i, p in enumerate(pcfg["identity_only"]):
        prompts.append({"key": f"identity_{i}", "category": "identity", "template": p})

    age = pcfg["age"]
    for a in age["ages"]:
        prompts.append(
            {"key": f"age_{a}", "category": "age", "template": age["template"].replace("{age}", str(a))}
        )

    race = pcfg["race"]
    for r in race["races"]:
        key = r.lower().replace(" ", "_")
        prompts.append(
            {"key": f"race_{key}", "category": "race", "template": race["template"].replace("{race}", r)}
        )

    for i, p in enumerate(pcfg["accessory"]):
        prompts.append({"key": f"accessory_{i}", "category": "accessory", "template": p})

    for i, p in enumerate(pcfg["compositional"]):
        prompts.append({"key": f"compositional_{i}", "category": "compositional", "template": p})

    return prompts
