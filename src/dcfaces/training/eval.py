"""Mid-training task eval: does the LoRA respond to demographic tokens yet?

Generates a few images per eval prompt, scores them with FairFace, and reports
age MAE + race accuracy vs the *prompted* demographics. This is the task-aware
signal used to watch training (and later, in 06, to pick checkpoints) -- NOT
val-MSE (the Part B lesson). Age uses the FairFace age head's bucket midpoint as
a stand-in for MiVOLO; MiVOLO can be slotted in later for the final metrics.
"""

from __future__ import annotations

import re

import torch

AGE_MIDPOINTS = {
    "0-2": 1, "3-9": 6, "10-19": 15, "20-29": 25, "30-39": 35,
    "40-49": 45, "50-59": 55, "60-69": 65, "70+": 75,
}


def parse_prompted_age(prompt: str) -> int | None:
    m = re.search(r"\b(\d{1,3})\s*year", prompt)
    return int(m.group(1)) if m else None


def parse_prompted_race(prompt: str, races: list[str]) -> str | None:
    # longest race name first so "South Asian" wins over "Asian" substrings
    for race in sorted(races, key=len, reverse=True):
        if race.lower() in prompt.lower():
            return race
    return None


@torch.no_grad()
def run_task_eval(pipe, prompts, seeds, classifier, races, steps=25, guidance=6.0):
    """Generate prompts x seeds, score with FairFace -> {age_mae, race_acc, n}."""
    from PIL import Image  # noqa: F401  (pipe returns PIL images)

    age_errs, race_hits, race_total = [], 0, 0
    for prompt in prompts:
        target_age = parse_prompted_age(prompt)
        target_race = parse_prompted_race(prompt, races)
        for seed in seeds:
            g = torch.Generator(device=pipe.device).manual_seed(seed)
            # autocast so bf16 activations match the fp32 LoRA params (training does
            # the same; the pipeline doesn't autocast on its own).
            with torch.autocast(pipe.device.type, dtype=torch.bfloat16):
                image = pipe(
                    prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, generator=g
                ).images[0]
            fr = classifier.classify(image)
            if target_age is not None and fr.age_bucket in AGE_MIDPOINTS:
                age_errs.append(abs(AGE_MIDPOINTS[fr.age_bucket] - target_age))
            if target_race is not None:
                race_total += 1
                race_hits += int(fr.race == target_race)

    return {
        "age_mae": (sum(age_errs) / len(age_errs)) if age_errs else float("nan"),
        "race_acc": (race_hits / race_total) if race_total else float("nan"),
        "n": len(prompts) * len(seeds),
    }
