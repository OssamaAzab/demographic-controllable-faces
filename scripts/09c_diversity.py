"""09c: seed-diversity. For a fixed (method, identity, prompt), the 5 seeds should
give varied renderings of the *same* person (pose, expression, framing) rather than
identical clones. Measured as mean pairwise DreamSim distance across the 5 seeds on
the identity-only prompts (neutral, so the variation isn't an attribute artifact).

Read it ALONGSIDE identity: high diversity + high identity = genuine variety of the
same person; high diversity + low identity = identity drift (not a virtue). Writes
results/tables/diversity.csv; script 09's aggregation joins it.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from dcfaces.metrics.perceptual import DreamSimScorer
from dcfaces.paths import BENCHMARK_DIR, TABLES_DIR


def main() -> None:
    manifest = [json.loads(line) for line in (BENCHMARK_DIR / "manifest.jsonl").read_text().splitlines() if line.strip()]
    groups: dict = defaultdict(list)
    for r in manifest:
        if r["category"] == "identity":
            groups[(r["method"], r["id"], r["prompt_key"])].append(r["path"])

    dsim = DreamSimScorer("cuda")
    per_method: dict = defaultdict(list)
    for (method, _id, _pk), paths in groups.items():
        imgs = [Image.open(BENCHMARK_DIR / p).convert("RGB") for p in sorted(paths)]
        dists = [dsim.distance(a, b) for a, b in itertools.combinations(imgs, 2)]
        if dists:
            per_method[method].append(float(np.mean(dists)))

    table = pd.DataFrame(
        [{"method": m, "seed_diversity": float(np.mean(v))} for m, v in per_method.items()]
    ).set_index("method").sort_values("seed_diversity", ascending=False)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "diversity.csv")
    print(table.round(3).to_string())
    print("DIVERSITY_OK")


if __name__ == "__main__":
    main()
