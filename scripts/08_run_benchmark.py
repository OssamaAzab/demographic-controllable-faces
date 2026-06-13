"""08: Run the benchmark — generate every (method, identity, prompt, seed) image.

Walks the method list in configs/benchmark.yaml, and for each runnable method
generates the full prompt set across all identities and seeds. Output is laid
out so script 09 can score it directly, and a manifest row is written per image.

Inputs:  configs/benchmark.yaml
         data/identity_gallery/id_*/{ref.jpg,metadata.json}
         models/dreambooth_loras/id_*/  +  models/demo_lora_checkpoints/step_4000/
Outputs: results/benchmark/generations/{method}/{id}/{prompt_key}/seed_*.png
         results/benchmark/manifest.jsonl

Runnable today: the DreamBooth variants. The reference-image methods (PuLID,
PhotoMaker, IP-Adapter, HyperLoRA) are skipped with a note until each is wired
into dcfaces.benchmark.methods.

The run is resumable: an image that already exists is skipped, so the script can
be stopped and restarted in chunks. Examples:

    python scripts/08_run_benchmark.py --dry-run
    python scripts/08_run_benchmark.py --methods dreambooth_lora --limit-identities 1 --seeds 42
"""

from __future__ import annotations

import argparse
import json

import torch
import yaml

from dcfaces.benchmark import build_method, expand_prompts
from dcfaces.paths import BENCHMARK_DIR, CONFIG_DIR, IDENTITY_GALLERY, ensure_dirs


def load_identities(limit: int):
    ids = sorted(d for d in IDENTITY_GALLERY.glob("id_*") if (d / "ref.jpg").exists())
    if not ids:
        raise SystemExit(f"No identities in {IDENTITY_GALLERY} — run 03_build_identity_gallery.py first.")
    return ids[:limit] if limit else ids


def gender_of(id_dir) -> str:
    return json.loads((id_dir / "metadata.json").read_text())["fairface"]["gender"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default=None, help="Comma list of method names; default = all in config.")
    parser.add_argument("--limit-identities", type=int, default=0, help="Cap #identities (smoke).")
    parser.add_argument("--seeds", default=None, help="Comma list overriding config seeds, e.g. 42,43.")
    parser.add_argument("--categories", default=None, help="Comma list of prompt categories to keep.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images that already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    ensure_dirs()
    torch.set_grad_enabled(False)  # inference only; PuLID's custom sampler isn't internally no_grad
    cfg = yaml.safe_load((CONFIG_DIR / "benchmark.yaml").read_text())

    prompts = expand_prompts(cfg["prompts"])
    if args.categories:
        keep = {c.strip() for c in args.categories.split(",")}
        prompts = [p for p in prompts if p["category"] in keep]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["seeds"]
    identities = load_identities(args.limit_identities)

    methods = cfg["methods"]
    if args.methods:
        want = {m.strip() for m in args.methods.split(",")}
        methods = [m for m in methods if m["name"] in want]

    total = len(methods) * len(identities) * len(prompts) * len(seeds)
    print(f"plan: {len(methods)} methods × {len(identities)} ids × {len(prompts)} prompts × {len(seeds)} seeds "
          f"= {total} generations")
    for m in methods:
        print(f"  - {m['name']} ({m['identity_method']})")
    if args.dry_run:
        for p in prompts:
            print(f"    [{p['key']:<18}] {p['template']}")
        return

    gen_root = BENCHMARK_DIR / "generations"
    manifest = (BENCHMARK_DIR / "manifest.jsonl").open("a")
    device = "cuda"
    made = skipped = 0
    try:
        for mcfg in methods:
            try:
                method = build_method(mcfg, cfg["model"], cfg["inference"], device)
            except NotImplementedError as e:
                print(f"skip {mcfg['name']}: {e}")
                continue
            print(f"=== {mcfg['name']} ===")
            for id_dir in identities:
                id_name = id_dir.name
                gender = gender_of(id_dir)
                phrase = method.identity_phrase(gender)
                for p in prompts:
                    prompt = p["template"].replace("{identity}", phrase)
                    out_dir = gen_root / mcfg["name"] / id_name / p["key"]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for seed in seeds:
                        out_path = out_dir / f"seed_{seed}.png"
                        if out_path.exists() and not args.overwrite:
                            skipped += 1
                            continue
                        img = method.generate(id_name, gender, id_dir / "ref.jpg", prompt, seed)
                        img.save(out_path)
                        manifest.write(json.dumps({
                            "method": mcfg["name"], "id": id_name, "gender": gender,
                            "prompt_key": p["key"], "category": p["category"], "prompt": prompt,
                            "seed": seed, "path": str(out_path.relative_to(BENCHMARK_DIR)),
                        }) + "\n")
                        manifest.flush()
                        made += 1
                print(f"  {id_name} done")
            method.close()
    finally:
        manifest.close()
    print(f"done: {made} generated, {skipped} already present -> {gen_root}")


if __name__ == "__main__":
    main()
