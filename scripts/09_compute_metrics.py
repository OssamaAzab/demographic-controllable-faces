"""09: Compute the 10-metric suite over the benchmark outputs.

Inputs:  results/benchmark/generations/...
         results/benchmark/manifest.jsonl
         data/ffhq_test.jsonl  (FID/KID reference set)
         data/identity_gallery/id_*/ref.jpg  (identity refs)
Outputs: results/results.csv  (long-format: one row per (method, id, prompt, seed, metric))
         results/tables/headline_table.csv
         results/tables/by_demographic.csv
         results/figures/identity_vs_control_pareto.png
         results/figures/by_method_radar.png
         results/figures/failure_grid.png

Metrics:
    Identity:         AdaFace cossim, ArcFace cossim
    Demographic:      MiVOLO age MAE, FairFace race accuracy
    Accessory:        CelebA classifier, CLIP zero-shot
    Prompt alignment: CLIP-Score
    Quality:          HPSv2, PickScore
    Distribution:     CLIP-FID, KID  (both with face-crop preprocessing)
    Perceptual:       DreamSim

Pre-processing for FID/KID:
    Crop both generations and FFHQ reference to face-only at 224×224 aligned,
    using MediaPipe FaceMesh + 5-point alignment. Removes background confound.
"""

from dcfaces.paths import (
    BENCHMARK_DIR,
    FFHQ_TEST,
    FIGURES_DIR,
    IDENTITY_GALLERY,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dirs,
)


def main() -> None:
    ensure_dirs()
    print(
        f"TODO: compute metrics — bench={BENCHMARK_DIR}, "
        f"fid_ref={FFHQ_TEST}, ids={IDENTITY_GALLERY}, "
        f"csv={RESULTS_DIR}/results.csv, tables={TABLES_DIR}, figs={FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()
