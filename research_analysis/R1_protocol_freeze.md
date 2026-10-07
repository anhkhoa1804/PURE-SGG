# R1 protocol freeze — fixed geometry, same seed

Frozen before any R1 training command. Run identity: `runs/R1_fixed_geometry_C1a_20261007T021152Z/`.

## Hypothesis and estimand

H1: correcting the geometry input-units contract alone materially changes the fixed-seed C1 result. H0: the correction does not produce a material WPRD improvement under the registered same-seed protocol.

The repository's existing, identifiable units-only contrast is C1a minus C0: C1a receives pixel-scale boxes and Fourier scale 1.0; C0 receives normalized boxes and Fourier scale 1.0. Both start from the same frozen PURE initialization and use seed 1234. This is the already registered R1 estimand in `research_analysis/R1_fixed_geometry_C1.json`. Historical C1 (WPRD 0.5749881522) is retained as context, not the primary parent: it already uses pixel-scale geometry and additionally changes Fourier scale to 0.01, so a rerun compared to it would not isolate the units correction.

## Frozen protocol

- Seed: 1234; dataset/splits and sampling: exact existing canonical C1/C0 launcher protocol.
- Architecture/objective: unchanged existing C1 architecture and stage-3 full objective.
- Treatment: C1a, `geom_input_pixel_space=true`, `geom_fourier_scale=1.0`.
- Parent: frozen C0, `geom_input_pixel_space=false`, `geom_fourier_scale=1.0`; `runs/eval_C0/result.json`, WPRD `0.5667271196244782`.
- Historical C1 context: `runs/eval_C1/result.json`, WPRD `0.5749881522134409`; not a valid isolated units-only comparator because Fourier scale differs.
- Initialization: load the common frozen PURE initialization `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` (SHA256 `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442`) using the same registered launcher path; reset epoch count and start a fresh optimizer. Do not resume either C0 or C1 trained checkpoint. This matches the historical C0/C1 protocol.
- Training recipe: `scripts/train/run_c0_c1.sh`; 3 epochs, 12,000 samples/epoch, batch 6, accumulation 4 (effective 24), AdamW per the registered stage-3 trainer, LR 2e-5 cosine, 300 warmup steps, seed 1234, AMP bfloat16, gradient checkpointing, full registered objective/loss weights, fixed final-epoch checkpoint and no validation checkpoint selection. The launcher is the executable source of every option.
- Evaluation: one full existing validation evaluation via `tools/c0_c1_evaluate.py --arm C1a`, GT pairs, full split, score mode ensemble with ensemble alpha 0, frequency prior alpha 3.75, cap 64. Compare identical cell values against frozen C0 with `tools/c0_c1_compare.py`; registered 2,000-resample paired-cell bootstrap, seed 11. No tuning or model selection uses validation.
- Primary metric: macro WPRD; primary materiality threshold: `delta_WPRD >= +0.010`, frozen from `docs/PAPER_C_C0_C1_PREREGISTRATION.md` §“Success threshold” and applied here as an R1 decision margin. This is a materiality decision threshold, not a test of statistical significance.
- Secondary diagnostics: the evaluator's registered R@50/mR@50, geometry channel summaries, fusion-gate summaries, prior agreement, and population/cell identity checks. No new analysis campaign.
- Uncertainty: report the existing paired-cell bootstrap CI if the standard comparison tool completes. It is cell-resampling uncertainty, not between-seed uncertainty. This is one same-seed comparison; no seed variance is estimated.
- Stop: exactly one training arm and one full evaluation. Any contention, OOM, non-finite training, integrity mismatch, unsafe disk margin, or other technical failure blocks the run; do not retry or change settings. GPU cost cap: 12 hours total.

No hyperparameter search, second seed, N4, CLIP replay, open-vocabulary work, or C1 residual training is included.
