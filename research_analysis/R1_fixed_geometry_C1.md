# R1 — Fixed-geometry C1 condition (pre-execution registration)

## Status before run

The scientific question met the R1 justification criteria in `portfolio_gate_decision.md`, but execution is `R1_BLOCKED` by the GPU gate below. This is one seed-1234 diagnostic arm, not a sweep. The existing launcher already defines C1a (`geom_input_pixel_space=true`, `geom_fourier_scale=1.0`); no geometry implementation changes are needed. Existing regression tests explicitly establish that normalized boxes collapse six channels and pixel-scale boxes retain variation (`tests/test_geometry_contract_flags.py`).

## Hypothesis and estimand

Hypothesis: restoring pixel-scale box units alone, while retaining the C0 Fourier scale, changes within-pair predicate discrimination relative to frozen C0. Estimand: paired difference in macro WPRD across the same eligible validation cells for C1a versus C0, at seed 1234 and the fixed evaluator protocol. This is a diagnostic estimate for this checkpoint/data pipeline, not a population-level causal or semantic claim.

## Treatment and comparator

- C1a: `geom_input_pixel_space=true`; `geom_fourier_scale=1.0`.
- C0: `geom_input_pixel_space=false`; `geom_fourier_scale=1.0`.
- Their only intended treatment difference is the geometry input coordinate contract. Both use the same original seed-1234 launcher, source commit, initialization checkpoint, split, sampler, optimizer, losses, sample budget and evaluator.
- No C1 checkpoint is resumed. Both treatment paths start from the common frozen PURE initialization used by the registered C0/C1 experiment.

## Frozen run recipe

Launcher: `scripts/train/run_c0_c1.sh`; `ARM=C1a`, `SEED=1234`, 3 epochs, 12,000 samples/epoch, batch 6 × accumulation 4, LR 2e-5 cosine, 300 warmup steps, AMP bfloat16, gradient checkpointing, full historical stage-3 objective, final fixed-budget checkpoint (no validation selection). Evaluation: `tools/c0_c1_evaluate.py --arm C1a`, full validation, GT pairs, alpha 3.75 frequency prior, ensemble alpha 0, cap 64; compare against frozen `runs/eval_C0/result.json` with `tools/c0_c1_compare.py`.

Primary: paired-cell macro WPRD difference and existing 2,000-replicate paired cell bootstrap, seed 11, as implemented by the comparison tool. Secondary: R@50, mR@50, gate and geometry channel summaries, and exact population/cell identity checks. The prior study’s +0.010 WPRD threshold is shown only as context; C1a was documented as diagnostic and this does not create a new confirmatory success threshold.

## Stopping rule and resource limit

Run this one training arm and one full evaluation only. Before each GPU launch require a fresh idle-L4 check. If GPU contention, OOM, output collision, or unsafe disk margin occurs, stop and preserve artifacts; no retry with changed settings. Expected cost is one prior-comparable C1 training (historically about 2.4 hours) plus one full evaluation (historically several hours); combined budget is capped at 12 GPU-hours for this gate. No second seed, architecture change, N4 work, validation tuning or additional analysis run is allowed.

## Interpretation matrix

| Outcome | Interpretation |
|---|---|
| Positive paired WPRD shift with interval excluding zero | Evidence that the pixel-unit geometry change alone affects within-pair score discrimination in this pipeline; magnitude vs +0.010 remains descriptive. |
| Interval includes zero or effect is small | The bundled C0→C1 result cannot be attributed to the units repair alone; separate Paper C support weakens. |
| Negative shift | Restored units alone do not reproduce the bundled positive shift; Fourier change or interaction is a live alternative. |
| Any integrity/resource failure | R1 blocked; no scientific inference. |

## Claim boundary

Even a positive result does not show the model uses geometry through the hypothesized fusion mechanism, does not rule out box-geometry shortcuts, does not establish relation semantics, and does not validate WPRD as a general SGG metric. This result can only isolate one factor in the historical bundled intervention.

## Run record

Execution status: **`R1_BLOCKED`**. A fresh check at 2026-10-06 07:53:38 UTC found an L4 workload (PID 24793, 5,794 MiB reported process memory, 15% utilization). No training/evaluation process was launched and no wait/retry was performed. The baseline init checkpoint hash was `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442`; free space was 24 GiB. The registration is preserved, but the scientific result is **not observed**.
