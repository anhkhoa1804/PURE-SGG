# Readout v2 — pilot manifest (written before GPU launch)

This is the second launch attempt. The first (same experiment ID) was
discarded after training completed: its checkpoint had no `clip` key
(`freeze_clip=true` also gates whether CLIP is written into the checkpoint —
see commit `f747671`), which would have invalidated any WPRD comparison had
it been evaluated. It was never evaluated; its artifacts were deleted.

| field | value |
|---|---|
| experiment ID | `readout_v2_pilot_seed1234` |
| branch | `research/pure-complete-readout-v2` |
| git SHA at launch | `f747671b62243476f43f2fb47794817bc586f21a` |
| git dirty | clean (`git status --porcelain` empty at launch) |
| seed | `1234` |
| base checkpoint | `checkpoints/C1_seed1234.pt` |
| base checkpoint SHA256 | `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` |
| dataset | VG150, `datasets_vg150_clean`, `local-jsonl` source |
| split (training) | train, GT-positive pairs only |
| split (endpoint) | validation, full 10,401 images / 132,556 pairs / 20,016 cells (same population as every other C1 WPRD number) |
| geometry | `geom_input_pixel_space=true`, `geom_fourier_scale=0.01` — fixed, never varied |
| readout config | `readout_v2_enabled=true`, `readout_v2_lambda_anchor=0.5`, `readout_v2_lr=2e-3` |
| corrected isolation flags | `explicit_spoa_enabled=false`, `text_conditioned_projection_enabled=false`, `predicate_label_relaxation_enabled=false` |
| CLIP-freeze flags | `freeze_clip=false`, `clip_unfreeze_after_epochs=999999`, `progressive_unfreeze=false` — CLIP saved into the checkpoint but never trainable, decoupled per commit `f747671` |
| inert flags (unchanged from C1 baseline, verified to not touch the v2 pathway) | `adaptive_calibration_enabled=true`, `adaptive_prior_enabled=true`, `bias_residual_enabled=true`, `relationness_enabled=true` — all classifier-branch/relationness-head only, zero-weighted, unused by `adaptive_predicate_logits` |
| optimizer | AdamW, resumed with `--reset_epoch true` (fresh optimizer state — a stale state for a from-scratch parameter would be meaningless) |
| learning rate | `2e-3`, applied via `base_lr` override in the `readout_v2_enabled` block (`train.py`); inert for every other parameter since none of them receive gradients |
| anchor coefficient | `λ_anchor = 0.5` |
| active loss terms | `L = l_readout_v2_ce + 0.5·l_readout_v2_anchor` — every other lambda explicitly zeroed (`lambda_predicate_ce=0.0`, `lambda_spoa_alignment=0.0`, `lambda_dense_grounding=0.0`, `lambda_counterfactual=0.0`, `lambda_text_predicate_ce=0.0`, `lambda_relationness=0.0`, `lambda_calibration_reg/kl/rank=0.0`) |
| sample budget | `samples_per_epoch=250`, `epochs=1` (≈3,000+ GT-positive pairs at this dataset's ~12.7 pairs/image density) |
| pilot budget enforcement | `scripts/train/run_readout_v2.sh` refuses to run without `PILOT=1` (exit 2 otherwise) — regression-tested |
| trainable parameters | exactly 1 tensor, `predicate_prototypes`, 39,168 elements (51×768) — everything else, including CLIP, frozen |
| training output dir | `runs/readout_v2_pilot_seed1234/` |
| training checkpoint | `checkpoints/readout_v2_pilot_seed1234.pt` |
| evaluator (endpoint, to run after training) | `tools/readout_v2_evaluate.py` — mirrors `tools/c0_c1_evaluate.py`'s structure exactly, geometry hardcoded to the C1 contract |
| WPRD estimator | `tools/within_pair_discrimination.py`, unmodified — `cap=64`, macro over cells, seed 0 |
| R0 comparator (already exists, not re-run) | `runs/eval_C1/result.json`, WPRD `0.5749881522134409` |

## Pre-launch checks completed

- Full test suite: 437 passed, 1 skipped (pre-existing, unrelated), 0 failed.
- 18/18 readout_v2 regression tests pass, including runtime-resolved checks
  (not string-grep) for all four corrected flags and the CLIP-freeze
  mechanism.
- Flag-off equivalence: proven at the row level against real data (all
  132,556 GT rows of the existing `runs/eval_C1/pair_logits.pt`, and again
  against a fresh 1,156-row end-to-end smoke run of
  `tools/readout_v2_evaluate.py --readout_v2_enabled false`) — both within
  fp16 storage tolerance (`atol=2e-3`, max observed diff 1.4e-4), the same
  tolerance already established and used throughout this programme's own
  bit-exactness checks.
- Real checkpoint round-trip (CPU-only, `checkpoints/C1_seed1234.pt`): all
  251/251 tensors load with matching shapes; `P`/`E` round-trip exactly
  through save/load.
- `nvidia-smi`: NVIDIA L4, 0 MiB / 23,034 MiB used, 0% utilization, no
  compute processes, immediately before this launch.
