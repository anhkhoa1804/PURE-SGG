# `p71` preregistration — does `p69` replicate on the held-out TEST split?

Committed **before** `runs/p71_pure_visible_geometry_test` is run. CPU only.

## Why

`p69` returned **MATERIAL** (`delta_missing = +0.0341`) and is the sole
justification for spending GPU on `C1a`. It is **validation-only**: it reads
`runs/p36_relfeat_cache` (VG150 validation, 10,401 images / 132,556 pairs).
Every load-bearing claim in this programme that has been promoted past
screening has been replicated on the held-out TEST split (`p59`, `p61`, `p62`,
`p64`); `p69` has not. Promoting a validation-only discovery into a held-out
claim without running the held-out test is exactly the failure mode this
programme's own discipline forbids.

The TEST-split cache **already exists** — `runs/p54_test_relfeat_cache`
(10,403 images / 132,334 pairs), same checkpoint
(`8845c3af…`), same eval configuration (`ensemble_alpha = 0.0`,
`explicit_spoa_enabled false`, `freq_bias_path
datasets_vg150_clean/frequency_prior_train.json`). So this costs **no GPU** and
no new extraction.

## Design

Identical to `p69` in every respect except the dump. Same tool
(`tools/pure_visible_geometry.py`), same estimator (AdamW MLP, hidden 256,
20 epochs, lr 2e-3, l2 1e-4, softmax CE), same 5-fold cross-fitting with salt 0,
same column indices, same arms:

| arm | features |
|---|---|
| `A_relfeat` | 768-d `rel_feat`, standardised |
| `B_geometry` | all 19 geometry numbers |
| `P_pure_visible` | cols 6,7 — the only channels PURE receives non-degenerately |
| `Q_visible_plus_sizes` | cols 6,7 + cols 8–11 (the four size channels) |
| `N_shuffled` | `rel_feat` against permuted labels (null) |
| `P_prior` | prior control (must read exactly 0.5000) |

The tool gains three backwards-compatible flags (`--anchor-a`, `--anchor-b`,
`--skip-anchor-gates`, `--split-label`) whose defaults reproduce `p69` bit-for-bit.
`runs/p69_pure_visible_geometry/est.json` is not touched.

## Primary quantity and decision rule

```
delta_missing_TEST = WPRD(B_geometry) - WPRD(P_pure_visible)
```

The thresholds are **quoted unchanged from `p69`**, not re-chosen:

- **REPLICATES / MATERIAL:** `delta_missing_TEST >= +0.03`.
  `p69` is promoted from a validation finding to a held-out one. `C1a`'s
  motivating measurement stands on TEST.
- **DIRECTIONALLY REPLICATES, WEAKER:** `+0.01 <= delta_missing_TEST < +0.03`.
  The effect is real but smaller out of sample. `C1a` may proceed, but `p69`
  must be quoted with its TEST value, never the validation value alone.
- **FAILS TO REPLICATE:** `delta_missing_TEST < +0.01`. The `p69` result is
  validation-specific. `C1a`'s motivation collapses and the Track C ranking
  must be reversed again — a NO-GO signal that must be honoured.

Secondary (no decision): `delta_size_TEST`, and the `p69` reframing check
`A_relfeat > P_pure_visible` (validation: +0.0097), which is the evidence for
"input starvation, not representational destruction".

## Validity gates

- **G4** `P_prior` reads exactly 0.5000.
- **G5** `N_shuffled` within [0.49, 0.51].
- **G6** every arm scores every GT row, all finite.
- **G1/G2** anchor gates are **explicitly skipped and marked as skipped**: no
  `p60`-equivalent estimator-matched anchor exists for TEST, so this run
  *establishes* the TEST anchors rather than reproducing them. Registered in
  advance so the skip cannot be mistaken for a pass.
- Fold-size gate `G3` is not applicable (TEST has its own fold sizes); the
  observed sizes are recorded so any future TEST run can gate against them.

## Stated in advance

TEST is 10,403 images against validation's 10,401, drawn from the same
distribution by the same pipeline, so a *large* discrepancy would itself be
evidence of an estimator or cache problem rather than of a genuine
generalisation failure. Both readings are reportable; neither is to be
suppressed.
