# R2 — Properly Nested Nuisance Readout

## 1. Scientific Question

Does frozen `rel_feat` add incremental predictive utility beyond a fixed FIT-trained ordered-object-pair prior (`O`) and beyond the registered FIT-trained geometry-plus-pair parent (`G+O`), under an exactly nested score construction? This is a bounded CAL-CHECK predictive analysis, not a semantic test.

## 2. Why the Previous R2 Was Insufficient

The prior R2 report's headline `ΔLL = −0.622110` compared raw pair-prior logits with an added rel-feature offset and did not use the calibrated pair-prior comparator. That number is retracted as the intended R2 result. The underlying score values remain preserved in the JSON with their historical status. The previous analysis also omitted the primary `G+O` parent contrast.

The old `O=1.632362` score is a temperature-calibrated score on the fixed validation population (1,182 images / 14,991 rows); the previous R2 `O=1.807878` is the raw same-FIT count lookup on 250 CAL-CHECK images / 3,142 rows. Their underlying pair counts, 51-label vocabulary, Laplace smoothing, and unsupported-pair FIT-global backoff agree, but population and calibration do not. The values are not directly comparable. Likewise, historical `O+rel_feat=1.266329` is full-validation/full-CAL-calibrated, whereas `1.185767` was CAL-CHECK after CAL-FIT residual fitting. Those values are also not directly comparable. See [R2_population_reconciliation.md](R2_population_reconciliation.md).

## 3. Population Reconciliation

| Quantity | Old result | Previous R2 | Match? |
|---|---:|---:|---|
| O pair prior LL | 1.632362; validation 1,182 images / 14,991 rows; full-CAL temperature | 1.807878; CAL-CHECK 250 / 3,142; raw logits | No: same fixed pair table/support rule, different population and calibration |
| O + rel_feat LL | 1.266329; validation 1,182 / 14,991; full-CAL fitting/calibration | 1.185767; CAL-CHECK 250 / 3,142; CAL-FIT residual fit and different calibration | No |

All current corrected arms below use the *same* CAL-FIT fitting rows and CAL-CHECK evaluation keys. The exact row key is `(image_id, subject_index, object_index, normalized predicate)`; targets are checked against the frozen 51-class vocabulary. No validation outcomes, E2 rows, or CAL-CHECK labels enter fitting.

## 4. Canonical O Baseline

`O` is the saved fullscale FIT-only ordered subject/object first-label count prior: 51-class Laplace-1 smoothing per pair, with unsupported pairs backed off to the FIT-global Laplace-1 prior. The count table and pair support are fixed; the previous independently fit O-only neural approximation is not used. A single positive temperature is fitted on the 750-image / 9,078-row CAL-FIT subset. Its calibrated logits are the fixed parent. The CAL-CHECK population has 2,610 supported pair rows and 532 backoff rows.

On all 14,991 saved validation rows, the already-published calibrated prior reproduces at LL 1.6323624327, accuracy 0.645854, macro recall 0.200061. This is a reproduction check only; validation was not used to fit this correction.

## 5. G+O Parent

`G+O` reuses the registered saved FIT-only estimator; no new geometry definition or learned weights are introduced. It is the fixed pair-log-probability offset plus the exact registered 19-dimensional geometry feature vector, standardized with FIT mean/sample standard deviation (scale floor 1e-6), an intercept, and the saved 20×51 matrix (1,020 FIT-fitted weights). Its positive temperature is fitted on CAL-FIT only. Geometry construction is the frozen `geometry19` in `runs/paper_c_fullscale_residual_audit_20261003T084403Z/fullscale_residual_runner.py`; model definition and optimizer are in `fullscale_residual_registration.json`.

The saved validation parent reproduces at LL 1.3242384157, accuracy 0.656127, macro recall 0.121893. Again this is a saved-parent integrity check, not new validation scoring for model selection.

## 6. Nested rel_feat Extension

For each parent `P ∈ {O, G+O}`, CAL-FIT first fits a parent temperature `T_P` and a rel-feature temperature `T_R`. Then one nonnegative scalar `α_P` is fitted by minimizing CAL-FIT cross-entropy for:

```text
z_P = fixed_parent_logits / T_P
z_P+R = z_P + α_P * log_softmax(frozen_relfeat_logits / T_R)
```

The optimizer is CPU LBFGS with softplus-constrained α. The exact parent (`α=0`) is explicitly included as a candidate and selected whenever the optimized extension does not improve the *same CAL-FIT objective*. CAL-CHECK logits are joined by identity; the saved CAL-CHECK pilot sidecar's rel-feature offset is recovered algebraically from its saved raw combined and N4 logits. The recovered log-probabilities reproduce the saved offset to max absolute error below `1e-8`. The rel-feature predictor is frozen.

## 7. Nesting Check

The parent is not refit or allowed to drift. By construction `z_P+R(α=0) == z_P` exactly. On CAL-FIT, selected extension cross-entropy is no worse than the fixed parent for both O and G+O; the recorded optimizer/objective check is true in [R2_nested_nuisance_analysis.json](R2_nested_nuisance_analysis.json). Thus there is no comparison against an independently trained replacement parent. Held-out log-loss may of course be worse in another dataset; that would not invalidate nesting.

## 8. Results

All four arms share CAL-CHECK: 250 images / 3,142 rows. The calibrated/final column is the parent CAL-FIT-temperature-scaled score plus the frozen CAL-FIT-fitted offset (no post-hoc CAL-CHECK calibration).

| Parent / extension | α | Raw parent LL | Raw extended LL | Calibrated parent LL | Calibrated extended LL | Calibrated ΔLL (95% image CI) | Accuracy parent → extended | Macro recall parent → extended |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O → O + rel_feat | 0.410449 | 1.807878 | 1.208272 | 1.598961 | 1.280646 | −0.318314 [−0.357666, −0.276130] | .663590 → .682050 | .227013 → .200346 |
| G+O → G+O + rel_feat | 0.196844 | 1.313680 | 1.281671 | 1.315260 | 1.255252 | **−0.060008 [−0.074317, −0.044480]** | .667091 → .668364 | .137086 → .135990 |

The G+O contrast is primary. Raw ΔLL is −0.032009 (95% image-cluster CI [−0.046121, −0.016764]); accuracy delta +0.002546; macro-recall delta −0.000597. The O contrast's raw ΔLL is −0.599605 (95% CI [−0.642659, −0.555209]); accuracy delta +0.018778; macro-recall delta −0.034440. CI protocol: paired pooled row-mean loss difference, resampling images with replacement, 2,000 replicates, seed 20261003. These intervals describe this pilot population and are not full-scale confirmatory intervals.

## 9. Per-Predicate Effects

Per-predicate support, recall, and precision changes for every predicate present on CAL-CHECK are in [R2_per_predicate_effects.csv](R2_per_predicate_effects.csv), sorted by contrast then support. Accuracy rises while macro recall declines for both O and G+O: the aggregate change is not a uniform predicate gain. Under O, recall improves for 6 classes and declines for 14 (26 unchanged); under G+O it improves for 8, declines for 6 (32 unchanged). For the largest G+O classes, `on` (support 1,124) recall changes .946619→.948399, `has` (418) .892344→.901914, `in` (404) .529703→.514851, `wearing` (258) .821705→.837209, and `of` (249) .526104→.522088. This mix does not establish broad or tail-predicate improvement.

## 10. Interpretation

**SUPPORTED, for the bounded nested predictive estimand.** On this 250-image CAL-CHECK sample, adding the frozen rel-feature log-probability offset to the fixed G+O parent improves calibrated LL by 0.060008 nats/row, with the exploratory image-cluster interval below zero. The same-direction O contrast is larger, consistent with the G+O parent absorbing some predictive utility. Since this is CAL-CHECK from a bounded pilot and only one frozen rel-feature readout, the result is exploratory and does not establish population-wide effect size or semantic specificity.

U/S remain unavailable under the current clean protocol: the identity-proven historical CLIP cache covers only a 1,200-image pilot, not the frozen full FIT/CAL populations. No U/S model is claimed here.

## 11. What This Does Not Establish

This does not establish relational semantics, causality, information unavailable from images, relation-specific representation, improvement on interactional predicates, or residual utility beyond full-population G+O+U+S. It does not justify C1 residual retraining or a fullscale N4 reconstruction. The pilot is small, predicate effects are heterogeneous, and U/S were not evaluated.

## 12. Publication Consequence

Corrected R2 replaces the erroneous comparator interpretation and supports only a bounded, exploratory nested G+O+rel_feat predictive increment. `ΔLL = −0.622110` is explicitly retracted as the intended R2 headline. R1 was not launched in this correction; the previous portfolio decision and publication scope should remain conservative. Any future R1 requires a separate frozen design and fresh resource/process audit, including PID 24793 ownership/tree confirmation before any decision to stop it.

Reproduce the correction with `CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/r2_nested_correction.py`; all source artifact hashes are recorded in the companion JSON.
