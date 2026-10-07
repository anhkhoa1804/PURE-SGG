# R2 Correction Summary

## VERDICT

**SUPPORTIVE, pilot-only:** the corrected nested G+O→G+O+rel_feat CAL-CHECK contrast is −0.060008 calibrated log-loss/row (image-cluster exploratory 95% CI [−0.074317, −0.044480]). This supports incremental predictive utility for this bounded comparison; it does not establish semantic specificity or full-population residual signal. The prior −0.622110 headline is retracted for the intended R2 question.

| Question | Status | Evidence |
|---|---|---|
| Were the old and previous-R2 O scores comparable? | NO | Same FIT count prior, labels, smoothing/backoff; different scored population and calibration (validation calibrated vs CAL-CHECK raw). |
| Does frozen rel_feat improve the nested O parent on CAL-CHECK? | YES, pilot | 1.598961→1.280646 calibrated LL; Δ −0.318314, CI [−0.357666, −0.276130]. |
| Does it improve the nested G+O parent on CAL-CHECK? | YES, pilot | 1.315260→1.255252 calibrated LL; Δ −0.060008, CI [−0.074317, −0.044480]. |
| Is the gain uniform by predicate? | NO | Accuracy rises while macro recall falls; see complete predicate table CSV. |
| Are U/S represented in this clean comparison? | NO | Historical identity-safe cache covers only 1,200 images, not current full FIT/CAL. |
| Was R1 run or authorized here? | NO | R1 remains outside this correction; any future launch needs a frozen design and fresh resource/process gate. |

## What changed

The corrected analysis holds the FIT-trained pair-prior or registered G+O logits fixed, fits temperature and one nonnegative rel_feat offset only on CAL-FIT, and evaluates all four arms on the same identity-joined CAL-CHECK rows. The exact parent is available at α=0, and the fitted extension's CAL-FIT objective is checked against that parent.

## What did not change

No historical result, checkpoint, representation, manifest, validation outcome, or training run changed. No GPU was used; no model was trained; no feature extraction, R1, seed 5678, B2, or validation fitting occurred. The 1.632362 full-validation O LL remains a valid historical metric for its own population/calibration—not the right comparator for pilot raw O LL 1.807878.

## What remains unresolved

The comparison is a 250-image pilot, the predicate profile is heterogeneous, U/S are absent, and the residual can reflect any predictive information in the frozen relation feature—including appearance/context/geometry. The result does not establish full-population conditional information, semantics, causality, or an interactional-family effect.

## Decision boundary

The portfolio remains `MERGE_A_B_AND_FOLD_C`; R2 alone does not establish a separate C geometry contribution. R1 was not started in this task. The current analysis says only that a nested G+O predictive residual appears on CAL-CHECK; it does not authorize a GPU run. See the [full R2 report](R2_nested_nuisance_analysis.md), [population reconciliation](R2_population_reconciliation.md), and [portfolio decision](portfolio_gate_decision.md).

## Tests

The R2 correction regression tests pass (8 passed); modified Python files pass `compileall`; `git diff --check` passes. A full `pytest -q` attempt reached 541 passes, then three existing `tests/test_readout_v2.py` launcher dry-run tests timed out at their 60-second limit. Those dry runs set `PYTHON=/bin/echo` but the shell launcher still calls `nvidia-smi` while recording provenance; that read-only utility did not return in time in this environment. Pytest was interrupted after the timeouts, so the remaining full suite is unverified. These tests and launcher files were not changed. No training process or GPU job was launched.
