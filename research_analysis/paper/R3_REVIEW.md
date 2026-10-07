# R3 Review — WPRD

## What does WPRD establish?

The saved implementation computes a macro-average of within-ordered-pair predicate-score AUCs over eligible cells (with a 64-row per-class cap). A score constant within a pair yields WPRD=.5 in the saved pair-prior controls. This supports describing WPRD as a conditional within-pair score-discrimination statistic for the eligible cells and implementation used. It does not establish human semantic correctness.

The held-out test arm table reports Spearman rho +.9142 between R@50 and WPRD and −.7273 between mR@50 and WPRD across 12 complete deterministic scoring arms. These arms share checkpoint/cache lineage and are dependent. Their historical permutation p-values are not independent-model inference. Leave-one-arm-out stability is descriptive and does not fix that dependence.

## What remains unvalidated?

- Planted-shortcut sensitivity is not done.
- A strong-VLM ceiling is not available.
- Dependence-aware inference across independent model families is absent.
- The exact held-out WPRD eligible-cell vectors/counts are not retained in the correlation JSON, limiting independent reconstruction.
- Pair-prior and random controls do not rule out geometry, local appearance, global context, or dataset regularities.
- Human-verified semantic validity and interactional evaluation are absent.

## Which claims must be narrowed?

“WPRD measures relational quality” should become “WPRD summarizes within-pair predicate-score discrimination over eligible cells.” Correlations with R/mR should be described as associations among 12 dependent arms, not evidence that WPRD is an independent or generally valid measure of SGG quality. The chance pair-prior result is a construction-specific control, not proof of semantic validity.

## Does WPRD belong as a main result or diagnostic?

Retain WPRD as a diagnostic endpoint that motivates the measurement audit, with the exact definition and prior-only control in the main methods/results if central to the thesis. Keep cross-arm correlations and their incomplete inference in supplementary material or a compact caveat. It should not anchor a broad metric-validation claim.

## Does the current evidence justify any broad metric-validity conclusion?

No. Current evidence supports a narrow operational interpretation only. R3 remains PARTIAL; no missing control should be described as a negative result.
