# R2-OOS — Encoder-Held-Out Nested Readout

## 1. Provenance

**Status: `PROVENANCE_INCONCLUSIVE`; no R2-OOS evaluation was run.** The current CAL-CHECK is held out from fitting the R2 offset and temperatures, but encoder exposure is unresolved. The C1 fine-tuning run used a local train JSONL loader with 12,000 samples per epoch; comparison to the current train file yields a candidate overlap of 37 CAL-CHECK images/430 rows. The C1 run did not record the train-file hash or consumed IDs. C1 also inherited `pure_best_adapt_light_mR50.pt`, whose training and checkpoint-selection populations are unknown (`docs/HISTORICAL_CHECKPOINT_MANIFEST.md`). Full detail: `R2_ENCODER_PROVENANCE_AUDIT.md/json`.

## 2. Protocol Freeze

No R2-OOS protocol was frozen because no population could be proved outside the full C1 encoder-training and model-selection lineage. A five-fold adapter cross-fit cannot correct unknown encoder exposure. No practical LL threshold was selected: the eligibility gate failed before evaluation, and no pre-existing incremental-LL threshold was found that could be applied to a valid OOS population.

## 3. Population

No eligible R2-OOS population was established. Current validation (10,401 images/132,556 rows) and test (10,403/132,334) are disjoint from the current train IDs, but they cannot be certified against the historical initializer's unknown training or checkpoint-selection populations. No new image population was constructed.

## 4. Nested Formulation

Not executed. The existing R2 nested construction remains documented in `research_analysis/R2_nested_nuisance_analysis.md`: fixed parent, nonnegative frozen-feature offset, exact alpha-zero parent recovery. No new fit or score was performed.

## 5. Cross-Fitting

Not run. Fold assignment would not make an encoder-in-sample feature encoder-held-out.

## 6. Results

No R2-OOS metrics. Delta, accuracy, macro recall, and practical-threshold decision are unavailable.

## 7. Fold Consistency

No folds were created; fold signs, median, and range are unavailable.

## 8. Statistical Uncertainty

No bootstrap was run. The specified 2,000-replicate, seed-11 image bootstrap was not applied because there is no valid OOS estimate to bootstrap.

## 9. Predicate Effects

No new predicate-level analysis. Existing corrected R2 class effects remain descriptive and are limited to its original CAL-CHECK population.

## 10. Interpretation

The existing nested R2 result remains a decoder-held-out predictive comparison on its CAL-CHECK rows, but the representation's encoder exposure is unresolved and candidate direct C1 overlap is present. It must not be described as encoder-held-out generalization. This is a provenance limitation, not evidence that the nested formulation failed.

## 11. What This Establishes

The provenance audit establishes that CAL-FIT/CAL-CHECK image IDs are frozen, all belong to the current canonical train split, and the current-file reconstruction places 119 CAL-FIT and 37 CAL-CHECK images in C1's 12k-per-epoch sample frame. It establishes that C1 fine-tuning used a fixed three-epoch budget with no best-checkpoint selection or early stopping.

## 12. What It Does Not Establish

It does not establish exact historical C1 consumed-ID overlap, upstream checkpoint training/selection overlap, encoder-held-out residual utility, or absence of leakage. It does not produce a new effect estimate.

## 13. Publication Consequence

Retain the corrected R2 point estimate and interval only as an exploratory, decoder-held-out comparison with unresolved encoder provenance. Do not call it encoder-generalization evidence. R1 remains closed; R3 remains partial; the portfolio remains `MERGE_A_B_AND_FOLD_C`. No further experiment is authorized.
