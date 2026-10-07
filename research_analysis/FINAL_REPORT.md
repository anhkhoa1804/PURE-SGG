# Portfolio Gate — Final Report

## Verdict

Corrected R2 supports an exploratory, bounded nested predictive increment for frozen `rel_feat` beyond the saved FIT-trained G+O predictor. It does not establish semantic specificity or full-population residual signal. The prior `−0.622110` R2 headline is retracted. Portfolio remains `MERGE_A_B_AND_FOLD_C`; R1 was not run and remains blocked/not authorized by this correction.

## What Was Verified

- Frozen full nuisance split: FIT 75,066 images / 943,815 rows; CAL 8,183 images / 102,612 rows; 170 E2 image IDs excluded.
- Bounded N4-FIT pilot partition: CAL-FIT 750 / 9,078 rows, CAL-CHECK 250 / 3,142 rows. The four corrected arms use identical CAL-CHECK relation keys and frozen 51-class labels.
- Ordered pair prior: full FIT pair table, Laplace-1 smoothing, FIT-global Laplace-1 backoff; 197,661 observed ordered pairs. CAL-CHECK support is 2,610/3,142.
- R3 correlation tables: 12 complete arm rows in each p49/p53/p61 artifact, all from a shared checkpoint/cache family. Prior-only WPRD is 0.5; planted shortcut and strong-VLM ceiling evidence are absent.
- Historical geometry study bundled pixel-unit and Fourier-scale changes. No corrected fixed-geometry R1 outcome exists.

## R2 — Nested Nuisance

On the common CAL-CHECK sample, calibrated O LL is 1.598961 and O+rel_feat is 1.280646 (Δ −0.318314; exploratory image-cluster 95% CI [−0.357666, −0.276130]). Primary: calibrated G+O LL is 1.315260 and G+O+rel_feat is 1.255252 (Δ **−0.060008**; exploratory image-cluster 95% CI **[−0.074317, −0.044480]**). Alpha values were 0.410449 for O and 0.196844 for G+O, fit on CAL-FIT only. Accuracy increases slightly for both contrasts, while macro recall decreases for both; gains are not uniform. The parent logits remain fixed, and alpha=0 recovers them exactly. Full raw/calibrated metrics, per-predicate values, identity checks, and source hashes are in `R2_nested_nuisance_analysis.json` and `R2_per_predicate_effects.csv`.

The older O LL 1.632362 is a calibrated score on 14,991 validation rows, while the previous R2 O LL 1.807878 is raw on 3,142 CAL-CHECK rows. Same prior recipe; different population/calibration; not directly comparable. The old `O+rel_feat=1.266329` validation result and previous `1.185767` CAL-CHECK result also differ in population and calibration fitting. See `R2_population_reconciliation.md`. The calibrated old O result is reproduced from saved validation logits only as an integrity check.

## R3 — WPRD Construct Validity

WPRD remains a narrow within-pair predicate discrimination statistic, not a general SGG quality or semantic metric. Spearman sample size is 12 dependent scoring arms and one underlying checkpoint/cache family. Historical permutation p-values are not model-level inference. Prior-only and random-null controls exist; no planted relational-shortcut sensitivity or strong-VLM ceiling is available. See `R3_WPRD_construct_validity.md`.

## R1 — Fixed Geometry C1

Not run. The previous gate recorded an occupied GPU resource (PID 24793); no process was killed. No R1 outcome is inferred. This correction does not authorize an R1 launch. Before any future R1, freeze the same-seed fixed-geometry design, success margin, metric, stop rule, baseline and artifact paths; perform a fresh `nvidia-smi`; identify PID 24793's command/owner/process tree and confirm relation to this workload before considering any process action.

## Publication Decision

Remain with `MERGE_A_B_AND_FOLD_C`. Corrected R2 is a bounded predictive pilot, not the missing distinct C geometry result. Fullscale N4-FIT and U/S were not evaluated; no full-population G+O+U+S residual inference or final interactional-family endpoint exists.

## What We Should Not Do

- Do not cite −0.622110 as the intended R2 contrast.
- Do not infer semantic, causal, or interaction-specific information from these predictive results.
- Do not launch R1, fullscale N4, seed 5678, B2, open-vocabulary work, or C1 residual retraining from this correction.
- Do not terminate PID 24793 or touch historical artifacts.

## Next Finite Step

No new experiment is authorized here. Any future R1 must return through its own frozen design and resource gate; otherwise retain the present merge/close decision.

## Reproducibility

- Corrected CPU analysis: `tools/r2_nested_correction.py`.
- Corrected outputs: `research_analysis/R2_nested_nuisance_analysis.json`, `R2_nested_nuisance_analysis.md`, `R2_population_reconciliation.json/.md`, `R2_per_predicate_effects.csv`.
- Full split and saved G+O parent: `runs/paper_c_fullscale_residual_audit_20261003T084403Z/`.
- Bounded saved predictions: `runs/paper_c_n4fit_pilot_training_20261003T182118Z/{calfit_predictions.json,calcheck_predictions.json}`.
- Canonical source commit: `ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`; canonical C1 SHA256 `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`; train JSONL SHA256 `306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4`.
- R2 bootstrap: paired pooled row-mean differences, image resampling, 2,000 replicates, seed 20261003; pilot-only/exploratory.
- Tests: R2 correction tests 8 passed; modified Python files compile; `git diff --check` passed. Full-suite run was interrupted after 541 passes and three existing readout-v2 launcher dry-run timeouts (each hits a 60-second limit while the shell invokes read-only `nvidia-smi` for provenance). Remaining suite tests are unverified; the affected test and launcher files were not changed.
