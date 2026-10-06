# Portfolio Gate — Final Report

## Verdict

**Merge A/B and fold C into the measurement/audit paper (`MERGE_A_B_AND_FOLD_C`).** R2 found a strong bounded held-out predictive increment beyond an ordered object-pair prior. WPRD is defensible only as a narrow within-pair discrimination statistic; its 12-arm association p-values are not model-level inference. A single geometry-only C1a run was scientifically justified but could not be launched because the fresh L4 check found an active workload; R1 is blocked, not negative.

## What Was Verified

- Frozen full nuisance split: FIT 75,066 images / 943,815 rows; CAL 8,183 images; 170 E2 image IDs excluded. R2 read no final validation outcomes.
- Bounded CAL pilot: CAL-FIT 750 images / 9,078 rows; CAL-CHECK 250 / 3,142; disjoint and joined by `(image_id, subject_index, object_index, predicate)` after checking target IDs.
- The R2 ordered-pair prior used 51 classes, FIT-only Laplace-1 pair counts and the FIT-global prior for unsupported pairs. It observed 197,661 ordered noun pairs; CAL-CHECK support was 2,610/3,142 rows.
- WPRD tables: 12 arm rows in each of p49, p53 and p61; all compared metrics were present. The test cache population is 10,403 images / 132,334 rows. The p61 table does not save test cell-level counts/vectors.
- The historical geometry study changed two flags together. The available C1a launcher condition changes the pixel-unit input flag with Fourier scale fixed at 1.0.

## R2 — Nested Nuisance

On the 3,142 CAL-CHECK rows, O-only log-loss was **1.807878** and nested O+rel_feat was **1.185767**, delta **−0.622110**; paired image-cluster 95% CI **[−0.672765, −0.569993]**. Alpha **0.3976523** was fit only on CAL-FIT. Accuracy changed 0.663590→0.679504; macro recall changed 0.227013→0.184865. Thus the loss/accuracy gain is not uniform across predicates. This bounded estimate identifies predictive residual beyond pair counts for the frozen predictor, but cannot attribute it to geometry, appearance, scene context, or relation-specific content. It does not supersede the previous full-population G+O result or the N4-FIT pilot.

## R3 — WPRD Construct Validity

The saved R@50/WPRD, mR@50/WPRD rank patterns replicate descriptively across validation raw-name, validation VG150-only and held-out test tables. Test correlations are +0.9142 (historical arm-permutation p=.0005) and −0.7273 (p=.0080); leave-one-arm-out ranges remain positive/negative respectively. However, 12 rows are correlated scorer variants from one checkpoint family, grouped into three treatment families, so those p-values are optimistic and not general model-level inference. Pair-prior-only WPRD is exactly 0.5 in all three tables; p70 shows sensitivity to geometry input on paired cells. WPRD is suitable here only for the narrow within-pair score-discrimination estimand—not as a generic SGG-quality or semantic metric.

## R1 — Fixed Geometry C1

**Blocked.** R1 was registered as one C1a training arm at seed 1234 (pixel-space geometry true, Fourier scale 1.0) and one existing-protocol full evaluation against frozen C0. The fresh GPU check found PID 24793 using 5,794 MiB with 15% utilization (5,802/23,034 MiB total reported); no process was launched and no retry occurred. Therefore no R1 measurement exists. The current C1 geometry evidence remains weak and bundled; it should not anchor a separate Paper C.

## Publication Decision

The current portfolio decision is **`MERGE_A_B_AND_FOLD_C`**. C1 may appear as a carefully qualified measurement/diagnostic result, not a standalone semantic geometry contribution. The R2 result is bounded and concerns pair-prior conditioning only. Fullscale N4-FIT was not run; the bounded N4-FIT pilot does not provide full-population inference. No C1 residual retraining, open-vocabulary result, or final interactional-family endpoint is established.

## What We Should Not Do

- Do not treat the R2 pair-prior residual as semantic or geometry-specific.
- Do not present WPRD-arm permutation p-values as independent-model inference.
- Do not infer an outcome for blocked R1 or launch it automatically when the GPU later becomes free.
- Do not run full 83k N4, seed 5678, B2, open-vocabulary work, or residual-C1 training in this gate.
- Do not delete pre-existing untracked historical run artifacts.

## Next Finite Step

No further experiment is authorized by this result. If the program is later reopened, the only already-specified decision-changing experiment is the registered single C1a condition; it requires a fresh idle-GPU gate and separate authorization to resume. Otherwise retain the portfolio decision and close this branch of analysis.

## Reproducibility

- Analysis source: `tools/portfolio_gate_analysis.py`; R2/R3 structured outputs: `research_analysis/R2_nested_nuisance_analysis.json`, `research_analysis/R3_WPRD_construct_validity.json`.
- Full split: `runs/paper_c_fullscale_residual_audit_20261003T084403Z/train_calibration_manifest.json` (SHA in R2 JSON).
- Saved pilot predictions: `runs/paper_c_n4fit_pilot_training_20261003T182118Z/{calfit_predictions.json,calcheck_predictions.json}`.
- WPRD correlation sources: `runs/p49_metric_grounding/corr.json`, `runs/p53_metric_grounding_vg150only/corr.json`, `runs/p61_test_metric_grounding/corr.json`.
- Canonical source snapshot commit: `ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`; canonical C1 SHA256: `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`; canonical train JSONL SHA256: `306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4`.
- R1 resource gate: 2026-10-06 07:53:38 UTC; NVIDIA L4 occupied by PID 24793. R1 result unavailable by design.
- Tests: CPU-only full suite `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q` → 618 passed, 25 warnings; focused gate suite → 2 passed; `compileall` and `git diff --check` passed. The warnings are existing PyTorch transformer/autograd warnings.
