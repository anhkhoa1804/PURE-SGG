# R3 — WPRD construct-validity audit

## 1. Question

Does existing evidence justify interpreting WPRD as a measure of within-object-pair predicate discrimination, and is it adequate for one narrow fixed-geometry C1 comparison? It does not ask whether WPRD is a general SGG quality metric.

## 2. Frozen Evidence Rechecked

The three saved correlation tables each contain 12 arms and three complete metrics per arm. Validation/raw-name: R@50 rho **+0.7413**, permutation p **0.0080**; mR@50 rho **−0.6503**, p **0.0205**; Pareto rho −0.3706, p 0.2359. Validation/VG150-only: +0.9510 (p .0005), −0.6294 (p .0255), −0.1119 (p .7241). Held-out test/raw-name: +0.9142 (p .0005), −0.7273 (p .0080), −0.4476 (p .1439). These exact values are stored in `runs/p49_metric_grounding/corr.json`, `runs/p53_metric_grounding_vg150only/corr.json`, and `runs/p61_test_metric_grounding/corr.json`.

The held-out test cache is documented as 10,403 images / 132,334 relation rows in `docs/TEST_SPLIT_REPLICATION_RESULT.md`. Validation WPRD’s earlier 10,401 images / 132,556 rows / 20,016 cells are not transferable to test. The VG150-restricted validation run reports 37,121 eligible object rows and 31,125 decidable rows in `runs/p53_metric_grounding_vg150only/stdout.log`; its cell count is not saved in the correlation JSON. The test correlation artifact does not retain per-arm cell counts or cell vectors, so its exact WPRD cell population cannot be independently reconstructed from `corr.json` alone.

## 3. Effective Analysis Population

The nominal correlation sample is 12 deterministic scoring arms evaluated on one shared cache/checkpoint—not 12 independently sampled models. Arms cluster into prior/random controls (2), a PURE alpha sweep (5), and geometry-containing models (5). All 12 entries are complete; no missing-value filtering changed the correlation population. R, mR, Pareto and WPRD are all present for each arm in each correlation file.

WPRD itself, implemented in `tools/within_pair_discrimination.py`, computes a macro mean of AUCs for predicate-score differences among rows sharing an ordered raw-name subject/object category group, for predicate pairs both present in that group; per-class rows are capped at 64. This is a within-pair conditional discrimination estimand on the selected eligible cells, not an overall image-level metric.

## 4. Statistical Dependence Audit

Historical p-values are permutation p-values over the 12 arm ranks (2,000 random permutations, seed 0, plus-one correction; see `tools/metric_grounding_correlation.py`). They do not preserve scorer-family blocks or independent model variation. Leave-one-arm-out correlations are descriptively stable in sign: on held-out test, R@50 rho ranges **+0.888 to +0.943**, mR@50 **−0.809 to −0.691**, and Pareto **−0.809 to −0.364**. This only says no single arm drives the rank pattern; it does not repair pseudoreplication.

Reducing to three arm-family means yields only three dependent design families. Any corresponding rank coefficient has no useful inferential resolution and is descriptive. The test correlation is replicated across held-out images but not across independently trained model families. Do not interpret the historical p-values as evidence for a field-wide association.

## 5. Construct-Validity Diagnostics

- **Prior/shortcut control:** pair-prior-only WPRD is exactly **0.5000** in each of the three stored tables; the random null is near chance (.5041, .497995, .506317). This supports the narrow claim that an object-pair-constant score cannot create within-pair discrimination under this implementation. It does not rule out appearance, geometry, or scene context.
- **Responsiveness to geometry:** the registered `p70` inference-time geometry ablation reports 20,016 paired validation cells; full versus no-geometry WPRD delta is **−0.008675** when geometry is removed (`runs/p70_geometry_causal_ablation/ablation_wprd.json`). This is direct evidence the score responds to geometry input in that model/cache, not a semantic validity test and not an independently trained model replication.
- **Scoring-family associations:** positive R@50 and negative mR@50 ranks recur across the three saved tables, but mostly reflect between-family differences; WPRD cannot therefore be presented as empirically orthogonal to ordinary SGG metrics.
- **Ceiling/discriminative capacity:** the score is bounded by AUC construction and prior-only gives chance. No oracle/perfect-within-pair score or independently established attainable ceiling is saved.
- **Protocol effects:** eligible cells, raw-name versus VG150 object filters, row cap, and one shared cache affect interpretation. Test cell-level vectors are absent, so no new clustered CI is computed.
- **Planted relational change:** no synthetic/controlled semantic relation alteration with matched object/geometry/appearance controls is present.

## 6. What the Current Evidence Supports

WPRD is a coherent, prior-canceling, conditional discrimination statistic for the eligible object-pair/predicate cells it scores. The exact prior-only 0.5 control and the `p70` geometry response make it sufficiently interpretable for one narrowly framed geometry-input comparison, provided claims stay at “within-pair score discrimination on this VG population.” The correlations are robust descriptively to leaving out one arm and replicate on held-out images.

## 7. What It Does Not Support

It does not validate WPRD as a general SGG benchmark, establish that WPRD measures semantic understanding, show independence from generic image-conditioned prediction quality, or warrant interpreting the 12-arm permutation p-values as model-level inference. Nor does it show that geometry-driven gains are semantic rather than box-geometry exploitation.

## 8. Missing Experiments

Independent checkpoint/model families with a pre-registered dependence-aware analysis; saved test cell-level bootstrap outputs; a known planted relational-signal control and matched appearance/context/geometry shortcuts; and an eligible human-verified interaction evaluation. These are absent, not negative results. No new experiment is authorized here beyond the single R1 gate.

## 9. Publication Consequence

WPRD can be reported as a specialized within-pair discrimination endpoint with its population and limitations. The cross-arm correlations belong in supplementary/internal analysis, not as a general metric-validity claim. A geometry effect is interpretable narrowly if a one-factor geometry-only condition is compared with the frozen C0 on exactly paired cells; no semantic claim follows.

## 10. Recommendation

The one bounded R1 condition was justified because R2 gives a clean nested predictive contrast, WPRD’s narrow target is supported by its construction and prior null, and the existing geometry comparison bundled pixel-unit restoration with Fourier bandwidth. The L4 resource gate then blocked execution; do not use the historical rank-correlation p-values to justify broader claims.

Full structured audit and leave-one-arm-out numbers: [R3_WPRD_construct_validity.json](R3_WPRD_construct_validity.json).

## Completeness Check for Paper A Rating

This is an artifact audit only; no new R3 analysis or experiment was run.

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Exact Spearman sample size recovery | DONE | Each saved correlation table has exactly 12 complete arm rows (`runs/p49_metric_grounding/corr.json`, `runs/p53_metric_grounding_vg150only/corr.json`, `runs/p61_test_metric_grounding/corr.json`). These are 12 deterministic scoring arms from one shared cache/checkpoint, not 12 independent model draws. |
| Planted-shortcut sensitivity | NOT DONE | No controlled planted relation signal with matched shortcut features is present in the saved R3 artifacts. |
| Strong-VLM ceiling | NOT AVAILABLE | No saved strong-VLM scoring arm or ceiling artifact in the Paper A WPRD correlation bundles. Existing oracle/ceiling-style projects elsewhere are not a strong-VLM construct-validity control and are not substituted. |
| Dependence-aware significance | PARTIAL | `R3_WPRD_construct_validity.json` records leave-one-arm-out ranges and historical arm-permutation p-values. The latter do not preserve scorer-family dependence; there is no independent-model cluster bootstrap or valid model-level inferential test. |
| Prior-only / null controls | DONE | The three saved correlation artifacts give pair-prior WPRD exactly 0.5 and random-null WPRD near 0.5; this addresses an object-pair-constant shortcut only, not appearance, geometry, or context. |

The historical reported correlations are recoverable with **n=12 arms**, but their nominal permutation p-values must not be read as independent-model evidence. No result from this completeness check changes WPRD's narrow descriptive interpretation or authorizes an experiment.
