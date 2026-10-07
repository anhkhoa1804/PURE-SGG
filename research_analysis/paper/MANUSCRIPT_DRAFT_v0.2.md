# Predictive Relation Scores Under Priors and Measurement Audits

> Historical internal draft. Superseded by `MANUSCRIPT_DRAFT_v0.3.md`, which incorporates the final encoder-provenance audit. The R2 estimate below is not encoder-held-out evidence.

## Abstract

Predicate-supervised scene-graph generation (SGG) assigns labels to object pairs, but predictive scores can reflect pair frequencies and image-conditioned cues without identifying which source supports a decision. We report a measurement and predictive audit of a frozen representation and its evaluation. Historical readouts show predicate decodability, while a pair-constant prior obtains chance WPRD, consistent with WPRD's narrow interpretation as within-pair score discrimination. Its broader construct validity remains partial: reported correlations use 12 dependent scoring arms, and planted-shortcut and strong-VLM ceiling controls are absent. The primary corrected nuisance analysis is exploratory and evaluates 250 CAL-CHECK images (3,142 rows). With a fixed G+O parent, a frozen-feature log-probability offset changes calibrated log-loss from 1.315260 to 1.255252 nats/row (Δ=−0.060008; paired image-cluster percentile interval [−0.074317,−0.044480]). The interval is conditional on one fit and split. Accuracy rises slightly while macro recall falls, and predicate-level recall changes are mixed. A geometry input-units contract failure is verified, but the attempted corrected C1 run did not reach endpoint evaluation. The evidence supports an exploratory predictive increment for this bounded comparison and a narrow measurement account; it does not establish semantic specificity, causal geometry effects, full-scale residual utility, or open-vocabulary generalization.

## 1. Introduction

Scene-graph generation predicts predicates for ordered object pairs. A predicate score may be useful while remaining ambiguous about the evidence that supports it. Object co-occurrence, box geometry, local appearance, and scene context are plausible contributors; the present analyses do not isolate each source. This distinction matters because decoding a predicate label from a representation is not, by itself, evidence that the representation captures a human-interpretable relation. [CITATION NEEDED]

We consolidate a set of readout, metric, geometry, and nuisance analyses around a limited question: what do these saved predictive measurements justify? The paper's empirical center is not a new semantic model. It is an audit of how pair priors, score construction, feature visibility, and a fixed-parent residual comparison constrain interpretation.

Three observations motivate the audit. First, predicates are decodable from the frozen representation, but geometry readouts are also competitive on the historical WPRD ladder. Second, the WPRD pair-prior control behaves as expected for its within-pair construction, yet associations with ordinary SGG metrics are computed across only twelve dependent scoring arms. Third, a corrected nested analysis finds a lower estimated calibrated log-loss when a frozen `rel_feat` offset is added to a fixed G+O parent on one bounded check population. The population is 250 CAL-CHECK images and 3,142 rows; the estimate is exploratory, and its predicate-level pattern is not uniform.

The geometry audit is a separate result. Historical preprocessing divided boxes by 336 before a helper that expects pixel-scale coordinates, collapsing six engineered geometry channels. A same-seed correction run was attempted but interrupted before endpoint evaluation. Thus the feature-construction defect is established; whether correcting it changes measured C1 performance is unknown.

Our contribution is a provenance-aware account of the boundary between predictive measurements and semantic claims. We report what the saved controls support, identify unresolved construct-validity gaps, and avoid interpreting a predictive residual as semantic or causal evidence.

## 2. Research question

The primary estimand is the relation-row-weighted mean calibrated log-loss difference on the registered CAL-CHECK population between a fixed FIT-derived G+O parent and that same parent plus a frozen `rel_feat` log-probability offset. Parent temperatures and the nonnegative offset coefficient are fit on CAL-FIT; CAL-CHECK is evaluation-only. The check set contains 250 images/3,142 rows, while CAL-FIT contains 750 images/9,078 rows. Image identity is the cluster for the paired bootstrap. This is a predictive estimand conditional on the fixed parent, fitted parameters, and split—not a causal or semantic estimand.

A secondary question concerns WPRD: what operational discrimination property is supported by its construction and available controls, and what broader validity claims remain unsupported?

## 3. Experimental framework

### 3.1 Historical readouts and priors

The saved A1–A6 and null ladder uses a historical validation population of 10,401 images, 132,556 relation rows, and 20,016 WPRD cells. WPRD values are: A1 frozen-text .574988; A2 linear .580128; A3 GELU MLP .585669; A4 cosine .575834; A5a geometry cross-fit .588144; A5b geometry train-fit .596235; A6 fusion .585503; N1 shuffled null .505451; and N2 pair prior .500000. A5a and A5b differ in fit protocol, so the ladder is descriptive rather than a matched-capacity comparison. Exact sources are `paper_package/TABLES/readout_ladder.csv` and `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`.

### 3.2 WPRD

WPRD is implemented as a macro-average of within-ordered-pair predicate-score AUCs over eligible cells, with a 64-row per-class cap. A score constant within an ordered pair gives .5 in the saved pair-prior controls. This supports a narrow interpretation as conditional score discrimination within the eligible cells. It does not establish human semantic correctness or independence from geometry, appearance, or context. The implementation and audit are `tools/within_pair_discrimination.py` and `research_analysis/R3_WPRD_construct_validity.md`.

On held-out test, Spearman associations across 12 complete scoring arms are +.9142 for R@50 and −.7273 for mR@50. The arms share model/cache lineage and are dependent; the saved permutation p-values do not provide independent-model inference. These associations are descriptive and are not used as a broad validation claim.

### 3.3 Geometry/input-visibility audit

The historical path normalized boxes by 336 before calling `geom_feats_torch()`, whose minimum-size clamp assumes pixel-scale coordinates. The six channels `rw`, `rh`, `ar1`, `ar2`, `a1`, and `a2` collapsed under that contract. CPU source tracing and regression tests verify the construction issue. They do not estimate a performance effect. R1 completed epoch 0 but stopped during an in-training diagnostic after separate GPU contention appeared; the evaluator did not run (`research_analysis/R1_artifact_audit.md`).

### 3.4 Nested nuisance readout

For each parent, calibrated parent logits are held fixed. The extension adds `alpha * log p_rel_feat`, with nonnegative alpha. Alpha zero exactly recovers the parent; the same CAL-FIT objective selects the parent if the fitted offset does not improve it. O is the ordered object-pair prior; G+O is the registered pair prior plus FIT-standardized geometry correction. The primary comparison is G+O versus G+O+`rel_feat`. No U/S full-population features are included. All details and identity checks are in `research_analysis/R2_nested_nuisance_analysis.md/json`.

### 3.5 Statistical protocol

The primary R2 point estimate is pooled row-weighted log-loss difference over the shared CAL-CHECK rows. The exploratory interval resamples image identities with replacement, retaining each sampled image's rows; 2,000 replicates, seed 20261003. It quantifies image-cluster variation conditional on the one fitted model and split, not seed/refit uncertainty. No validation labels are used in R2 fitting.

## 4. Results

### 4.1 Prior baselines and historical readouts

The historical ladder shows that several readouts decode predicate labels above the saved null/pair-prior levels, while geometry-only values are competitive. These values concern historical WPRD, not the current R2 CAL-CHECK log-loss estimand. They do not identify the source of decoded information.

### 4.2 WPRD audit

The pair-prior WPRD of .5 is consistent with a pair-constant score providing no within-pair ranking. It is one shortcut control, not a broad construct-validation result. The +.9142/−.7273 correlations describe twelve dependent scoring arms, not independent models. Planted-shortcut sensitivity, a strong-VLM ceiling, and independent-family dependence-aware inference are absent. WPRD is retained as a diagnostic statistic, not a general semantic-quality measure.

### 4.3 Nested predictive utility

On CAL-CHECK (250 images/3,142 rows), calibrated/final LL for O is 1.598961 and for O+`rel_feat` is 1.280646 (Δ=−0.318314; exploratory image-cluster interval [−0.357666,−0.276130]). For the primary fixed G+O parent, LL is 1.315260 and for G+O+`rel_feat` is 1.255252 (Δ=−0.060008; exploratory interval [−0.074317,−0.044480]). These are nested fixed-parent comparisons, not independently fit decoder comparisons.

For G+O, accuracy changes from .667091 to .668364 (+.001273), whereas macro recall changes from .137086 to .135990 (−.001096). Predicate-level results are mixed: for example, recall rises for `on` (support 1,124; .946619→.948399) and `has` (418; .892344→.901914), but falls for `in` (404; .529703→.514851) and `of` (249; .526104→.522088). Across all observed predicates, 8 recall changes are positive, 6 negative, and 32 unchanged. There is no registered head/mid/tail threshold; support ordering is descriptive. Full per-predicate values are in `research_analysis/R2_per_predicate_effects.csv`.

The smaller G+O offset estimate than O offset estimate is descriptive and consistent with a more informative parent absorbing some predictive utility. It is not a causal decomposition, nor does it show that geometry specifically explains the difference.

### 4.4 Geometry result unavailable

The geometry feature-construction units defect is verified, but R1 has no corrected endpoint. No WPRD delta or materiality decision is available. The incomplete checkpoint and training trace are excluded from result claims.

## 5. Discussion

The bounded R2 result supports a limited statement: on this registered check set, the fitted frozen-feature offset has lower calibrated log-loss than the fixed G+O parent. The image-cluster interval is exploratory and conditional on the single fit. Concurrent accuracy and macro-recall movements point in different directions, and predicate-level changes are heterogeneous. This should be reported as predictive complementarity under the specified parent, not as relational semantics.

Several alternatives remain compatible with the result: the offset may capture complementary appearance or scene-context cues, residual dataset/annotation regularities, or other features not represented by the fixed parent. The current evidence does not separate these explanations. The separate N4-FIT pilot was small and did not justify fullscale reconstruction; because its protocol/population differs, it is not a complete next rung in this analysis.

For WPRD, the pair-prior result validates one property of its construction. The dependent-arm correlations do not establish broad metric validity. A within-pair score can still depend on visual/context features and need not track human semantic judgments.

The geometry audit illustrates an additional distinction: verifying that an input contract was violated does not estimate the effect of correcting it. Since R1 did not reach evaluation, the performance consequence remains unresolved.

## 6. Limitations

R2 evaluates 250 images and 3,142 rows, with uncertainty conditional on one fit/split. It uses a fixed G+O parent and does not include full-population U/S features. Per-predicate estimates are descriptive and some supports are limited; no group thresholds were predeclared. WPRD's broader construct validity remains partial: scoring arms are dependent, exact held-out cell vectors are unavailable in the correlation table, and planted-shortcut/strong-VLM controls are missing. R1's corrected endpoint is unavailable. The project does not provide a human-verified semantic evaluation, predicate-disjoint evaluation, or eligible interactional-family endpoint. Historical and pilot metrics come from different populations and must not be numerically conflated.

## 7. Conclusion

The saved evidence supports predicate-score decodability and a narrow operational WPRD interpretation, and it documents a geometry feature-construction defect. In one exploratory nested CAL-CHECK comparison, a frozen `rel_feat` offset lowers calibrated log-loss relative to fixed G+O, while macro recall decreases and predicate effects are mixed. The result is bounded to this pilot and does not identify semantic specificity, causal geometry contribution, full-population residual utility, or open-vocabulary generalization. The paper's contribution should remain a measurement and predictive audit with these limits explicit.

## References

[CITATION NEEDED: SGG task and benchmark foundations]

[CITATION NEEDED: relation representation and predicate evaluation]

[CITATION NEEDED: within-group discrimination metrics and clustered uncertainty]
