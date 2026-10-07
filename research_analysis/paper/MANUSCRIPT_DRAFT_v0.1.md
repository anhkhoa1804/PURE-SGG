# Predictive Relation Scores Under Priors and Measurement Audits

## Abstract

Predicate-supervised scene-graph models are evaluated through scores that may reflect object co-occurrence, geometry, appearance, and contextual regularities as well as relation-relevant structure. We audit a frozen relation representation and the metrics used to interpret it. A historical readout ladder shows that predicate labels are decodable from the representation, while geometry-only and fused readouts are competitive. A within-pair discrimination statistic has a chance-level pair-prior control, supporting a narrow conditional interpretation, but correlations across twelve dependent scoring arms do not provide independent-model inference. In a corrected nested nuisance analysis, adding a frozen relation-feature offset to a fixed G+O parent reduces calibrated log-loss by 0.060008 nats per row on a 250-image, 3,142-row CAL-CHECK pilot (exploratory image-cluster 95% interval [−0.074317, −0.044480]). Predicate-level recall changes are mixed. A verified geometry units-contract failure motivated a fixed-geometry rerun, but that run was interrupted before endpoint evaluation; its effect is unknown. These results establish bounded predictive utility and measurement constraints, not semantic relational understanding, causal geometry effects, or open-vocabulary generalization.

## 1. Introduction

Scene-graph generation (SGG) assigns predicates to ordered pairs of objects. A high predicate score may reflect several sources of regularity: which objects co-occur, their relative boxes, local appearance, global scene context, or cues tied to the relation itself. Prediction can be useful without identifying which source supports it. Accordingly, an evaluation metric or a successful decoder should not be treated as direct evidence of semantic relational understanding. [CITATION NEEDED]

This study consolidates an empirical audit of one frozen predicate-supervised representation. The work began with readout comparisons and geometry controls, then moved to nuisance prediction and a nested residual formulation. The operational question is whether the measured predictive contribution survives increasingly informative controls and what measurement claims are justified by the saved data. It is not a causal or human-semantic evaluation.

The evidence has three main parts. First, historical readouts show that predicates are decodable, but geometry-based readouts are competitive with a learned feature head. Second, WPRD—the within-pair ranking/discrimination metric used in this project—has a narrow interpretation: an object-pair-constant prior obtains 0.5, but observed correlations with ordinary SGG scores are over dependent scoring arms, not independent model replications. Third, a corrected nested comparison holds the nuisance parent fixed and adds a nonnegative `rel_feat` log-probability offset. On a bounded check set this improves calibrated log-loss beyond G+O, while macro recall slightly declines.

The geometry result must be read separately. Source tracing and regression tests show a units-contract violation in the historical geometry input path. The one same-seed corrected-geometry run did not finish because another GPU workload appeared during an in-training diagnostic. No corrected endpoint was obtained. An implementation defect is established; the performance consequence is not.

Our contribution is therefore an evidence-bounded measurement and predictive audit: the record distinguishes decodeability from semantics, predictive residual from semantic specificity, and an implementation correction from an estimated effect. We do not claim a new state of the art, causal mechanism, or open-vocabulary capability.

## 2. Research Question and Scope

The primary operational question is: does frozen `rel_feat` add predictive utility beyond a fixed ordered object-pair plus geometry predictor under a nested score construction? A secondary measurement question is what WPRD can and cannot measure given its construction, controls, and dependent scorer population.

The target of the nested analysis is the relation-row mean calibrated log-loss difference between a fixed parent and that same parent plus a fitted frozen-feature offset, evaluated on common CAL-CHECK identities. It is a predictive estimand conditional on the registered split and fitted models. It is not mutual information, causal information, or a semantic estimand.

## 3. Experimental Framework

### 3.1 Historical readout ladder

Saved A1–A6 and null results use a historical validation set with 10,401 images, 132,556 relation rows, and 20,016 WPRD cells. A1 is a frozen text-score baseline; A2 is a linear head; A3 is a GELU MLP; A4 is cosine; A5a/A5b are geometry readouts with distinct fit protocols; A6 fuses relation features and geometry. N1 is a shuffled-label control and N2 a pair-prior control. Exact values and definitions are in `paper_package/TABLES/readout_ladder.csv` and `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`. Because A5b differs in fitting population, ladder values should not be mistaken for a single controlled capacity comparison.

### 3.2 WPRD

WPRD averages within-pair predicate score discrimination over eligible ordered object-pair/predicate cells with a per-class row cap. It is conditional on the selected eligible cells and is not a general image-level SGG score. Pair-prior-only WPRD is 0.5 in the saved controls, as expected for scores constant within a pair. The cross-arm Spearman tables contain twelve deterministic scoring arms from shared model/cache lineage, so their permutation p-values are not independent-model inference. See `research_analysis/R3_WPRD_construct_validity.md` and its linked source JSON files.

### 3.3 Corrected nested nuisance analysis

The corrected R2 uses a fixed FIT-derived O or G+O parent. O uses ordered subject/object labels with Laplace smoothing and FIT-global backoff for unsupported pairs. G+O adds the registered geometry correction. Parent temperature and the frozen `rel_feat` temperature/offset are fit on CAL-FIT; CAL-CHECK is evaluation-only. The residual coefficient is nonnegative. At alpha zero, the extension recovers its parent logits exactly. The selected CAL-FIT objective is no worse than the parent objective. No validation or CAL-CHECK labels fit the model.

The common CAL-CHECK set has 250 images and 3,142 relation rows. Uncertainty is an exploratory paired image-cluster bootstrap with 2,000 replicates and seed 20261003. It is conditional on this training/calibration fit and does not capture seed or model-fitting uncertainty. Full protocol and checks appear in `research_analysis/R2_nested_nuisance_analysis.md`.

### 3.4 Geometry audit

The historical path divided boxes by 336 before a geometry helper that expects pixel-scale coordinates and clamps dimensions at 1.0. Six size/aspect channels collapsed. CPU regression tests verify pixel-scale inputs reach the helper and produce finite, varying channels. This verifies input-contract correction, not its effect on WPRD. R1 was interrupted after epoch 0 and has no endpoint. Its partial checkpoint is excluded from metric evidence (`research_analysis/R1_artifact_audit.md`).

## 4. Results

### 4.1 Decodability and geometry competition

On the historical WPRD ladder, A1 is 0.574988, A2 0.580128, A3 0.585669, A4 0.575834, A5a 0.588144, A5b 0.596235, and A6 0.585503. N1 is 0.505451 and N2 is 0.500000. These results show that predicate scores can be decoded and that geometry readouts are competitive. They do not identify whether the decoded signal is semantic, nor do they make A5b a matched-fit comparator to A5a. Source and population are specified in `research_analysis/paper/TABLES_v0.1.md` Table 1.

### 4.2 WPRD validity boundary

On held-out test, Spearman correlation across twelve complete scoring arms is +0.9142 for R@50 and −0.7273 for mR@50. This is a descriptive association among dependent arms, not a model-level relationship. Prior-only WPRD is 0.5, a useful but limited shortcut control. The repository has no planted relational-signal sensitivity experiment and no strong-VLM ceiling. Thus WPRD can remain as a specialized within-pair discrimination endpoint, while broad metric-validity and semantic-quality claims must be removed or qualified.

### 4.3 Nested predictive utility

On the common 250-image / 3,142-row CAL-CHECK, O changes from LL 1.598961 to 1.280646 with the `rel_feat` offset (Δ −0.318314; exploratory 95% image-cluster interval [−0.357666, −0.276130]). G+O changes from 1.315260 to 1.255252 (Δ −0.060008; interval [−0.074317, −0.044480]). Exact values are in `research_analysis/R2_nested_nuisance_analysis.json` and Table 3.

For the primary G+O contrast, accuracy changes from 0.667091 to 0.668364 while macro recall changes from 0.137086 to 0.135990. The likelihood improvement is not uniform across metrics or predicates. The O contrast has a larger accuracy increase but a lower macro recall after extension. These paired predictive results support an increment on this check population, not semantic specificity or generalization to a full-population nuisance ladder.

The full predicate table shows mixed class-level recall and precision changes, particularly across the largest support classes. No predeclared head/middle/tail grouping exists, so we report support ordering descriptively and do not infer a head/tail interaction.

### 4.4 Geometry result unavailable

R1's corrected C1 WPRD, delta, and confidence interval are unavailable. The L4 was idle at launch preflight; later, the log records a separate OmniDocBench process and 100% utilization during the in-training diagnostic. The R1 process was interrupted and no evaluation command ran. This is a resource-blocked experiment, not a null result.

## 5. Discussion

The results demonstrate why predictive success and semantic interpretation should be separated. A pair prior predicts labels well, learned and geometric readouts both decode predicates, and the remaining `rel_feat` contribution contracts substantially when the parent includes geometry. Corrected R2 nevertheless finds a nonzero bounded predictive increment beyond its fixed G+O parent under the evaluated CAL-CHECK setup. The result deserves reporting, with its exploratory status and small population visible.

The strongest alternative interpretation is that the offset extracts complementary image-conditioned cues, estimation differences, or residual annotation regularities not captured by O and the registered G+O features. The analysis does not distinguish those from relationally specific content. The bounded N4-FIT pilot had a much smaller residual and did not justify fullscale reconstruction; because its fit scale and protocol differ, it is not a strict next rung or a proof that appearance/context fully explains the effect.

The metric evidence is similarly limited. Pair-prior chance behavior establishes a property of WPRD's within-pair construction; it does not show invariance to geometry, appearance, or scene context. The dependence among twelve scorer arms prevents the nominal correlation p-values from supporting broad claims about WPRD's relationship to SGG quality. [EVIDENCE GAP]

## 6. Limitations

First, R2's test population is only 250 images / 3,142 rows; uncertainty is conditional on one fitted model and split. Second, no full-scale G+O+U+S comparison was completed. Third, R1 did not yield an endpoint, so the geometry implementation defect has no corrected-performance estimate. Fourth, WPRD's correlation analyses use dependent arms, with incomplete independent-model inference; planted-shortcut and strong-VLM ceiling controls are missing. Fifth, class-level effects vary and tail supports are small. Sixth, the data do not provide a dedicated human-verified semantic test, predicate-disjoint protocol, or eligible frozen interactional endpoint. Seventh, the evidence is drawn from historical, current validation, and bounded pilot populations; values across those populations are not interchangeable.

## 7. Conclusion

The evidence supports a bounded predictive statement: frozen `rel_feat` improves calibrated log-loss over the fixed G+O parent on the corrected CAL-CHECK pilot. It also supports a narrow WPRD interpretation and verifies a geometry feature-construction units defect. It does not establish semantic relational understanding, causal geometry contribution, full-scale residual after appearance and context, or open-vocabulary generalization. The merged measurement/audit paper should present these distinctions as its central result rather than claim an independent semantic method contribution.

## References

[CITATION NEEDED: SGG task and benchmark foundations]

[CITATION NEEDED: relation representation and predicate evaluation]

[CITATION NEEDED: conditional ranking/AUC and clustered uncertainty]
