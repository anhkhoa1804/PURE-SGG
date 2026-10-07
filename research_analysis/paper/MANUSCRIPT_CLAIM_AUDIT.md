# Manuscript Claim Audit — v0.1

This sentence-level audit covers each substantive empirical or interpretive sentence in `MANUSCRIPT_DRAFT_v0.1.md`. Purely organizational sentences are omitted. “SUPPORTED” means the statement matches a saved artifact at its stated scope; it does not upgrade exploratory evidence.

| Section / original sentence | Status | Review |
|---|---|---|
| Abstract: “Predicate-supervised scene-graph models are evaluated through scores that may reflect object co-occurrence, geometry, appearance, and contextual regularities as well as relation-relevant structure.” | NARROWLY_SUPPORTED | Appropriate motivation, but phrase as possible contributors, not measured decomposition. |
| Abstract: “We audit a frozen relation representation and the metrics used to interpret it.” | SUPPORTED | Scope is accurate. |
| Abstract: “A historical readout ladder shows that predicate labels are decodable from the representation, while geometry-only and fused readouts are competitive.” | NARROWLY_SUPPORTED | Values support decodability and competitive WPRD, but A5 fit protocols/populations differ; “competitive” is descriptive. |
| Abstract: “A within-pair discrimination statistic has a chance-level pair-prior control, supporting a narrow conditional interpretation…” | NARROWLY_SUPPORTED | Pair-prior WPRD=.5 supports invariance to a pair-constant score, not broad metric validity. |
| Abstract: “…correlations across twelve dependent scoring arms do not provide independent-model inference.” | SUPPORTED | Matches R3 audit. |
| Abstract: “In a corrected nested nuisance analysis, adding a frozen relation-feature offset to a fixed G+O parent reduces calibrated log-loss by 0.060008 nats per row on a 250-image, 3,142-row CAL-CHECK pilot…” | EXPLORATORY | Numerically correct. Move “exploratory” immediately before result and identify one split/fit. |
| Abstract: “Predicate-level recall changes are mixed.” | SUPPORTED | R2 CSV shows mixed changes; aggregate macro recall decreases. |
| Abstract: “A verified geometry units-contract failure motivated a fixed-geometry rerun, but that run was interrupted before endpoint evaluation; its effect is unknown.” | SUPPORTED | R1 audit confirms no endpoint. |
| Abstract: “These results establish bounded predictive utility and measurement constraints…” | INCONCLUSIVE | “Establish” sounds too strong across exploratory pilot; replace with “provide exploratory evidence of a bounded predictive increment and document…” |
| Abstract: “…not semantic relational understanding, causal geometry effects, or open-vocabulary generalization.” | SUPPORTED | Correct claim boundary. |
| Introduction: “A high predicate score may reflect several sources of regularity…” | NARROWLY_SUPPORTED | Plausible framing; should avoid implying each source was separately quantified. |
| Introduction: “Prediction can be useful without identifying which source supports it.” | SUPPORTED | Conceptual claim, consistent with design. |
| Introduction: “Accordingly, an evaluation metric or a successful decoder should not be treated as direct evidence of semantic relational understanding.” | SUPPORTED | Correct methodological boundary. |
| Introduction: “The work began with readout comparisons and geometry controls, then moved to nuisance prediction and a nested residual formulation.” | SUPPORTED | Timeline matches records. |
| Introduction: “The operational question is whether the measured predictive contribution survives increasingly informative controls…” | NARROWLY_SUPPORTED | Do not imply a complete ordered ladder; U/S not fullscale. Say “selected controls.” |
| Introduction: “The evidence has three main parts…” | SUPPORTED | Organizational. |
| Introduction: “A1–A6 show that predicates are decodable, but geometry-based readouts are competitive…” | NARROWLY_SUPPORTED | Keep qualification on differing fit protocol and metric scope. |
| Introduction: “WPRD … has a narrow interpretation…” | NARROWLY_SUPPORTED | Accurate if “within-pair score discrimination” and no generalized metric-validity claim. |
| Introduction: “a corrected nested comparison holds the nuisance parent fixed…” | SUPPORTED | Correct R2 construction. |
| Introduction: “On a bounded check set this improves calibrated log-loss beyond G+O, while macro recall slightly declines.” | EXPLORATORY | Correct but needs exact n (250/3,142) and the estimate is row-weighted; class effects mixed. |
| Introduction: “Source tracing and regression tests show a units-contract violation…” | SUPPORTED | Geometry docs/test support construction defect. |
| Introduction: “An implementation defect is established; the performance consequence is not.” | SUPPORTED | Correct. |
| Introduction: “Our contribution is therefore an evidence-bounded measurement and predictive audit…” | NARROWLY_SUPPORTED | Defensible contribution framing; avoid implying validated measurement framework. |
| Scope: “The target … is the relation-row mean calibrated log-loss difference…” | SUPPORTED | Add image-cluster resampling and conditional fit scope. |
| Scope: “It is a predictive estimand conditional on the registered split and fitted models.” | SUPPORTED | Correct. |
| Historical ladder section: “Saved A1–A6 and null results use a historical validation set…” | SUPPORTED | Counts/source match saved table. |
| Historical ladder section: “A5a/A5b are geometry readouts with distinct fit protocols…” | SUPPORTED | Correct caveat. |
| WPRD: “WPRD averages within-pair predicate score discrimination…” | NARROWLY_SUPPORTED | Define exact eligible-cell macro AUC/cap with source implementation. |
| WPRD: “Pair-prior-only WPRD is 0.5…” | SUPPORTED | Narrow control only. |
| WPRD: “cross-arm Spearman tables contain twelve deterministic scoring arms…” | SUPPORTED | Correct and important. |
| R2 methods: “Parent temperature and frozen rel_feat temperature/offset are fit on CAL-FIT; CAL-CHECK evaluation-only.” | SUPPORTED | Confirmed R2 source. |
| R2 methods: “The residual coefficient is nonnegative…” | SUPPORTED | Softplus constraint and alpha=0 candidate. |
| R2 methods: “No validation or CAL-CHECK labels fit the model.” | SUPPORTED | R2 JSON and split. |
| R2 methods: “The common CAL-CHECK set has 250 images and 3,142 relation rows.” | SUPPORTED | Correct. |
| R2 methods: “Uncertainty is an exploratory paired image-cluster bootstrap…” | SUPPORTED | Add pooled row-weighted aggregation and conditional limitation. |
| Geometry section: “Six size/aspect channels collapsed.” | SUPPORTED | Correct. |
| Geometry section: “CPU regression tests verify pixel-scale inputs…” | SUPPORTED | Tests establish implementation behavior, not model result. |
| Geometry section: “R1 was interrupted after epoch 0 and has no endpoint.” | SUPPORTED | Exact artifact. |
| Results: historical WPRD values A1–N2. | SUPPORTED | Values match saved table; label historical population and distinguish A5 fit. |
| Results: “These results show that predicate scores can be decoded and that geometry readouts are competitive.” | NARROWLY_SUPPORTED | Keep tied to WPRD and historical setup, not semantic performance. |
| Results: “They do not identify whether the decoded signal is semantic…” | SUPPORTED | Correct. |
| R3: “Spearman … across twelve complete scoring arms…” | SUPPORTED | But emphasize descriptive dependent-arm association; no valid model-level p-value. |
| R3: “Prior-only WPRD is 0.5, a useful but limited shortcut control.” | SUPPORTED | Limit accurately. |
| R3: “Thus WPRD can remain as a specialized within-pair discrimination endpoint…” | NARROWLY_SUPPORTED | It can be reported operationally; construct validity remains partial. |
| R2: O LL change and interval. | EXPLORATORY | Exact corrected values; secondary contrast. |
| R2: G+O LL change and interval. | EXPLORATORY | Exact primary pilot result, not confirmatory. |
| R2: accuracy rises and macro recall declines. | SUPPORTED | Exact aggregate values; should give the values and predicate examples. |
| R2: “The likelihood improvement is not uniform across metrics or predicates.” | SUPPORTED | Correct; show per-predicate evidence. |
| R2: “No predeclared head/middle/tail grouping exists…” | SUPPORTED | Do not invent group thresholds. |
| R1: “The L4 was idle at launch … another workload…” | SUPPORTED | Resource record supports contention; do not claim owner identity. |
| R1: “This is a resource-blocked experiment, not a null result.” | SUPPORTED | Correct. |
| Discussion: “The results demonstrate why predictive success and semantic interpretation should be separated.” | NARROWLY_SUPPORTED | Rhetorical but “demonstrate” too strong; replace with “illustrate in this case.” |
| Discussion: “A pair prior predicts labels well…” | NARROWLY_SUPPORTED | Use “provides a useful baseline”; metrics/populations vary. |
| Discussion: “the remaining rel_feat contribution contracts substantially when the parent includes geometry.” | EXPLORATORY | Descriptive comparison of nested deltas (−.318314 vs −.060008); not a mediation/decomposition claim. |
| Discussion: “Corrected R2 nevertheless finds a nonzero bounded predictive increment…” | EXPLORATORY | Replace “finds” with “estimates”; interval is conditional/exploratory. |
| Discussion: “The result deserves reporting…” | SUPPORTED | Editorial judgment, not empirical result. |
| Discussion: “The strongest alternative … complementary image-conditioned cues…” | INCONCLUSIVE | Clearly label as live alternatives, not observed explanations. |
| Discussion: N4-FIT pilot residual “much smaller” and did not justify fullscale. | EXPLORATORY | Valid resource decision, but omit from central ladder or state it is a separate pilot and not a matched rung. |
| Discussion: pair-prior chance behavior does not show invariance to geometry/appearance/context. | SUPPORTED | Correct. |
| Limitations: R2 population, fitting uncertainty, fullscale N4, R1, WPRD gaps, heterogeneity, no semantic/human tests. | SUPPORTED | Accurate; “tail supports are small” should cite support table or be removed if no threshold is defined. |
| Conclusion: “frozen rel_feat improves calibrated log-loss over the fixed G+O parent on the corrected CAL-CHECK pilot.” | EXPLORATORY | Replace with “the exploratory estimate is lower…” to avoid generalization. |
| Conclusion: “It also supports a narrow WPRD interpretation and verifies a geometry feature-construction units defect.” | NARROWLY_SUPPORTED | WPRD scope remains partial; call it consistent with a narrow operational interpretation. |
| Conclusion: no semantic, causal, full-scale nuisance, or open-vocabulary claims. | SUPPORTED | Correct. |

## Required wording corrections

| Original | Problem | Replacement |
|---|---|---|
| “These results establish bounded predictive utility and measurement constraints” | “Establish” overstates an exploratory single-split check. | “The analyses provide exploratory evidence of a bounded predictive increment on one check set and document specific measurement constraints.” |
| “The results demonstrate why predictive success and semantic interpretation should be separated.” | Broad, rhetorical inference. | “In this case, the results illustrate why predictive performance and semantic interpretation should be reported separately.” |
| “Corrected R2 nevertheless finds a nonzero bounded predictive increment…” | Sounds population-level/confirmatory. | “On the registered CAL-CHECK pilot, the estimated row-weighted loss difference favors the offset; the result remains exploratory and conditional on one fit and split.” |
| “the remaining rel_feat contribution contracts substantially when the parent includes geometry” | Could imply geometry explains/mediates the effect. | “The estimated offset benefit is smaller for G+O than for O on this pilot; this contrast does not identify a causal or additive decomposition.” |
| “tail supports are small” | No declared head/tail partition or threshold. | “Support varies by predicate; effects are shown in support order without a predeclared head/tail classification.” |
