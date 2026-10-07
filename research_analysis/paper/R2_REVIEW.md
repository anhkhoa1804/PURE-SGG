# R2 Review

## 1. Identification

The corrected construction addresses the main identification flaw in the preliminary R2: it does not compare separately trained O and O+relation decoders as though they formed a nested test. O and G+O parent logits are fixed; the extension adds a frozen `rel_feat` log-probability offset with a nonnegative scalar coefficient. CAL-FIT fits parent/feature temperatures and alpha; CAL-CHECK is not used for fitting. The exact zero coefficient reproduces the parent logits. This identifies predictive complementarity for this fixed parent and fitting protocol, not semantic content or a causal effect.

## 2. Population

CAL-FIT: 750 images/9,078 rows. CAL-CHECK: 250 images/3,142 rows. All four arms share the same check identities. The primary estimate is pooled row-weighted mean log-loss difference; image identity is the resampling cluster. Report the population in abstract, results, and table caption. This is neither the full validation set nor a full-population N4 evaluation.

## 3. Nesting

The parent is fixed. `alpha=0` gives exactly the parent; the optimizer's selected CAL-FIT solution is compared to the included parent candidate under the same objective. The CAL-FIT objective is no worse than parent for both O and G+O. This is the correct nesting check. It does not imply that an independently fitted expanded model must outperform its parent on held-out data.

## 4. Confidence Interval

For G+O, calibrated/final LL is 1.315260 for parent and 1.255252 for extension; Δ=−0.060007679 nats/row. The exploratory paired image-cluster percentile interval is [−0.074317092, −0.044480410], with 2,000 replicates and seed 20261003. Each sampled image contributes all its rows; the statistic remains pooled row-weighted. The interval is conditional on the one FIT/calibration fit and this split; it excludes fitting/seed uncertainty and does not establish large-scale generalization. Raw ΔLL is −0.032009 with interval [−0.046121, −0.016764].

## 5. Predicate Heterogeneity

Primary G+O accuracy rises from .667091 to .668364, while macro recall falls from .137086 to .135990. In the per-predicate CSV, recall improves for 8 predicates, declines for 6, and is unchanged for 32. Largest-support examples are mixed: `on` (1,124) +.001780 recall; `has` (418) +.009570; `in` (404) −.014852; `wearing` (258) +.015504; `of` (249) −.004016. No head/mid/tail cutoffs were registered. The evidence is not a uniform predicate gain; pooled accuracy should never stand alone.

## 6. Interpretation

The primary comparison is an exploratory bounded predictive gain beyond a fixed G+O parent on this check population. It is not evidence of semantic specificity, geometry independence, causality, broad predicate improvement, or residual utility beyond full G+O+U+S. The smaller estimated delta relative to O is descriptive; it does not quantify a causal share “explained” by geometry.

## 7. Required Manuscript Changes

1. Put 250 images/3,142 rows near the first statement of the effect.
2. Use “exploratory estimate” and “image-cluster interval” consistently.
3. Explain fixed-parent alpha-zero nesting explicitly.
4. State interval conditioning and row-weighted aggregation.
5. Surface accuracy-up/macro-recall-down values together.
6. Show the mixed predicate table and avoid “uniform,” “robust,” or “significant.”
7. Do not combine R2 with the distinct N4-FIT pilot into a complete nuisance ladder.
