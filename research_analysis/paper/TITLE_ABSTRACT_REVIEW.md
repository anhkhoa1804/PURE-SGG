# Title and Abstract Review

## Existing title options

1. **Auditing Predictive Relation Scores in Scene-Graph Generation** — strongest umbrella title; says audit and prediction, not semantics.
2. **Priors, Geometry, and the Interpretation of Predicate Scores** — accurate but should signal measurement/SGG in subtitle or abstract.
3. **A Measurement Audit of Within-Pair Predicate Discrimination** — precise for WPRD, too narrow for the full A+B+C manuscript.
4. **Predictive Residuals Under Nuisance Controls for Scene-Graph Predicates** — over-centers the exploratory R2 pilot and may imply broader nuisance coverage.
5. **When Predicate Decodability Is Not Semantic Evidence** — clear cautionary framing, but rhetorically negative and narrower than the full audit.

## Three recommended titles

1. **Auditing Predictive Relation Scores in Scene-Graph Generation**
2. **Priors, Geometry, and Measurement in Scene-Graph Predicate Prediction**
3. **What Predicate Scores Measure: A Predictive Audit for Scene Graphs**

These are editorial options only; novelty and related-work positioning require literature verification.

## Revised abstract

Predicate-supervised scene-graph generation assigns predicates to object pairs, but predictive scores can reflect pair frequencies and image-conditioned cues without identifying which source supports a decision. We present a measurement and predictive audit of a frozen representation and its evaluation. Historical readouts show predicate decodability, while a pair-constant prior obtains chance WPRD, consistent with a narrow interpretation of WPRD as within-pair score discrimination. Its broader construct validity remains partial: reported correlations use 12 dependent scoring arms, and planted-shortcut and strong-VLM ceiling controls are absent. The primary corrected nuisance analysis is exploratory and evaluates 250 CAL-CHECK images (3,142 rows). With a fixed G+O parent, adding a frozen-feature log-probability offset changes calibrated log-loss from 1.315260 to 1.255252 nats per row (Δ=−0.060008; paired image-cluster percentile interval [−0.074317,−0.044480]). This interval is conditional on one fit and split. Accuracy rises slightly while macro recall falls, with mixed predicate-level changes. A geometry input-units contract failure is verified, but the attempted corrected C1 run did not reach endpoint evaluation. The evidence supports an exploratory predictive increment for this bounded comparison and a narrow measurement account; it does not establish semantic specificity, causal geometry effects, full-scale residual utility, or open-vocabulary generalization.
