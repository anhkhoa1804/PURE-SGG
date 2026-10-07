# Contribution Statement

## 1. One-sentence contribution

We provide a provenance-aware audit showing that predicate-score decodability, within-pair discrimination, and bounded predictive residuals are distinct from semantic relational understanding, and that nuisance and input-contract controls materially constrain interpretation.

## 2. 100-word abstract-style contribution

This work audits the interpretation of predicate-supervised scene-graph scores using readout, metric, geometry, and nuisance analyses. Decoders show predicates are decodable, while geometry-only readouts are competitive. WPRD behaves as a narrow within-pair discrimination statistic under a pair-prior chance control, but correlations across twelve dependent scoring arms do not support independent-model inference. The nested readout holds the G+O parent fixed and adds frozen relation-feature log-probabilities; calibrated loss improves on a bounded CAL-CHECK pilot, with heterogeneous class-level recall changes. A geometry units-contract defect is verified, but the corrected C1 run did not reach evaluation. No semantic or causal claim is established here.

## 3. Conference-paper contribution bullets

- We separate predicate-score decodability and within-pair discrimination from claims of semantic understanding, using the project's saved readout and prior/null controls.
- We report a corrected nested predictive comparison in which the G+O parent is fixed and the frozen relation-feature offset has an exact zero-effect parent configuration; the held-out pilot improvement is exploratory and class-heterogeneous.
- We trace a geometry input-units contract defect and preserve the interrupted fixed-geometry rerun as blocked provenance, preventing an implementation correction from being mistaken for evidence of a semantic effect.
