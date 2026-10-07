# Core Contribution Review

## Minimal Defensible Contribution

This work provides a provenance-aware empirical audit showing that predicate decodability, a within-pair discrimination score, and a bounded residual log-loss change answer different questions. On one 250-image/3,142-row exploratory CAL-CHECK population, a frozen feature offset improves calibrated log-loss over a fixed G+O parent under an exactly nested construction, while macro recall decreases and predicate-level changes are mixed. The work also documents a geometry feature-construction units defect without claiming a corrected performance effect.

## Strongest Supporting Evidence

- Corrected R2 artifact uses a fixed parent and explicit alpha-zero configuration; G+O LL changes 1.315260→1.255252, Δ=−0.060008, exploratory image-cluster interval [−0.074317,−0.044480]. Source: `research_analysis/R2_nested_nuisance_analysis.json`.
- Aggregate accuracy rises slightly (.667091→.668364) while macro recall falls (.137086→.135990), corroborated by `research_analysis/R2_per_predicate_effects.csv`.
- WPRD pair-prior control is .5, supporting only its narrow within-pair discrimination interpretation; `research_analysis/R3_WPRD_construct_validity.md`.
- Geometry contract failure and incomplete R1 are documented with source/test/process records; `research_analysis/R1_artifact_audit.md`.

## Evidence That Is Only Exploratory

- R2's check population and interval: one bounded split and fit, conditional image-cluster bootstrap.
- R3's 12-arm correlations: dependent deterministic arms, descriptive association only.
- Historical WPRD readout ladder: useful evidence of decodability/geometry competition but not a matched model-capacity experiment.
- N4-FIT result: separate pilot, not fullscale nuisance evidence.

## Claims That Must Be Removed

- Semantic or relational understanding established.
- Geometry's causal or semantic contribution established.
- Full-population residual beyond G+O+U+S established.
- Open-vocabulary or interactional-family advantage established.
- General WPRD validity or independent model-level association established.
- R1 has a corrected C1 endpoint.

## Why A+B+C Still Form One Paper

The defensible unifying question is measurement: how priors, score construction, feature visibility, and nuisance conditioning constrain interpretation of predicate prediction. A/B readouts and WPRD describe measurement behavior; C's geometry and nested nuisance work tests specific limits on interpretation. C does not supply an independent semantic method claim, so folding it into the audit gives a coherent boundary-setting paper rather than an unsupported standalone contribution.

## What Would Make the Contribution Stronger

The most immediate improvement is editorial and evidentiary: cite verified source artifacts beside each result, provide the predicate-level table, and clearly distinguish historical, exploratory, and unavailable endpoints. No new experiment is authorized by this internal review.
