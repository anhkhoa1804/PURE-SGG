# Geometry Review

## Finding

Source tracing identified a units-contract failure: boxes were divided by 336 before entry to `geom_feats_torch()`, whose `clamp_min(1.0)` behavior assumes pixel-scale coordinates. Under that path, `rw`, `rh`, `ar1`, `ar2`, `a1`, and `a2` collapsed to constants/zero. The source/test evidence is in `research_analysis/R1_protocol_freeze.md`, `research_analysis/R1_fixed_geometry_C1.md` §§4–5, and `tests/test_geometry_contract_flags.py`.

The supported sentence is: “The geometry path was shown to violate its feature-construction units contract.”

The separate sentence “Geometry was shown to cause the observed relational effect” is NOT ESTABLISHED.

## R1 status

The corrected same-seed run started under a frozen protocol, completed epoch 0, and wrote a partial checkpoint. It stopped during an in-training diagnostic after unrelated GPU contention appeared; no endpoint evaluator ran. The exact record is `research_analysis/R1_artifact_audit.md` and `research_analysis/R1_fixed_geometry_C1.json`. Corrected C1 WPRD, delta, interval, and materiality are unavailable. The partial checkpoint/loss trace is not evidence for or against an effect.

## Manuscript consequences

- State the implementation finding in the audit/results section.
- State R1's unavailable endpoint in limitations, not as a result.
- Do not say geometry improved, harmed, explained, or had no effect.
- Do not call the defect a causal explanation of previous scores.
- Keep geometry as a measurement/input-visibility audit, not semantic contribution evidence.
