# Geometry Status for Paper

## 1. Historical implementation problem

The audited path divided boxes by image resolution 336 before calling `geom_feats_torch()`. That helper clamps side lengths with `clamp_min(1.0)` and assumes pixel-scale image-space coordinates. Under normalized coordinates, the size/aspect channels `rw`, `rh`, `ar1`, `ar2`, `a1`, and `a2` collapsed to constants/zero. Source tracing and tests: `research_analysis/R1_protocol_freeze.md`, `research_analysis/R1_fixed_geometry_C1.md` §§4–5, `tests/test_geometry_contract_flags.py`.

## 2. Why it matters

The model did not receive the intended variation in these engineered geometry channels on that path. This weakens interpretations of historical comparisons as clean evidence about the effect of informative geometry. It does not by itself determine what the corrected path would do to WPRD or another endpoint.

## 3. What was learned from forensic tracing

The feature-construction contract is now explicit and covered by CPU regression tests: image-space pixel boxes are passed to the helper, output is finite, shapes remain compatible, and the formerly collapsed channels vary on representative valid boxes. This is an implementation/input-visibility audit.

**The geometry path was shown to violate its feature-construction units contract.**

## 4. R1 status

R1 was one seed-1234 fixed-protocol C1a attempt with pixel-scale inputs and Fourier scale held at 1.0, compared to frozen C0 as the units-only parent. Epoch 0 completed and an intermediate checkpoint was saved. The run was interrupted during an in-training diagnostic after a separate GPU process appeared. The full evaluator did not run. Status: `BLOCKED_TECHNICAL`; corrected C1 result and delta are unavailable. See `research_analysis/R1_artifact_audit.md` and `runs/R1_fixed_geometry_C1a_20261007T021152Z/`.

## 5. What cannot be claimed

**Geometry was shown to cause the observed relational effect.** — NOT ESTABLISHED.

The quoted sentence is not supported. Do not claim geometry improved performance, had no effect, or failed. The +0.010 WPRD decision threshold was frozen but cannot be assessed without an endpoint. The partial checkpoint/loss trace is not a result. R1 is optional future strengthening and is not required for the core paper.
