# Final Claim Boundary

| Claim | Status | Evidence | Safe wording | Unsafe wording |
|---|---|---|---|---|
| `rel_feat` incremental predictive utility beyond G+O | `SUPPORTED — BOUNDED / EXPLORATORY` | Corrected nested R2, 250 CAL-CHECK images / 3,142 rows; Δ calibrated LL −0.060008, exploratory image-cluster 95% CI [−0.074317, −0.044480], in `R2_nested_nuisance_analysis.md` | “A frozen `rel_feat` offset improved calibrated predictive log-loss over the fixed G+O parent on this bounded held-out pilot.” | “A full-population residual effect is established” or “the residual is semantic.” |
| WPRD within-pair discrimination | `NARROWLY SUPPORTED` | Saved WPRD correlation/control artifacts and R3 audit; pair-prior-only WPRD = 0.5 under this implementation | “WPRD measures within-pair predicate-score discrimination on eligible VG cells.” | “WPRD measures relational understanding or general SGG quality.” |
| Prior dependence | `PARTIAL / NARROW CONTROL` | Pair-prior-only and random-null controls in `R3_WPRD_construct_validity.md` | “An object-pair-constant score does not yield above-chance within-pair WPRD in the saved control.” | “All dataset priors and shortcuts are ruled out.” |
| Geometry-path implementation failure | `ESTABLISHED AS IMPLEMENTATION DEFECT` | Geometry unit-contract forensic evidence, regression tests, and frozen R1 protocol; pre-correction normalized coordinates caused six size/aspect channels to collapse under a pixel-scale clamp | “The audited path passed normalized boxes to a helper whose contract expects pixel-scale boxes, collapsing the affected channels.” | “This proves geometry caused a semantic performance gain.” |
| Geometry semantic contribution | `NOT ESTABLISHED` | R1 was `BLOCKED_TECHNICAL`; no completed corrected C1 WPRD/delta/CI; see `R1_artifact_audit.md` | “The semantic contribution of corrected geometry remains undetermined.” | “Geometry improved performance,” “geometry had no effect,” or “geometry failed.” |
| Semantic specificity | `NOT ESTABLISHED` | No human-verified or otherwise decisive semantic-specific evaluation in the closed evidence | “Predictive results do not identify semantic specificity.” | “Predictive utility proves semantic understanding.” |
| Open-vocabulary generalization | `NOT ESTABLISHED` | Current C1 supervision/evaluation does not constitute a predicate-disjoint open-vocabulary protocol | “Open-vocabulary generalization was not established.” | “C1 demonstrates open-vocabulary relation understanding.” |
| Causal interpretation | `NOT ESTABLISHED` | Saved interventions/ablations are diagnostic and do not isolate a causal semantic mechanism; R1 endpoint unavailable | “Observed predictive differences are associational under the evaluated protocol.” | “The representation contains causal relational information.” |

Interpretation guardrails:

- predictive utility != semantic understanding
- implementation correction != causal geometry evidence
- within-pair discrimination != relational semantics
- historical result != directly comparable current result

R1's partial checkpoint and interrupted training diagnostics are provenance artifacts only. They must never be used as corrected-C1 endpoint evidence.
