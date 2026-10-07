# Introduction

- **Purpose:** Motivate why predictive predicate scores should be separated from semantic interpretation.
- **Claims:** Predicate prediction can exploit several correlated visual and dataset cues; the present study audits measurement and nuisance structure rather than proving semantics.
- **Exact evidence:** `runs/paper_c_final_evidence_audit_20261003T185925Z/evidence_ledger.md`; final corrected sources `research_analysis/R2_nested_nuisance_analysis.md` and `research_analysis/R3_WPRD_construct_validity.md`.
- **Planned figure/table:** Figure 1 (audit framework); Table 5 (claim/evidence boundary).

# Related Work

- **Purpose:** Position scene-graph predicate prediction, conditional ranking/discrimination, and evaluation validity.
- **Claims:** Literature context only; avoid novelty assertions until references are verified.
- **Exact evidence:** Repository provides methods and prior internal references, not a verified bibliography adequate for publication.
- **Planned figure/table:** None.
- **Drafting marker:** Add verified literature and citations; retain `[CITATION NEEDED]` meanwhile.

# Research Question

- **Purpose:** State the operational question and distinguish it from the aspirational semantic claim.
- **Claims:** What is decodable? What persists against a fixed pair/geometry parent? What does the metric measure?
- **Exact evidence:** `research_analysis/paper/PAPER_EVIDENCE_DOSSIER.md` §§1–2.
- **Planned figure/table:** Figure 1.

# Experimental Framework

## Models

- **Purpose:** Describe frozen C1, saved readout heads, and the bounds of the evaluation.
- **Claims:** The ladder varies readout families on saved relation features; the corrected R2 residual model freezes its parent.
- **Exact evidence:** `paper_package/METHODS.md`; `paper_package/TABLES/readout_ladder.csv`; `research_analysis/R2_nested_nuisance_analysis.md` §§4–7.
- **Planned figure/table:** Table 1.

## Priors

- **Purpose:** Define ordered subject/object class prior and support/backoff.
- **Claims:** The train-fitted pair table with Laplace smoothing has predictive utility; unsupported pairs use train-global backoff.
- **Exact evidence:** `runs/paper_c_fullscale_residual_audit_20261003T084403Z/`; `research_analysis/R2_nested_nuisance_analysis.md` §§3–4.
- **Planned figure/table:** Table 3.

## WPRD

- **Purpose:** Specify the conditional within-pair statistic and controls.
- **Claims:** The statistic is narrow; pair-only score is chance because it cannot vary within the pair.
- **Exact evidence:** `tools/within_pair_discrimination.py`; `research_analysis/R3_WPRD_construct_validity.md` §§3–5.
- **Planned figure/table:** Figure 2; Table 1.

## Geometry Audit

- **Purpose:** Trace input units, geometry features, and what the R1 resource stop implies.
- **Claims:** A feature-contract failure is verified; R1 endpoint is unavailable.
- **Exact evidence:** `research_analysis/R1_protocol_freeze.md`; `research_analysis/R1_artifact_audit.md`; `tests/test_geometry_contract_flags.py`.
- **Planned figure/table:** Figure 3; Table 2.

## Nested Nuisance Readout

- **Purpose:** Define common populations, frozen parents, offset, calibration, and zero-offset nesting.
- **Claims:** The bounded G+O+rel contrast is a predictive estimate; it is not semantic identification.
- **Exact evidence:** `research_analysis/R2_nested_nuisance_analysis.md` §§3–8; `.json`.
- **Planned figure/table:** Figure 4; Table 3.

## Statistical Protocol

- **Purpose:** Clarify units, bootstrap, multiplicity/exploration, and the lack of independent-arm inference.
- **Claims:** Image bootstrap for R2 is conditional/exploratory; WPRD arm correlations have dependent n=12.
- **Exact evidence:** `research_analysis/R2_nested_nuisance_analysis.md` §8; `research_analysis/R3_WPRD_construct_validity.md` §4.
- **Planned figure/table:** Figure 4 intervals; Table 3.

# Results

## Prior Baselines

- **Purpose:** Show the strength of pair-frequency prediction.
- **Claims:** O is a strong predictive nuisance baseline, not a complete explanation.
- **Exact evidence:** `research_analysis/R2_nested_nuisance_analysis.md` §§4,8; fullscale audit run.
- **Planned figure/table:** Table 3.

## WPRD Audit

- **Purpose:** Report historical score ladder and construct-validity scope.
- **Claims:** Readout decodability and geometry competition; no independent-model correlation claim.
- **Exact evidence:** `paper_package/TABLES/readout_ladder.csv`; R3 audit JSON/MD.
- **Planned figure/table:** Figure 2; Tables 1–2.

## Geometry Contract Failure

- **Purpose:** Report the defect and distinguish it from an endpoint result.
- **Claims:** Geometry path violated units contract; corrected C1 effect is undetermined.
- **Exact evidence:** `research_analysis/R1_artifact_audit.md`; run stop evidence.
- **Planned figure/table:** Figure 3; Table 2.

## Nested Predictive Utility

- **Purpose:** Report corrected R2 O and G+O parent/extension comparisons.
- **Claims:** Lower calibrated LL in this held-out pilot under exact nested offset.
- **Exact evidence:** `research_analysis/R2_nested_nuisance_analysis.json`.
- **Planned figure/table:** Figure 4; Table 3.

## Predicate-Level Heterogeneity

- **Purpose:** Show that pooled LL changes are not uniform across classes.
- **Claims:** Head support dominates the count; observed recall/precision changes have mixed signs.
- **Exact evidence:** `research_analysis/R2_per_predicate_effects.csv`.
- **Planned figure/table:** Figure 5; Table 4.

# Discussion

## What the Results Establish

- **Purpose:** State decodability, predictive nuisance effects, and bounded residual.
- **Claims:** Predictive evidence only, with exact populations and uncertainty.
- **Exact evidence:** Dossier §§3–4.
- **Planned figure/table:** Figure 1; Table 5.

## What They Do Not Establish

- **Purpose:** Bound semantic, causal, open-vocabulary, and interactional claims.
- **Claims:** None of those stronger claims is identified.
- **Exact evidence:** Dossier §§5–9; `research_analysis/FINAL_CLAIM_BOUNDARY.md`.
- **Planned figure/table:** Table 5.

## Measurement vs Semantics

- **Purpose:** Explain why WPRD and predicted predicates are measurement outputs, not semantic ground truth.
- **Claims:** The defect and score responsiveness are distinct from causal semantics.
- **Exact evidence:** R3 and R1 audits.
- **Planned figure/table:** Figure 1.

## Limitations

- **Purpose:** Make small check set, missing full U/S, dependence, heterogeneity, and interrupted R1 explicit.
- **Claims:** These constrain interpretation and publication scope.
- **Exact evidence:** Dossier §§6–9.
- **Planned figure/table:** Table 5.

# Conclusion

- **Purpose:** End with the narrow audit/predictive conclusion and portfolio boundary.
- **Claims:** Reported relation quality requires nuisance and measurement scrutiny; this evidence does not prove semantic relational understanding.
- **Exact evidence:** Entire dossier; portfolio `MERGE_A_B_AND_FOLD_C`.
- **Planned figure/table:** Figure 1; Table 5.
