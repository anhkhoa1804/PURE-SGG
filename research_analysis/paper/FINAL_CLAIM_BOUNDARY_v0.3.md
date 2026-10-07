# Final Claim Boundary v0.3

| Claim | Status | Evidence | Safe wording | Boundary |
|---|---|---|---|---|
| Prior sensitivity | SUPPORTED | Pair-prior LL and pair-prior WPRD controls; `research_analysis/R2_nested_nuisance_analysis.json`, `research_analysis/R3_WPRD_construct_validity.md` | Ordered object-pair priors predict labels under the evaluated population; a pair-constant score has chance WPRD. | Does not explain all prediction or all semantics. |
| WPRD | NARROWLY_SUPPORTED | Metric implementation and pair-prior/random controls; `tools/within_pair_discrimination.py`, `research_analysis/R3_WPRD_construct_validity.md` | WPRD summarizes within-pair score discrimination over eligible cells. | Broader construct validity is partial; 12 scoring arms are dependent. |
| Geometry | NARROWLY_SUPPORTED | Units-contract source audit and CPU regression tests; `research_analysis/R1_artifact_audit.md`, `tests/test_geometry_contract_flags.py` | The historical geometry path violated its feature-construction units contract. | Corrected performance effect unavailable; R1 blocked. |
| Nested predictive utility | EXPLORATORY | Corrected R2: G+O 1.315260→1.255252; Δ −0.060008; interval [−0.074317,−0.044480]; `research_analysis/R2_nested_nuisance_analysis.json` | The frozen offset lowered calibrated LL on a 250-image/3,142-row decoder-held-out CAL-CHECK pilot with the fixed G+O parent. | Encoder exposure unresolved; candidate C1 direct overlap 37 CAL-CHECK images in current-file reconstruction. |
| Encoder generalization | INCONCLUSIVE | R2 encoder provenance audit; `research_analysis/paper/R2_ENCODER_PROVENANCE_AUDIT.md/json` | No encoder-held-out conclusion is available. | C1 run lacks exact training-file hash/consumed IDs; upstream initializer training and selection populations are unknown. |
| Semantic specificity | NOT_SUPPORTED | No human-verified semantic endpoint or isolated relation-only branch; `PAPER_EVIDENCE_DOSSIER.md` §§5,7,9 | Predictive utility is not semantic specificity. | Existing residual cannot identify relation-specific semantic content. |
| Causal interpretation | NOT_SUPPORTED | R1 blocked before endpoint; `research_analysis/R1_artifact_audit.md` | The implementation defect is verified; its corrected performance effect is unknown. | No causal geometry effect can be claimed. |
| Open-vocabulary generalization | NOT_SUPPORTED | No predicate-disjoint evaluation; `research_analysis/paper/PAPER_EVIDENCE_DOSSIER.md` §5 | Open-vocabulary generalization was not evaluated. | No unseen-predicate claim. |

The encoder-provenance update does not change the numerical corrected R2 record; it narrows the interpretation from bounded predictive evidence to decoder-held-out evidence with unresolved encoder exposure. No new R2-OOS result exists.
