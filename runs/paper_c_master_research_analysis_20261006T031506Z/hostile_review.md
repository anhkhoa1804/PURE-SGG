# Reviewer premortem

**Summary.** The submission audits predicate-supervised scene-graph representations using readouts, nuisance estimators, source/provenance checks and a bounded richer-nuisance pilot. The strongest result is predictive utility against object-pair priors, followed by pronounced contraction with geometry. The work does not yet identify semantic relation content.

**Strengths.** Transparent identity joins and immutable lineage; appropriate image clustering for current residuals; disclosure of invalid comparisons and stopped experiments; a held-out CAL-CHECK pilot; independently recomputable probabilities. The negative or bounded conclusions are potentially useful if the contribution is framed as an empirical audit.

**Weaknesses.** Heterogeneous populations/offset families; no full appearance/context comparison; one residual split/seed; no semantic endpoint; finite-probe geometry claims sometimes overstated historically; no demonstrated new method or global novelty. These are fatal for a semantic-method paper but not automatically fatal for a carefully scoped audit paper.

| Reviewer concern | Severity under a predictive-audit submission | Answer / needed evidence |
| --- | --- | --- |
| 1 Just pair prior? | MINOR | O combination improves held-out NLL substantially; residual source remains unidentified. Existing artifacts answer this restricted objection. [E2] |
| 2 Just geometry? | MAJOR | G+O accounts for most measured increment; CI includes zero. Current evidence supports geometry as a competing explanation. [E2] |
| 3 Just CLIP appearance? | MAJOR | Full U/S is absent; pilot insufficient for final exclusion. Requires full comparator or a new controlled evaluation. [E4] |
| 4 Just global context? | MAJOR | Context enters rel_feat and is bundled with crop features in pilot. Remains alive. [E4,E11] |
| 5 Why relational? | FATAL if semantic claim retained | Name denotes model output/function, not identified semantic content. Title and claims must use predictive scope. [E11] |
| 6 Full nuisance missing? | MAJOR | Historically blocked coverage, then resource-stopped below the frozen trigger. Disclose; do not extrapolate. [E3,E4] |
| 7 Pilot not fullscale? | MINOR for resource decision; MAJOR for residual claim | Resource choice is coherent; it does not test a full-population null. [E4] |
| 8 Enough power? | MAJOR | 250 independent image clusters; bootstrap narrowness is conditional, not power certification. No seed or population uncertainty. [E4] |
| 9 C1 not retrained? | NOT A REAL ISSUE for audit | Constructive hypothesis is outside completed evidence; retraining lacked justification, not refutation. [E4] |
| 10 Interactional evaluation absent? | FATAL for interaction-specific claim | Zero eligible paired endpoint; missing human gold. Requires a new evaluation. [E10] |
| 11 Open-vocabulary implication? | FATAL if retained | Every real predicate is supervised; no unseen test. Remove implication. [E12] |
| 12 Subset dependence? | MAJOR | M2 bridge demonstrates sample sensitivity, N4 pilot is smaller and frozen rel_feat has larger training scale. [E4,E5] |
| 13 Practical meaning? | MAJOR | Report nats/row and resource rule; no deployment or semantic benefit established. [E4] |
| 14 Dependence unit? | MINOR for residuals; MAJOR for WPRD interpretation | Current residuals cluster images. Historical WPRD intervals cluster cells, which can share images. No silent relabeling. [E1,E2] |
| 15 Shortcut encoding? | MAJOR | Entirely consistent with source and results. Explicitly retain as alternative. [E11] |
| 16 Falsification? | MAJOR but fixable in design | Frozen controlled semantic discrimination beyond nuisance with independent gold, or stable full-nuisance residual, would weaken shortcut-only prediction. [E10] |

**Technical concerns:** exact output vocabulary/background handling; source-specific geometry-8 versus geometry-19; scaler lineage; different offset orientations; historical N4 CAL contamination. Existing artifacts resolve identity/protocol details, not model adequacy. **Statistical concerns:** exploratory branching, clustered rows, confidence conditional on fitted models, lack of second seed, no equivalence margin. **Data concerns:** missing semantic labels, ambiguous/coexistent predicates, rare classes and exact-nuisance-match sparsity. **Novelty concerns:** no literature-wide claim justified; empirical audit must show reusable methodological insight. **Reproducibility concerns:** external payload availability and runtime-sensitive exactness, addressed by hashes and code snapshots but not a Git-only clone.

**Verdict:** Reject a submission claiming relational semantics or a validated residual method. Consider a scoped predictive/provenance audit after clear population/offset tables and claim revisions; venue suitability and novelty remain unassessed. The highest-value remedy is human-verified controlled evaluation. An expensive fullscale nuisance replay primarily refines a predictive question and is lower priority given the pilot and absent semantic endpoint. No extra model training is recommended solely to make the submission stronger.
