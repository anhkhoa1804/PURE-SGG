# Internal Review Decision

## Verdict

`READY_FOR_INTERNAL_REVIEW_ONLY` after the v0.2 revisions. The merged paper has a defensible measurement/predictive-audit core but is not ready for external review or submission.

## Must Fix Before External Review

1. Add verified related-work citations and a literature-grounded novelty statement; currently marked `[CITATION NEEDED]`.
2. Keep the abstract and all summaries explicit that R2 is exploratory, based on 250 CAL-CHECK images/3,142 rows, and conditional on one fit/split.
3. Include the complete predicate-effect table in the paper or supplement and retain the accuracy-up/macro-recall-down contrast.
4. Audit every use of “semantic,” “relation,” “robust,” “significant,” “explains,” and “generalization” against `MANUSCRIPT_CLAIM_AUDIT.md`.
5. Explain WPRD's exact eligible-cell estimand and report n=12 dependent arms wherever the correlations appear.
6. Keep historical readout, current R2, N4-FIT pilot, and R1 populations/results visibly distinct.
7. Ensure final tables/figures carry artifact source, population, and exploratory/historical status.

## Should Fix

1. Add a concise population/protocol map near the start of Methods.
2. Move dependent-arm correlations and N4-FIT resource-pilot detail to supplementary/limitations unless needed for the main thesis.
3. Add the missing transition that pair-prior cancellation does not remove geometry, appearance, or context dependence.
4. Replace the repeated general caveats with concise, section-specific limitations.
5. Verify the WPRD test-cell count/vector gap is stated consistently.

## Do Not Touch

No new experiment is authorized. Keep R1 closed; do not rerun it. Do not rerun R2, expand WPRD, run planted-shortcut or VLM-ceiling experiments, start open-vocabulary work, alter datasets, or add baselines.

## Paper's Defensible Core

The paper can argue that predictive predicate scores require careful interpretation: historical readouts establish decodability but do not identify semantic content; WPRD has a narrow within-pair discrimination interpretation under a pair-constant prior control, while broader construct validation is partial; and a corrected fixed-parent nested comparison estimates an exploratory log-loss improvement from a frozen feature offset on 250 CAL-CHECK images, with mixed predicate-level effects. A geometry feature-construction units defect is verified, but the corrected performance consequence is unavailable. This supports a measurement/predictive audit, not a semantic or causal claim.

## Remaining Fatal Risk

The fatal risk is claim-category mismatch: if the manuscript's contribution depends on semantic specificity, general WPRD validity, or a geometry performance effect, the repository does not support it. For the narrower audit paper, these are scope boundaries rather than fatal defects, but novelty remains unverified until literature review.

## Next Step

Complete one citation-verified editorial revision of the manuscript using v0.2 and this claim audit; do not reopen experiments.
