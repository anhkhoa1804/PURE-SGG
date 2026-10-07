# Paper Construction Package

## Verdict

The experiment phase is closed. The package supports a measurement/predictive audit paper, not a standalone Paper C semantics claim. Portfolio remains `MERGE_A_B_AND_FOLD_C`. R2's corrected nested CAL-CHECK result is bounded and exploratory; R3 completeness is partial; R1 is `BLOCKED_TECHNICAL` with no corrected C1 endpoint.

## What was investigated

The project audited whether predicate-score decodability and within-pair ranking support semantic interpretations once pair priors, geometry inputs, readout choices, and nuisance prediction are made explicit.

## Why

The original readout and geometry results could be overread as relation-specific evidence. The later nested analysis was designed to ask a narrower predictive question with a fixed nuisance parent and held-out CAL-CHECK rows.

## What was already known

Historical WPRD readouts showed decodability and strong geometry readouts. The prior-only WPRD control was at 0.5. Pair priors had high predictive utility. A historical geometry units-contract defect was traced in the input path.

## What was newly checked here

No experiment was rerun. This package reconciles saved outputs, reproduces no new model, and makes the corrected nested R2 plus R1 blocked status explicit. R1 training was interrupted before evaluation; do not interpret its partial checkpoint.

## What changed numerically

Corrected nested R2 reports G+O LL 1.315260 and G+O + `rel_feat` LL 1.255252 on the same 250-image / 3,142-row CAL-CHECK set: Δ −0.060008, exploratory image-cluster 95% CI [−0.074317, −0.044480]. This is a bounded predictive result, not semantic identification.

## What remains uncertain

WPRD construct completeness; independent-model inference; residual beyond full-scale appearance/context; corrected geometry performance; semantic specificity; and open-vocabulary generalization.

## Publication boundary

Publishable with qualification: historical decoder ladder, narrow WPRD measurement result, corrected nested predictive result, and geometry implementation audit. Not publishable as established: semantic relational understanding, causal geometry contribution, open vocabulary, uniform predicate benefit, or completed R1.

## Structure

- `PAPER_EVIDENCE_DOSSIER.md`: factual source ledger and status.
- `CLAIM_EVIDENCE_MATRIX.md`: safe/unsafe phrasing by claim.
- `MANUSCRIPT_DRAFT_v0.1.md` and `MANUSCRIPT_OUTLINE.md`: first draft and structure.
- `TABLES_v0.1.md`, `FIGURES_v0.1.md`: publication table/figure specifications.
- `PREDICATE_ANALYSIS.md`, `WPRD_STATUS_FOR_PAPER.md`, `GEOMETRY_STATUS_FOR_PAPER.md`: focused evidence notes.
- `REVIEWER_ATTACK.md`, `CONTRIBUTION_STATEMENT.md`, `TITLE_OPTIONS.md`, `PAPER_READINESS.md`: editorial review.

All source paths point to the repository record. Literature references remain marked `[CITATION NEEDED]` until verified; no references are invented.

## Next step

Construct the merged paper from this frozen evidence package. R1 remains optional future strengthening and is not automatically authorized.
