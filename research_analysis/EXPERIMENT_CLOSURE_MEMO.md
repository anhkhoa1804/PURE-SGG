# Experiment Closure Memo

## Closed Experiments

R2:
Corrected nested R2 is complete for its bounded CAL-CHECK predictive estimand. With a fixed FIT-trained G+O parent and a nonnegative `rel_feat` offset fitted on CAL-FIT, calibrated log-loss changed by −0.060008 nats/row on 250 CAL-CHECK images / 3,142 rows (exploratory image-cluster 95% CI [−0.074317, −0.044480]). Accuracy rose slightly while macro recall fell slightly; this is not a uniform predicate gain and does not establish semantics. The previous −0.622110 comparison is preliminary/comparator-invalid for the intended R2 question and is not used.

R3:
Evidence-completeness status is PARTIAL. The exact correlation sample is recoverable as 12 dependent scoring arms; pair-prior and random-null controls are present. Planted-shortcut sensitivity and a strong-VLM ceiling are absent. Dependence-aware inference is partial: leave-one-arm-out checks are descriptive, while historical permutation p-values do not address scorer-family dependence or independent-model uncertainty. See `R3_WPRD_construct_validity.md`, especially “Completeness Check for Paper A Rating.”

R1:
`BLOCKED_TECHNICAL`. The corrected-geometry R1 run was initiated under a frozen protocol but did not complete because unrelated GPU contention appeared. Consequently no corrected C1 WPRD, delta, or materiality conclusion is available. Epoch 0 and its intermediate checkpoint exist, but the full evaluator was not run. PID 32655 is evidenced as a separate OmniDocBench command tree; its owner is not established. The partial run artifacts are classified in `R1_artifact_audit.md` and must not be interpreted as a completed experiment.

## Evidence We Now Trust

- Canonical C1 checkpoint and train JSONL hashes verified unchanged in this closure pass.
- Corrected R2's common-population, nested comparison and its explicitly exploratory image-cluster interval, as recorded in `R2_nested_nuisance_analysis.md` and `.json`.
- R3's saved 12-arm tables and narrow prior-only/random-null control interpretation, subject to the dependence limitation above.
- R1's protocol, process/resource history, checkpoint hash, and terminal blocked status as provenance only. There is no R1 final metric.

## Evidence We Explicitly Do Not Use

- The incomplete R1 checkpoint, partial training losses, or any output from the in-training diagnostic as a corrected C1 result.
- The historical/non-comparable R2 scores, including −0.622110, as the intended nested incremental effect.
- Semantic interpretations unsupported by semantic evaluation.
- Dependent arm-correlation permutation p-values as independent-model inference.
- Historical C1 WPRD as the primary R1 comparator, because its Fourier scale differs from the units-only treatment.

## Final Research Boundary

The defensible claims are predictive and measurement-specific: `rel_feat` is decodable; corrected bounded R2 supports an incremental predictive improvement over its fixed G+O parent on the CAL-CHECK pilot; WPRD provides a narrow within-pair discrimination statistic with prior-only chance behavior under the saved implementation; and the historical geometry path had a verified units-contract implementation failure. These do not establish semantic specificity, a causal geometry contribution, open-vocabulary generalization, or full-population residual information after G+O+U+S. R1's geometry contribution remains undetermined.

## Experiments Not Authorized

- No more R1 reruns now.
- No seed sweep.
- No open-vocabulary campaign.
- No large feature extraction.
- No new architecture.

## Paper Transition

The program now moves from:

    experiment exploration

to:

    paper construction / evidence consolidation
