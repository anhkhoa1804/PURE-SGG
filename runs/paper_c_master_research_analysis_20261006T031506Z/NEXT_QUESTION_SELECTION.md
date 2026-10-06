# Highest-value next question

**Question:** Does frozen C1 distinguish human-verified contrasting relation truth across images with identical ordered object labels, beyond frozen geometry and appearance/context competitors?

**Why selected:** the largest gap is semantic identification, not another score on the current annotations. O utility and geometry competition are already measured; fullscale N4 would sharpen a small predictive residual while retaining ambiguous labels. Human-controlled evaluation can change interpretation whether positive or negative. Existing model-blind E2 development candidates make instrument validation feasible, but no human gold currently exists. [E2,E4,E10]

**Hypothesis:** on accepted independent contrast blocks, a frozen focal model has higher both-correct accuracy than the strongest predeclared nuisance competitor. **Null:** no positive paired advantage. The claim is bounded controlled image-conditioned discrimination, not global semantics. Final effect/precision thresholds and sample size must be fixed from human development, not from focal-model results.

**Decision tree:** human gold absent → design only; development not solvable/support too small → revise evaluation in a new registration or stop; development passes and all controls frozen → freeze an untouched final population; positive final difference with adequate uncertainty/positive-control behavior → evidence for bounded discrimination and consider a new representation objective; negative difference with a successful positive control → current C1 fails the targeted task and further residual training needs a new mechanism; all controls fail → task/model interface inconclusive.

**Rejected now:** fullscale N4 (small resource-trigger failure; still only predictive), residual C1 (premise not identified), seed5678/B2 (implementation replication and nuisance specificity cannot supply semantic truth), architecture isolation (expensive and evaluation remains ambiguous), predicate-disjoint training (large distribution/alias/supervision changes before semantic ground truth), additional decoder sweeps (low decision value). Cross-dataset and synthetic tests are useful later if a controlled semantic target is credible.

**Cost and stop:** this task uses zero GPU-hours. Human development and adjudication are the binding prerequisites; no timing or final N is fabricated. No experiment executes now. Stop before inference until gold, human gate, final sample-size plan, nuisance controls, independent positive visual control and frozen final manifest exist. See `semantic_evaluation_protocol.md`.

**Execution decision:** `NO_NEW_EXPERIMENT_JUSTIFIED_WITH_CURRENT_ASSETS`. This does not reject all future research; it rejects another autonomous compute campaign on present labels. Paper C remains closed; a separately registered Paper D is the recommended path if human development passes.
