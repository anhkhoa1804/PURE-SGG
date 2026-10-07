# Predictive Relation Scores Under Priors and Measurement Audits

## Abstract

Predicate-supervised scene-graph generation (SGG) predicts predicates for object pairs, but score quality alone does not identify what evidence supports a prediction. We present a measurement and predictive audit of a frozen representation. Historical readouts show predicate decodability, while a pair-constant prior obtains chance WPRD, supporting only a narrow within-pair score-discrimination interpretation. Broader WPRD construct validity remains partial: the reported associations use 12 dependent scoring arms, and planted-shortcut and strong-VLM ceiling controls are absent. In the corrected nested nuisance analysis, a frozen-feature offset changes calibrated log-loss from 1.315260 to 1.255252 nats per row beyond a fixed G+O parent (Δ=−0.060008; exploratory image-cluster 95% interval [−0.074317,−0.044480]) on 250 CAL-CHECK images (3,142 rows). The CAL-CHECK labels were not used to fit the R2 offset, but encoder-held-out status is not established: a current-file reconstruction places 37 CAL-CHECK images in C1's direct fine-tuning sample frame, while the run's exact data identity and the initializer's training/selection lineage are unavailable. Accuracy rises slightly while macro recall falls, with mixed predicate changes. A geometry units-contract failure is verified, but the corrected C1 run did not reach endpoint evaluation. The evidence supports an exploratory decoder-held-out predictive comparison, not encoder generalization, semantic specificity, causal geometry effects, or open-vocabulary generalization.

## 1. Introduction

Scene-graph generation assigns predicates to ordered object pairs. A successful prediction may draw on pair frequencies, geometry, local appearance, global context, or annotation regularities. A decoder's ability to recover predicate labels therefore does not, by itself, identify the source or semantic character of its signal. [CITATION NEEDED]

This paper consolidates readout, metric, geometry, and nuisance analyses to ask what the saved measurements justify. Its contribution is an evidence-bounded audit, not a new semantic model. We distinguish decodability from semantics, a within-pair ranking statistic from broad metric validity, and a predictive residual from encoder-held-out generalization.

Historical WPRD readouts show predicates are decodable, and geometry readouts are competitive under their recorded protocols. A pair-only control yields WPRD=.5, as expected when scores are constant within a pair. However, the cross-arm metric associations use 12 dependent deterministic scoring arms. WPRD is therefore treated as a narrow conditional discrimination diagnostic, not a general semantic-quality measure.

The corrected nested R2 comparison fixes the G+O parent and adds a frozen `rel_feat` log-probability offset with an exact zero-effect parent setting. On 250 CAL-CHECK images/3,142 rows, calibrated LL changes from 1.315260 to 1.255252 (Δ=−0.060008; exploratory paired image-cluster interval [−0.074317,−0.044480]). This is decoder-held-out from the adapter fitting, but the C1 encoder exposure audit is inconclusive. The current R2 images come from the train split; 37 CAL-CHECK images overlap C1's current-file reconstructed 12k sample frame. The C1 run has no run-bound train-file hash or consumed-ID log, and it began from a historical checkpoint with unknown training and checkpoint-selection populations. Thus this result is not encoder-held-out evidence.

A separate audit found that the historical geometry path scaled boxes before a helper that expects pixel coordinates, collapsing six geometry channels. The corrected same-seed C1 run did not reach evaluation. The implementation defect is established; its performance consequence remains unavailable.

## 2. Research question and estimand

The nested R2 estimand is the row-weighted calibrated log-loss difference on CAL-CHECK between a fixed G+O parent and that parent plus a frozen-feature offset. CAL-FIT (750 images/9,078 rows) fits the R2 temperatures and alpha; CAL-CHECK (250/3,142) is not used for those fits. The paired interval resamples image clusters and retains all rows. This identifies a decoder-held-out predictive comparison conditional on the saved features, fixed parent, fitting procedure, and selected split. Because encoder exposure is unresolved, it does not identify encoder-held-out generalization. It is not a semantic or causal estimand.

## 3. Methods and evidence populations

### Historical readouts and WPRD

The historical readout ladder uses 10,401 validation images, 132,556 rows, and 20,016 WPRD cells. A1–A6 scores are .574988, .580128, .585669, .575834, .588144, .596235, and .585503; N1=.505451 and N2=.500000. A5a and A5b have different fit protocols. Sources: `paper_package/TABLES/readout_ladder.csv` and `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`.

WPRD averages within-ordered-pair predicate-score AUCs over eligible cells under the recorded per-class cap. Pair-prior chance performance supports that narrow implementation property. Held-out test correlations +.9142 (R@50) and −.7273 (mR@50) are calculated over n=12 dependent arms and are descriptive only. No planted-shortcut sensitivity or strong-VLM ceiling is available. Sources: `tools/within_pair_discrimination.py` and `research_analysis/R3_WPRD_construct_validity.md`.

### Corrected nested R2

For each arm, calibrated fixed-parent logits are preserved; the extension adds a nonnegative alpha times frozen `rel_feat` log-probabilities. Alpha zero exactly recovers the parent, and the parent is included as a candidate under the same CAL-FIT objective. For G+O, alpha≈.196844. The exploratory image-cluster bootstrap uses 2,000 replicates, seed 20261003, pooled row-weighted differences. It is conditional on the one fit and split and does not include refit or seed uncertainty. Source: `research_analysis/R2_nested_nuisance_analysis.json`.

### Encoder provenance

C1's direct run records seed 1234, train split, 12,000 samples per epoch, three epochs, and initialization from `checkpoints/demo_best/pure_best_adapt_light_mR50.pt`. The C1 fine-tune used a fixed final epoch, no best-checkpoint selection, and no early stopping. Current-file index reconstruction suggests 119 CAL-FIT images and 37 CAL-CHECK images are in the local JSONL 12,000-index sample frame. This is not an exact historical overlap count because C1's train-file hash and consumed IDs were not recorded. The base checkpoint's own training and selection populations are unknown; see `research_analysis/paper/R2_ENCODER_PROVENANCE_AUDIT.md/json` and `docs/HISTORICAL_CHECKPOINT_MANIFEST.md`.

## 4. Results

On common CAL-CHECK, O LL is 1.598961 and O+`rel_feat` LL is 1.280646 (Δ=−.318314; exploratory interval [−.357666,−.276130]). The primary G+O parent is 1.315260 and G+O+`rel_feat` is 1.255252 (Δ=−.060008; exploratory interval [−.074317,−.044480]). Both are decoder-held-out pilot contrasts, not full-scale or encoder-held-out estimates.

For G+O, accuracy changes .667091→.668364 while macro recall changes .137086→.135990. Class recall is mixed: `on` (support 1,124) .946619→.948399; `has` (418) .892344→.901914; `in` (404) .529703→.514851; `wearing` (258) .821705→.837209; `of` (249) .526104→.522088. The complete table is `research_analysis/R2_per_predicate_effects.csv`. No predeclared head/tail thresholds exist, so support ordering is descriptive.

The 37 current-file candidate overlaps prevent a claim of complete C1 fine-tune separation for CAL-CHECK; missing base-checkpoint lineage prevents certification even for images outside that candidate set. No R2-OOS experiment was run because no eligible population could be proven outside the full encoder training and selection lineage. R2-OOS status is `BLOCKED_PROVENANCE_INCONCLUSIVE`.

## 5. Geometry audit

The historical geometry path divided boxes by 336 before a helper whose clamp presumes pixel-scale values, collapsing `rw`, `rh`, `ar1`, `ar2`, `a1`, and `a2`. R1 completed epoch 0 but stopped before the full evaluator; no corrected WPRD endpoint exists. This is an implementation/input-contract finding, not a measured performance effect. [EVIDENCE GAP]

## 6. Discussion

The strongest defensible positive result is an exploratory reduction in calibrated log-loss for the frozen-feature offset beyond a fixed G+O parent on the R2 CAL-CHECK population. The parent/offset construction is nested, and the CAL-CHECK labels do not fit the offset. However, the evaluation is not encoder-held-out: partial direct-training overlap is suggested by current data reconstruction, while exact exposure and upstream checkpoint-selection exposure remain unknown. The effect must not be presented as generalization of the encoder or as semantic relational information.

The WPRD control shows that a pair-constant score cannot create within-pair ranking under this metric. It does not exclude geometry, appearance, or context. The geometry audit identifies a verified input-contract error but provides no corrected endpoint. Together these findings support an audit of measurement interpretation, not a semantic model claim.

## 7. Limitations and conclusion

R2 is a 250-image/3,142-row pilot with one fixed fit and an image-cluster interval conditional on that fit. Encoder and checkpoint-selection exposure cannot be completely reconstructed. WPRD construct validity remains partial; R1 has no endpoint; no full-population G+O+U+S, human semantic endpoint, or predicate-disjoint evaluation is available.

The evidence shows predicate decodability and a bounded decoder-held-out predictive comparison under the corrected nested formulation. It does not show that the C1 encoder generalizes this increment to unseen images, that the increment is semantically specific, that geometry has a causal performance effect, or that open-vocabulary behavior exists. The merged paper should present the R2 estimate as exploratory with unresolved encoder provenance, keep WPRD secondary, and treat geometry as an implementation audit with an unavailable performance consequence.

## References

[CITATION NEEDED: SGG foundations and predicate evaluation]

[CITATION NEEDED: within-group discrimination metrics and clustered uncertainty]
