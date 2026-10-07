# Paper Evidence Dossier

## 1. Central Research Question

When predicate-supervised scene-graph representations and metrics appear to capture relations, how much of the measured signal is attributable to object-pair priors, geometry/input visibility, readout choices, or other predictive nuisance structure—and what conclusions remain defensible after those alternatives are audited?

## 2. Main Thesis

The repository supports predictive decodability and an exploratory decoder-held-out predictive comparison from frozen `rel_feat` beyond a fixed G+O parent on one CAL-CHECK pilot. Encoder-held-out status is **not established**: the C1 training-file identity and consumed image IDs are not run-bound, a current-file reconstruction indicates candidate overlap, and the upstream initializer's training/selection populations are unknown. The evidence does not identify semantic relational understanding, causal geometry effects, or open-vocabulary generalization. See `R2_ENCODER_PROVENANCE_AUDIT.md/json` and `R2_OOS_FINAL.md/json`.

## 3. Evidence Chain

### 3.1 Prior effects

The calibrated ordered object-pair prior had LL **1.6323624327**, accuracy **0.6458541792**, and macro recall **0.2000605986** on the fixed validation population of **1,182 images / 14,991 relation rows**. **12,330 / 14,991 rows** had a train-supported ordered pair; unsupported pairs backed off to the train-global prior. Source: `runs/paper_c_fullscale_residual_audit_20261003T084403Z/` saved pair-prior results and `research_analysis/R2_nested_nuisance_analysis.md` §4. Unit: relation row for LL/accuracy and predicate macro recall. Status: saved full-scale historical nuisance result, reused as a frozen comparator; current corrected R2's fitting/evaluation population is different.

On corrected R2 CAL-CHECK, fixed calibrated O scored LL **1.598961** and O + `rel_feat` scored **1.280646**, Δ **−0.318314**; the exploratory image-cluster 95% CI is **[−0.357666, −0.276130]**. Both arms used the same **250 images / 3,142 rows**; alpha was fit on CAL-FIT only. Source: `research_analysis/R2_nested_nuisance_analysis.json`, keys `O` and corrected contrast; unit: relation-row mean log-loss, bootstrap unit: image. Status: corrected bounded exploratory analysis. Accuracy changed **0.663590→0.682050** while macro recall changed **0.227013→0.200346**, so this is not a uniform class gain. Source: same artifact, `research_analysis/R2_nested_nuisance_analysis.md` §8.

### 3.2 WPRD measurement audit

The historical readout ladder reports WPRD A1–A6 values **0.574988, 0.580128, 0.585669, 0.575834, 0.588144, 0.596235, 0.585503**; shuffled null N1 **0.505451** and prior-only N2 **0.500000**. Source: `paper_package/TABLES/readout_ladder.csv` and `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`; population: **10,401 validation images / 132,556 relation rows / 20,016 WPRD cells**, across **50 predicates**, per `paper_package/METHODS.md` §Historical readout experiment. Unit: aggregate WPRD per arm over eligible within-pair predicate cells; status: historical. A5b used a different fit population than A5a, as recorded in the same methods section.

Three saved cross-arm correlation tables each contain **12 complete deterministic scoring arms**: `runs/p49_metric_grounding/corr.json`, `runs/p53_metric_grounding_vg150only/corr.json`, and `runs/p61_test_metric_grounding/corr.json`. For held-out test/raw-name, the recorded correlations are Spearman(R@50,WPRD) **+0.9142** and Spearman(mR@50,WPRD) **−0.7273**; status: historical descriptive arm association. Unit: scoring arm, not independently trained model. R3 audit `research_analysis/R3_WPRD_construct_validity.md` §§2–4 records that arm dependence invalidates interpreting the stored permutation p-values as independent-model inference.

### 3.3 Geometry/input-visibility audit

The forensic record found boxes divided by **336** before `geom_feats_torch()`, whose `clamp_min(1.0)` presumes pixel-scale coordinates; the resulting size/aspect channels `rw`, `rh`, `ar1`, `ar2`, `a1`, `a2` collapsed. Source: `research_analysis/R1_protocol_freeze.md`, `research_analysis/R1_fixed_geometry_C1.md` §§4–5, and regression `tests/test_geometry_contract_flags.py`. Unit: geometry-feature channels under the audited code path; status: implementation defect verified by source tracing and CPU regression, not a performance effect.

Historical C0 WPRD is **0.5667271196** and historical C1 WPRD is **0.5749881522**. Sources: `runs/eval_C0/result.json` and `runs/eval_C1/result.json`; historical evaluation population/protocol, WPRD endpoint. C1 is not a clean units-only contrast because its Fourier scale differs. R1 used C0 as the matched parent, froze a **+0.010** absolute WPRD materiality threshold, and completed no endpoint. Source: `research_analysis/R1_protocol_freeze.md` and `research_analysis/R1_artifact_audit.md`. Status: corrected result unavailable; geometry materiality undetermined.

### 3.4 Nested nuisance readout

Corrected nested R2 fixes the FIT-trained parent and adds a nonnegative scalar multiple of frozen `rel_feat` log-probabilities. On common CAL-CHECK (**250 images / 3,142 rows**): O LL **1.598961→1.280646**, Δ **−0.318314**, CI **[−0.357666, −0.276130]**; G+O LL **1.315260→1.255252**, Δ **−0.060008**, CI **[−0.074317, −0.044480]**. Source: `research_analysis/R2_nested_nuisance_analysis.json` and `.md` §§6–8. Unit: relation-row mean calibrated log-loss; CIs: paired image-cluster bootstrap, **2,000** resamples, seed **20261003**. Status: corrected, bounded, exploratory. The zero-offset configuration recovers the parent exactly; selected CAL-FIT objective is no worse than parent. The fixed parent is not refit in the extension. The CAL-CHECK labels are held out from fitting the R2 offset, but encoder-held-out evaluation is not established; see the provenance audit.

For G+O, accuracy changes **0.667091→0.668364** and macro recall **0.137086→0.135990**. Source: same R2 artifacts; same rows/units. Thus reduced log-loss does not imply a broad argmax or class-balanced gain.

The separate N4-FIT feasibility pilot scored N4-FIT LL **1.5984459578**, N4-FIT + `rel_feat` **1.5959997348**, Δ **−0.0024462230**, exploratory image-cluster CI **[−0.0026268349, −0.0022835433]**, on **250 CAL-CHECK images / 3,142 rows**. Source: `runs/paper_c_n4fit_pilot_training_20261003T182118Z/pilot_results.json`, `pilot_bootstrap.json`, summarized in `paper_package/RESULTS.md` §Richer nuisance pilot. Unit: relation-row mean loss, resampled by image. Status: pilot-only; not the full 83,249-image N4-FIT run and not a matched next rung for corrected R2. It did not justify the planned full-scale resource cost.

### 3.5 Predicate-level heterogeneity

Corrected R2's full per-predicate table is `research_analysis/R2_per_predicate_effects.csv`. Population: CAL-CHECK **3,142 relation rows**, with support varying by predicate. Unit: one-vs-rest predicate recall and precision; status: bounded exploratory. For G+O, the largest class `on` (**1,124** rows) recall changes **0.946619→0.948399**; `has` (**418**) **0.892344→0.901914**; `in` (**404**) **0.529703→0.514851**; `wearing` (**258**) **0.821705→0.837209**; `of` (**249**) **0.526104→0.522088**. Corresponding precision changes are mixed. These examples and the complete table show heterogeneity, not uniform improvement or a preregistered head/tail effect.

## 4. Claims We Can Make

- The tested `rel_feat` supports predicate decoding under the recorded readouts.
- Ordered object-pair priors carry substantial predictive value under the evaluated label distribution.
- WPRD is a narrow, conditional within-pair score-discrimination statistic; the pair-prior-only control is at its chance value in the saved implementation.
- Corrected nested R2 supports a bounded incremental predictive improvement in calibrated log-loss beyond the fixed G+O parent on this CAL-CHECK pilot.
- The historical geometry feature path violated its coordinate-units contract.
- The R1 correction's performance effect remains unknown.

## 5. Claims We Cannot Make

We cannot claim semantic relational understanding, semantic specificity, causal geometry contribution, geometry-independent relation information, full-population residual beyond G+O+U+S, open-vocabulary generalization, or interactional-family superiority. We cannot use the 12-arm correlation p-values as independent-model evidence. No completed R1 WPRD result exists.

## 6. Statistical Qualifications

R2 uncertainty resamples images and retains relation rows, but is conditional on one fitted set of weights, one selected split, and calibration. It is exploratory, not confirmatory; it does not estimate seed/training uncertainty. In addition, C1 encoder exposure is unresolved, so this is not encoder-held-out evidence. The earlier WPRD cell-based intervals and R2 image-cluster intervals use different units and answer different questions. The R3 nominal permutation p-values shuffle dependent scorer arms and are not valid independent-model inference. Per-predicate estimates have variable support; no new thresholds or groups are introduced here.

## 7. Experimental Limitations

R2 uses a 250-image CAL-CHECK pilot. Full-scale U/S nuisance reconstruction was not completed; N4-FIT was bounded and its cross-level comparison differs in training scale/offset orientation. R1 was blocked during a diagnostic after epoch 0; no full evaluation ran. Current validation has zero rows eligible under the frozen interactional-family definition (`runs/paper_c_final_evidence_audit_20261003T185925Z/split_integrity_audit.md`); this is an eligibility count on the fixed validation rows. No human semantic adjudication or predicate-disjoint test supports semantic/open-vocabulary claims.

## 8. Frozen / Exploratory / Historical Evidence

| Evidence | Status | Population / unit | Source |
|---|---|---|---|
| A1–A6, N1–N2 WPRD ladder | Historical | 10,401 images; 132,556 rows; 20,016 WPRD cells; arm-level aggregate | `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json`; `paper_package/TABLES/readout_ladder.csv` |
| R3 correlations | Historical descriptive | n=12 dependent scorer arms; correlations over arm-level summaries | `runs/p49_metric_grounding/corr.json`; `runs/p53_metric_grounding_vg150only/corr.json`; `runs/p61_test_metric_grounding/corr.json` |
| Full-scale O / G+O parents | Frozen saved nuisance predictors | 1,182 validation images; 14,991 rows | `runs/paper_c_fullscale_residual_audit_20261003T084403Z/`; audit summary `research_analysis/R2_nested_nuisance_analysis.md` |
| Corrected nested R2 | Exploratory bounded | FIT/CAL-FIT/CAL-CHECK split; check=250 images/3,142 rows; row-mean LL with image bootstrap | `research_analysis/R2_nested_nuisance_analysis.json` |
| N4-FIT residual pilot | Pilot-only | CAL-CHECK=250 images/3,142 rows | `runs/paper_c_n4fit_pilot_training_20261003T182118Z/pilot_results.json` |
| R1 corrected geometry | Blocked; result unavailable | One seed-1234 attempt; no valid endpoint | `research_analysis/R1_artifact_audit.md`; `runs/R1_fixed_geometry_C1a_20261007T021152Z/` |

## 9. Remaining Evidence Gaps

The full-scale G+O+U+S contrast; encoder-held-out R2-OOS estimate; completed corrected-geometry R1 endpoint; independent-model WPRD validity evidence; controlled separation of appearance, scene context, and geometry; adjudicated semantic relation labels; and an eligible frozen interactional-family evaluation remain absent. C1's upstream checkpoint training/selection population is not recoverable from current artifacts. These are gaps, not null results.

## 10. Publication Risk Assessment

The paper is viable as a measurement/predictive audit if its title, abstract, and claims stay narrow. Main reviewer risks are WPRD's dependence on dataset annotation and eligibility, dependent-arm correlation inference, the small R2 check set, mixed evidence populations across historical/pilot arms, and the lack of semantic ground truth. R1 is optional strengthening, not required to write the consolidated paper; its blocked status must be visible rather than converted into a result.
