# Paper C evidence ledger

Values below are tied to the cited artifacts. WPRD is reported as stored in
the 50-class ladder artifact; residual deltas are calibrated multiclass
log-loss contrasts on the stated fixed validation population unless noted.

| Experiment / artifact | Hypothesis and population | Primary result | Interpretation | Does not prove |
| --- | --- | --- | --- | --- |
| A1–A6, N1–N2; `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/result.json` | Frozen `rel_feat` readouts; 20,016 matched WPRD cells, 50-class WPRD | A1 .574988; A2 .580128; A3 .585669; A4 .575834; A5a .588144; A5b .596235; A6 .585503; N1 .505451; N2 .500000 | Readout decodability is present under these protocols; geometry-only was competitive and the shuffled null was near chance. | Semantics, generalization, or a uniquely relational source of the signal. |
| Canonical C1 representation; `runs/canonical_train_representation_full_20261001T020630Z/representation/canonical_train_relfeat_manifest.json` | Frozen C1 on canonical train; 83,249 images / 1,046,427 rows | 17 shards, width 768, float16, manifest complete | Provides a frozen feature artifact for identity-safe downstream diagnostics. | Predictive validity by itself or any semantic interpretation. |
| Same-validation M2; `runs/paper_c_balanced15k_m2_20261002T075232Z/` | Balanced 15k training images, fixed E2-excluded validation (1,182 images / 14,991 rows) | Log-loss 1.6304535145; accuracy .5614035088; macro recall .1412721492 | Result is specific to balanced sampling and the registered M2. | Direct comparison with historical prefix metrics evaluated on a different validation-row population; semantic claims. |
| Nuisance residual audit; `runs/paper_c_nuisance_ladder_v2_20261003T092837Z/nuisance_ladder_v2_results.json` | O and G+O fit/calibrated under frozen image-disjoint split; fixed validation 14,991 rows | O→O+rel_feat delta −.3660331323, CI [−.3966375880,−.3361649032]; G+O→G+O+rel_feat delta −.0051288329, CI [−.0263599217,+.0158628067] | Contribution contracted markedly with G+O; G+O residual interval crosses zero. G+O+U+S was blocked by incomplete identity-safe U/S cache coverage. | Full nuisance-gate result, conditional mutual information, relational semantics. |
| N4 vocabulary contract and bounded pilot; `runs/paper_c_n4fit_pilot_training_20261003T182118Z/` | FIT-only vocabulary/model, CAL-FIT calibration, CAL-CHECK evaluation; 5,000 FIT / 750 CAL-FIT / 250 CAL-CHECK images | CAL-CHECK LL 1.5984459578 vs 1.5959997348; delta −.0024462230; exploratory image-bootstrap CI [−.0026268349,−.0022835433] | Small pilot residual was below the prior G+O residual magnitude; registered resource decision did not support fullscale. | Final full-population G+O+U+S result or absence/equivalence of residual. |
| Matched-supervision smoke; `runs/paper_c_matched_supervision_smoke_20261002T111542Z/` | FULL/A/B on frozen 750/200/200 pilot, seed 1234 | Persisted `final_status.json` is reported as `SMOKE_GO`; artifact records intervention and pipeline checks. | Engineering/scientific-design feasibility only. | Confirmatory effect, generalization, or an intervention winner. |

### Population and protocol cautions

The N4-FIT pilot used FIT-only object vocabulary (6,909 labels), 16,900 input
width and 16,900→768→51 model. Its 250-image CAL-CHECK was evaluation-only;
fullscale N4-FIT was deliberately not run. The historical 2,907-label N4
checkpoint is reproduction-only and inadmissible for current CAL because its
training population overlaps current CAL. The current validation interactional
eligibility count is zero.
