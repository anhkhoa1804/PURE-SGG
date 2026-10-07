# R2 Encoder Provenance Audit

## Verdict

`PROVENANCE_INCONCLUSIVE`. The current R2 CAL-CHECK is held out from fitting the R2 offset/temperature parameters, but it is not certified encoder-held-out. The exact overlap with the full C1 encoder lineage cannot be determined because C1 was initialized from a historical checkpoint with unrecoverable training and model-selection populations, and the C1 run did not bind its local training JSONL by hash or save consumed image IDs.

The exact CAL-FIT and CAL-CHECK identities remain in `runs/paper_c_n4fit_pilot_training_20261003T182118Z/population_manifests/calfit_calcheck_manifest.json`; per-row identities are in `cal_row_manifest.jsonl`. The R2 manifest is hashed in the audit JSON.

## Training and evaluation lineage

The R2 images are sampled from the current `datasets_vg150_clean/train.jsonl` population. It contains 83,249 unique images and 1,046,427 relation rows. CAL-FIT contains 750 images/9,078 rows; CAL-CHECK contains 250 images/3,142 rows. Neither R2 population overlaps the current validation or test JSONL by image ID.

The C1 seed-1234 launcher (`runs/C1_seed1234/provenance.txt`; launch command in `logs/C1_seed1234.log`) specifies `vg150_source=local-jsonl`, `vg150_root=datasets_vg150_clean`, split `train`, seed 1234, 12,000 samples/epoch, three epochs, and initialization from `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` (SHA256 `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442`). The source at C1 commit `2c35bba26baced2b63e5afc41b4693e349e9889` makes the local-JSONL training dataset length equal `samples_per_epoch`; with shuffle and no weighted sampler, the training indices span positions 0–11,999 each epoch. Thus the *current* train-file reconstruction gives a 12,000-image sample frame, not the full 83,249-image split.

The overlap calculation against the currently available train file is 119 CAL-FIT images (1,400 rows) and 37 CAL-CHECK images (430 rows), 156/1,000 total. These are **candidate overlaps only**, not a verified historical consumed-ID intersection: the C1 provenance does not include a SHA256 for the `train.jsonl` present at run time, and does not store per-batch image IDs. The saved historical manifest records a different train-file hash (`36bc2923…`), while the current canonical train JSONL is `306fc0db…`; no artifact binds the former file's ordered image identities to this run. The 156 count must not be represented as the exact C1 exposure count.

## Selection and stopping audit

The C1 fine-tuning run used a fixed three-epoch budget. `save_best_checkpoints=false`; the script states that arms are compared at a fixed final budget, so the C1 fine-tune had no best-epoch checkpoint selection and no early stopping. It did run a 60-batch validation diagnostic each epoch and a full validation endpoint later; those are validation observations, but the saved protocol does not select the final C1 checkpoint from them.

This does not resolve upstream selection. C1 starts from `pure_best_adapt_light_mR50.pt`, a historical checkpoint whose embedded name is “best” and whose config records a best mR@50 claim, but `docs/HISTORICAL_CHECKPOINT_MANIFEST.md` says its original training dataset/split and evaluation split/image count are unknown. Its predecessor checkpoint is absent and original dataset preparation is unavailable. Therefore the pretraining image set and checkpoint-selection population cannot be enumerated or compared against CAL-FIT/CAL-CHECK. No zero-overlap claim is justified.

## Cached feature extraction is not training exposure

The canonical C1 `rel_feat` extraction manifest records frozen-checkpoint inference on **83,249 train images / 1,046,427 relation rows** (`runs/canonical_train_representation_full_20261001T020630Z/representation/canonical_train_relfeat_manifest.json`; source/input hashes in `representation/input_artifacts.json`). The bounded N4-FIT feature cache used by the R2 residual sidecar records **5,000 FIT images/63,286 rows and 1,000 CAL images/12,220 rows** (`runs/paper_c_n4fit_pilot_training_20261003T182118Z/extraction_provenance.json`). These are inference/cache populations, not proof those images trained C1; they show that identities and features were materialized for evaluation. The R2 offset and calibration parameters were fit on CAL-FIT only and scored on CAL-CHECK. Neither cache can recover the unavailable historical C1 training IDs.

## Required population reconciliation

| Population | Number of images | Number of rows | Source artifact | In C1 encoder-training set? | Used for checkpoint selection? | Used for early stopping? | Overlap count | Overlap percentage | Verdict |
|---|---:|---:|---|---|---|---|---:|---:|---|
| R2 CAL-FIT | 750 | 9,078 | Frozen R2 CAL-FIT/CAL-CHECK manifest and CAL row manifest | All 750 are in current canonical train split; exact run-time sample overlap unknown. Current-file 12k-frame candidate: 119 images/1,400 rows. | C1 fine-tune: no best-checkpoint selection. Upstream init: unknown. | C1 fine-tune: no. Upstream init: unknown. | 119/750 candidate, **not run-bound**; full-lineage exact count unknown | 15.87% candidate | R2 offset/calibration-fit population; encoder holdout not established. |
| R2 CAL-CHECK | 250 | 3,142 | Frozen R2 CAL-FIT/CAL-CHECK manifest and CAL row manifest | All 250 are in current canonical train split; exact run-time sample overlap unknown. Current-file 12k-frame candidate: 37 images/430 rows. | C1 fine-tune: no best-checkpoint selection. Upstream init: unknown. | C1 fine-tune: no. Upstream init: unknown. | 37/250 candidate, **not run-bound**; full-lineage exact count unknown | 14.80% candidate | Decoder-held-out from R2 fitting; not certified encoder-held-out. |
| C1 configured training split | 83,249 | 1,046,427 | Current `datasets_vg150_clean/train.jsonl`; C1 command/config | Configured split is `train`; effective 12k sample frame can be reconstructed only conditionally because run-time file hash is absent. | No C1 fine-tune best-checkpoint selection. | No C1 fine-tune early stopping. | CAL split membership: 750/750 and 250/250; exact consumed exposure is not run-bound | 100% split-membership overlap | Training split membership must not be conflated with realized sample exposure. |
| C1 validation diagnostics / endpoint | 10,401 | 132,556 | Current `datasets_vg150_clean/validation.jsonl`; C1 run config/logs | Not the configured C1 training split; no current CAL ID overlap. | C1 final checkpoint was fixed-budget, not selected by these diagnostics. Historical initializer's selection split remains unknown. | No C1 fine-tune early stopping. | 0 by current image IDs | 0% | No overlap with current R2 CAL, but does not settle upstream checkpoint selection. |
| Candidate held-out test split | 10,403 | 132,334 | Current `datasets_vg150_clean/test.jsonl` | Disjoint from current train and validation IDs; upstream initializer lineage unknown. | Unknown for upstream checkpoint. | Unknown for upstream checkpoint. | 0 with current R2 CAL; full C1-lineage exclusion unprovable | 0% only against current R2 CAL | Not eligible as provably encoder-held-out under the required full-lineage rule. |

The 119 and 37 image IDs in the current-file candidate intersection are stored in `R2_ENCODER_PROVENANCE_AUDIT.json`. All selected CAL identities are preserved in the frozen R2 manifest linked above.

## Decision

The correct status is `PROVENANCE_INCONCLUSIVE`, not `ENCODER_HELD_OUT`. The R2-CAL-CHECK prediction is decoder-held-out from the R2 adapter fitting, while at least a candidate subset overlaps the C1 fine-tuning sample frame and upstream encoder/selection exposure is unknown.

No R2-OOS fitting is run. Although current validation/test files exist, neither can be proven outside the historical initializer's training and checkpoint-selection populations. The existing cached feature artifacts do not repair missing lineage. A five-fold cross-fit on such a population would not solve encoder exposure and would create a false OOS label. No practical-effect threshold is frozen because the eligibility/provenance gate failed before protocol registration.

`R2_OOS_STATUS = BLOCKED_PROVENANCE_INCONCLUSIVE`

This audit does not change the R2 point estimate as a bounded decoder-held-out predictive result, but it narrows its interpretation: it is not evidence of encoder-held-out generalization. R1 remains `BLOCKED_TECHNICAL`; R3 remains `PARTIAL`; portfolio decision remains `MERGE_A_B_AND_FOLD_C`.
