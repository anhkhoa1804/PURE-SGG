# Historical artifact index

This index locates major preserved Paper C artifacts. Paths may refer to
ignored/external local files and may not be present in a Git-only clone. Do not
rewrite historical records to match current interpretation.

| Artifact | Path | Purpose / status | Preservation note |
| --- | --- | --- | --- |
| Canonical source snapshot | `runs/paper_c_canonical_train_relfeat_20260930T062248Z/code` | Source snapshot at `ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`. | Immutable provenance; no tracked source changes made. |
| Canonical C1 checkpoint | `checkpoints/C1_seed1234.pt` | Frozen C1 checkpoint. | SHA256 recorded in reproducibility page; do not overwrite. |
| Canonical train representation | `runs/canonical_train_representation_full_20261001T020630Z/representation` | 83,249 images / 1,046,427 rows / 17 shards, 768-D float16. | Manifest and shard hashes are provenance-critical. |
| A1–A6 readout ladder and nulls | `runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/` | Readout, geometry, fusion, shuffled-label, prior controls. | Historical exploratory result; result JSON is metric source. |
| Geometry/diagnostic series | `runs/p57_*` through `runs/p70_*`; dated geometry docs in `docs/` | Geometry and nuisance diagnostics. | Preserve preregistrations, failures, and negative results. |
| Readout-v2 | `runs/eval_readout_v2_*`, `runs/readout_v2_pilot_*`; `docs/PAPER_C_READOUT_V2_*` | Readout treatment and fidelity work. | Protocol-specific; do not merge across populations. |
| Cross-seed exploration | `runs/eval_C1_seed5678/`, related historical logs | Historical cross-seed artifacts. | Distinct from the later authorized matched-supervision smoke; preserve as-is. |
| M2 prefix/balanced studies | `runs/paper_c_balanced15k_m2_20261002T075232Z/`, same-validation bridge directories | Sampling-regime diagnostics and rescoring audit. | Validation row populations matter; historical metrics are not automatically comparable. |
| FULL/A/B smoke | `runs/paper_c_matched_supervision_smoke_20261002T111542Z/` | Seed-1234 engineering/design smoke, final status reported `SMOKE_GO`. | Not confirmatory; retain mappings, logs, predictions, checkpoints, and audit. |
| Fullscale residual audit | `runs/paper_c_fullscale_residual_audit_20261003T084403Z/`; `runs/paper_c_nuisance_ladder_v2_20261003T092837Z/` | O/G+O residuals; full G+O+U+S blocked for identity coverage. | Exploratory/blocked records are part of the research trail. |
| Strategically stopped U/S replay | `runs/paper_c_nuisance_ladder_v2_final_20261003T101500Z/` | Partial replay stopped at 4,100 images / 51,573 relation rows / 121,401 crops. | Explicitly aborted strategically; not a completed N4 test. |
| U/S feasibility pilot | `runs/paper_c_nuisance_ladder_v2_us_pilot_20261003T113200Z/` | Bounded pilot/resource allocation artifacts. | Preserve without upgrading to a final nuisance result. |
| N4 vocabulary forensic | `runs/paper_c_nuisance_vocab_contract_20261003T114634Z/` | Historical/current vocabulary contract analysis. | Historical N4 checkpoint is reproduction-only for current CAL. |
| N4-FIT contract and pilot | `runs/paper_c_n4fit_contract_20261003T115720Z/`; `runs/paper_c_n4fit_pilot_training_20261003T182118Z/` | Leakage-safe FIT-only vocabulary/model bounded pilot. | Pilot only; fullscale not run; checkpoint/result records remain immutable. |

`runs/` itself is not a supported active workflow. Small tracked records and
large ignored payloads coexist; consult `.gitignore` and the per-run README,
manifest, and hashes before reuse.
