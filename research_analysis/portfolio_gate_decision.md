# Portfolio gate decision

## Decision: `R1_BLOCKED_TECHNICAL`; portfolio `MERGE_A_B_AND_FOLD_C`

Corrected R2 uses fixed FIT-trained O and G+O parent scores, calibration and a nonnegative rel_feat offset fitted only on CAL-FIT, and the same 250-image / 3,142-row CAL-CHECK for all arms. The primary G+O→G+O+rel_feat calibrated LL change is **−0.060008** (image-cluster exploratory 95% CI **[−0.074317, −0.044480]**). Accuracy rises slightly, while macro recall falls slightly; this is not a uniform predicate gain. See `R2_nested_nuisance_analysis.md`.

R3 remains limited: the Spearman analysis has 12 dependent scoring arms from one underlying checkpoint/cache family; prior-only and random-null controls exist; planted-shortcut and strong-VLM ceiling controls are absent; dependence-aware inference remains incomplete. WPRD is a narrow within-pair statistic, not a general semantic metric.

## R1 final status

The corrected-geometry R1 run was initiated under a frozen protocol but did not complete because unrelated GPU contention appeared. Consequently no corrected C1 WPRD, delta, or materiality conclusion is available. Epoch 0 completed and an intermediate checkpoint was written; the run stopped during its in-training diagnostic and the full endpoint evaluator was not launched. The recorded process evidence shows PID 32655 running a separate OmniDocBench command tree, but does not establish its owner. PID 24793 was absent at preflight and was never signalled. Therefore R1 is **`BLOCKED_TECHNICAL`**, not positive, negative, or null. Historical C1 WPRD 0.5749881522 is context only because its Fourier scale is 0.01 and it does not isolate the units correction. The partial checkpoint is preserved at `runs/R1_fixed_geometry_C1a_20261007T021152Z/checkpoint.pt` and classified `PARTIAL_NONFINAL`; do not treat it as a corrected C1 result. See `R1_artifact_audit.md`.

PID 24793 was absent at preflight and was never signalled. No further run is authorized by this finite gate.

## Publication consequence

Portfolio remains **`MERGE_A_B_AND_FOLD_C`**. R1 did not provide a completed result supporting a separate geometry contribution; it also did not establish that the effect is null. Paper C standalone, open vocabulary, and residual-C1 standalone remain STOP / insufficient independent evidence. R1 is an optional future strengthening experiment, not required for paper completion. Do not rerun R1 or reopen Paper C. Continue with merged A+B+C consolidation and describe R1 as blocked.
