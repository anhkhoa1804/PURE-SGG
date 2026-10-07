# R1 — Fixed Geometry C1 Same-Seed Rerun

## 1. Scientific Question

Does correcting the geometry input-units contract alone materially change C1 under the frozen seed-1234 protocol? This question was not answered because an unrelated GPU workload appeared during the run.

## 2. Frozen Protocol

The pre-launch protocol is recorded at `research_analysis/R1_protocol_freeze.md`. The units-only registered contrast is C1a minus frozen C0: C1a uses pixel-scale boxes and Fourier scale 1.0; C0 uses normalized boxes and Fourier scale 1.0. Both use seed 1234 and the same frozen PURE initialization, launcher, stage-3 architecture/objective, dataset/split, optimizer recipe, fixed three-epoch budget, and evaluator.

Historical C1 has WPRD 0.5749881522134409, but is not the primary comparator: it already uses pixel-scale boxes while changing Fourier scale to 0.01. Comparing C1a directly to historical C1 would not isolate the units intervention. Frozen C0 WPRD is 0.5667271196244782.

The decision margin was frozen as +0.010 absolute WPRD, from `docs/PAPER_C_C0_C1_PREREGISTRATION.md`. It is a materiality threshold, not a significance test. The standard paired-cell bootstrap (2,000 replicates, seed 11) would be reported if evaluation completed. No between-seed uncertainty is estimated.

## 3. Resource / Process Check

Initial and immediate pre-launch checks found an L4 with 0 MiB reported, 4% utilization, and no compute process. PID 24793 was absent on both `ps` checks and was never signalled. Pre-launch disk availability was 24 GiB. During training, an unrelated process appeared: PID 32655, running `experiments/005_omnidocbench/prepare.py`, with 186 MiB GPU memory; at 02:57:19 UTC GPU utilization was 100%. The process was not touched. The full preflight evidence is in `research_analysis/R1_resource_check.md` and the timestamped run directory.

## 4. Geometry Contract Correction

No geometry source modification was needed. The existing C1a path sets `geom_input_pixel_space=true`, passing image-space pixel boxes to `geom_feats_torch`; Fourier scale remains 1.0. C0’s parent path divides boxes by image resolution, which causes six geometry channels to collapse under the helper’s pixel-scale clamp.

## 5. Regression Test

Added an integration regression that captures the actual `RelationalModel` geometry-helper inputs, verifies pair identity independent of internal row ordering, confirms coordinates remain pixel-scale, checks finite 8-D outputs with all channels varying, and checks finite relation features. Existing geometry tests also cover normalized-box collapse, pixel-box variation, and scale-invariant feature behavior. Targeted command: `.venv/bin/python -m pytest tests/test_geometry.py tests/test_geometry_contract_flags.py tests/test_model_forward.py -q` — **18 passed**, 5 existing transformer warnings. `py_compile` and `git diff --check` passed before launch.

## 6. Run Provenance

The single C1a training command started at 2026-10-07 02:18:22 UTC from commit `93495d0932ff3d7ad275931d139d12d24c01f3e7`. It loaded the registered frozen PURE initialization (SHA256 `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442`), reset the epoch count, and used a fresh AdamW optimizer. It did not resume C0 or C1. After the first epoch’s checkpoint save, the launcher entered its 60-batch in-training validation diagnostic. When unrelated PID 32655 appeared, only the R1 training process (PID 17600) received SIGINT. It exited with status 130 after `KeyboardInterrupt`; no retry was made.

The epoch-0 checkpoint is a partial/intermediate artifact, SHA256 `1bebd6104688c4c50484fe0d1a507caed2e23c23f5592628490ff697fd0cda82`, size 5,293,175,545 bytes. It is not a completed three-epoch model and was not evaluated. The log and resource/process evidence are retained under `runs/R1_fixed_geometry_C1a_20261007T021152Z/`.

## 7. Results

- Frozen C0 parent WPRD: **0.5667271196244782**.
- Historical C1 context WPRD: **0.5749881522134409** (not the isolated units-only parent).
- Corrected C1a WPRD: **NOT AVAILABLE**; full evaluation was not launched.
- Absolute delta and 95% CI: **NOT AVAILABLE**.
- Predefined materiality margin: **+0.010 WPRD**.
- Decision: **BLOCKED_TECHNICAL**; the success criterion cannot be assessed.

No completed validation result, geometry endpoint statistics, fusion-gate endpoint statistics, or predicate/error result is claimed. Training samples had finite reported loss through the epoch-0 progress logs; interruption occurred during the subsequent diagnostic evaluation.

## 8. Predicate / Error Pattern

Not run. No completed standard evaluator output exists, and no post-hoc error analysis was added.

## 9. Success Criterion

**BLOCKED_TECHNICAL.** The fixed +0.010 threshold was not changed. The evaluation needed to compute the contrast never completed.

## 10. Interpretation

The geometry path was configured to use pixel-scale boxes for C1a, and the CPU regression test verifies that the model passes pixel-scale boxes to the geometry helper. This establishes the intended correction was exercised by the training configuration; it does not establish that corrected geometry caused any semantic improvement.

## 11. What This Establishes

The frozen C1a training configuration and units correction were executable; training reached its first epoch checkpoint with finite progress losses. An unrelated GPU process appeared, and the R1 job was stopped as required. The only authorized run was not completed.

## 12. What This Does Not Establish

No R1 WPRD effect, confidence interval, materiality decision, semantic interpretation, causal contribution, or geometry-specific publication claim is established. This is not a null result.

## 13. Publication Consequence

Portfolio remains **`MERGE_A_B_AND_FOLD_C`**. The interrupted pilot does not justify reopening C as a separate paper, nor does it support a claim that the units correction is immaterial. Do not retry or launch another GPU experiment under this authorization. The partial checkpoint is preserved for provenance only.
