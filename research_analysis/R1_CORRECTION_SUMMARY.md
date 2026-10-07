# R1 Verdict

| Item | Frozen C1 | Corrected C1 | Delta | Threshold | Decision |
|---|---:|---:|---:|---:|---|
| WPRD | 0.5749881522 (historical C1; context only) | NOT AVAILABLE (C1a run interrupted) | NOT AVAILABLE | +0.010 absolute WPRD on the identified C1a−C0 contrast | BLOCKED_TECHNICAL |

The corrected-geometry R1 run was initiated under a frozen protocol but did not complete because unrelated GPU contention appeared. Consequently no corrected C1 WPRD, delta, or materiality conclusion is available. The valid units-only comparator was frozen C0 at 0.5667271196 because historical C1 also changed Fourier scale. Epoch 0 completed and an intermediate checkpoint was written; the run was interrupted during its in-training diagnostic. See [R1_artifact_audit.md](R1_artifact_audit.md). The partial checkpoint is not a corrected C1 result.

## Scientific Interpretation

The corrected pixel-scale input path passed the CPU integration regression. No completed WPRD evidence exists, so R1 is neither positive nor negative for materiality. The interruption is not evidence that geometry had no effect, improved performance, or failed.

## Portfolio Consequence

`MERGE_A_B_AND_FOLD_C` remains the portfolio decision. Paper C standalone, open vocabulary, and residual-C1 standalone remain STOP / insufficient independent evidence. R1 is an optional future strengthening experiment, not required for paper completion; this closure pass authorizes no rerun.

## What Remains Unresolved

Whether the units-only correction changes WPRD by at least +0.010 under the fixed same-seed comparison remains unanswered. No second run or retry is authorized by this one-run gate.

## Next Step

Proceed with merged A+B+C paper consolidation; do not launch another GPU run under this authorization.
