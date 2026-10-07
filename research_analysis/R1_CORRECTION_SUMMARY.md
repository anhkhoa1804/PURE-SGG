# R1 Verdict

| Item | Frozen C1 | Corrected C1 | Delta | Threshold | Decision |
|---|---:|---:|---:|---:|---|
| WPRD | 0.5749881522 (historical C1; context only) | NOT AVAILABLE (C1a run interrupted) | NOT AVAILABLE | +0.010 absolute WPRD on the identified C1a−C0 contrast | BLOCKED_TECHNICAL |

The valid units-only comparator is frozen C0 at 0.5667271196 because historical C1 also changed Fourier scale. The R1 training run reached the epoch-0 checkpoint, then was interrupted during its in-training diagnostic when unrelated GPU PID 32655 appeared. No final evaluation ran.

## Scientific Interpretation

The corrected pixel-scale input path passed the CPU integration regression. No completed WPRD evidence exists, so R1 is neither positive nor negative for materiality.

## Portfolio Consequence

`MERGE_A_B_AND_FOLD_C` remains the portfolio decision. This blocked run does not establish an independent geometry contribution and does not establish a null effect.

## What Remains Unresolved

Whether the units-only correction changes WPRD by at least +0.010 under the fixed same-seed comparison remains unanswered. No second run or retry is authorized by this one-run gate.

## Next Step

Proceed with merged A+B+C paper consolidation; do not launch another GPU run under this authorization.
