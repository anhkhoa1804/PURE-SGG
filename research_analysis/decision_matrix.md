# Paper portfolio decision matrix

| Candidate | Independence | Novelty | Current Evidence | Major Confound | Missing Evidence | Publication Strength | Recommended Action |
|---|---|---|---|---|---|---|---|
| Paper A | CONDITIONAL | CONDITIONAL | CONDITIONAL | Shared evaluation/data lineage with B | Independent external validation | CONDITIONAL | MERGE_A_B_AND_FOLD_C |
| Paper B | CONDITIONAL | CONDITIONAL | CONDITIONAL | Benchmark/protocol construct validity and shared model families | Cross-model evidence and external validation | CONDITIONAL | MERGE_A_B_AND_FOLD_C |
| Paper C | WEAK | CONDITIONAL | CONDITIONAL | Geometry+Fourier bundled; nuisance residual is pair-prior conditional, not semantics | R1 geometry-only condition is resource-blocked | WEAK | MERGE_A_B_AND_FOLD_C |
| Paper D | STRONG | STRONG | NOT_READY | Requires genuinely new semantic evaluation | Human-verified controlled relation dataset | NOT_READY | STOP |
| Paper E / Open Vocabulary | CONDITIONAL | CONDITIONAL | NOT_READY | Current predicate training overlaps the fixed 51-class output | Predicate-disjoint task and protocol | NOT_READY | STOP |
| Paper F / Residual C1 | CONDITIONAL | CONDITIONAL | WEAK | N4-FIT pilot is small; full nuisance signal unresolved | No current decision to train; pilot delta below resource trigger | STOP | STOP |

Portfolio recommendation: **`MERGE_A_B_AND_FOLD_C`**. C1 remains a result/qualification within the merged measurement and audit paper. R1 was scientifically justified but could not run because the fresh L4 check found an active workload; no outcome is imputed. This is not a claim that A/B are ready without their own manuscript-level review.
