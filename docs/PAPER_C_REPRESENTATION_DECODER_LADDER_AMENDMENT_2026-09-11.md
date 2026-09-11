# Paper C — representation decoder ladder additive protocol amendment

**Status: registered before the corrected ladder execution.** This is an
additive clarification to
`docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md`. The original
preregistration and its historical result are unchanged and remain immutable.
This amendment resolves the output-column mapping and registers a new output
location for the corrected rerun. It does not change any scientific
hyperparameter, arm, seed, fold, estimator, bootstrap, or decision threshold.

## 1. Exact 51/50 decoder contract

The closed-set predicate vocabulary has exactly 51 total columns in decoder
index order:

| decoder indices | meaning | ladder treatment |
|---|---|---|
| `0..49` | the 50 canonical VG150 foreground predicates, in the dump's canonical vocabulary order | trained/scored as the positive predicate targets |
| `50` | synthetic `relation` placeholder/background | present in every fitted decoder's output width, never a positive GT target, and excluded from WPRD; CE arms train it indirectly as the non-target output, while ridge arms have an all-zero one-hot target column |

The background index is therefore exactly `50`; the foreground index list is
exactly `[0, 1, ..., 49]`. The background is not a real VG150 predicate and
must not enter the 50-class WPRD denominator or any WPRD cell comparison.

For the raw50 evaluator view, `B.gt_y` is a 0-based class id over the sorted
50 foreground class names, while `B.col_to_class` maps each foreground dump
column to that class id. The registered decoder target mapping is:

```text
class_id = B.gt_y[i]
decoder_column[i] = inverse(B.col_to_class)[class_id]
```

This produces a target in decoder columns `0..49`; decoder column `50` is
never selected for a positive WPRD target. Fitted decoder outputs have shape
`(n_rows, 51)`. Immediately before WPRD, every fitted output is projected to
`scores[:, [0..49]]`. A1, A4, and N2 already expose the same 50-column
foreground score view and require no synthetic background column.

The background column is therefore not a new background data population. It is
the registered 51-way output column: its ridge target is identically zero, and
the cross-entropy arms see it only as the non-target class while fitting the
positive-row labels.

The arms obey this contract as follows:

- **A1:** stored text channel, foreground 50 columns.
- **A2/A3:** 51-output decoders trained on mapped positive targets; foreground
  50 columns passed to WPRD.
- **A4:** recomputed 51-column cosine scores, then foreground 50 columns;
  this remains the identity gate against A1.
- **A5a/A5b:** 51-output geometry probes; foreground 50 columns passed to
  WPRD.
- **A6:** 51-output fusion decoder; foreground 50 columns passed to WPRD.
- **N1:** 51-output A2-form decoder fitted on permuted mapped targets;
  foreground 50 columns passed to WPRD.
- **N2:** prior-only raw50 foreground score, exactly as before.

This is a representation/output bookkeeping repair only. It does not add a
background training population, change the positive-only ladder target
population, or alter the registered estimator.

## 2. Corrected rerun output registration

The historical path below is permanently preserved and must not be written:

```text
runs/paper_c_representation_decoder_ladder.json
```

The corrected rerun is registered at this new dated directory:

```text
runs/paper_c_representation_decoder_ladder_rerun_20260911T105648Z/
  result.json
  console.log
  provenance.json
```

The directory is new and does not collide with any historical ladder, R0, R2,
R2c, checkpoint, or preregistration artifact. The corrected run must be
invoked with explicit `--out` and `--log` paths under this directory.

## 3. Execution provenance

The corrected run writes `provenance.json` inside the registered directory
before loading or fitting any arm. It records only execution and protocol
provenance required by the R2 treatment-fidelity execution policy and relevant
to this CPU-only ladder:

- UTC start/completion timestamps and status;
- repository path, branch, HEAD, and `git status --porcelain`;
- this amendment and the original ladder preregistration;
- exact command, fixed ladder configuration, input paths, output path, and
  console-log path;
- Python, PyTorch, host, kernel, CPU count, and platform information;
- whether `nvidia-smi` is available and its captured output when available;
  unavailable GPU information is recorded as unavailable, never fabricated.

R2-specific checkpoint-P hashes, optimizer state, trainable-tensor, and
prototype-source fields are not copied into this ladder provenance record.

## 4. Execution status

This amendment registers readiness requirements only. No ladder arm has been
run, no decoder has been trained, no scientific result has been generated,
and no historical artifact has been modified.
