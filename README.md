# Research-No.1 / Paper C

## Current status

**`PAPER_C_DIRECTION_CLOSED`.** The current Paper C research direction was
closed after controlled representation/readout diagnostics, nuisance
comparisons, intervention smoke tests, and a bounded leakage-safe N4-FIT
pilot. There is no currently authorized next Paper C experiment.

The evidence does **not** prove relational semantics, causal information,
open-vocabulary relation understanding, interactional-family superiority, or
predictive residual beyond a full-population G+O+U+S nuisance model. Fullscale
N4-FIT was not run, and no C1 retraining was authorized from this branch.

## Research question

The evolved question was whether predicate-supervised relation representations
retain predictive residual structure beyond nuisance variables, and whether
such residual structure would justify a constructive residual-relation model.
The experiments support limited claims about decodability and predictive
utility under specified populations and decoder families; they do not identify
the source of that utility as relational semantics.

## Main empirical findings

The registered 50-class WPRD readout ladder reported these macro values:

| Arm | WPRD macro |
| --- | ---: |
| A1 frozen text | 0.574988 |
| A2 linear | 0.580128 |
| A3 GELU MLP | 0.585669 |
| A4 cosine | 0.575834 |
| A5a geometry cross-fit | 0.588144 |
| A5b geometry train-fit | 0.596235 |
| A6 fusion | 0.585503 |
| N1 shuffled null | 0.505451 |
| N2 prior | 0.500000 |

These are experiment-specific WPRD values, not generalization or semantic
claims. A5b is the geometry train-fit arm and is not the cross-fit estimator
used in A5a; the ladder is not a simple ranking of matched treatment arms.
Their source is the decoder-ladder rerun result artifact listed in the
[evidence ledger](docs/research/PAPER_C_EVIDENCE_LEDGER.md).

The later residual comparisons were:

| Comparison | Calibrated held-out log-loss change | Image-clustered 95% CI |
| --- | ---: | ---: |
| O → O + rel_feat | −0.3660331323 | [−0.3966375880, −0.3361649032] |
| G+O → G+O + rel_feat | −0.0051288329 | [−0.0263599217, +0.0158628067] |
| N4-FIT pilot → N4-FIT + rel_feat | −0.0024462230 | [−0.0026268349, −0.0022835433] exploratory |

The final saved pair-prior row audit identifies 12,330 of 14,991 validation
rows as supported by a training pair (82.25%). An earlier summary reported
12,402; that count is superseded by the identity-checked saved-row artifact
and must not be used as the current figure. See
[`posthoc_metrics.json` in the final evidence audit](runs/paper_c_final_evidence_audit_20261003T185925Z/posthoc_metrics.json)
and its documented consistency note in `evidence_index.json`.

The balanced-15k M2 run reported log-loss 1.6304535145, accuracy
0.5614035088, and macro recall 0.1412721492 on 1,182 validation images / 14,991
rows. Historical prefix M2 metrics used 129,826 validation rows, so those
published aggregates are not directly comparable to the balanced result.
This is a sampling-regime diagnostic, not a clean monotonic scaling curve.

For N4-FIT, the bounded pilot reported CAL-CHECK log-loss 1.5984459578 for
N4-FIT and 1.5959997348 with the frozen rel_feat offset. The pilot did not
justify spending the estimated fullscale resources because its observed
improvement was smaller than the pre-existing G+O residual reference. This is
not the final full-population nuisance-gate result: fullscale N4-FIT was not
run, so residual beyond full-population G+O+U+S remains unestablished.

## What is established—and what is not

| Established within the tested protocols | Not established |
| --- | --- |
| `rel_feat` is decodable by specified readouts. | Relational semantics or semantic understanding. |
| `rel_feat` has predictive utility relative to the tested pair-prior baseline. | Causal information or information unavailable from the image. |
| Much of the apparent advantage contracts after nuisance structure is added; geometry accounts for a substantial portion under the tested comparison. | Open-vocabulary relation understanding. |
| A bounded N4-FIT pilot showed a small CAL-CHECK residual, insufficient for the registered fullscale resource decision. | Residual beyond a full-population G+O+U+S model. |
| Seed-1234 FULL/A/B smoke completed as an engineering/design check. | Interactional-family superiority; the frozen current validation has zero eligible interactional rows. |

Interpret the smoke interventions only at their registered narrow estimands:
geometry-at-fusion-gate conditioning (A) and correctly aligned pair-state
input to edge updates (B). The smoke was not confirmatory effect estimation.

## Experimental chronology

The detailed chronology is in
[PAPER_C_EXPERIMENTAL_TIMELINE.md](docs/research/PAPER_C_EXPERIMENTAL_TIMELINE.md).
At a high level it covers A1–A6 and null baselines, geometry controls,
readout-v2 forensic work, cross-seed exploration, M2 sampling diagnostics,
FULL/A/B intervention smoke, the fullscale residual audit, the N4 vocabulary
forensic, and the bounded N4-FIT pilot followed by the strategic stop.

## Reproducibility

The canonical C1 source snapshot is preserved at
`runs/paper_c_canonical_train_relfeat_20260930T062248Z/code`, commit
`ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`. The canonical checkpoint is
`checkpoints/C1_seed1234.pt`, SHA256
`79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`. The
canonical train JSONL is `datasets_vg150_clean/train.jsonl`, SHA256
`306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4`.
The canonical 83,249-image, 1,046,427-row, 17-shard `rel_feat` manifest and
shard hashes are preserved under
`runs/canonical_train_representation_full_20261001T020630Z/representation/`.
See [reproducibility](docs/research/PAPER_C_REPRODUCIBILITY.md) for the
artifact boundaries and environment notes.

The local environment inspected for this closeout was Python 3.10.12 and
PyTorch 2.9.1+cu129. The checked-in requirements declare lower bounds that do
not exactly match this environment; test status records that limitation.

## Historical artifacts

`runs/` contains completed, failed, blocked, and exploratory runs, plus source
snapshots and provenance. Treat these as archival evidence, not as a single
active workflow. Large checkpoints, datasets, tensor caches, and many run
payloads are excluded from Git by `.gitignore`; a clone alone is not a complete
artifact archive. The [historical artifact index](docs/archive/HISTORICAL_ARTIFACTS.md)
maps major preserved runs without moving or rewriting them.
The [closeout codebase inventory](docs/research/CLOSEOUT_CODEBASE_INVENTORY.md)
classifies active, historical, generated, and retained legacy areas.

## Active workflow

There is no active Paper C experiment launcher. The repository still contains
the inherited `openvocab_rel/` model, dataset, training, evaluation, and data
preparation utilities, but their presence is not authorization to continue
this research direction. Use the synthetic regression suite for maintenance:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest
```

See [development](docs/engineering/DEVELOPMENT.md) for setup and supported
maintenance checks. Commands copied inside historical run snapshots are not
current workflow instructions.

## Future research

This repository has no authorized next Paper C experiment. Any future work
should begin with a new research hypothesis and a newly registered protocol,
not an unbounded continuation of the diagnostic ladder documented here.
