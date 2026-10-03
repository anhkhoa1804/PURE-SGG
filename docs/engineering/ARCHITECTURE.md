# Repository architecture

## Active maintenance surface

- `openvocab_rel/`: inherited Python package. `models/relational_model.py`
  defines the model; `datasets/vg150_loader.py` loads VG150 JSONL; `train.py`
  and `evals.py` implement training/evaluation; `geometry.py`, `losses.py`,
  `predicate_metadata.py`, and `config.py` hold supporting logic.
- `tools/`: data preparation, validation, reproducibility, and analysis
  utilities. Many scripts are experiment-specific; inspect their dated
  provenance before use.
- `scripts/`: shell entrypoints grouped partly by train/eval/notebook role.
  They include historical launch recipes and are not an active Paper C
  experiment workflow.
- `configs/`: metadata and presets. Verify whether a preset is consumed by
  runtime code; some files are documentary only.
- `tests/`: offline synthetic regression suite.

## Data flow

The inherited training/evaluation path loads VG150 relationship rows and
images, constructs ordered object-pair features with visual/context and
geometry inputs, predicts predicate scores, and records evaluation artifacts.
Paper C’s later `rel_feat` and nuisance/readout results are frozen run outputs,
not a separate current production package. Their identity, configuration, and
hash lineage live under the corresponding `runs/` directories and manifests.

## Historical and generated areas

`runs/` is an archive of immutable/provenance-critical results, copied source
snapshots, blocked/failed attempts, and local generated payloads. `checkpoints/`,
`datasets_vg150_clean/`, caches, and feature shards are external/ignored local
artifacts. `docs/archive/HISTORICAL_ARTIFACTS.md` is the index. Do not infer
that a script is current merely because it remains in the tree.

There is no active Paper C CLI or authorized experiment launcher.
