# Development and maintenance

## Environment

Use the repository-local `.venv` when available. The tested closeout
environment was Python 3.10.12 and PyTorch 2.9.1+cu129; `requirements.txt`
currently declares `torch>=2.10.0` and should be reconciled in a future
environment-specific maintenance change. Do not install or download model
weights merely to run the offline suite.

## Supported maintenance checks

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m compileall -q openvocab_rel tools scripts
```

For a CLI, use `--help` only after checking that the entrypoint has no import-
time model/data side effects. No research/training command is endorsed here.

## Artifact conventions

- Research runs are additive and timestamped; never overwrite a historical
  directory or canonical artifact.
- Persist identity manifests and SHA256 hashes for research-critical inputs
  and outputs.
- Use explicit split manifests; do not join by ordinal when IDs are available.
- Large datasets/checkpoints/tensors are local or external and ignored by Git.
  Check `.gitignore` before staging; do not add `runs/` or checkpoints with a
  broad `git add`.
- Historical commands and paths under `runs/**/code` are provenance, not
  supported commands.

## Repository changes

Prefer narrow diffs and regression tests. Avoid broad refactors of model or
evaluation code without a registered need and explicit compatibility tests.
