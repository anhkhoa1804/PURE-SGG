# Repository closeout audit

Audit date: 2026-10-03 (UTC). This is a read-only inventory and classification
of the starting repository, followed by an additive documentation closeout.

## Structure and classification

| Area | Classification | Finding / risk | Action |
| --- | --- | --- | --- |
| `openvocab_rel/` | ACTIVE (maintenance) | Main Python package: model, dataset, training/evaluation, geometry, losses, metadata, and supporting utilities. It is inherited SGG code, not an active Paper C experiment. | Retain; no algorithmic changes in this closeout. |
| `scripts/` | ACTIVE + LEGACY | Dataset/environment checks and several launch/evaluation scripts coexist with old experiment entrypoints. | Keep in place; do not advertise experiment launchers as current Paper C workflow. |
| `tools/` | ACTIVE + HISTORICAL | Data preparation, validation, reproducibility, and analysis utilities coexist with experiment-specific helpers. | Preserve; classify usage in engineering docs rather than deleting without dependency proof. |
| `configs/` | HISTORICAL / SUPPORT | Dataset metadata and experiment presets; some preset documentation is not loaded by runtime code. | Preserve and document this distinction. |
| `tests/` | ACTIVE | Synthetic/offline regression suite, intentionally does not require real VG150 images or model weights. | Retain; default pytest discovery now targets only this tree. |
| `docs/` | MIXED | Large body of dated preregistrations, results, audits, and older living status documents. Some old state maps conflict with the final 2026-10-03 status. | Add a current authoritative research set; mark the two old living maps superseded without rewriting their dated contents. |
| `runs/` | HISTORICAL / PROVENANCE | Tracked small records plus ignored/untracked run directories, copied source snapshots, checkpoints/logits, and negative/blocked records. | Do not move, delete, or rewrite. Index major Paper C artifacts. |
| `checkpoints/`, `datasets_vg150_clean/`, `downloads/`, caches | GENERATED / EXTERNAL | Large local payloads ignored by Git; a clone does not contain all inputs/outputs. | Keep external; report hashes and archive expectations. |
| root `runs_p70_smoke*.log`, caches, `.venv`, bytecode | GENERATED | Ignored transient artifacts. | Leave untouched; existing ignore rules are appropriately scoped. |

## Technical debt and risk

1. **Mixed-era documentation — high communication risk.** `README.md`,
   `docs/PROJECT_STATUS.md`, and `docs/RESEARCH_STATE.md` previously described
   an earlier PURE/CORE or pre-closeout research state. The README is replaced;
   the dated status maps are explicitly superseded while their historical text
   remains intact.
2. **Experiment artifacts live beside code snapshots — medium test/tooling
   risk.** Run directories may contain copied repositories and duplicate test
   module names. Bare pytest discovery previously traversed these snapshots
   and errored. `pytest.ini` now restricts normal discovery to top-level
   `tests/`; the snapshots remain untouched.
3. **Artifact availability — high reproducibility risk.** Important checkpoints,
   datasets, and tensors are ignored or external. The Git tree is not a
   self-contained reproduction bundle. Artifact paths and hashes are indexed;
   no large files are newly committed.
4. **Environment bounds — medium reproducibility risk.** Current venv is Python
   3.10.12 / PyTorch 2.9.1+cu129; checked-in requirements specify `torch>=2.10`
   and newer lower bounds for other dependencies. The suite is tested in the
   observed environment, not certified against a fresh installation.
5. **Entrypoint breadth — medium operational risk.** Many scripts are dated or
   tied to specific machines, data mounts, and historical protocols. No
   unified active Paper C CLI exists; no such CLI is invented in this closeout.
6. **Large-file policy — medium operational debt.** Multi-GB checkpoints and
   feature payloads remain outside Git under established ignore rules. There
   is no evidence supporting history rewriting or bulk externalization.

## Scope and provenance safety

No source module, checkpoint, dataset, historical run artifact, canonical
representation shard, or historical result was cleaned up or regenerated.
Changes are limited to current documentation, pytest discovery configuration,
and this closeout report. The canonical C1 checkpoint and train JSONL hashes
were checked against their frozen values; the canonical source snapshot HEAD
is recorded in the reproducibility document.
