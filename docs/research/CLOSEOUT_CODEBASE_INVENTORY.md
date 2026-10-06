# Closeout codebase inventory

This is the concise current index for the engineering closeout. The original
read-only inventory and technical-risk review is preserved in
[`CLOSEOUT_AUDIT.md`](CLOSEOUT_AUDIT.md). Historical experiment content stays
in place; this classification describes how to treat it, not a request to
move or delete it.

| Area | Classification | Current use / disposition | Provenance sensitivity |
| --- | --- | --- | --- |
| `openvocab_rel/` | ACTIVE (maintenance) | Core inherited model, data, losses, training and evaluation package. No active Paper C experiment is authorized. | High for reproducing prior code paths; avoid algorithmic edits without a new protocol. |
| `tools/`, `scripts/` | ACTIVE + HISTORICAL | Dataset validation/preparation and small maintenance utilities coexist with experiment-specific tools and launchers. Use only documented maintenance commands. | Medium to high; individual scripts may be the sole reproduction path for a run. |
| `configs/` | SUPPORT / HISTORICAL | Dataset and experiment presets; not all are current runtime configuration. | Medium; preserve with the runs that cite them. |
| `tests/` | ACTIVE | Offline/synthetic regression suite, scoped by root `pytest.ini` to avoid copied run snapshots. | High for current maintenance; historical copied tests remain untouched. |
| `docs/` | MIXED | Current Paper C status and engineering docs are authoritative; dated plans, queues, and reports remain archival and may be marked superseded. | High for scientific interpretation; do not rewrite historical reports. |
| `runs/` | HISTORICAL / PROVENANCE | Completed, exploratory, blocked and failed experiment records; canonical source snapshots and final evidence audit. | Critical. Preserve manifests, hashes, checkpoints, predictions, reports, and negative results. |
| `checkpoints/`, `datasets_vg150_clean/`, local caches | GENERATED / EXTERNAL INPUTS | Large payloads are local or ignored and are not a complete Git-delivered dataset bundle. | Critical inputs; verify hashes and retain under the established external storage policy. |
| `.venv/`, bytecode, pytest cache, logs | GENERATED | Local environment and transient execution products; ignored by Git. | Low, except logs explicitly copied into an experiment run as provenance. |
| Root closeout reports and `pytest.ini` | ACTIVE DOCUMENTATION / TEST CONFIG | Explain the closed research state and constrain default test discovery to active tests. | High for maintenance and interpretation; update only with traceable evidence. |

## Cleanup disposition

No experiment or provenance-bearing source was deleted or moved. No safe
deletion candidate was established by repository-wide reference checks, so
legacy scripts remain in place and are not presented as the supported Paper C
workflow. Large checkpoints, datasets, run payloads, and the additive final
evidence-audit directory are preserved outside the closeout commit unless
already intentionally tracked; the many pre-existing untracked run artifacts
were not staged.

## Current authoritative sources

- Research state and claim boundary: `PAPER_C_STATUS.md` and
  `PAPER_C_CLAIM_AUDIT.md`.
- Experiment evidence and limitations: `PAPER_C_EVIDENCE_LEDGER.md` and
  `runs/paper_c_final_evidence_audit_20261003T185925Z/`.
- Engineering layout and commands: `../engineering/ARCHITECTURE.md` and
  `../engineering/DEVELOPMENT.md`.
- Preserved historical run map: `../archive/HISTORICAL_ARTIFACTS.md`.
