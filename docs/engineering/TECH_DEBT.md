# Technical debt register

| Item | Risk | Handling / follow-up |
| --- | --- | --- |
| Dated experiment scripts and machine-specific paths | Stale commands can be mistaken for active recipes. | Retained for provenance; consult their run/config records. No active Paper C experiment is authorized. |
| Large local artifacts excluded from Git | Clone is not self-contained. | Preserve external payloads and manifests; establish an explicit archival distribution before claiming clean-room reproducibility. |
| Requirements/environment mismatch | Fresh installation may resolve a different stack; current torch is below declared minimum. | Record exact env for each result; reconcile dependency floor in a separately tested change. |
| Mixed historical docs | Old current-state prose conflicts with final state. | New `docs/research/PAPER_C_*` pages are authoritative; older living maps carry a supersession note. Dated result records remain unchanged. |
| Duplicated tests/source under run snapshots | Default test discovery could collide with active tests. | `pytest.ini` scopes normal collection to root `tests/`; snapshots are untouched. |
| Script conventions and output safety vary by era | Some old scripts may overwrite or assume mounted paths. | Do not promote historical scripts; any reactivation requires a new wrapper/provenance review. |
| No unified packaging/CI metadata | Developer checks are local/manual; import and lint guarantees vary. | Keep current lightweight test command documented; add CI only as a separately scoped engineering task. |
| Large checkpoint versioning | Multi-GB files are outside Git and no history migration is justified. | Continue SHA256-based external artifact manifests; do not rewrite Git history. |

## Intentionally retained

Do not “clean” the canonical source snapshot, canonical checkpoint, canonical
representation, historical runs (including failures and negative results),
dataset manifests/checksums, or dated preregistrations/results. Their
provenance value outweighs aesthetic tree simplification.
