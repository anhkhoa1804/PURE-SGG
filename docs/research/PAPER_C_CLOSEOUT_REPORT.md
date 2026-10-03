# Paper C repository closeout report

Date: 2026-10-03 UTC.

## Starting state

Branch `research/pure-complete-readout-v2`, HEAD
`ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`, tracking the same branch at
`origin`. The tracked tree was clean at start; many run directories, local
logs, checkpoints, datasets, and later forensic tools were untracked or
ignored. They were treated as user research artifacts and not staged.

## Changes in this closeout

- Replaced the stale root README with the Paper C closed-state summary and
  links to current research/engineering documentation.
- Added the closeout audit, evidence ledger, claim audit, timeline,
  reproducibility, closeout decision, and archive index.
- Added architecture, development, test-status, and technical-debt docs.
- Restricted default pytest discovery to top-level `tests/` so copied
  historical `runs/**/code/tests` do not collide with active tests.
- Code removed: none. Legacy and experiment-specific scripts were retained
  because they have historical reproduction/provenance value or no safe
  deletion proof was established.
- No model/source algorithm, canonical artifact, historical run, or large
  generated file was removed or altered.

Generated/untracked items such as `runs/` outputs, root smoke logs, `.venv`,
checkpoints, local datasets, caches, bytecode, and prior forensic tools/tests
were left in place and were not staged. Existing root-anchored `.gitignore`
rules exclude large payload directories and tensor/checkpoint/cache/log
extensions while intentionally leaving small provenance text visible for
explicit review. No broad ignore rule was added.

## Checks and limitations

Both the active-tree and default test commands passed 611 tests with 25
PyTorch warnings (346.96 s and 315.62 s, respectively). The workspace
contained three pre-existing untracked L4 audit tests, which were run locally
but not staged. The initial bare pytest attempt failed during collection
because it walked copied historical source trees; the new `pytest.ini` scopes
default discovery. Compileall, core imports, and dataset subset, dataset
validation, and training `--help` checks passed. No test was weakened.
Targeted supersession notices mark older live status maps, queues, and
launch/design documents while preserving their dated bodies.

## Provenance

Canonical checkpoint SHA256 verified as
`79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`.
Canonical train JSONL SHA256 verified as
`306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4`.
Canonical source snapshot HEAD is
`ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`; canonical representation
manifest and shard hashes are preserved in its run directory. No historical
artifact was overwritten.

## Git delivery

Closeout content commit: `213d116fbcfdf6933f884fd7577f8397ed1c98e3`.
Remote/branch: `origin` /
`research/pure-complete-readout-v2`. Push status: initial closeout commit was
successfully pushed. This delivery record is maintained in a follow-up,
documentation-only commit; the final branch state is verified from Git after
that push.
