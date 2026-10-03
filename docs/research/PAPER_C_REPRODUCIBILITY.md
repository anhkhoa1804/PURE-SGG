# Paper C reproducibility and provenance

## Frozen references

| Artifact | Location | Verified identifier |
| --- | --- | --- |
| Canonical source snapshot | `runs/paper_c_canonical_train_relfeat_20260930T062248Z/code` | Git HEAD `ec4cca6ff01ab46929b7b428220fe0df2ba00f2a`; tracked tree clean at snapshot audit, with only untracked later helper files noted in the snapshot directory. |
| Canonical C1 checkpoint | `checkpoints/C1_seed1234.pt` | SHA256 `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` |
| Canonical train JSONL | `datasets_vg150_clean/train.jsonl` | SHA256 `306fc0db96ca716bbc78c2cf32d49fdc695ab0074004a99c7399c850a1c605a4` |
| Canonical representation | `runs/canonical_train_representation_full_20261001T020630Z/representation/` | 83,249 images; 1,046,427 rows; 17 shards; 768-D float16; manifest records shard hashes. Manifest SHA256 `f70b819eb22a40b431f32280fd6b04d74caeb821fc37764fdb288d6abdb9c8d4`. |

The current repository HEAD at closeout start was `ec4cca6` on
`research/pure-complete-readout-v2`, tracking the same-named origin branch.
The artifact payloads are local/external and are not all tracked in Git.

## Environment and tests

Observed environment: Python 3.10.12; PyTorch 2.9.1+cu129. `requirements.txt`
declares `torch>=2.10.0`; this installed environment is below that declared
minimum, so passing local tests are not a fresh-install compatibility
certification. The smoke suite uses synthetic fixtures and does not require
GPU, full data, or real model downloads. Test commands and outcomes are in
`docs/engineering/TEST_STATUS.md`.

## Reproduction boundaries

- Preserve the canonical source snapshot, C1 checkpoint, representation
  manifest, frozen split/index artifacts, and historical result directories.
- Verify artifact hashes before any future reuse; do not join by tensor row
  ordinal when identity-bearing keys are available.
- Validation/E2 boundaries are experiment-specific; follow each run’s frozen
  manifest rather than inferring populations from filenames.
- The bounded N4-FIT pilot is not fullscale and does not resolve the final
  G+O+U+S residual question.
- See `docs/archive/HISTORICAL_ARTIFACTS.md` for artifact locations/status and
  `.gitignore` for local large payload policy.
