# Paper C — Readout v2, invalid pilot attempts (quarantine record)

This document exists so neither invalid attempt below is ever mistaken for
scientific evidence. Neither attempt's checkpoint, loss curve, or WPRD is
usable. **No artifacts from either attempt exist on disk** — both
checkpoints were deleted by the sessions that diagnosed them; this document
is the only remaining record.

## Attempt 1 — CLIP not serialized (`freeze_clip=true`)

- **Git SHA at launch**: prior to `f747671b62243476f43f2fb47794817bc586f21a`.
- **Failure**: `train.py`'s checkpoint-save block only writes
  `ckpt["clip"] = ...` `if not bool(cfg.freeze_clip)`. The launch script used
  `freeze_clip=true` (intending only to keep CLIP's *gradient* frozen), which
  also silently skipped writing CLIP's weights into the checkpoint at all.
- **Symptom**: resulting checkpoint was ~413 MB (vs. the base checkpoint's
  5.29 GB) and had no `"clip"` key. A fresh-process reload would silently
  fall back to un-finetuned pretrained CLIP — a different model than the one
  that actually produced the training loss.
- **Disposition**: training completed, but the checkpoint was **never
  evaluated**. Its artifacts were deleted before any WPRD number was
  computed. Do not evaluate it, compare its loss curve, or treat it as
  evidence of anything, because none was ever produced.
- **Fix**: commit `f747671` decoupled "CLIP stays frozen" from "CLIP gets
  serialized" via the pre-existing `clip_unfreeze_after_epochs` warmup gate
  (`clip_warmup_frozen` in `train.py`, checked before the `freeze_clip`
  toggle), so `freeze_clip=false` + `clip_unfreeze_after_epochs=999999` +
  `progressive_unfreeze=false` now serializes CLIP into the checkpoint while
  keeping `requires_grad=False` on every CLIP parameter for the pilot's
  entire (1-epoch) duration.

## Attempt 2 — optimizer group stuck at `min_lr`

- **Git SHA at launch**: `f747671b62243476f43f2fb47794817bc586f21a` (the fix
  for Attempt 1, believed correct at the time — see
  `runs/readout_v2_pilot_seed1234_manifest.md`, written immediately before
  this launch).
- **Base checkpoint**: `checkpoints/C1_seed1234.pt`
  (`sha256=79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`,
  re-verified untouched this session).
- **Failure**: `train.py` constructs the AdamW optimizer with two param
  groups (model, CLIP) before the `readout_v2_enabled` setup block adds
  `predicate_prototypes` as a third group via `add_param_group` (no explicit
  `lr`). The per-step LR update inside the training loop's `do_step` block
  only ever wrote `param_groups[0]`/`param_groups[1]` by hardcoded index.
  Group 2 was therefore touched exactly once, by a one-time pre-loop
  `pg["lr"] = min_lr` reset, and never again — regardless of
  `readout_v2_lr`.
- **Symptom**, diagnosed from the real saved optimizer state: after 20 real
  optimizer steps, group 2's `lr` was exactly `1e-07` and
  `predicate_prototypes`' row norms had moved only ~2e-6 from unit norm.
  Gradients *were* flowing (a real, non-trivial `exp_avg` of `0.0248`
  confirmed this) — the parameter simply never received a meaningful update.
  The observed loss decrease (`1.062 → 0.474`) was batch-to-batch CE
  variance on an effectively-frozen `P`, not learning.
- **Disposition**: checkpoint (`checkpoints/readout_v2_pilot_seed1234.pt`)
  discarded, never evaluated. This is the run cited in commit `d4fe687`'s
  message.
- **Fix**: commit `d4fe687` adds
  `if readout_v2_enabled and len(optim.param_groups) > 2:
  optim.param_groups[2]["lr"] = readout_v2_lr` inside the same per-step
  block, plus a regression test
  (`tests/test_readout_v2.py::test_optimizer_group2_gets_readout_v2_lr_not_stuck_at_min_lr`)
  reproducing the exact three-group construction and reset/update sequence.

## Current status (this recovery session, HEAD `d4fe687`)

Both bugs are fixed at HEAD. Full test suite: 438 passed, 1 skipped
(pre-existing, unrelated), 0 failed — including 19/19 `test_readout_v2.py`
tests. No checkpoint files from either invalid attempt remain on disk
(verified via filesystem search). The clean pilot (this session) uses a
distinct run name (`readout_v2_pilot_seed1234_v3`) so its artifacts are
never confused with either attempt above, and resumes fresh
(`--reset_epoch true`) from the untouched `checkpoints/C1_seed1234.pt`.
