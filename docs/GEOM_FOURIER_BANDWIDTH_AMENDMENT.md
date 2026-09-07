# `p73` — amendment to `p72`: is the null gate mis-specified, and is the Fourier map really information-destroying?

Written and committed **before** `runs/p73_fourier_null_and_invertibility` is
run. CPU only, no GPU. Does not modify `tools/geom_fourier_bandwidth.py`, does
not touch `runs/p72_VOID_null_gate_failed/`, does not touch the checkpoint.

## Why this exists

`p72` ran to completion and returned `delta_fourier = +0.0892` → **DESTRUCTIVE**,
which by its own registration makes `C1b` mandatory and forbids running `C1a`
alone first. But `p72` **failed gate G4**: its shuffled null read **0.4873**
against a registered window of `[0.49, 0.51]`. By the registration, *no number
in that run is reportable*, including the verdict that would otherwise decide
how the entire `C` factorial is ordered. `p72` is therefore preserved as
`runs/p72_VOID_null_gate_failed/` and is **not** cited as evidence anywhere.

The `p72` null is a **single realisation** — one label shuffle at seed 7, fitted
on the 9-d `R_raw_fixed` design. `p69` (0.4992) and `p71` (0.5063) drew their
nulls from the same seed but on the **768-d `relfeat`** design, so they are not
evidence about *this* design's null variance. Nothing in the programme has
measured how much a `p72`-shaped null moves seed to seed. The `[0.49, 0.51]`
window was carried over from `p69`/`p71` by analogy, not calibrated.

Two possibilities, and they have opposite consequences:

1. the window is **too tight** for a 9-d design — 0.4873 is an ordinary draw,
   `G4` was mis-specified, and `p72`'s primary stands; or
2. 0.4873 is a genuine **downward bias** in this estimator — in which case
   every `p72` arm may be shifted and the reading needs care.

Rerolling the seed until the gate passes would be gate-shopping and is
explicitly **not** what this does. The null's *distribution* is measured, and
the decision rule below is fixed in advance.

## Part A — null calibration (decides `p72`'s fate)

Fit `N_shuffled` on `R_raw_fixed` exactly as `p72` does — same estimator, folds,
rows, cap, salt — at shuffle seeds `{7, 11, 13, 17, 19, 23, 29, 31}` (seed 7 is
`p72`'s own draw and is included so `p72` is reproduced, not replaced).

Report mean, SD and range of the null WPRD.

**Decision rule, fixed in advance:**

- **G4-MISSPECIFIED** — the 8-seed null mean lies in `[0.49, 0.51]` **and**
  `p72`'s 0.4873 lies within 2 SD of that mean. The window was too tight for
  one draw of a low-dimensional design. `p72`'s `G4` is amended to *"the null
  **mean** lies in [0.49, 0.51]"*, `p72` is **un-voided**, and its
  `DESTRUCTIVE` primary becomes reportable.
- **NULL-BIASED** — the 8-seed null mean lies **below 0.49**. The estimator has
  a real downward bias on this design. `p72` stays void, every arm is re-read
  relative to the measured null rather than to 0.5, and the primary is
  recomputed as a null-referenced contrast before any conclusion is drawn.
- **NULL-UNSTABLE** — null SD `> 0.01`. The null is too noisy for a ±0.01 gate
  of any kind on this design; `p72` stays void and the bandwidth question is
  answered by Part B alone.

## Part B — invertibility probe (independent of WPRD and of the null)

`p72`'s primary asks whether *predicate discrimination* survives the Fourier
map. That routes the question through labels, an MLP, and WPRD's cell
weighting — the same machinery whose null just failed. Part B asks the
mechanism directly, with **no labels involved at all**:

> Can the 8 raw geometry channels be **decoded back** out of their own Fourier
> encoding?

If the map is information-preserving, an MLP of the same class as `geom_mlp`
must be able to recover its own input. If it cannot, the map is destroying the
information before any predicate head ever sees it — and that conclusion does
not depend on WPRD, on the label shuffle, or on `G4`.

Design: cross-fitted (same 5 folds, same salt 0) ridge-regularised MLP,
hidden 256, 20 epochs, lr 2e-3, l2 1e-4 — the `p69`/`p72` estimator with an
8-output regression head and MSE loss instead of softmax CE. Targets are the 8
standardised raw channels. Metric is per-channel out-of-fold **R²**.

| arm | input |
|---|---|
| `I_raw` | `R_raw_fixed` itself — **identity control, must read R² ≈ 1.0** |
| `I_s1` | `fourier(R_raw_fixed, geom_B)` — the checkpoint's own bandwidth |
| `I_s0.25`, `I_s0.1`, `I_s0.05` | same, `geom_B` scaled down |
| `I_s1_prod` | `fourier(R_raw_prod, geom_B)` — what production actually encodes |
| `I_shuffled_rows` | `I_s1` with its rows permuted — **null control, R² ≈ 0** |

**Decision rule, fixed in advance,** on `mean_R2(I_s1)` over the 8 channels:

- **DESTRUCTIVE-CONFIRMED** — `mean_R2(I_s1) < 0.30` while `mean_R2(I_raw) >
  0.95`. The checkpoint's own bandwidth is information-destroying by a
  label-free measurement. `C1b` is mandatory; `C1a` alone is predicted null for
  a reason unrelated to the units hypothesis, and must not be run first.
- **BENIGN-CONFIRMED** — `mean_R2(I_s1) > 0.70`. The map preserves its input;
  bandwidth is not the bottleneck and `C1a` alone is the correct first
  intervention.
- **PARTIAL** — in between. Report the recovery curve against scale and choose
  the factorial order on the strength of `delta_bandwidth`.

Validity gates (all must pass or Part B is void):

- **H1** `mean_R2(I_raw) > 0.95` — the probe can solve the trivial case.
- **H2** `mean_R2(I_shuffled_rows) < 0.05` — the probe does not manufacture R².
- **H3** every arm scores every GT row, all finite.
- **H4** the `prod` contract still reproduces `p68`'s 6-of-8 degeneracy, and the
  `fixed` contract still has 0 constant channels (re-asserted here so Part B
  stands alone).

## Stated in advance

1. **Part B is the stronger evidence and Part A cannot override it.** If Part A
   returns `NULL-BIASED` and Part B returns `DESTRUCTIVE-CONFIRMED`, the
   bandwidth conclusion holds on Part B alone.
2. **Decodability is not use.** Part B measures whether the information
   survives the map, recoverable by an MLP. It does not measure whether PURE's
   training would recover it — that remains `C1`'s job. A `BENIGN` reading
   would license running `C1a` first, not predict that `C1a` works.
3. **`a1`/`a2` are computed in raw VG pixel units**, as in `p72`; the six other
   channels are exactly scale-invariant
   (`tests/test_geometry_contract_flags.py`). This shifts those two channels'
   absolute phase, not their information content.
4. Part B's targets are *standardised* channels, so R² is comparable across
   channels of very different native scale.

---

## Part C — bandwidth curve (added after Part B returned DESTRUCTIVE-CONFIRMED)

Registered before running. Part B measured the registered grid
`{1.0, 0.25, 0.1, 0.05}` and found the first three on the null floor and only
0.05 above it (+0.106), with `dx`/`dy` still negative there. The registered grid
is therefore uninformative about where the encoder becomes usable, and `C1b`'s
scale cannot be chosen from it.

Part C reruns Part B's probe, unchanged in every other respect, on
`geom_fourier_scale ∈ {1.0, 0.05, 0.02, 0.01, 0.005, 0.002}` (1.0 retained so
the `I_s1` anchor and the DESTRUCTIVE verdict are reproduced, not replaced).

No new decision rule and no new gates: the same H1–H4 apply, and the reported
quantity is the recovered mean R² and the per-channel curve. The purpose is to
**select `geom_fourier_scale` by measurement before any GPU is spent**, rather
than to test a hypothesis. `I_raw` is the `fourier-disabled` reference — the
ceiling any bandwidth choice is competing against.

Stated in advance: a scale low enough to be invertible may be *too* low to be
useful — as `geom_B * s -> 0` the Fourier features degenerate toward a constant
plus a linear function of the input, so high R² at very small `s` indicates the
map has become trivial, not good. The selection criterion is therefore the
**knee** of the curve — the largest scale that recovers the input — not the
argmax.
