# `p73` — the frozen Fourier encoder destroys its own input completely, and the units bug is the only reason any geometry survives

Pre-registered in `docs/GEOM_FOURIER_BANDWIDTH_AMENDMENT.md`, committed
**before** the run. CPU only, no GPU. Run:
`runs/p73_fourier_null_and_invertibility`. Tool:
`tools/fourier_null_and_invertibility.py`. `runs/p72_VOID_null_gate_failed/`
and `tools/geom_fourier_bandwidth.py` were not modified.

**Part B gates: 4/4 PASS.** Part A returns `INCONCLUSIVE` by its own rule.

---

## Part A — null calibration: `p72` stays VOID

`N_shuffled` refitted on `p72`'s exact design at eight shuffle seeds:

| seed | 7 (p72's) | 11 | 13 | 17 | 19 | 23 | 29 | 31 |
|---|---|---|---|---|---|---|---|---|
| null WPRD | 0.4864 | 0.4982 | 0.5007 | 0.5029 | 0.4992 | 0.4962 | 0.5001 | 0.4986 |

```
mean 0.4978   sd 0.0050   range [0.4864, 0.5029]
p72's own draw within 2 sd of the mean:  NO  (2.28 sd)
PART A VERDICT -> INCONCLUSIVE
```

The estimator is **not** biased — the null mean sits at 0.4978, essentially
chance, and the registered `NULL-BIASED` branch is excluded. The `[0.49, 0.51]`
window is almost exactly ±2 sd, so it is a defensible gate on the *mean* but
too tight for a *single draw*: roughly one draw in twenty will miss it, and
seed 7 is one of those.

But the amendment's `G4-MISSPECIFIED` branch required the `p72` draw to lie
within 2 sd, and it lies at 2.28 sd. **The rule was fixed in advance and is
honoured: `p72` is NOT un-voided, and no `p72` number is cited as evidence
anywhere in this programme.** Its `+0.0892` is discarded, not rescued.

(The seed-7 null reads 0.4864 here against `p72`'s 0.4873 because `p72` fitted
its null last, sharing one `Generator` across eight prior arms, so the
minibatch permutation stream differed. Same design, different nuisance draw —
which is itself a small illustration of how much a single null realisation can
move.)

Part A therefore decides nothing. Everything below rests on Part B, which was
registered in advance as the stronger evidence precisely because it does not
route through WPRD, labels, or the shuffled null.

---

## Part B — invertibility: the encoder is exactly as informative as noise

Label-free. Cross-fitted MLP (same folds, salt, hidden 256, 20 epochs, lr 2e-3,
l2 1e-4 as `p69`/`p72`), 8-output regression head, MSE, out-of-fold R² per
channel against the 8 standardised raw geometry channels.

| arm | dims | **mean R²** | dx | dy | rw | rh | ar1 | ar2 | a1 | a2 |
|---|---|---|---|---|---|---|---|---|---|---|
| `I_raw` (identity control) | 9 | **0.9997** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **`I_s1`** (checkpoint's own `geom_B`) | 513 | **−0.3314** | −0.12 | −0.20 | −0.55 | −0.55 | −0.22 | −0.22 | −0.39 | −0.40 |
| `I_s0.25` | 513 | −0.3354 | −0.12 | −0.20 | −0.56 | −0.56 | −0.23 | −0.22 | −0.40 | −0.40 |
| `I_s0.1` | 513 | −0.2631 | −0.10 | −0.16 | −0.47 | −0.47 | −0.18 | −0.18 | −0.21 | −0.33 |
| `I_s0.05` | 513 | **+0.1064** | −0.09 | −0.14 | +0.14 | +0.19 | +0.12 | +0.07 | **+0.35** | +0.21 |
| `I_s1_prod` (what production encodes) | 513 | +0.0920 | **+0.39** | **+0.33** | −0.01 | −0.03 | −0.02 | +0.00 | −0.01 | +0.08 |
| **`I_shuffled_rows`** (null control) | 513 | **−0.3197** | −0.12 | −0.19 | −0.53 | −0.53 | −0.22 | −0.22 | −0.38 | −0.39 |

```
PART B VERDICT -> DESTRUCTIVE-CONFIRMED   (mean R2 at the checkpoint's own bandwidth = -0.3314)
gates: H1 identity 0.9997 PASS   H2 null -0.3197 PASS   H3 PASS   H4 PASS
```

### The decisive comparison

**`I_s1` (−0.3314) and `I_shuffled_rows` (−0.3197) are the same number, channel
by channel.** The null arm is `I_s1` with its rows randomly permuted — inputs
with *provably zero* relationship to the targets. The checkpoint's own Fourier
encoding of the geometry is, to this probe's precision, **exactly as
informative about the geometry as a random permutation of itself**: a gap of
0.0117 in mean R², against a per-channel spread of 0.4.

(The shared −0.32 floor is the out-of-fold overfitting cost of 513 useless
dimensions, which is why both sit below zero. `I_raw` at 9 dimensions reaches
0.9997, so the probe solves the problem trivially when the information is
there. The absolute value of the floor is an artifact; the *equality* of `I_s1`
with its own null is the finding.)

This is `p69`'s "random hash" observation upgraded from a kernel profile to a
direct, gated, label-free measurement of information loss.

## The result that changes the plan: the bug is load-bearing

Compare the two rows that describe real configurations:

- **`I_s1_prod`** — what production encodes today. Recovers **dx at R² 0.39 and
  dy at 0.33**, and nothing else (the other six are `p68`-constant, so there is
  nothing to recover).
- **`I_s1`** — what `C1a` alone would encode: the same frozen `geom_B`, fed the
  units-corrected 8-vector. Recovers **dx at −0.12 and dy at −0.20**, i.e.
  nothing at all, on the null floor.

The mechanism is arithmetic. Production's `/336` normalisation compresses `dx`
into std 0.177; the units-fixed contract expands it to std 2.09 against a
`clamp(-10, 10)` domain. At `2π·|B|` ≈ 41 rad per unit, a std-0.177 input wraps
the phase a few times — locally injective, and the probe recovers it. A std-2.09
input wraps it ~85 times — a hash.

> **The `p68` units bug is the only reason any geometry information reaches the
> model at all.** It destroys six channels, and it simultaneously compresses the
> surviving two into the narrow range where the frozen encoder is still
> invertible. Correcting the units *alone* would restore six channels' worth of
> information at the input and then destroy all eight at the encoder.

**`C1a` alone is therefore predicted to be null or harmful**, for a reason that
has nothing to do with the units hypothesis — exactly the misattribution the
`p72` registration was written to prevent, now established on evidence that
survived its gates.

## Consequence for the `C` factorial (this reverses the planned order)

- **`C1a` must not be run first.** A null would be uninterpretable.
- **`C1b` is mandatory**, and its useful range is far lower than registered.
  `geom_fourier_scale` at 0.25 (−0.335) and 0.1 (−0.263) are still on the null
  floor. Only 0.05 (+0.106) shows any recovery, and even there `dx`/`dy` stay
  negative while the *size* channels come back first (`a1` +0.35, `rh` +0.19).
  The registered scale grid `{1.0, 0.25, 0.1, 0.05}` is mostly wasted; the
  informative grid starts at **0.05 and goes down**.
- **The correct first GPU experiment is `C1` (both fixes together)**, or a
  bounded CPU bandwidth search first. `C1b` alone is also defensible — it would
  test whether un-hashing the two channels PURE *does* receive is worth
  anything — but it cannot deliver the +0.034 that `p69`/`p71` measured,
  because those channels are still zeroed by the units bug.

## What this does NOT establish

- **Decodability is not use.** This shows the information survives (or does
  not survive) the map, recoverable by an MLP of `geom_mlp`'s class. It does
  not show PURE's training would recover it. `C1` remains the test.
- It does **not** establish the best value of `geom_fourier_scale`. The 0.05
  point is a single grid point that happens to be above the floor; a proper
  bandwidth curve has not been run.
- It does **not** rescue `p72`. Part A failed its own rule; `p72`'s `+0.0892`
  remains void and uncited. The DESTRUCTIVE conclusion here is established
  independently, by a different measurement, against its own gated null.
- It does **not** contradict `p69`/`p71`. Those measure the information content
  of the raw channels, upstream of the encoder. This measures what the encoder
  does to them. Both hold simultaneously, and together they say the geometry
  pathway is broken in **two independent places**.
