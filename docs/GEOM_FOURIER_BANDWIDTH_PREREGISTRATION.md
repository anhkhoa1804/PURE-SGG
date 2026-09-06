# `p72` preregistration — does PURE's frozen Fourier encoder destroy the geometry information `p69` found?

Committed **before** `runs/p72_geom_fourier_bandwidth` is run. CPU only, no GPU.
Reads the `p36` cache and the historical checkpoint's `geom_B` (read-only).

## Why this must run before `C1a`

`p69` measured that the six geometry channels PURE cannot see are worth
**+0.0341 WPRD**. `C1a` restores them by fixing the units contract. But they
would then be delivered through

```
fourier = [sin, cos]( 2*pi * x @ geom_B )      geom_B frozen, requires_grad=False
```

with `geom_B` at **std 9.91, |max| 35.9** (verified from the checkpoint this
session). The phase rate is `2*pi*|B|`: **median 40.6 rad per unit of `dx`**, a
full cycle every **0.155** of `dx` — and `dx` spans [−1, 1] in the current
contract.

Worse, the units fix *increases* the input's dynamic range. `a1 = log(w*h)` in
pixel units sits near **+9**, against near **0** today. At `|B| ≈ 10` that is a
phase of ~580 radians. **If the Fourier map is information-destroying at this
bandwidth, `C1a` alone cannot work, and running it first would waste GPU and
produce a null that would be misattributed to the units hypothesis rather than
to the encoder.** This experiment decides the order of the factorial.

## Design

Same estimator, folds, rows and reporting path as `p60`/`p69` (AdamW MLP,
hidden 256, 20 epochs, lr 2e-3, l2 1e-4, softmax CE, 5-fold CV on validation,
salt 0, `cap = 64`). Every arm is fitted on identical rows.

Two input contracts, both built with the **production function**
`geom_feats_torch`:

- **fixed** — boxes in native pixel units, the domain the function was written
  for; `clamp_min(1.0)` does not bind. This is `C1a`'s contract.
- **prod** — boxes divided by their own per-image extent so every width/height
  is ≤ 1; `clamp_min(1.0)` binds on all four, reproducing `p68`'s
  6-of-8-constant degeneracy.

| arm | features |
|---|---|
| `R_raw_fixed` | the 8 raw channels, units fixed — information ceiling of the 8-channel contract |
| `R_raw_prod` | the 8 raw channels as production delivers them (6 constant) |
| `F_s1_fixed` | `fourier(R_raw_fixed, geom_B)` — **the checkpoint's own bandwidth** |
| `F_s0.25_fixed`, `F_s0.1_fixed`, `F_s0.05_fixed` | same, with `geom_B` scaled down |
| `F_s1_prod` | `fourier(R_raw_prod, geom_B)` — what the geometry branch actually encodes today |
| `N_shuffled` | null |
| `P_prior` | prior control (must read exactly 0.5000) |

Plus a descriptive kernel profile per bandwidth: mean cosine similarity of the
Fourier features against input distance, bucketed.

## Primary quantity and decision rule

```
delta_fourier = WPRD(R_raw_fixed) - WPRD(F_s1_fixed)
```

how much of the units-fixed geometry information the checkpoint's own frozen
Fourier map costs.

- **DESTRUCTIVE** — `delta_fourier >= +0.02`. The encoder throws away a large
  fraction of the very information `C1a` would restore. **`C1b` becomes
  mandatory and the factorial must run `C1` (both fixes) or `C1b` first**; a
  units-only run would be predicted-null for a reason that has nothing to do
  with the units hypothesis.
- **BENIGN** — `delta_fourier < +0.005`. Bandwidth is not the bottleneck.
  **`C1a` alone is the correct first intervention** and `C1b` is deprioritised.
- **PARTIAL** — in between. Report both; run `C1a` first but register the
  expected attenuation before, not after, seeing its result.

Secondary, no decision: `delta_bandwidth = best low-bandwidth arm − F_s1_fixed`,
i.e. whether a deliberately smoother map recovers what the checkpoint's loses.

## Validity gates

- **G1** the `prod` contract reproduces `p68`'s degeneracy: exactly **6 of 8**
  channels constant. (If this fails, the reconstruction is not production's.)
- **G2** the `fixed` contract has **0** constant channels.
- **G3** `P_prior` reads exactly 0.5000.
- **G4** `N_shuffled` within [0.49, 0.51].
- **G5** every arm scores every GT row, all finite.

## Stated in advance

1. **The estimator sees 512 Fourier dims against 8 raw dims.** More capacity and
   more overfitting risk favour the Fourier arms, which biases *against* the
   DESTRUCTIVE verdict. The conservative direction is the one that would license
   the more expensive conclusion, which is the correct way round.
2. **A learnable readout is not the model's readout.** This measures whether the
   information *survives* the map, recoverable by an MLP of the same class as
   `geom_mlp`. It does not measure whether PURE's actual training would recover
   it. That remains `C1`'s job.
3. **The `prod` arm uses per-image extent normalisation, not PURE's fixed /336.**
   Same registered caveat as `p69`: it is mildly *more* informative than what
   PURE receives, so `R_raw_prod` is an upper bound on production.
4. `a1`/`a2` under the `fixed` contract are computed in raw Visual Genome pixel
   units rather than PURE's 336-crop units. The two differ by a per-image
   additive constant `2*log(s)`; the other six channels are exactly
   scale-invariant. This affects the absolute phase of `a1`/`a2` but not the
   information content or the comparison between bandwidths, and is recorded
   here rather than discovered later.
