# Paper C — `C0` / `C1` preregistration

Committed **before** either arm is trained. This document fixes the hypothesis,
the arms, the endpoints, the thresholds and the decision rule. Nothing below is
tuned after seeing a result.

## Hypothesis

> Restoring the intended geometry units **and** reducing the pathological
> Fourier bandwidth will improve image-conditioned relational discrimination in
> PURE, measured as WPRD.

## Why both fixes together, and not `C1a` or `C1b` alone

Established, gated, and not re-litigated here:

- `p68` — 6 of 8 geometry channels reaching `forward_pairs` are bit-exactly
  constant, caused by a units-contract violation at `relational_model.py:738`.
- `p69` / `p71` — the missing channels carry **+0.0341** (val) / **+0.0366**
  (test) WPRD of information; ~90–95% of it is box size.
- `p70` — the geometry pathway is causally **live but WEAK** in the deployed
  model: `delta_D = −0.0087`, CI [−0.0127, −0.0049], 6/6 gates.
- `p73` — the frozen Fourier encoder at the checkpoint's own bandwidth is
  **exactly as informative about its input as a random permutation of itself**
  (mean R² −0.331 vs a −0.320 null, against a 0.9997 identity control), and the
  units bug is **load-bearing**: production's `/336` compresses `dx` into the
  narrow range where the encoder is still locally invertible (R² +0.39), while
  the units-fixed contract expands it and the encoder destroys everything
  (R² −0.12).

`C1a` alone is therefore **predicted null or harmful** and `C1b` alone cannot
deliver the +0.034, because its six informative channels remain zeroed. The two
defects interact, so the coherent minimal intervention is both at once.

## Arms

Both arms are `scripts/train/run_c0_c1.sh` with **one argument changed**. Data,
split, seed, optimizer, LR, schedule, warmup, batch size, accumulation, budget,
losses, sampler, predicate embeddings, text model, prior, layer counts, hidden
sizes and evaluator are pinned identically in that script.

| arm | `geom_input_pixel_space` | `geom_fourier_scale` |
|---|---|---|
| **`C0`** (control) | `false` | `1.0` |
| **`C1`** (treatment) | `true` | `0.01` |
| `C1a` (diagnostic, conditional) | `true` | `1.0` |
| `C1b` (diagnostic, conditional) | `false` | `0.01` |

`geom_fourier_scale = 0.01` was selected by `p73` Part C's pre-registered knee
criterion (input recovery R² 0.984; 0.02 is channel-selective, ≤0.005 has
degenerated the encoder toward a linear map). It is **not** tuned here.

## Protocol

- **Initialisation** — both arms resume the frozen historical checkpoint
  `checkpoints/demo_best/pure_best_adapt_light_mR50.pt`
  (sha256 `8845c3af…`), `--reset_epoch true`. The checkpoint file is never
  modified.
- **Seed** — `1234` for both arms (paired intervention comparison).
- **Budget** — 3 epochs × 12,000 samples/epoch = **36,000 samples**, batch 12,
  accum 2, lr 2e-5 cosine, 300 warmup steps, bf16, `clip_input_res 336`.
  Measured throughput ≈ 3.7 img/s ⇒ ≈ 3 h/arm.
- **Sequential, never concurrent.** `nvidia-smi` is checked before each launch.
- **Evaluated checkpoint** — the **final epoch** of each arm, at the identical
  fixed budget. **Never a per-arm "best" epoch**: that selection would differ
  between arms and contaminate the paired comparison. The four `best_*`
  selection checkpoints are suppressed for both arms via
  `--save_best_checkpoints false` (new flag, default `True` = historical, tested).

## Endpoints

**PRIMARY**

```
delta_WPRD = WPRD(C1) - WPRD(C0)
```

on the full VG150 **validation** split (10,401 images / 132,556 pairs — the
`p33`/`p36`/`p60`/`p69`/`p70` population), model term only, `ensemble_alpha =
0.0`, `cap = 64`, macro over cells, via `tools/within_pair_discrimination.py::wprd`
— the same function every existing anchor uses. Paired bootstrap over identical
cells, 2,000 resamples.

**SECONDARY** — R@50, mR@50, under the checkpoint's own composition
(`model_term + 3.75 · prior`, background masked).

**MECHANISTIC** — measured for both arms with `tools/c0_c1_preflight.py` and
`tools/geometry_causal_ablation.py`:

1. geometry channel variance / number of constant channels
2. Fourier decodability (the `p73` invertibility probe, on each trained model)
3. `fusion_gate` mean, std, per-pair std, min, max, quantiles
4. geometry ablation sensitivity (`delta_D`, `delta_B`, `delta_C`)
5. prior/calibration shift: agreement with the prior's argmax, score
   distributions, `model_term_std`

## Success threshold (fixed in advance)

| `delta_WPRD` | classification |
|---|---|
| **≥ +0.010** | **MATERIAL** — the registered success threshold |
| +0.005 … +0.010 | WEAK-POSITIVE — reportable, but not a Paper C success |
| −0.005 … +0.005 | NULL |
| ≤ −0.005 | HARMFUL |

**A first positive result is NOT Paper C `GO`.** `GO` additionally requires:

1. the paired 95% CI excludes 0;
2. the gain is **not** explained by a prior/calibration shift — specifically, if
   R@50/mR@50 improve while WPRD is flat, the result is classified as a
   **calibration/composed-metric effect and NOT a Paper C success**;
3. replication on a **second seed** with the identical `C1` intervention.

Until (3) is satisfied the best available verdict is **CONDITIONAL GO**.

## Stop conditions (fixed in advance)

Abandon the `C1` branch, without hyperparameter sweeping, if any of:

- training is unstable (non-finite loss, divergence);
- geometry channels are still degenerate in the trained `C1` model;
- `geom_fourier_scale` demonstrably fails to reach the live path;
- `delta_WPRD < +0.005`;
- `delta_WPRD` is negative;
- the positive result exists only in R@K/mR@K;
- the improvement is attributable only to prior/calibration.

If `C1` is null **and** the implementation is verified correct, the registered
conclusion is:

> *Input-path correction alone does not produce a meaningful learned relational
> improvement.*

That is a valid result and the branch stops there. `C1a`/`C1b` are then run as
**diagnostic factorisation** — not as a sweep — only if needed to attribute why.

## Preflight, already executed and passed

`tools/c0_c1_preflight.py`, bounded eval on 33,698 real validation pairs per
arm, no training:

| | `C0` | `C1` |
|---|---|---|
| constant geometry channels | **6 of 8** (reproduces `p68`) | **0 of 8** |
| live `decoder.geom_fourier_scale` | 1.0 | **0.01** |
| `fusion_gate` mean / per-pair std | 0.50155 / 0.000126 | 0.50156 / 0.000129 |
| `geom_B` | std 9.908, `requires_grad=False` | identical |
| gates | **5/5 PASS** | **5/5 PASS** |

`C0` reproduces `p68`'s degeneracy and its gate mean 0.50155 to five decimals,
so it is a genuine control and has not inherited the `C1` flags. `C1`'s eight
channels are all live and the scale reaches the live decoder.

Clamp saturation under the `C1` contract, measured on the full cache and
recorded here rather than discovered later: `dx` 1.38%, `dy` 0.72%, `rw`/`rh`
0.01%, `ar1`/`ar2` ~0.00% of rows sit at `geom_feats_torch`'s own ±10 / ±5
bounds. These are the function's designed bounds, not a new artifact.

## Stated in advance: what this cannot show

1. **`C0` is not the historical checkpoint.** It is the historical checkpoint
   fine-tuned 36,000 further samples. `C0`'s WPRD will differ from the frozen
   0.5542 anchor for that reason alone, which is exactly why `C0` exists. All
   comparisons are `C1` vs `C0`, never `C1` vs 0.5542.
2. **36,000 samples may be too small a budget** for `geom_mlp` and the fusion
   gate to learn a newly informative input distribution from weights tuned on
   hashed noise. A null at this budget does not prove a null at every budget; it
   proves a null at *this* budget, and will be reported with that scope.
3. **One seed is one seed.** Registered above as insufficient for `GO`.
4. **Paper A is untouched.** No frozen metric is altered. If `C1` explains why
   geometry previously underperformed, that is recorded as a new mechanistic
   explanation, not a revision.
