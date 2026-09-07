# `p70` — the geometry pathway is causally live, but WEAK: `delta_D = -0.0087`

Pre-registered in `docs/GEOMETRY_CAUSAL_ABLATION_PREREGISTRATION.md`, commit
`db66b49`, **before** the run. GPU run:
`runs/p70_geometry_causal_ablation` (VG150 validation, 10,401 images,
132,556 pairs, 2 h 53 m on the idle L4, finished 2026-09-06 23:16 UTC).
Offline analysis: `tools/geometry_ablation_wprd.py` (CPU only),
`runs/p70_geometry_causal_ablation/ablation_wprd.json`.

> **Recovery note.** The GPU forward pass completed and wrote every artifact
> before the previous session hit its quota; the CPU analysis step never ran.
> This document reports the analysis of those recovered artifacts. No GPU work
> was repeated — the `.pt` dumps are the originals, timestamped 23:16.

**Gates: 6/6 PASS.** Arm `A_full` reproduces `p33`'s published validation
model-term WPRD **0.5542 to ±0.0000**; the population is the `p36`/`p60`/`p69`
population exactly (10,401 / 132,556); the prior control reads exactly 0.5000.

## Result

| arm | WPRD | weighted | cells>.5 | R@50 | mR@50 | agree@1 vs A | pred = prior argmax |
|---|---|---|---|---|---|---|---|
| `A_full` | **0.5542** | 0.5266 | 49.7% | 0.6717 | 0.2321 | 1.0000 | 0.9244 |
| `B_no_fusion_geom` | 0.5506 | 0.5252 | 49.3% | 0.6716 | 0.2286 | 0.9891 | 0.9256 |
| `C_no_edge_geom` | 0.5504 | 0.5240 | 49.0% | 0.6718 | 0.2266 | 0.9751 | 0.9281 |
| `D_no_geom` | **0.5455** | 0.5211 | 48.4% | 0.6722 | 0.2248 | 0.9687 | **0.9302** |
| `E_dx_only` (dy removed) | 0.5471 | 0.5232 | 48.9% | 0.6717 | 0.2293 | 0.9831 | 0.9259 |
| `F_dy_only` (dx removed) | 0.5519 | 0.5281 | 49.3% | 0.6713 | 0.2306 | 0.9839 | 0.9246 |

Paired bootstrap over identical cells (2,000 resamples):

| arm | ΔWPRD | 95% CI | P(Δ<0) | ΔR@50 | ΔmR@50 |
|---|---|---|---|---|---|
| `B_no_fusion_geom` | −0.0036 | [−0.0061, −0.0013] | 1.000 | −0.0001 | −0.0035 |
| `C_no_edge_geom` | −0.0038 | [−0.0074, −0.0004] | 0.987 | +0.0001 | −0.0055 |
| **`D_no_geom`** | **−0.0087** | **[−0.0127, −0.0049]** | **1.000** | +0.0005 | −0.0073 |
| `E_dx_only` | −0.0071 | [−0.0109, −0.0032] | 1.000 | −0.0000 | −0.0028 |
| `F_dy_only` | −0.0023 | [−0.0061, +0.0011] | 0.901 | −0.0004 | −0.0015 |

```
PRIMARY   delta_D = WPRD(D_no_geom) - WPRD(A_full) = -0.0087   -> WEAK
PATHS     fusion-only -0.0036   edge-only -0.0038   additivity residual -0.0013
```

## What this establishes (MEASURED)

1. **The pathway is live, not inert.** `delta_D`'s 95% CI is
   [−0.0127, −0.0049] and excludes zero with P(Δ<0) = 1.000. Deleting the
   geometry *information* while holding its magnitude and unit-norm fusion
   contribution fixed costs **0.0087 WPRD**. The pre-registered `INERT` band
   (|Δ| < 0.003) is excluded.
2. **But it is WEAK, not LIVE.** It falls short of the registered `LIVE`
   threshold (≤ −0.010) by 0.0013. This was registered in advance as
   "geometry contributes, but little. Report the small effect honestly." It is
   reported as such.
3. **The two injection points contribute almost equally and additively.**
   Fusion −0.0036, edge layers −0.0038, residual −0.0013. Neither path
   dominates; a fix must reach both, which the `C1a` call-site change does by
   construction (it changes the input, upstream of both).
4. **`dy` carries more than `dx`.** Removing `dy` costs 0.0071; removing `dx`
   costs 0.0023 and its CI touches zero. This independently corroborates `p66`,
   which found `dy_rel` nonlinearly decodable from `rel_feat` at R² = 0.292
   against `dx_rel`'s 0.073 — two unrelated estimators, same asymmetry.

## Prior / calibration analysis (Phase 6 discipline)

The effect is **not** a calibration artifact, and the diagnostics say so in an
unusually clean way:

- **R@50 is flat to ±0.0005 across every arm** — and moves the *wrong* way
  (`D_no_geom` scores +0.0005 *higher* than `A_full`). Removing all geometry
  information does not change R@50 at all. A result read off R@50 would have
  concluded "geometry does nothing."
- **WPRD falls monotonically with geometry removed**, from 0.5542 to 0.5455.
- **Agreement with the frequency prior's argmax rises monotonically** as
  geometry is deleted: 0.9244 (`A_full`) → 0.9302 (`D_no_geom`). Geometry is
  precisely the thing pushing this model *off* the prior; take it away and the
  model collapses further onto the prior.
- `model_term_std` is constant to 4 significant figures across arms, so no arm
  is rescaled relative to another.

This is the textbook shape of a **genuine within-pair discrimination effect
that a recall metric cannot see**, and it is the strongest single piece of
evidence this programme has produced for WPRD's construct validity as a metric
separate from R@50.

## The gate is a constant (VERIFIED, full population)

The same run measured `fusion_gate` over all **40,201,728** gate elements
(132,556 pairs × 768 dims for real, on the full split — `p68` measured 17,196
pairs):

| statistic | value |
|---|---|
| elementwise mean | 0.50153 |
| elementwise std | 0.00485 |
| min / max | 0.47852 / 0.52734 |
| q25 / median / q75 | 0.49805 / 0.50000 / 0.50391 |
| fraction in [0.45, 0.55] | **1.0000** |
| fraction < 0.25 or > 0.75 | **0.0000** |
| per-pair mean, std across pairs | 0.50153, **0.00013** |

`p68`'s pilot finding replicates on the full split to four decimals. **Not one
of 40 million gate elements falls outside [0.478, 0.528].** The gate is
architecturally input-dependent and empirically a constant ½.

Also measured: `cos(sem, geom) = -0.0015` (the two fused branches are
essentially orthogonal), `geom_proj_prenorm_mean = 27.67`.

## Reading this against `p69`/`p71`

These measure different things and both are correct:

- `p69`/`p71`: the geometry PURE is denied is worth **+0.0341 / +0.0366 WPRD**
  as *information*, on validation and held-out test.
- `p70`: the geometry PURE *does* receive is worth **0.0087 WPRD** of *current
  causal use*.

The registration stated in advance that an inference-time ablation bounds
**current reliance**, not the value of a corrected input — a network trained
without a signal cannot be shown, by deleting it, to be unable to use it. So
`p70` neither confirms nor refutes `C1a`. What it does is set the prior
honestly: the pathway is open and carrying a small real signal, so a fix has
somewhere to act — but the deployed model's demonstrated appetite for geometry
is 0.0087, not 0.034.

## What this does NOT establish

- It does **not** predict `C1a`'s effect. Registered as such in advance.
- It does **not** show the gate's constancy is *caused* by the degenerate
  input; that is a hypothesis `C1` tests.
- It does **not** touch any frozen Paper A claim. `A_full` reproducing 0.5542
  exactly is, if anything, an independent re-verification of `p33`.
