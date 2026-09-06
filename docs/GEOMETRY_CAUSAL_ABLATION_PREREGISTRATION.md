# `p70` preregistration — does geometry causally affect the deployed model, and through which path?

Committed **before** `runs/p70_geometry_causal_ablation` is launched. Tool
`tools/geometry_causal_ablation.py` already exists and has passed two bounded
smoke tests (`runs/p70_smoke`, `runs/p70_smoke2`); this document registers the
**hypotheses, endpoints and decision rules**, which are not yet settled.

## Motivation

`p68` measured that 6 of 8 geometry channels reaching `forward_pairs` are
bit-exactly constant; `p69` measured (pre-registered, 6/6 gates) that the
missing channels are worth **+0.0341 WPRD** as raw information. Neither result
shows that geometry *currently does anything* inside the deployed checkpoint.

Before any GPU is spent retraining (`C1a`/`C1b`/`C1`), this experiment asks the
prior question: **is the geometry pathway causally live at all, and if so
through which of its two injection points?** A pathway that is measurably inert
at inference reframes what a units fix could plausibly buy.

## The two pathways (VERIFIED FACT, source-level)

`relational_model.py:441-467` and `:537-541`. `geom_feat_proj` is injected
**twice**:

1. **fusion** — `geom_norm = normalize(geom_feat_proj)`, then
   `fused_feat = gate * sem_feat + (1 - gate) * geom_norm` (`:453-458`).
2. **edge layers** — the *unnormalised* `geom_feat_proj` is concatenated into
   every `ProgressiveEdgeConditionedLayer.rel_update`
   (`:65`, 2 layers, `progressive_edge_layers = 2`).

## Arms

"Neutralised" = `dx = dy = 0` in the 8-vector. `p68` established the other six
channels are already bit-exactly constant in production, so this makes the whole
geometry vector **constant across pairs**: zero information, full magnitude,
unchanged unit-norm contribution to the fusion. This is an *information*
ablation, not a magnitude ablation.

| arm | `forward_pairs` geom input | edge-layer geom injection |
|---|---|---|
| `A_full` | real | real (untouched) |
| `B_no_fusion_geom` | neutralised | real (overridden) |
| `C_no_edge_geom` | real | neutralised (overridden) |
| `D_no_geom` | neutralised | neutralised |
| `E_dx_only` | `dy` neutralised | follows input |
| `F_dy_only` | `dx` neutralised | follows input |

Every arm calls the **unmodified** `forward_pairs`; arms differ only in what is
fed to it and what is injected into the edge layers. There is no
reimplementation of the fusion equation.

## Estimator and rows

Identical rows for every arm, by construction — one forward pass emits all six
`rel_feat` tensors per pair. Full VG150 **validation** split, 10,401 images,
GT rows only, exactly the population `p33`/`p60`/`p69` use.

Deployed model term is reconstructed offline as
`layer_norm_51(normalize(rel_feat) @ normalize(pred_emb).T)` — verified against
the `p36` cache to 1.1e-4 (fp16 storage error only), and `ensemble_alpha = 0.0`
so the classifier branch contributes exactly zero.

WPRD via `tools/within_pair_discrimination.py::wprd`, `cap = 64`, `seed = 0`,
macro over cells — the same function `p33`/`p59`/`p60`/`p69` call. Because arms
are **paired on identical cells**, the primary test is a paired bootstrap over
cells, not a comparison of independent CIs.

## Primary endpoint

```
delta_D = WPRD(D_no_geom) - WPRD(A_full)
```

the total causal contribution of the geometry pathway, as deployed.

## Secondary endpoints

- `delta_B`, `delta_C` — path localisation (fusion vs edge layers).
- `delta_E`, `delta_F` — single-channel attribution (`dx` alone, `dy` alone).
- R@50 / mR@50 recomputed offline per arm under the checkpoint's own
  composition (`model_term + 3.75 * prior_rows`, background masked).
- Top-1 prediction agreement with `A_full`.
- Prior-alignment / calibration shift: KL of each arm's mean predicted
  distribution against `A_full`'s, and against the frequency prior.

## Pre-registered decision rules

On `delta_D` (sign convention: negative = removing geometry *hurts*, i.e.
geometry is being used):

- **LIVE** — `delta_D <= -0.010`. The geometry pathway materially carries
  image-conditioned discrimination in the deployed model. Restoring the six
  destroyed channels then has a mechanism to act through, and `C1a` is
  strongly justified.
- **WEAK** — `-0.010 < delta_D <= -0.003`. Geometry contributes, but little.
  `C1a` remains justified by `p69` alone; report the small effect honestly.
- **INERT** — `|delta_D| < 0.003`. The pathway is causally dead at inference.
  This does **not** falsify `p69` (which measures information content, not
  current use), but it **does** mean any `C1a` gain must come from *learning to
  use* newly available inputs, not from unblocking an already-used channel. It
  is a materially weaker prior for `C1a`, and it must be reported as such
  before, not after, the training run.
- **INVERTED** — `delta_D >= +0.003`. Removing geometry *improves* within-pair
  discrimination. This would be a genuinely surprising result implying the
  degenerate geometry vector is actively harmful, and would make a units fix a
  *repair* rather than an *addition*.

Path localisation is descriptive and carries no independent decision, but the
registered expectation is `delta_D ≈ delta_B + delta_C` if the paths are
approximately additive; a large departure indicates interaction.

## Validity gates (all must pass or the run is void)

- **G1** Arm `A_full` reproduces the evaluator's own dumped `rel_feat`
  **bit-exactly** (`max_abs_diff == 0.0`). *Already passed on both smoke runs.*
- **G2** Arm `A_full`'s recomputed WPRD reproduces the published PURE
  validation model-term WPRD **0.5542** (`p33`) within ±0.005.
- **G3** Row/image counts equal the `p36` cache's (10,401 images / 132,556
  pairs), i.e. this is the same population as every existing anchor.
- **G4** Prior control on the same rows reads exactly 0.5000.
- **G5** Every arm scores every GT row, all finite.
- **G6** Arms `B` and `C` differ from `A` and from each other in `rel_feat`
  (a null intervention would be a bug, not a finding).

## Stated in advance: what this cannot show

An inference-time ablation removes an input from a network **trained with that
input present**. It measures current reliance, which can *overstate* the
information's value (the network may route around it if retrained) and can also
*understate* the value of a *corrected* input (the network never had the chance
to learn from a non-degenerate one). `delta_D` is therefore a bound on current
use, **not** a prediction of `C1a`'s effect. It is registered here as a prior on
`C1a`, not a substitute for it.

## Cost

Measured marginal throughput 0.956 s/image with all six arms (from
`runs/p70_smoke`, 48 images / 92.9 s, and `runs/p70_smoke2`, 300 images /
333.8 s). Full validation split ≈ **2 h 46 m** on the idle L4. GPU verified
idle (0 MiB, 0 %, no compute apps) immediately before launch.
