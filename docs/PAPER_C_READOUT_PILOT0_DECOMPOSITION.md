# Pilot 0 — decomposing the PURE readout bottleneck

**CPU-only, analysis-only. No training, no GPU, no fitting, no source change, no
registered experiment.** Every number below is computed from artifacts that
already existed — `runs/eval_C1/pair_logits.pt`, `runs/eval_C0/pair_logits.pt`,
`checkpoints/C1_seed1234.pt`, `checkpoints/C0_seed1234.pt` — using the
**registered** WPRD estimator (`within_pair_discrimination::wprd`, `cap = 64`,
macro, the same 20,016 cells).

The running `C1` seed-2 endpoint (PID 61013) was not touched, and **no seed-2
value was inspected or used anywhere in this analysis**. Its liveness was checked
and nothing else.

> **Result: the bottleneck is the FIXED PROTOTYPE CONSTRAINT, not nonlinearity.**
> A learned **linear** map on `rel_feat` recovers **96.3%** of the readout span on
> `C1` (88.4% on `C0`); adding the nonlinearity contributes **3.7%** (11.6%).
> **This reverses the primary recommendation in
> `docs/PAPER_C_PURE_READOUT_V2_DESIGN.md` §11**, which proposed a nonlinear MLP
> plus learnable prototypes. The MLP is not justified by the evidence.

---

# 1. Objective

Separate the two explanations for the `≈ +0.0244` readout gap that the previous
ladder could not distinguish:

- **(A) learnable predicate prototypes** — learned class-specific directions
  replacing the frozen CLIP set;
- **(B) relational nonlinearity** — the `GELU` inside `predicate_classifier`.

# 2. Existing readout ladder (locked, `C1` seed 1)

| readout | WPRD |
|---|---|
| frozen CLIP cosine (deployed) | `0.5749881522` |
| + trained `text_space_projection` adapter | `0.5750821265` |
| ZCA-whitened prototypes | `0.5769953892` |
| centred + whitened prototypes | `0.5771811900` |
| learned nonlinear classifier, prior-free | `0.5979433958` |

# 3. Exact mathematical definitions

With `r = rel_feat` (n, 768), `E = normalize(pred_emb)` (51, 768), and the
trained `predicate_classifier` parameters
`LN(·)`, `W1 ∈ R^{768×768}, b1`, `W2 ∈ R^{51×768}, b2`:

```
z(r) = W1 · LN(r) + b1                       the learned linear map
h(r) = GELU(z(r))                            the learned nonlinear map

R0 = rowstd( normalize(r) @ E^T )            frozen prototypes, fixed linear form
R1 = rowstd( normalize(r + 0.35·adapter(r)) @ E^T )
RH = rowstd( normalize(h(r)) @ E^T )         learned map vs FROZEN prototypes
RL = rowstd( W2 · z(r) + b2 )                LESION: GELU removed, learned params
R3 = rowstd( W2 · h(r) + b2 )                the full trained classifier
```

`rowstd` is `evals.py:975-983`. All five are **pure functions of `rel_feat`**;
none contains any prior term.

# 4. Existing-artifact controls — what is and is not identifiable

| control | status |
|---|---|
| `R0`, `R1`, `R3` | **directly measured** from stored weights |
| `RL` | **measured, but a LESION** — the weights were trained *with* `GELU`; removing it is a perturbation, not a retrained linear head |
| `RH` | **measured but INVALID as a control** — see §7 |
| **`R2` — a properly *trained* linear head on `rel_feat`** | **NOT IDENTIFIABLE FROM EXISTING ARTIFACTS.** No such head exists in any checkpoint, and no repository run contains a linear probe on `rel_feat` (`p60`'s `A_relfeat` is an **MLP**; `D_geometry_linear` is linear but on the 20 *geometry* features, not `rel_feat`). Per the pilot constraint it was **not fitted.** |

# 5. `R0` / `RL` / `R3` results

**`C1` seed 1**, prior-free, 20,016 cells:

| control | WPRD | weighted |
|---|---|---|
| `R0` deployed frozen CLIP cosine (stored) | `0.5749881522` | `0.533987` |
| `R0` reconstructed | `0.5749353114` | `0.537035` |
| `R1` + trained text adapter | `0.5750821265` | `0.537416` |
| `RH` `cos(h, CLIP)` — **invalid, see §7** | `0.4881843634` | `0.491209` |
| **`RL` LESION, purely linear in `r`** | **`0.5965890043`** | `0.548633` |
| `R3` full trained classifier | `0.5974257146` | `0.549509` |

(The stored-vs-reconstructed `R0` differ by `5.3e-05`, consistent with the dump's
stored precision; the reconstruction is used for all deltas so that every term in
the decomposition is computed identically.)

### Replication across checkpoints — same procedure, both arms

| arm | `R0` frozen CLIP | `RL` linear lesion | `R3` nonlinear | `RL − R0` | `R3 − RL` | linear span | nonlinearity |
|---|---|---|---|---|---|---|---|
| `C0` | `0.5666999240` | `0.5873943370` | `0.5901189879` | `+0.020694` | `+0.002725` | **88.4%** | 11.6% |
| `C1` | `0.5749353114` | `0.5965890043` | `0.5974257146` | `+0.021654` | `+0.000837` | **96.3%** | 3.7% |

# 6. Prototype contribution

```
Delta_proto  =  WPRD(RL) - WPRD(R0)  =  +0.0216536928     96.3% of the R0->R3 span
```

For contrast, **re-basing** the frozen prototypes without learning them —
centring plus ZCA-whitening to exact orthonormality, effective dimension
`1.41 → 50.00` — bought only `+0.0022459`. So it is **not** the conditioning or
geometry of the CLIP prototype set that binds: it is that the directions are
**inherited from CLIP's language space rather than learned for the relational
task**. Learning them is worth roughly **10×** re-basing them.

# 7. Nonlinearity contribution

```
Delta_nonlinear  =  WPRD(R3) - WPRD(RL)  =  +0.0008367103    3.7% of the span
```

`C0` gives `+0.0027246509` (11.6%). Small in both.

**`RH` is discarded as an invalid control.** Scoring `h = GELU(...)` against CLIP
prototypes composes two spaces that were never aligned — `h` is
non-negative-dominated (`frac(h > 0) = 0.5868`) and its mean cosine to the mean
CLIP prototype is `0.0088`, against `0.1746` for `rel_feat`. `RH = 0.4882` is
below chance and reflects the space mismatch, not a prototype effect. Any
decomposition routed through `RH` is meaningless and is not reported as one.

**Caveat on `RL`, stated plainly.** `RL` is a lesion, not a trained linear head,
so it is not `R2`. Its interpretation rests on one observation: removing the
nonlinearity from a network *trained with it* costs only `0.0008` — i.e. the
trained classifier is operating in a near-linear regime for this task. A head
actually *optimised* within the linear function class would not plausibly do
worse than a lesioned one, so `RL` is best read as a **lower bound** on trained
linear performance. That reading is an **inference**, not a measurement.

# 8. Spatial-predicate breakdown

Registered grouping (`configs/predicate_metadata_vg150.json`), same size bands
as seed 1:

| stratum | cells | `R0` | `RL` | `R3` | `R3 − R0` |
|---|---|---|---|---|---|
| both spatial | 6,762 | `0.59118` | `0.61076` | `0.60950` | `+0.01832` |
| not both spatial | 13,254 | `0.56665` | `0.58936` | `0.59127` | `+0.02462` |
| ALL | 20,016 | `0.57494` | `0.59659` | `0.59743` | `+0.02249` |
| `na·nb ≥ 25` | 2,823 | `0.55523` | `0.56892` | `0.57068` | `+0.01545` |

**Nonlinearity is not especially useful for directional decoding** — it *lowers*
the spatial stratum slightly (`0.61076 → 0.60950`) and raises the non-spatial one
(`0.58936 → 0.59127`). The readout fix is general capacity, larger off the
spatial group, exactly as the earlier audit found. No new partition was
introduced.

# 9. Prior-free validation

```
prior control WPRD = 0.5000000000   (exactly)
```

`R0`, `R1`, `RH`, `RL`, `R3` are all pure functions of `rel_feat` and contain
**no** prior term — the `gate·prior` branch of `calibrated_predicate_logits`
(`:962-968`) is excluded from every control here. The `+0.0217` prototype gain
therefore survives prior removal in full; it is not a calibration effect.

# 10. Function-class conclusion

**CASE A — primarily a fixed-prototype constraint.**

A learned **linear** readout on `rel_feat` recovers 96.3% (`C1`) / 88.4% (`C0`)
of the span between the deployed head and the full trained classifier.
Nonlinearity contributes 3.7% / 11.6%. Re-basing the frozen prototypes without
learning them contributes ~10%.

This is coherent with the wider record rather than in tension with it: `p60`
found nonlinearity worth `+0.0235` on the **20 raw geometry features**
(`D_geometry_linear` 0.5741 → `B_geometry` 0.5976), where the input is raw
numbers. `rel_feat` is already the output of the entire network — a deeply
nonlinear function of the image and boxes — so a linear readout on top of it
suffices, and the binding constraint is only *which directions* that linear
readout uses.

# 11. Minimum justified Readout v2

Per the "smallest function class that explains the evidence" rule:

> **Replace the frozen CLIP cosine with a learned LINEAR predicate readout whose
> prototypes are initialised from `pred_emb` and are trainable, anchored to CLIP
> by a regularisation term.**

```
logits = rowstd( normalize(rel_feat) @ normalize(P)^T ),
         P in R^{51x768},  P initialised = pred_emb,  requires_grad = True
loss  += lambda_proto * || P - pred_emb ||_F^2
```

New parameters: **39,168** (51 × 768). No MLP, no hidden layer, no extra
nonlinearity, no change to the relational trunk, no change to the geometry
contract.

**A nonlinear relational projection is NOT justified by this evidence** and
should not be included in the first Readout v2. The `~0.6M`-parameter MLP
proposed as PRIMARY in `PAPER_C_PURE_READOUT_V2_DESIGN.md` §11 buys an estimated
3.7–11.6% of the span and should be **demoted to a later ablation**, to be
considered only if the linear version underperforms its prediction.

**Is a hybrid justified?** Only in the narrow sense that the prototypes should be
**CLIP-initialised and anchored** — which preserves open-vocabulary scoring and
the semantic prior — rather than randomly initialised. A hybrid in the sense of
*CLIP semantics plus a nonlinear discriminative branch* is **not** justified. A
logit-level blend was separately measured to be monotone in `α` with no interior
optimum, so it is rejected as well.

# 12. Unresolved ambiguity

1. **`R2` proper was never measured.** The linear conclusion rests on a lesion
   (`RL`) plus the inference in §7. A properly trained linear head on frozen
   `rel_feat` would settle it and requires **fitting**, which this pilot forbade.
2. **How much of `Delta_proto` needs full freedom?** Whitening gave ~10%,
   learning gave 96%; nothing here interpolates between "re-based CLIP" and
   "freely learned", e.g. a low-rank learned correction to `pred_emb`.
3. **Anchoring strength `lambda_proto` is unconstrained** by any measurement here.
4. **Open-vocabulary cost is unmeasured.** No artifact quantifies what learnable
   prototypes cost in open-vocabulary generalisation, which is CLIP's reason for
   being in the readout at all.
5. `RL`'s weights were trained under the full objective including a `text_ce`
   term (`lambda_text_predicate_ce = 0.4`) that shapes `rel_feat` toward CLIP
   space; a readout trained *without* the frozen cosine might reshape `rel_feat`
   itself, which none of these frozen-feature controls can predict.

# 13. Next experiment

**Pilot 0b (CPU, would require fitting — therefore NOT run here and needing its
own registration):** fit a linear head and an MLP head on frozen `C1` `rel_feat`
under `p60`'s exact estimator (AdamW, hidden 256, 20 epochs, lr 2e-3, l2 1e-4,
softmax CE, 5-fold CV on validation, salt 0). This yields `R2` properly, converts
the §7 inference into a measurement, and costs no GPU.

Only after that should Readout v2 be implemented, and then as the **linear,
CLIP-initialised, anchored prototype head** of §11 — behind a default-off flag,
with the geometry contract fixed at the `C1` values, and with its own
pre-registration stating the threshold in advance.

---

## Status

Analysis only. Nothing implemented, nothing registered, nothing trained. The
locked `C1` seed-1 result and the `+0.010` registered threshold are unchanged,
and the seed-2 replication was neither touched nor consulted.
