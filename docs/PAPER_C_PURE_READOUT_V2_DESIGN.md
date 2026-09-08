# PURE Complete — Readout v2 design

**Design document. Nothing here is a result, and nothing here has been run as an
experiment.** All measurements below are CPU-only re-derivations from artifacts
that already existed (`runs/eval_C1/pair_logits.pt`, `checkpoints/C1_seed1234.pt`)
using the **registered** WPRD estimator on the same 20,016 cells. No model was
trained, no probe was fitted, no GPU was used for any of it.

Written while the registered `C1` seed-2 replication was running. **The seed-2
experiment was not touched**: no source, config, checkpoint, seed or output path
was modified, and no runtime code was changed.

> Every alternative readout measured here is **EXPLORATORY** and is **NOT** the
> registered endpoint. The registered `C1` endpoint remains the frozen-CLIP
> cosine at `ensemble_alpha = 0.0`, and the locked seed-1 result
> (`delta_WPRD = +0.008261`, **THRESHOLD NOT MET**) is unchanged.

---

# 1. Problem statement

After the `C1` geometry repair, the dominant measured gap in PURE is not the
relational representation but the **readout that decodes it**. On the *same*
`C1` `rel_feat`, the deployed head scores `0.5750` and the model's own stored
classifier head scores `0.5994`.

The design question is **not** "how do we raise WPRD". It is: *what readout lets
the repaired relational representation be decoded with minimal information loss,
while keeping the semantic and auditability properties PURE was built for?*

# 2. Locked `C1` evidence (unchanged)

| | |
|---|---|
| `C0` WPRD | `0.5667271196244782` |
| `C1` seed-1 WPRD | `0.5749881522134409` |
| `delta_WPRD` | `+0.008261032588962776` |
| paired 95% CI | `[+0.0038311178, +0.0126973772]` |
| registered threshold | `+0.010` — **NOT MET** |

The geometry repair raised **both** readouts by nearly the same amount — text
`+0.0082610326`, classifier `+0.0085018590` — which is why it is classified as a
**representation** effect and is independent of everything below.

# 3. Current readout architecture (exact)

```
rel_feat  (n, 768)          relational_model.py:552   out_norm = LayerNorm(768)

TEXT HEAD (deployed, ensemble_alpha = 0.0)
  text_relation_features(r)  :851-857   = r + 0.35 * text_space_projection(r)
                                          -- but DISABLED at the endpoint, so = r
  score(r, pred_emb)         :984-990   = normalize(r) @ normalize(pred_emb)^T
  _normalize_eval_logits     evals.py:975-983   per-row standardisation over 51 cols

CLASSIFIER HEAD (computed, then DISCARDED at alpha = 0)
  predicate_classifier(r)    :606-612   LayerNorm -> Linear(768,768) -> GELU
                                        -> Dropout(0.1) -> Linear(768,51)
  + 0.25 * tanh(bias_residual_head(r))  :958-961
  + 1.0  * gate(r) * centred_log_prior  :962-968   (adaptive_prior_scale = 1.0)

COMPOSITION               evals.py:1444-1449
  model_term = a * rowstd(cls) + (1-a) * rowstd(text),  a = 0
  score      = model_term + 3.75 * log_prior
```

Parameter counts read from `checkpoints/C1_seed1234.pt`:
`predicate_classifier` **631,347**, `bias_residual_head` **631,347**,
`calibration_gate` **149,377**, `text_space_projection` **1,182,720**.

# 4. The text-head limitation — measured, not assumed

### 4.1 `F.normalize` is a mathematical no-op here

`_normalize_eval_logits` standardises **per row**, and per-row standardisation is
invariant to any positive per-row rescaling. So `normalize(rel_feat)` cannot
change the endpoint.

```
max | rowstd(normalised) - rowstd(un-normalised) |  =  1.410e-05   (float noise)
```

There was no amplitude to lose in the first place: `out_norm` is a LayerNorm, so

```
||rel_feat|| : mean 27.7315   std 0.0043   min 27.7118   max 27.7431
```

a relative spread of `1.5e-4`. **Hypothesis "cosine normalisation destroys
amplitude" is refuted twice over.**

### 4.2 The CLIP prototype set is globally degenerate

50 foreground prototypes in 768 dimensions:

| | |
|---|---|
| mean pairwise cosine | **0.9519** (min 0.8666, max 0.9939) |
| participation-ratio effective dimension | **1.41 of 50** |
| spectral energy in the top 1 direction | **> 90%** |

Directional antonyms are nearly collinear — `above`/`under` 0.9828,
`in front of`/`behind` 0.9802, `over`/`under` 0.9884, `on`/`under` 0.9874.

### 4.3 But collinearity does **not** predict where the text head fails

Over 100 predicate pairs with ≥40 cells:

```
Spearman(prototype cosine, TEXT head WPRD)  = +0.1670
Spearman(prototype cosine, CLS  head WPRD)  = +0.1017
Spearman(prototype cosine, cls advantage)   = -0.0416
```

This **replicates `p51`** (`docs/READOUT_GEOMETRY_RESULT.md`), which refuted the
pairwise embedding-geometry hypothesis on the historical checkpoint. The
resolution: the degeneracy is a **global rank limit**, roughly uniform across
pairs, so it caps capacity everywhere without predicting any particular pair.

# 5. Classifier-head evidence, and the prior caveat is now closed

The previous audit could only bound the prior-gated term by signature. It has now
been **decomposed directly** by re-running the stored heads from checkpoint
weights on the stored `rel_feat`:

| readout on the same `C1` `rel_feat` | WPRD | Δ vs deployed | % of gap |
|---|---|---|---|
| **deployed text head** (frozen CLIP cosine) | `0.5749881522` | — | 0% |
| `predicate_classifier` only — **prior-free** | `0.5974257146` | `+0.0224376` | **91.8%** |
| `+ bias_residual_head` — **prior-free** | `0.5979433958` | `+0.0229552` | **93.9%** |
| stored classifier head (incl. `gate·prior`) | `0.5994290264` | `+0.0244409` | 100% |

**93.9% of the classifier's advantage is prior-free.** The `gate·prior` term
contributes only `+0.0014856`. Failure mode I ("prior interaction") is therefore
**ABSENT as an explanation**, established by decomposition rather than inference.

# 6. Information-bottleneck analysis — the ablation ladder

Every row is the same `C1` `rel_feat`, the same registered estimator, the same
20,016 cells. Nothing is fitted; each row is a deterministic transform.

| # | readout | WPRD | % of the `+0.0244` gap |
|---|---|---|---|
| 0 | deployed frozen-CLIP cosine | `0.5749881522` | 0% |
| 1 | + the **already-trained** `text_space_projection` adapter (0.35) | `0.5750821265` | **0.4%** |
| 2 | prototypes **centred** | `0.5747837787` | −0.6% |
| 3 | prototypes **ZCA-whitened** (exactly orthonormal) | `0.5769953892` | 8.4% |
| 4 | prototypes **centred + whitened** | `0.5771811900` | **9.2%** |
| 5 | **learned nonlinear head, prior-free** | `0.5979433958` | **93.9%** |

Row 4 makes the 50 prototypes *perfectly* orthonormal (effective dimension
50.00/50, mean pairwise cosine 0.0000) and recovers **only 9.2%**.

| hypothesis | verdict | confidence | evidence |
|---|---|---|---|
| **B** cosine normalisation destroys amplitude | **REFUTED** | **HIGH** | no-op to `1.4e-05`; `‖rel_feat‖` constant to `1.5e-4` |
| **D/E** features need adaptation before projection | **REFUTED** | **HIGH** | the trained 1.18M-param adapter buys **0.4%** |
| **I** prior interaction | **REFUTED** | **HIGH** | 93.9% of the advantage is prior-free |
| **A/C** prototype geometry is misaligned / too rigid | **PRESENT BUT MINOR** | **HIGH** | perfect orthonormalisation buys **9.2%** |
| pairwise prototype collinearity | **REFUTED** | **HIGH** | ρ = −0.04; replicates `p51` |
| **G** the readout is a **fixed linear form** where a **learned nonlinear** map is needed | **PRIMARY** | **MEDIUM-HIGH** | rows 0→5 localise **91.8%** to swapping the linear cosine for the learned nonlinear head; `p60` independently found linear→MLP worth `+0.0235` on geometry features (`D_geometry_linear` 0.5741 → `B_geometry` 0.5976) |
| **E/F** classifier preserves order / geometry better | **NOT SEPARATED** | **LOW** | the classifier advantage is present in *both* strata — spatial `+0.0207`, non-spatial `+0.0263` — so it is not specifically a directional fix |

**What is NOT yet separated:** how much of row 5 is *learned prototypes* versus
*the nonlinearity*. `predicate_classifier`'s final `Linear(768,51)` is itself a
learned prototype set, so rows 3–4 (better *fixed* prototypes) are only a proxy
for the linear ceiling. Separating them requires **fitting a linear head**, which
is Pilot 0 (§15) and is CPU-only.

# 7. Directionality

`sem_feat = normalize(sub + obj)` (`:429`) is symmetric, and the SPOA role
machinery is disabled at the endpoint, so at the seed the only order-bearing
signal is pair-relative geometry (`geometry.py:34-62`), with the edge-layer
`cat([sub_upd, obj_upd, rel_feat, geom_feat])` (`:65`) adding a second route.

Does the text head add a *further* directional bottleneck? The evidence says
**not specifically**:

```
classifier advantage, both-spatial cells      +0.020735  [+0.01282, +0.02902]
classifier advantage, not-both-spatial cells  +0.026332  [+0.02103, +0.03176]
```

The advantage is, if anything, **larger off** the spatial group. Individual
directional pairs are dramatic — `above`/`behind` text **0.5091** (chance) vs
classifier **0.6911** — but the pattern is not confined to directional pairs, so
the readout limitation is **general capacity**, not a directional failure. Any
claim that CLIP prototypes specifically destroy relational asymmetry is **NOT
SUPPORTED** by these artifacts.

# 8. CLIP audit — verdict

Frozen CLIP provides a **useful semantic prior** (it is what makes open-vocabulary
predicate scoring possible at all, `:862-894`) but an **unnecessarily restrictive
discriminative readout**: a fixed linear form onto a set whose effective
dimension is 1.41. The correct conclusion is *not* "delete CLIP" — deleting it
costs the open-vocabulary property and, per §6, prototype geometry was only 9.2%
of the problem anyway. The conclusion is that CLIP should not be the **only**,
**fixed**, **linear** readout.

# 9-10. Candidate architectures and ranking

A logit-level blend was measured and is **not** supported — the mixture is
monotone in `α` with no interior optimum:

```
alpha      0.0     0.2     0.4     0.6     0.8     1.0
prior-free 0.5750  0.5848  0.5928  0.5957  0.5972  0.5979
```

so candidate **I (mixture of semantic and discriminative logits) is REJECTED on
evidence**: blending only interpolates, it never exceeds the classifier.

| rank | candidate | input | computation | new params | fixes | preserves | new failure mode | expected benefit | cost |
|---|---|---|---|---|---|---|---|---|---|
| **1 (PRIMARY)** | **C — hybrid: CLIP-initialised *learnable* prototypes + nonlinear relational projection** | `rel_feat` (n,768) | `g(r) = MLP(r)`; `logits = normalize(g(r)) @ normalize(P)^T`, `P` (51,768) **initialised from `pred_emb`**, trainable, anchored by `λ‖P − P₀‖²` | ~0.6M (MLP) + 39,168 (`P`) | the fixed-linear bottleneck (§6 row 5) *and* the rank-1.41 degeneracy (row 4) | open-vocabulary scoring (P stays in CLIP space, anchored) | `P` can drift off CLIP semantics if `λ` too small; new hyper-parameter | up to ~`+0.023` on WPRD by analogy with row 5 | one training run at the `C1` budget |
| **2 (FALLBACK)** | **A′ — deploy the existing trained classifier head, prior-free** | `rel_feat` | `rowstd(predicate_classifier(r) + 0.25·tanh(bias_res(r)))`, `adaptive_prior` **off** | **0 new** | the fixed-linear bottleneck | nothing semantic — closed 51-way vocabulary | loses open-vocabulary capability; head was trained under a different composition | `+0.0229552` **already measured** | **no training at all** — one endpoint re-run |
| 3 | B — CLIP-initialised learnable prototypes, **linear** | `rel_feat` | `normalize(r) @ normalize(P)^T`, `P` trainable | 39,168 | prototype rigidity only | open-vocab | still linear | ≥ 9.2%, ≤ the linear ceiling (unknown until Pilot 0) | one training run |
| 4 | G — asymmetric subject/object readout | `rel_feat` + roles | re-enable SPOA at eval, or an order-aware head | 0 (re-enable) or small | train/eval parity defect (§11-K) | everything | changes every existing WPRD anchor's comparability | unknown | endpoint re-run |
| 5 | H — geometry-aware predicate readout | `rel_feat` + `geom_feat_proj` | concatenate geometry at the head | ~0.6M | possible geometry dilution | geometry gain | risks making the head a box classifier (§13 audit item) | unknown | one training run |
| — | D/E — feature-side adapter / low-rank adapter | `rel_feat` | `r + α·adapter(r)` | 1.18M | **nothing** | — | — | **measured: 0.4%** | **REJECTED on evidence** |
| — | I — logit mixture | both heads | `α·cls + (1−α)·text` | 0 | nothing | — | — | **measured: monotone, no interior optimum** | **REJECTED on evidence** |
| — | F — relation-conditioned predicate decoder | `rel_feat` + context | a full decoder | large | unknown | — | violates §7 minimality | unknown | large — **deferred** |

# 11. Primary proposal, 12. Minimal fallback

**PRIMARY (candidate C).** Replace `score(rel_feat, pred_emb)` with

```
logits = normalize(MLP(rel_feat)) @ normalize(P)^T ,
         P ∈ R^{51×768}, P initialised = pred_emb, trainable,
         loss += lambda_proto * || P - pred_emb ||_F^2
```

`MLP = LayerNorm(768) → Linear(768,768) → GELU → Linear(768,768)` — deliberately
the **same shape** as the existing `text_space_projection`, so the change is
"make the adapter's output the thing that is scored, and let the prototypes
move", not a new subnetwork.

**FALLBACK (candidate A′).** Change nothing in the model. Evaluate `C1` with
`eval_sgg_predicate_ensemble_alpha = 1.0` and `adaptive_prior_enabled = false`.
Zero new parameters, **zero training**, one endpoint. Its value is already known
(`0.5979433958`), so it functions as a *calibration of the pilot machinery*
rather than a discovery.

# 13. Exact intervention

| | |
|---|---|
| files to modify | `openvocab_rel/models/relational_model.py` (add `predicate_prototypes` + `readout_v2_enabled` branch in `text_predicate_logits`), `openvocab_rel/config.py` (3 flags), `openvocab_rel/train.py` (arg parsing + `lambda_proto` term) |
| new flags | `readout_v2_enabled: bool = False`, `readout_v2_proto_anchor: float = 1e-3`, `readout_v2_hidden_mult: float = 1.0` |
| default | **off** — the `C0`/`C1` path must remain bit-identical when the flag is false |
| geometry | **untouched.** `geom_input_pixel_space` and `geom_fourier_scale` keep their `C1` values |
| training budget | identical to `C1`: 3 epochs × 12,000, batch 6 × accum 4, lr 2e-5 cosine, seed 1234 |
| initialisation | the same frozen historical checkpoint, sha `8845c3af…` |

# 14. Tensor contract (to be asserted by tests before any GPU run)

```
rel_feat            (n_pairs, 768)  float32, finite
MLP(rel_feat)       (n_pairs, 768)  float32, finite
P                   (51, 768)       trainable, P[i] finite
logits              (n_pairs, 51)   float32, finite
rowstd(logits)      (n_pairs, 51)   row mean ~0, row std ~1
model_term          (n_pairs, 50)   after foreground slice
at init:  || logits_v2 - logits_v0 ||_inf  <  1e-4     (P = pred_emb, MLP ~ identity-ish)
flag off: logits_v2 == logits_v0 bit-exactly
```

# 15. Pilot design

**Pilot 0 — CPU only, no GPU, no training.** Fit a *linear* and an *MLP* readout
on the frozen `C1` `rel_feat` using **`p60`'s exact registered estimator**
(AdamW, hidden 256, 20 epochs, lr 2e-3, l2 1e-4, softmax CE, 5-fold CV on
validation, salt 0). This separates *learned prototypes* from *nonlinearity* —
the one thing §6 could not resolve — and calibrates the ceiling before any GPU
time. Expected cost: minutes to a few hours of CPU.

**Pilot 1 — the fallback, one endpoint, no training.** `alpha = 1.0`,
`adaptive_prior` off. Confirms the pipeline end-to-end against the already-known
`0.5979433958`.

**Pilot 2 — Readout v2, short.** 1 epoch × 12,000 samples at the `C1` geometry
contract, then the full registered endpoint. Purpose: *does a trained v2 head
beat the deployed v0 head on the same representation?* No hyper-parameter tuning.

# 16. Full experiment design (clean comparator, per Phase 13)

Same backbone, same geometry repair (`pixel_space=true`, `scale=0.01`), same
data, same split, same seed, same budget, same WPRD estimator, same 20,016
cells. **Only the readout changes.**

```
PRIMARY   delta_WPRD_readout = WPRD(v2) - WPRD(v0)
```

with `v0` = the `C1` seed-1 endpoint `0.5749881522134409`, and the prior-free
classifier `0.5979433958` as a **REFERENCE TARGET, not a success threshold**.

Secondary: weighted WPRD, R@50, mR@50, prior agreement.
Mechanistic: spatial vs non-spatial WPRD, cell-size bands, constant geometry
channels (must stay 0/8), fusion-gate stats, prior control (must stay exactly
0.5).

# 17. Failure criteria

- geometry regression: any constant geometry channel returns, or the `C1`
  geometry gain is not preserved → **INVALID**;
- prior control ≠ 0.5000 → **INVALID**;
- `P` drifts so far from `pred_emb` that open-vocabulary scoring is destroyed
  (report `mean cos(P, pred_emb)`) → **degenerate to candidate A′**;
- WPRD gain accompanied by a large rise in prior-argmax agreement →
  **calibration effect, not a readout fix**;
- gain present only on cells with a single-row side → **macro artefact**, per the
  seed-1 mechanism audit.

# 18. Success criteria

`delta_WPRD_readout > 0` with a paired 95% CI excluding zero over the 20,016
cells, **and** the `C1` geometry signature preserved (0/8 constant channels,
spatial-group gain retained), **and** not explained by any §17 failure mode.
No numeric threshold is pre-committed here — this is a **new** question and its
threshold must be registered in its own pre-registration before the run.

# 19. Expected interpretation

If v2 approaches ~`0.598` it confirms the readout was the binding constraint and
that CLIP's semantics can be retained while fixing it. If v2 lands near `0.575`
the fixed-linear diagnosis is wrong and attention returns to the representation.
If v2 exceeds the prior-free classifier, the hybrid is doing something neither
pure head does, and that would need its own explanation before being believed.

# 20. Rollback

Every change sits behind `readout_v2_enabled`, default **false**. With the flag
off the forward pass must be bit-identical to `C1`, asserted by a unit test.
Rollback = set the flag false, or `git revert` the implementation commit; no
checkpoint, dump, or registered document is touched by the implementation.

---

## Status

**Design only. Not registered, not implemented, not run.** Implementation must
wait for (a) the seed-2 replication to be validated and documented, and (b) its
own pre-registration. The registered `C1` result and threshold are unchanged.
