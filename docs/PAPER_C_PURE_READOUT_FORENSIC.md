# Paper C — readout bottleneck: forensic audit

Analysis-only. **No training, no GPU, no new experiment, no probe fitted, no
architectural change.** Every number is either (a) read from an existing
artifact, or (b) a re-derivation using the **registered** WPRD estimator
(`tools/within_pair_discrimination.py::wprd`, `cap=64`, macro, the same 20,016
cells) applied to logits **already stored** in the endpoint dumps. No model was
executed.

State locked at `7dd1bd3`; working tree clean.

> **Headline.** The readout hypothesis is **confirmed, and it is larger than the
> geometry repair.** The same `rel_feat`, read by the model's own trained
> classifier head, scores **0.5994**; read by the deployed frozen-CLIP cosine it
> scores **0.5750**. That gap — **+0.0244** — is ~2.9× the entire `C1` geometry
> gain (+0.0083), is stable across both arms, and is **not** explained by prior
> leakage. Under its own classifier head, repaired PURE (0.5994) now **matches
> and slightly exceeds** the 20-dimensional geometry-only probe (0.5976).

`C1` remains **POSITIVE DIRECTION, THRESHOLD NOT MET**. Nothing here changes it.

---

# 1. Current state

| | |
|---|---|
| `C0` WPRD (deployed text head) | `0.5667271196244782` |
| `C1` WPRD (deployed text head) | `0.5749881522134409` |
| `delta_WPRD` | `+0.008261032588962776` |
| paired 95% CI | `[+0.0038311178, +0.0126973772]` |
| registered threshold | `+0.010` — **not met** |

Locked and unmodified: registration, `PAPER_C_C0_RESULT.md`,
`PAPER_C_C1_RESULT.md`, `PAPER_C_C1_MECHANISM_AUDIT.md`, checkpoints, dumps,
training code, evaluation code. One factual correction was made to
`PAPER_C_PURE_COMPLETE_ARCHITECTURE.md` §11 and is disclosed in §15 below.

---

# 2. The full readout path

From `rel_feat` to WPRD. `dim = 768`, 51 predicate columns (50 foreground + 1
background), `img_res = 336`.

| # | operation | source | shape | trainable | linear? | norm | residual | prior? | order preserved? | geometry present? | image present? | loss risk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | `rel_feat` leaves the decoder | `relational_model.py:552 out_norm` | `(n,768)` | yes | — | **LayerNorm** | — | no | yes (via geom + edge cat) | yes | yes | per-row scale erased by LayerNorm |
| R1a | text-space projection | `:851-857 text_relation_features` | `(n,768)` | yes | no | LayerNorm | `+0.35·proj` | no | yes | yes | yes | **DISABLED at endpoint** — returns `rel_feats` unchanged |
| R1b | **text head (deployed)** | `:984-990 score` | `(n,51)` | **`pred_emb` frozen** | **linear + L2 norm** | `normalize(rel)`, `normalize(emb)` | — | **no** | only as far as `pred_emb` span allows | ditto | ditto | **projection onto 51 fixed CLIP directions; row magnitude discarded** |
| R2a | classifier | `:606-612 predicate_classifier` | `(n,51)` | yes | **no** (GELU) | LayerNorm | — | no | yes | yes | yes | learned 51 directions, dropout 0.1 |
| R2b | bias residual | `:958-961` | `(n,51)` | yes | no (tanh) | LayerNorm | `+0.25·tanh(...)` | no | yes | yes | yes | bounded by tanh |
| R2c | **adaptive prior** | `:962-968` | `(n,51)` | gate trainable | no | — | `+1.0·gate(rel)·centred_prior` | **YES** | n/a | n/a | n/a | **injects the prior into the "model" head** — see §6 |
| R3 | branch normalisation | `evals.py::_normalize_eval_logits` over all 51 cols | `(n,51)` | no | linear | standardise | — | no | — | — | — | order-preserving |
| R4 | composition | `model_term = α·cls_norm + (1−α)·text_norm`, **α = 0** | `(n,51)` | no | linear | — | — | no | — | — | — | **α = 0 discards R2 entirely** |
| R5 | deployed score | `+ 3.75 · log_prior` (`freq_bias_alpha`) | `(n,51)` | no | linear | — | — | **YES** | — | — | — | drives argmax (R@50) |
| R6 | WPRD | `within_pair_discrimination::wprd` | 20,016 cells | no | rank-based | — | — | **cancels exactly** | — | — | — | macro over cells |

**The decisive structural facts.**

1. **The deployed model term is `normalize(rel_feat) @ normalize(pred_emb)ᵀ`** —
   a projection of a 768-dimensional feature onto **51 fixed, never-trained CLIP
   text directions**, with the row's magnitude discarded by `F.normalize`. Any
   relational information outside the span of those 51 directions is invisible
   to the endpoint.
2. **The classifier head reads the same `rel_feat`** through learned weights
   (`LayerNorm → Linear(768,768) → GELU → Dropout(0.1) → Linear(768,51)`), and is
   **discarded at `α = 0`**.
3. **WPRD is prior-free by construction at R6**, so R5's prior cannot affect the
   primary endpoint. It dominates R@50.

---

# 3. Estimator audit — what each number actually measures

| | estimator | input | target | population | training mode | split | cross-fitting | normalisation | output |
|---|---|---|---|---|---|---|---|---|---|
| `p60 A_relfeat` | AdamW MLP, hidden 256, 20 ep, lr 2e-3, l2 1e-4, softmax CE | historical `rel_feat` (769) | 50 predicate classes | 132,556 val rows / **20,016 cells** | probe fitted | validation | **5-fold CV, salt 0** | probe-internal | 51 logits |
| `p60 B_geometry` | **identical** | 20 raw geometry dims | same | same | probe fitted | validation | **5-fold CV, salt 0** | probe-internal | 51 logits |
| `p60 C_fusion` | identical | 789 = 769 + 20 | same | same | probe fitted | validation | 5-fold CV | probe-internal | 51 logits |
| `p60 ref_text_head` | **none** — stored logits | historical model | — | same | — | — | — | `_normalize_eval_logits` | 51 logits |
| `p60 ref_classifier_head` | **none** — stored logits | historical model | — | same | — | — | — | ditto | 51 logits |
| `C0`/`C1` text head | **none** — stored logits | `C0`/`C1` model | — | **same 20,016 cells** | trained head | **train split**, eval on val | n/a (out-of-sample) | ditto | 51 logits |
| `C0`/`C1` classifier head | **none** — stored logits | same `rel_feat` | — | same | trained head | train split | n/a | ditto | 51 logits |

Downstream of the 51 logits **every row uses the identical registered WPRD**
(`cap=64`, macro, same 20,016 cells, prior control exactly 0.5).

### Comparison classification

| comparison | class | justification |
|---|---|---|
| **text head vs classifier head, same arm** | **DIRECT** | identical dump, identical `rel_feat`, identical estimator; only which stored head is read differs |
| `C0` vs `C1`, same head | **DIRECT** | the registered paired design |
| `p60 A_relfeat` vs `p60 B_geometry` | **DIRECT** | p60's own matched design, gates 5/5 |
| **classifier head vs `B_geometry`** | **INDIRECT, but empirically calibrated** | see below |
| deployed text head vs `B_geometry` | **INDIRECT** | same calibration, different readout |

**The calibration that makes the indirect comparison usable is p60's own.** On
the historical checkpoint, on the same 20,016 cells:

```
A_relfeat        (5-fold CV MLP probe on rel_feat)   0.5731767780068251
ref_classifier_head (the model's trained head)        0.5728057489650741
difference                                            0.0003710290417510
```

A cross-fitted probe on `rel_feat` and the model's own train-fitted classifier
head agree to **3.7e-04**. The protocol difference (CV-on-validation probe vs
train-fitted head) is therefore worth ~0.0004 for this quantity — **measured, not
assumed**. That is what licenses comparing a trained head against `B_geometry`.

**Two limits stated plainly.** (i) The calibration was established on the
historical checkpoint only; that it transfers to `C1` is an inference, not a
measurement, and confirming it would require fitting a probe on `C1`'s `rel_feat`
— **a new experiment, not run**. (ii) `p60` is a frozen-encoder readout study; it
cannot say what a jointly trained encoder would do.

---

# 4. Candidate bottlenecks

The decisive new measurement, using the registered estimator on the stored heads:

| | historical (`p60`) | `C0` | `C1` |
|---|---|---|---|
| **deployed text head** | 0.5541946637 | 0.5667271196 | **0.5749881522** |
| **classifier head (discarded)** | 0.5728057490 | 0.5909271674 | **0.5994290264** |
| **readout gap (cls − text)** | **+0.0186110853** | **+0.0242000478** | **+0.0244408742** |
| MLP probe on `rel_feat` | 0.5731767780 | — | — |
| MLP probe on 20 geometry dims | 0.5976140505 | — | — |
| `C_fusion` probe | 0.5922138929 | — | — |
| prior control | 0.5 | 0.5 | 0.5 |
| shuffled null | 0.4992158539 | — | — |

Geometry repair, measured through **both** heads:

```
text head   C1 - C0 = +0.0082610326      (the registered endpoint)
classifier  C1 - C0 = +0.0085018590
```

**The repair adds the same ~+0.0084 to both heads.** The readout gap is
orthogonal to it and ~2.9× larger.

| # | candidate | evidence FOR | evidence AGAINST | confidence |
|---|---|---|---|---|
| **D/E** | **predicate readout: the deployed text head under-decodes `rel_feat`** | same `rel_feat` → 0.5994 (classifier) vs 0.5750 (text); gap +0.0244, replicated in three checkpoints (+0.0186, +0.0242, +0.0244); not prior-driven (§6) | the text head is the *registered* protocol and is provably prior-free; the classifier's extra freedom includes a prior-gated term whose contribution is bounded but not eliminated | **HIGH** |
| **A** | relational representation itself | `p60`: historical `rel_feat` probe ceiling 0.5732 **< geometry 0.5976** — a genuine representation deficit at that time; `p57`: `dx_rel` decodable at R² 0.052 | `C1`'s classifier head reads 0.5994 ≥ `B_geometry` 0.5976 — the deficit appears **closed** after the repair | **MEDIUM** (was HIGH pre-`C1`) |
| **J** | order/direction representation | `sem_feat = normalize(sub+obj)` is symmetric (`:429`); SPOA role machinery **disabled at endpoint**; gain concentrated in directional predicates | edge-layer concatenation is order-bearing (§8); no direct measurement of order confusion exists | **MEDIUM-HIGH** |
| **F** | frequency-prior composition | prior amplitude is **5.21×** the model term (α·prior std 5.179 vs 0.9937) | **the prior cancels exactly in WPRD** — control reads 0.5 in both arms; it cannot affect the primary endpoint | **HIGH for R@50, ABSENT for WPRD** |
| **G** | calibration / temperature | `adaptive_prior` injects `gate(rel)·prior` into the classifier head — a non-cancelling channel | empirical signature absent (§6): the advantage does not grow with prior contrast | **LOW as an explanation; MEDIUM as an auditing defect** |
| **B** | semantic/geometry fusion | `C_fusion` (0.5922) < `B_geometry` (0.5976) — concatenating 768-d onto boxes *hurts* under a matched probe | that is a probe result on the historical encoder, not the trained fusion | **MEDIUM** |
| **C** | edge-layer relational update | geometry gets 768 of 3072 input dims, ungated, ×2 (§8) | `p70`: the edge path was worth −0.0038 when removed — live, not inert | **LOW as a bottleneck** |
| **H** | dimensionality | text head projects 768 → **51 fixed** directions | the classifier also outputs 51 and does far better, so 51 is not the binding limit | **LOW** |
| **I** | normalisation | `F.normalize` at R1b discards row magnitude; `geom_mlp` ends in `LayerNorm` (`:260`), erasing per-row geometry scale before fusion | rank-based WPRD is invariant to monotone per-row scaling in many cases | **LOW-MEDIUM** |

---

# 5. Subject–object directionality

**Where order enters, in graph order (endpoint configuration):**

| stage | operation | order-bearing? |
|---|---|---|
| pair index select (`:727-728`) | `sub_feat`, `obj_feat` | yes — but only as two separate tensors |
| `_semantic_pair_feature` (`:422-429`) | `normalize(sub_norm + obj_norm)` | **NO — symmetric.** `asymmetric_pair_fusion_enabled = False`; `config.py:52-54` states it "cannot distinguish (a,pred,b) from (b,pred,a) on its own" |
| **`geom_feats_torch` (`geometry.py:34-62`)** | `dx=(c2x−c1x)/w1`, `rw=log(w2/w1)`, `ar1`/`ar2`, `a1`/`a2` | **YES — FIRST order-bearing pair operation.** Under swap: `rw→−rw`, `rh→−rh`, `ar1↔ar2`, `a1↔a2`, `dx` rescaled and sign-flipped |
| fusion (`:461`) | `gate·sem + (1−gate)·geom_norm` | order-bearing **only via `geom_norm`** |
| `rel_seed(fused_feat)` (`:487`) | seeds `rel_feat` | inherits order only from geometry |
| SPOA (`:475-486`) | `subject_role_embedding` / `object_role_embedding`, separate `subject_branch` / `object_branch` | **strongly order-bearing — DISABLED AT ENDPOINT** (`c0_c1_evaluate.py:118`) |
| edge layers (`:48-68`) | separate `sub_attn` / `obj_attn`; `cat([sub_upd, obj_upd, rel_feat, geom_feat])` | **YES — order-bearing by concatenation position** |

**First operation where order matters: `geom_feats_torch`, stage 11 — the
geometry construction.** Everything pair-level upstream of it is symmetric.

**Is it sufficiently expressive?** At the endpoint there are exactly **two**
order-bearing routes: geometry, and the edge-layer concatenation. The explicit
role machinery designed for this job is switched off. And `rel_feat` is *seeded*
from `rel_seed(fused_feat)` whose only asymmetry is geometric — so the edge
layers do not add order to a neutral representation, they must repair one that
was already collapsed to a symmetric sum plus a geometry vector.

In `C0`, six of the eight geometry channels were identically constant, leaving
`dx, dy` as the sole order carriers at the seed. `C1` restores `rw, rh, ar1,
ar2, a1, a2` — the size and aspect relations. For `above`/`under`,
`in front of`/`behind`, `over`/`under`, the discriminating quantity within a
fixed `(subject, object)` category group is exactly a signed offset plus a size
relation. **This is the structural reason the gain landed where it did** (§9).

*Not claimed:* that order is currently confused. No artifact measures
subject/object swap behaviour; the endpoint dumps contain GT pairs only.

---

# 6. Prior / readout interaction

**The composed score, exactly** (`evals.py:1288-1289`, `:1441-1446`; `R3`–`R5`):

```
model_term = α·norm(cls_logits)/cls_temp + (1−α)·norm(text_logits)/text_temp
             with α = 0, cls_temp = text_temp = 1.0   ->   model_term = norm(text_logits)

score      = model_term + 3.75 · log_prior          (freq_bias_alpha = 3.75)

text_logits = normalize(rel_feat) @ normalize(pred_emb)ᵀ          (prior-free)
cls_logits  = predicate_classifier(rel_feat)
            + 0.25 · tanh(bias_residual_head(rel_feat))
            + 1.0 · gate(rel_feat) · centred_log_prior            (adaptive_prior_scale = 1.0,
                                                                    gate ∈ [0,1])
```

**Measured amplitudes** (GT rows, both arms identical to 4 s.f.):

| | `C0` | `C1` |
|---|---|---|
| model term std | 0.993690 | 0.993760 |
| `3.75 · log_prior` std | 5.179402 | 5.179402 |
| **ratio prior / model** | **5.212×** | **5.212×** |
| model term per-row range | 5.4041 | 5.3995 |
| `3.75 · log_prior` per-row range | 15.4406 | 15.4406 |

**Answer to the posed question, split in two because the answer differs:**

- **For WPRD (the primary endpoint): NO.** The prior is constant within an
  `(s,o)` group (max deviation `9.441e-05`), so it cancels exactly in WPRD's
  double difference — prior control reads `0.500000` in both arms. Amplitude is
  **irrelevant** to the primary endpoint. The readout cannot be losing relational
  information *to the prior* in WPRD, because WPRD never sees the prior.
- **For R@50 / mR@50: YES, decisively.** The prior carries 5.2× the amplitude and
  2.9× the per-row range of the learned term. This is why prior-argmax agreement
  is 92.0%/91.9%, why `p70` found R@50 flat to ±0.0005 across every geometry
  ablation, and why `C1`'s WPRD gain is nearly invisible in R@50 (+0.00096).

### The one real prior-leak channel, and its test

`cls_logits` contains `gate(rel_feat) · centred_prior`. Within a cell,
`prior[a] − prior[b] = Δ_ab` is a **constant**, but `gate_i` **varies per row**,
so `s_i = f_i + gate_i·Δ_ab` does **not** cancel. This is a genuine route by
which prior-derived signal could inflate the classifier head's WPRD, and the
un-calibrated logits were never stored, so it cannot be subtracted.

It can, however, be tested by signature: **if the advantage were prior-driven it
must grow with `|Δ_ab|`.** Measured on the existing dumps:

| quartile of `|Δ_ab|` | `C0` advantage | `C1` advantage |
|---|---|---|
| Q1 `[0.000, 0.693]` | `+0.028299` | `+0.030447` |
| Q2 `[0.693, 1.386]` | `+0.026372` | `+0.026539` |
| Q3 `[1.386, 2.314]` | `+0.025492` | `+0.019516` |
| Q4 `[2.314, 9.095]` | `+0.016706` | `+0.021434` |
| Spearman(advantage, `|Δ_ab|`) | `+0.0216` | `+0.0032` |
| **lowest decile** (`|Δ_ab| ≤ 0.2007`, 2,008 cells) | **`+0.031030`** | **`+0.026142`** |

**The advantage does not grow with prior contrast — it mildly *shrinks*, and it
is at its largest where the prior can contribute least.** The prior-leak
explanation is **rejected on its own signature**. This is a signature test, not
a decomposition: it bounds the leak's explanatory role, it does not prove the
term contributes nothing.

---

# 7. Classifier head vs text head

| | text head (deployed) | classifier head (discarded) |
|---|---|---|
| formula | `normalize(rel) @ normalize(pred_emb)ᵀ` | `classifier(rel) + 0.25·tanh(bias_res(rel)) + gate(rel)·prior` |
| source | `:984-990`, `:859` | `:606-612`, `:948-972` |
| reads the same `rel_feat`? | **yes** — `evals.py:1493-1495` documents exactly this | **yes** |
| trainable degrees of freedom | **`pred_emb` frozen**; `text_space_projection` exists but is **off at endpoint** | `768×768 + 768×51` + bias-residual + calibration gate |
| implicit class geometry | **yes — the 51 CLIP text directions**, fixed by language, not by the task | learned |
| magnitude of `rel_feat` | **discarded** (`F.normalize`) | retained (LayerNorm inside) |
| prior contribution | **none** | `gate·prior` — non-cancelling in WPRD |
| WPRD, `C1` | **0.5749881522** | **0.5994290264** |

**Does `ensemble_alpha` change readout or representation? Readout only.** It is
applied at `R4` on already-computed logits (`evals.py:1444-1446`); `rel_feat` is
identical for every α. Both heads are stored precisely so this question is
answerable without rerunning the model (`evals.py:1245-1248`).

**Is the text head's weakness explained by CLIP embedding collinearity? No —
already refuted.** `docs/READOUT_GEOMETRY_RESULT.md` (`p51`) tested exactly that
hypothesis and it failed: column collinearity is **uncorrelated** with the text
head's WPRD (Spearman −0.073, p = 0.294), the association is *stronger* for the
learned head, the classifier has ~3× *less* contrast budget yet better WPRD, and
`above` vs `under` has one of the text head's smallest budgets (0.1345) and its
highest WPRD (0.8002). **The gap is real and replicated; its cause is
unexplained.** That remains open (§14).

---

# 8. The edge-layer geometry path

```
geom_feat_proj = geom_mlp(fourier)                       (:451)   (n, 768)
   geom_mlp = Linear(512,256) → GELU → LayerNorm(256)
              → Linear(256,768) → LayerNorm(768)          (:255-261)

for edge_layer in self.edge_layers:                       (:540-542)   edge_layers = 2
    ...
    rel_delta = rel_update(cat([sub_upd, obj_upd, rel_feat, geom_feat], dim=-1))   (:65)
    rel_upd   = rel_norm(rel_feat + rel_delta)                                     (:66)

    rel_update = Linear(3072, 1536) → GELU → Linear(1536, 768)   (:41-45)
```

| question | answer |
|---|---|
| dimensionality geometry receives | **768 of 3072 input dims = 25%**, twice |
| interaction with semantic features | **concatenation**, then a 2-layer MLP — so multiplicative interaction is possible but not built in |
| concatenated? | **yes**, at a fixed slice position |
| transformed again? | yes — `rel_update`, then residual `rel_norm(rel_feat + rel_delta)` |
| gated? | **no.** `gate_proj` (`:57-58`) gates `rel_feat` into the attention *queries*; the geometry slice is never multiplied by any gate |
| directionality preserved? | yes — `geom_feat_proj` inherits the order-bearing channels, and the `cat` position distinguishes `sub_upd` from `obj_upd` |
| can normalisation erase scale? | **yes, and earlier than expected.** `geom_mlp` **ends in `LayerNorm(768)`** (`:260`), so per-row magnitude is largely removed *before* either consumer. The fusion path's `F.normalize` (`:456`) therefore adds little further loss. Absolute geometric scale survives only as a *pattern* across dimensions (via `a1 = log(w₁h₁)`, `rw = log(w₂/w₁)` upstream of the Fourier map), not as a magnitude |

**Is this path the reason `C1` improves while the gate stays flat?** **Partly,
and it cannot be more precisely apportioned from existing evidence.** The gate
is not an obstacle in the first place — at `gate ≈ 0.5` the fusion path already
passes geometry at half weight on a unit vector (`:461`). So there are two live
routes, and the only path-level causal evidence is `p70`, on the *historical*
checkpoint under the *broken* contract:

```
B_no_fusion_geom  0.5505989364   Δ = -0.0036   (fusion path removed)
C_no_edge_geom    0.5504243409   Δ = -0.0038   (edge path removed)
D_no_geom         0.5455294887   Δ = -0.0087   (both)
```

Nearly equal and roughly additive. **No experiment separates them under the
repaired contract.** Attributing the `C1` gain to the edge path specifically is
**NOT SUPPORTED**; the defensible statement is that both routes are live and the
flat gate never blocked either.

---

# 9. The spatial-predicate effect: representation or readout?

The two hypotheses:

- **S1 — representation:** `C1` added directional geometry to `rel_feat`, so
  spatial contrasts became decodable at all.
- **S2 — readout:** the readout finds spatial predicates easier to separate once
  directional geometry exists.

**The head decomposition separates them.** If S2 were doing the work, the gain
should depend on which readout is used. It does not:

```
text head    C1 - C0 = +0.0082610326
classifier   C1 - C0 = +0.0085018590     difference: 0.0002408
```

**The repair delivers the same gain through two structurally different
readouts** — a frozen cosine against fixed CLIP directions, and a learned
two-layer MLP. A readout-side explanation would have to produce that agreement
by coincidence.

**Conclusion: S1 (representation) is supported; S2 is not required and is not
supported.** The gain is an increase in what `rel_feat` *contains*, not in what a
particular head can extract. This is consistent with §5: the restored channels
are the model's principal order carriers at the seed, and directional predicates
are the ones that need them.

**The two effects are independent and additive-looking:**

```
readout loss   (cls − text)  ≈ +0.0244, essentially constant across arms
geometry gain  (C1 − C0)     ≈ +0.0084, essentially constant across heads
```

---

# 10. Macro vs weighted, and the practical scale of a readout gain

`C1` vs `C0`, from `PAPER_C_C1_MECHANISM_AUDIT.md`:

```
macro    Δ = +0.008261      weighted Δ = +0.003036      ratio 2.72
```

The same quantities for the **readout gap** (`C1`: classifier vs text):

```
macro    gap = 0.5994290264 − 0.5749881522 = +0.0244408742
weighted gap = 0.549558     − 0.537023     = +0.0125350000      ratio 1.95
```

**Quantified, not adjectivised:**

| quantity | macro | weighted | macro/weighted |
|---|---|---|---|
| geometry repair (`C1` − `C0`) | `+0.008261` | `+0.003036` | 2.72 |
| readout gap (cls − text) | `+0.024441` | `+0.012535` | **1.95** |
| ratio readout / geometry | **2.96×** | **4.13×** | — |

**The readout gap is less macro-inflated than the geometry gain.** Its
macro/weighted ratio is 1.95 against the repair's 2.72, so a larger share of it
survives comparison-weighting — it is *not* a small-cell artifact to the degree
the geometry gain is. On the weighted metric, which weights each cell by its
`na·nb` comparisons, the readout gap is **4.1× the geometry gain**.

The geometry gain's macro inflation is fully characterised in the mechanism
audit: 30.53% of cells are 1-vs-1 and contribute 37.50% of the net gain while
holding 1.10% of all pairwise comparisons; 53.22% of cells are exactly unchanged;
gross positive `+1658.35` against gross negative `−1492.997` (cancellation
0.9003); and the gain is concentrated in the registered spatial group
(`+0.022055`, and `+0.019987` on cells with `na·nb ≥ 25`, CI excluding zero).

**Neither number is "large".** The repair is a standardised effect of ~0.027 per
cell. The readout gap is ~3× that and better-behaved under weighting, but it is
still a fraction of the distance from the 0.4992 shuffled null to a perfect
discriminator.

---

# 11. The PURE Complete readout contract

| | requirement | status | evidence |
|---|---|---|---|
| **A** | **preserve subject–object directionality** | **DEFECTIVE** | `sem_feat` symmetric (`:429`); SPOA role machinery disabled at the endpoint; only geometry and the edge `cat` carry order (§5) |
| **B** | **expose geometry-derived relational signal** | **SATISFIED** (post-`C1`) | 8/8 channels live; `+0.0084` through both heads |
| **C** | **retain image-conditioned relational signal** | **UNCERTAIN** | CLIP unfrozen and the deformable router is live, but no artifact isolates the image contribution to WPRD; geometry alone (`B_geometry`, 20 dims) still matches the full model |
| **D** | **separate learned signal from the frequency prior** | **SATISFIED for the deployed head; DEFECTIVE for the classifier head** | text head is provably prior-free; `cls_logits` carries `gate(rel)·prior`, non-cancelling in WPRD (§6) |
| **E** | **adequate predicate discrimination capacity** | **DEFECTIVE** | same `rel_feat`: 0.5994 vs 0.5750 across the two heads |
| **F** | **auditable under WPRD** | **SATISFIED** | prior control exactly 0.5; bit-exact reproduction; exact cell pairing |
| **G** | **train/eval parity** | **DEFECTIVE** | three trained components (SPOA, text-projection, relationness) disabled at the endpoint (`c0_c1_evaluate.py:118-121`) |
| **H** *(added; evidenced)* | **the deployed head must not be strictly dominated by a stored alternative** | **DEFECTIVE** | the discarded classifier head is better by `+0.0244`, replicated over three checkpoints |
| **I** *(added; evidenced)* | **per-row magnitude must not be discarded before class scoring unless shown irrelevant** | **UNCERTAIN** | `F.normalize` at `:989`; `geom_mlp` LayerNorm at `:260`; no measurement of what magnitude carried |

---

# 12. Bottleneck ranking

**HIGH CONFIDENCE**

1. **Predicate readout capacity (the deployed head).** The same `rel_feat` yields
   `+0.0244` more WPRD through the model's own classifier head; replicated at
   `+0.0186` / `+0.0242` / `+0.0244` across three checkpoints; the effect is
   ~3× the geometry repair and *less* macro-inflated; the prior-leak explanation
   is rejected on its signature (§6) and the embedding-collinearity explanation
   was refuted by `p51`. **This is now the dominant measured gap.**
2. **Prior dominance — for `R@50`/`mR@50` only, and absent for WPRD.** 5.21×
   amplitude; explains why no geometry intervention moves the deployed metric.

**MEDIUM CONFIDENCE**

3. **Relational order representation.** Symmetric `sem_feat`, SPOA disabled at
   the endpoint, order carried at the seed only by geometry. Structurally
   implicated by §5 and consistent with §9, but never directly measured.
4. **Train/eval architecture mismatch.** Three trained components are switched
   off at evaluation; this is a deliberate anchor-consistency choice, but it
   means no reported number describes the trained model.
5. **Semantic/geometry fusion.** `C_fusion` (0.5922) < `B_geometry` (0.5976) —
   adding the 768-d representation to boxes hurts under a matched probe.
6. **Auditability of the classifier head.** Its `gate·prior` term makes it
   unusable as a clean "model term" without a decomposition that the current
   dumps do not permit.

**LOW CONFIDENCE**

7. Relational representation itself — was HIGH before `C1`, now apparently
   closed (`C1` classifier 0.5994 ≥ `B_geometry` 0.5976), conditional on the §3
   calibration transferring.
8. Edge-layer capacity; 51-way dimensionality; normalisation scale loss.

**No intervention is proposed.**

---

# 13. What seed 2 answers

| question | answers it? |
|---|---|
| **A — is the geometry repair reproducible?** | **YES. This is what seed 2 is for**, and it remains the only outstanding registered obligation |
| **B — is the architecture complete enough to exploit the restored geometry?** | **NO.** §4 already answers B: it is not — the deployed readout leaves `+0.0244` unread. Seed 2 cannot change that |
| **C — what change would remove the next bottleneck?** | **NO.** Out of scope for any seed |

Seed 2 is **not** a second attempt at `+0.010`; the registered rule stands and
re-running until a seed crosses it would be a forking-paths failure. **Not run.**

---

# 14. What remains unknown

1. **Why the text head under-decodes `rel_feat`.** The gap is replicated;
   `p51` refuted both natural explanations (embedding collinearity, contrast
   budget). Mechanism open.
2. **Whether the classifier head's advantage is fully clean.** Its `gate·prior`
   term cannot be subtracted from the existing dumps; §6 bounds its role by
   signature only.
3. **Whether the §3 probe/head calibration (3.7e-04) transfers to `C1`.**
   Establishing it needs a probe fit on `C1`'s `rel_feat` — a new experiment.
4. **Whether `C1` closed the representation deficit or merely narrowed it.**
   Depends on (3).
5. **Which geometry route carries the gain** — fusion or edge. Unresolved (§8).
6. **Whether order is actually confused** — no swap-based artifact exists.
7. **Whether the endpoint's disabled components would change any conclusion.**
8. **Whether the repair replicates across seeds.**

---

# 15. Final architectural judgment

> **After the `C1` geometry repair, the dominant remaining limitation in PURE is
> the deployed predicate readout — the frozen-CLIP cosine head — because the
> identical `rel_feat` yields WPRD 0.5994 through the model's own trained
> classifier head and only 0.5750 through the deployed head, a gap of
> `+0.0244` that is ~3× the entire geometry repair, replicates across three
> checkpoints, survives comparison-weighting better than the repair does, and is
> explained neither by prior leakage (§6) nor by CLIP embedding geometry
> (`p51`).**

Two supporting statements, at their proper strength:

- **The representation deficit that motivated this cycle appears closed.** On the
  historical checkpoint `rel_feat`'s probe ceiling was 0.5732, *below* raw
  geometry's 0.5976. After the repair, `C1`'s own classifier head reads 0.5994 —
  at or slightly above the geometry probe. **This is conditional** on the
  probe/head calibration of §3 transferring to `C1`, which is an inference, not
  a measurement.
- **The prior is not a WPRD bottleneck and never was** — it cancels exactly. It
  *is* a decisive `R@50` bottleneck at 5.21× the model term's amplitude, which
  is why no geometry result has ever moved the deployed metric.

### Correction to a previously committed record

`docs/PAPER_C_PURE_COMPLETE_ARCHITECTURE.md` §11 stated that after the repair
"the full model still does not beat boxes". That was measured on the **deployed
text head** (0.5750 vs 0.5976) and is correct for that head, but as written it
generalises to the architecture, which this audit shows is wrong: through its own
classifier head the repaired model reads **0.5994 ≥ 0.5976**. A one-line
correction with a pointer to this document has been added at that location. No
other existing record was modified, and no result, threshold or verdict changed.

---

## Method note

CPU only. `runs/`, `checkpoints/` and all `.pt` dumps opened read-only; none
modified. The head comparison, amplitude audit and prior-leak test apply the
**registered** `within_pair_discrimination::wprd` to logits already present in
the endpoint dumps — `cls_logits` is stored precisely so this question can be
asked without rerunning the model (`evals.py:1245-1248`). No model was executed
and no probe was fitted; fitting a probe on `C1`'s `rel_feat` would be a new
experiment and was **not** done.
