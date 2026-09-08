# Paper C `C1` — forensic mechanism audit

Analysis-only addendum to `docs/PAPER_C_C1_RESULT.md`. **No training, no GPU
job, no new experiment.** Every number below is derived on CPU from artifacts
that already existed on disk (`runs/eval_C0/`, `runs/eval_C1/`,
`configs/predicate_metadata_vg150.json`, `datasets_vg150_clean/validation.jsonl`)
plus direct reading of the model source.

> **The `+0.00826` improvement is real under the registered paired WPRD
> analysis, but the originally hypothesised "adaptive fusion gate" mechanism was
> not observed.**

This audit **does not** change the registered result, threshold, or verdict.

---

## A. Locked result (unchanged)

| | |
|---|---|
| `C0` WPRD | `0.5667271196244782` |
| `C1` WPRD | `0.5749881522134409` |
| `delta_WPRD` | `+0.008261032588962776` |
| paired 95% CI | `[+0.0038311178, +0.0126973772]` |
| `P(delta > 0)` | `1.000` |
| paired cells | `20,016` |
| `delta >= +0.01` | **NO** |
| registered verdict | **POSITIVE DIRECTION, THRESHOLD NOT MET** |

Working tree clean at `a5ce243` before and after this audit. Both arms'
`cell_values` were re-derived from the raw dumps and again matched the stored
`result.json` **element-for-element**; cell identity and per-cell row membership
are identical across arms. The AUC helper used here was verified equal to
`within_pair_discrimination.auc` on 200 tie-heavy cases before use.

---

## B. Gain distribution — the effect is broad, thin, and heavily cancelled

| | |
|---|---|
| cells improved | 4,799 (23.98%) |
| cells unchanged (exactly 0) | **10,653 (53.22%)** — independently confirmed |
| cells worsened | 4,564 (22.80%) |
| mean | `+0.008261` |
| median | `0.000000` |
| std across cells | `0.3096` |

Quantiles are coarse — `p5 = −0.500`, `p10 = −0.250`, `p25…p75 = 0.000`,
`p90 = +0.333`, `p95 = +0.500`, `p99 = +1.000` — because most cells are tiny
and their AUC can only take a few values.

**The decisive structural fact: gross masses nearly cancel.**

```
gross positive   +1658.350  over 4,799 cells   (mean +0.3456)
gross negative   -1492.997  over 4,564 cells   (mean -0.3271)
net              + 165.353                      |neg| / pos = 0.9003
```

The headline `+0.00826` is the ~10% residual of two opposing masses each ~200x
larger. This is **A. broad and distributed**, not B. concentrated: no small set
of cells carries it (the top-100 cells "explain 60% of the net" only because the
net is minuscule relative to the gross; a 10% two-sided trim already removes 44%
of the mean).

### What macro-WPRD is actually averaging

| | |
|---|---|
| 1-vs-1 cells | **6,110 (30.53%)** |
| cells with at least one side of size 1 | **15,062 (75.25%)** |
| total pairwise comparisons | 556,672 |

| size band (`na*nb`) | cells | % cells | % of all comparisons | mean delta | share of net |
|---|---|---|---|---|---|
| 1 | 6,110 | 30.53% | 1.10% | `+0.010147` | 37.50% |
| 2–4 | 5,916 | 29.56% | 2.90% | `+0.005685` | 20.31% |
| 5–9 | 2,575 | 12.86% | 3.07% | `+0.017981` | 28.00% |
| 10–24 | 2,592 | 12.95% | 7.15% | `+0.007161` | 11.23% |
| 25–99 | 1,879 | 9.39% | 16.44% | `+0.001519` | 1.73% |
| 100+ | 944 | 4.72% | **69.34%** | `+0.002177` | **1.24%** |

Macro-WPRD gives a 1-vs-1 cell the same weight as a 64-vs-64 cell. The cells
holding 69% of all actual comparisons contribute 1.2% of the macro gain. This is
why `macro delta = +0.008261` while `weighted delta = +0.003036`.

### How much of the headline survives each restriction

| restriction | cells | mean delta | 95% CI | `P(>0)` |
|---|---|---|---|---|
| all cells (**registered headline**) | 20,016 | **`+0.008261`** | `[+0.0039, +0.0126]` | 1.000 |
| drop 1-vs-1 cells | 13,906 | `+0.007432` | `[+0.0031, +0.0119]` | 0.999 |
| drop any cell with a side of 1 | 4,954 | `+0.002804` | `[−0.0024, +0.0080]` | 0.853 |
| `min side >= 4` | 1,617 | `−0.000984` | `[−0.0071, +0.0049]` | 0.359 |
| `na*nb >= 25` | 2,823 | `+0.001739` | `[−0.0037, +0.0072]` | 0.723 |
| `na*nb >= 100` | 944 | `+0.002177` | `[−0.0043, +0.0087]` | 0.737 |

**The aggregate effect does not survive restriction to cells large enough for a
per-cell AUC to be statistically meaningful.** Section C shows why: it is not
that the effect disappears, but that it is confined to one predicate family
which is a minority of the large cells.

### Frequency band

Head/body/tail is defined in the repository (`tools/cprime_analysis.py:125-129`:
head = 15 most frequent GT classes, body = next 20, tail = remainder), so the
term is used here in that registered sense only.

| band | cell-slots | mean delta |
|---|---|---|
| head | 30,951 | `+0.009359` |
| body | 6,625 | `+0.004670` |
| tail | 2,456 | `+0.004107` |

**This is not a long-tail effect.** The gain is largest on the head classes.

---

## C. Is the gain geometry-linked?

### C.1 Registered partition — yes, strongly

`configs/predicate_metadata_vg150.json` assigns every VG150 predicate a coarse
group and is already consumed by the training pipeline
(`--predicate_metadata_path`). It is the only non-invented geometry-relevant
partition in the repository, so it is used here as-is. Its own description
warns the groups are "coarse and used for role-swap filtering and long-tail
diagnostics, not as ground-truth semantic ontology" — that caveat carries over.

| registered group | cell-slots | mean delta |
|---|---|---|
| **spatial** | 21,797 | **`+0.015704`** |
| other | 690 | `+0.006110` |
| contact | 5,449 | `+0.001771` |
| action | 1,761 | `+0.000940` |
| part-whole | 132 | `−0.001475` |
| possession | 7,581 | `−0.002567` |
| pose | 2,432 | `−0.002595` |
| attribute | 190 | `−0.006028` |

Partitioning cells by whether **both** predicates are spatial:

| stratum | cells | mean delta | 95% CI | share of net gain |
|---|---|---|---|---|
| both spatial | 6,762 | **`+0.022055`** | `[+0.0141, +0.0294]` | **90.19%** |
| mixed | 8,273 | `+0.005322` | — | 26.63% |
| neither spatial | 4,981 | `−0.005583` | — | −16.82% |

Per-predicate, the ranking is exactly what a geometry repair predicts:

```
most improved   across +0.0567   above +0.0379   behind +0.0376
                in front of +0.0335   over +0.0237   near +0.0169
                on +0.0135   in +0.0124            (all 'spatial')
most harmed     on back of -0.0333  to -0.0292  carrying -0.0229
                has -0.0140  attached to -0.0132  wearing -0.0105
                sitting on -0.0093                (possession/contact/pose)
```

(`covering`, `+0.0918` over 230 cells, is the largest single mover and is
grouped `contact`, not `spatial` — a reminder the grouping is coarse.)

### C.2 The spatial effect is robust to cell size — the aggregate null is a mix effect

| size band | both-spatial mean delta (95% CI) | not-both-spatial mean delta (95% CI) |
|---|---|---|
| 1 | `+0.029762` `[+0.014, +0.046]` | `−0.002129` `[−0.015, +0.010]` |
| 2–4 | `+0.007587` `[−0.006, +0.021]` | `+0.004627` `[−0.006, +0.015]` |
| 5–9 | `+0.032692` `[+0.016, +0.050]` | `+0.010719` `[−0.001, +0.022]` |
| 10–24 | `+0.027972` `[+0.013, +0.043]` | `−0.001732` `[−0.011, +0.008]` |
| 25–99 | `+0.023645` `[+0.009, +0.038]` | `−0.006724` `[−0.015, +0.002]` |
| 100+ | `+0.009328` `[−0.009, +0.027]` | `+0.000549` `[−0.006, +0.008]` |

Restricted to statistically meaningful cells (`na*nb >= 25`):

```
both-spatial      685 cells   +0.019987   95% CI [+0.0084, +0.0317]   P(>0) 0.999
not-both-spatial  2,138 cells -0.004108   95% CI [-0.0104, +0.0019]   P(>0) 0.094
all big           2,823 cells +0.001739   95% CI [-0.0037, +0.0072]   P(>0) 0.723
```

**This is the audit's central positive finding.** The spatial-predicate gain
(`≈ +0.020`) is stable across every cell-size band and its CI excludes zero even
on large cells alone. The *aggregate* effect washes out on large cells only
because large cells are 76% non-spatial, and non-spatial cells trend negative.
So the intervention did not produce a diffuse across-the-board improvement: it
**helped spatial relations and mildly hurt non-spatial ones**, and the registered
macro endpoint reports the net.

### C.3 Exploratory per-cell geometric separability — null

A per-cell descriptive statistic (**exploratory, not registered**): using the
exact 336-space geometry the model saw, `sep = max_k |2·AUC_k − 1|` over the 8
channels, computed on the same rows each WPRD cell uses.

The model's boxes were reconstructed exactly — the dumps store original-image
pixel boxes, while the model consumes `obj_boxes_224`, so
`geometry.preprocess_boxes_to_clip224` was re-applied using the image dimensions
in `validation.jsonl`. **Verification gate passed:** the reconstruction
reproduces the endpoint's own recorded contract — `C0` 6/8 constant channels and
`C1` 0/8, per-channel stds matching to ~0.5%, and `C1 a1 max = 11.634222` against
the recorded `11.634222` (`= log(336²)`).

| population | cells | Spearman(delta, sep) |
|---|---|---|
| all cells | 20,016 | `+0.129` — **saturated**, `sep = 1.0` for 67.8% of cells |
| `min side >= 4` | 1,617 | **`−0.004`** |
| `min side >= 8` | 519 | **`+0.015`** |

`sep` is degenerate on small cells (with one row on a side, some channel
separates perfectly by construction), so the `+0.129` on all cells is a cell-size
artifact, not a geometry signal. **On cells where the statistic is meaningful
there is no association at all**, and the quartile means are non-monotonic.

**Reading:** the gain tracks *which relation is being discriminated* (a
predicate-family property) but **not** *how geometrically separable the
particular instances are*. That is weaker than a per-instance geometry mechanism
would predict, and it is reported as a negative result.

---

## D. The flat-gate puzzle — resolved: the premise was wrong

The `C1` result document treated the flat `fusion_gate` as evidence against the
mechanism. Reading the forward path shows that inference was mistaken on three
independent counts.

### D.1 A gate at 0.5 is maximum geometry mixing, not zero

`openvocab_rel/models/relational_model.py:461`:

```python
fused_feat = gate * sem_feat + (1.0 - gate) * geom_norm
```

Both operands are unit vectors — `sem_feat` from `_semantic_pair_feature`
(`F.normalize`, line 429) and `geom_norm = F.normalize(geom_feat_proj)`
(line 456). At `gate ≈ 0.5`, `(1 − gate) ≈ 0.5`: **geometry supplies half of the
fused representation by norm.** The gate is a *mixing coefficient*, not an
on/off switch. "Gate ≈ 0.5 in both arms" therefore means geometry was weighted
identically and substantially in both arms — it says nothing about whether the
*content* of `geom_norm` changed. It changed completely (Section E).

### D.2 The gate is explicitly regularised to 0.5 by the training objective

`relational_model.py:462-463` and `train.py:2413-2415`:

```python
mean_gate = gate.mean(dim=-1)
self.last_gate_reg = ((mean_gate - 0.5) ** 2).mean()
...
gate_reg_weight = float(getattr(cfg, "gate_regularizer_weight", 0.01))
if gate_reg is not None and gate_reg_weight > 0.0:
    loss_total = loss_total + (gate_reg_weight * gate_reg)
```

`gate_regularizer_weight = 0.01` was active in **both** arms (recorded in each
training log's `[ActiveBranches]`). The loss **penalises the gate for leaving
0.5**. A flat gate is the enforced, designed behaviour — it is not an
observation about geometry, and the hypothesis that restored geometry would make
the gate "open" was inconsistent with the model's own objective from the start.

### D.3 Geometry bypasses the gate entirely via the edge layers

`relational_model.py:540-542` feeds `geom_feat_proj` — the raw `geom_mlp` output,
**before** normalisation and **never** multiplied by `gate` — into every edge
layer:

```python
for edge_layer in self.edge_layers:
    sub_feat, obj_feat, rel_feat, gate_strength = edge_layer(
        sub_feat, obj_feat, rel_feat, visual_tokens, geom_feat_proj)
```

and `ProgressiveEdgeConditionedLayer.forward` (line 65) concatenates it directly
into the relational update:

```python
rel_delta = self.rel_update(torch.cat([sub_upd, obj_upd, rel_feat, geom_feat], dim=-1))
rel_upd = self.rel_norm(rel_feat + rel_delta)
```

With `edge_layers = 2` (both arms), geometry enters the relational representation
at **three** points: once through the gated fusion and twice through ungated
concatenation. `fusion_gate` controls **one of three** injections.

**Answers to the posed questions:** (A) yes — the edge layers; (B) yes —
`geom_feat_proj` is concatenated at every edge layer; (C) yes — the units fix
changes the input to `geom_feats_torch` so its `clamp_min(1.0)` no longer binds,
and the bandwidth fix rescales the Fourier projection before `geom_mlp`;
(D) yes — the intervention acts *before* the `sin/cos` nonlinearity and before
`geom_mlp`; (E) yes — demonstrably, because `(1 − gate) ≈ 0.5` and because two of
three injections never touch the gate.

---

## E. The actual geometry computational path

| stage | source | changed by `C1`? | evidence |
|---|---|---|---|
| raw GT boxes | dump `obj_boxes` | **unchanged** | byte-identical across arms (verified) |
| box → 336-space | `geometry.preprocess_boxes_to_clip224` | **unchanged** | same function, same `img_res=336` in both |
| **units selection** | `relational_model.py:735` | **CHANGED — first divergence** | `C0` divides by `img_res`; `C1` does not |
| `geom_feats_torch` | `geometry.py:34-62` | **indirectly affected** | same code; `clamp_min(1.0)` binds on every w/h under `C0`, killing 6 of 8 channels; binds rarely under `C1` |
| clamp / `nan_to_num` | `relational_model.py:444-446` | indirectly affected | same bounds, different inputs |
| **Fourier projection** | `relational_model.py:448-450` | **CHANGED** | `2π · scale · x @ geom_B`; `scale` 1.0 → 0.01 |
| `geom_B` | `relational_model.py:254` | **unchanged** | `requires_grad=False`; std `9.90808391571045` bit-identical in both arms |
| `geom_mlp` | `relational_model.py:255-261` | indirectly affected (+ trained weights differ) | same architecture, different input distribution |
| `geom_norm` | line 456 | indirectly affected | unit-normalised either way; direction changes |
| fusion (gated) | line 461 | indirectly affected | mixing weight unchanged (`≈0.5`); mixed content changed |
| **edge layers ×2 (ungated)** | lines 540-542, 65 | indirectly affected | `geom_feat_proj` concatenated, gate-independent |
| `rel_feat` | line 552 | **CHANGED, large** | mean cosine `C0`↔`C1` = **0.8505** |
| predicate logits | text head, `alpha = 0` | changed | `model_term` differs |
| WPRD | `within_pair_discrimination::wprd` | changed | `+0.008261` |

**First point of divergence: `relational_model.py:735`** — the units branch, two
lines before `geom_feats_torch` is called. Everything downstream is
consequence.

### Representation drift is large

| | |
|---|---|
| cosine similarity `C0` vs `C1` `rel_feat` | mean **0.8505**, std 0.1144, min −0.2166, max 0.9763 |
| rows with cosine > 0.99 | **0.00%** |
| relative L2 drift `‖r1−r0‖/‖r0‖` | mean **0.5203** |

The relational representation moved substantially on essentially every row —
conclusive that the intervention propagated through the network despite the flat
gate.

**Caveat, and it is a serious one:** `C0` and `C1` are two independently trained
models. This drift conflates (i) the changed geometry input with (ii) three
epochs of divergent weight updates. It proves the arms differ; it does **not**
isolate the geometry input as the cause. Per-cell drift barely predicts per-cell
gain (Spearman `+0.012`), which is consistent with most of the drift being
generic training divergence rather than geometry-specific signal.

---

## F. Evidence-strength table

| # | claim | strength | basis |
|---|---|---|---|
| 1 | **geometry representation restoration** | **STRONG** | 6/8 → 0/8 constant channels, recorded by the endpoint and independently reproduced from the dumps through a passing verification gate; `p73` Part C bandwidth curve `−0.33 → +0.98` mean R² at the selected scale on the same frozen `geom_B` |
| 2 | **decoder access to geometry** | **STRONG** | source-level: `(1 − gate) ≈ 0.5` at the gated injection, plus two ungated concatenations into the edge layers; geometry is structurally half the fused vector |
| 3 | **geometry use by the model** | **MODERATE** | `rel_feat` moved (mean cos 0.8505) and the gain is confined to the registered spatial predicate family, stable across all cell sizes (`+0.020`, CI excludes 0 on large cells). Weakened by: drift confounded with training divergence, and the null per-cell separability association |
| 4 | **improved relational discrimination** | **MODERATE** | `+0.008261`, CI `[+0.0039, +0.0126]`, `P(>0) = 1.000` on 20,016 exactly-paired cells — but below the registered threshold, not robust to dropping single-row-side cells in aggregate, and n = 1 seed |
| 5 | **image-specific relational reasoning** | **NOT ESTABLISHED** | see Section G |

---

## G. Alternative explanations, and the one that survives

The ten-item audit in `docs/PAPER_C_C1_RESULT.md` (A–J) is unchanged; nothing
here disturbs it. This section addresses only the item left open there, **I —
geometry leakage / shortcut**, which this audit sharpens rather than closes.

**The geometry pathway is a pure function of the GT boxes.** `geom_feats_torch`
consumes box coordinates and nothing else — no pixels, no visual tokens. Under
PredCls the GT boxes are *given* to the model. So:

- WPRD is **prior-free** — proven, prior control exactly `0.5` in both arms.
- WPRD is **not shortcut-free**. A score that varies with the instance satisfies
  WPRD whether that variation comes from image content or from the supplied box
  coordinates.

The evidence pattern actively favours the box-exploitation reading:

1. The gain concentrates in `above`, `behind`, `in front of`, `over`, `across`,
   `near`, `on`, `in` — precisely the relations decidable from two boxes alone.
2. It is negative on `has`, `wearing`, `of`, `attached to`, `carrying` —
   relations requiring appearance or semantics, where an added box-only signal
   is noise.
3. The channel that was restored is, by construction, box-derived.

**Conclusion: the most parsimonious causal interpretation of `+0.00826` is that
`C1` restored the model's ability to exploit GT-box geometry for spatially
defined predicates.** Nothing in these artifacts demonstrates improved
*visual* discrimination, and this document does not claim it. Separating the two
would require an experiment not run here (e.g. box-only and box-ablated arms
under the identical estimator).

### A dissociation that must not be papered over

Per-predicate recall, computed from the same dumps under the same registered
partition:

| | WPRD delta | R@50 recall delta |
|---|---|---|
| spatial predicates | **`+0.0221`** (both-spatial cells) | **`−0.00020`** (78,708 GT rows) |
| non-spatial predicates | `+0.0012` (`−0.0041` on large cells) | **`+0.00266`** (53,848 GT rows) |
| overall | `+0.008261` | `+0.000958` |

**The WPRD gain and the R@50 gain come from different, nearly disjoint predicate
families and point in opposite directions within each.** They are not two views
of one improvement and must not be cited as mutually corroborating. This
dissociation is itself informative: a generic "the model got slightly better"
explanation predicts co-movement, and co-movement is absent. It also means the
WPRD gain is **not** visible in the deployed metric.

---

## H. What is proven

1. The two registered flags were applied, and only those two (2 of 101 config
   keys differ between the arms' endpoints).
2. `C1` restored all six geometry channels `C0` had destroyed; verified twice,
   independently, through a passing reconstruction gate.
3. At the selected bandwidth the frozen Fourier encoder is invertible
   (`p73`, mean R² `+0.9839`) where `C0`'s is not (`+0.0920`).
4. Geometry reaches the relational representation through three injections, only
   one of which is gated; `(1 − gate) ≈ 0.5` means the gated one passes geometry
   at half weight.
5. The flat gate is **enforced by a `0.01`-weighted regulariser toward 0.5**, is
   present identically in both arms, and carries no information about geometry
   use.
6. `rel_feat` differs substantially between arms (mean cosine 0.8505).
7. `delta_WPRD = +0.008261`, CI `[+0.0039, +0.0126]`, `P(>0) = 1.000`, over
   20,016 exactly-paired cells with bit-exact reproduction.
8. The gain is confined to the registered **spatial** predicate group
   (`+0.0221`; `+0.0200` on large cells alone, CI excluding zero) and is mildly
   negative on non-spatial predicates.
9. The gain is a **head-class**, not a tail-class, effect.

## I. What is NOT proven

1. **That the gain is caused by geometry rather than by training divergence.**
   The arms are two separately trained models; `rel_feat` drift and the
   predicate pattern are consistent with a geometry mechanism but do not isolate
   it. n = 1 seed cannot separate the intervention from seed variance.
2. **That geometry is used per-instance.** The exploratory per-cell separability
   association is null (`ρ = −0.004` on meaningful cells). The effect tracks
   predicate family, not instance geometry.
3. **Image-specific relational reasoning** — not established, and the evidence
   leans the other way (Section G).
4. **That the aggregate endpoint is robust.** It does not survive dropping cells
   with a single-row side (`+0.0028`, CI includes 0). Only the spatial subgroup
   survives that cut.
5. **That the WPRD gain has any deployed consequence.** R@50 moved in a
   different predicate family entirely.
6. **The registered threshold.** `+0.008261 < +0.010`. Unchanged.

## J. Implications for the second seed

The registered plan is unchanged and this audit creates no new criterion.
`delta_WPRD >= +0.01` remains the threshold and `C1` remains **POSITIVE
DIRECTION, THRESHOLD NOT MET**. Three points bear on how seed 2 is read:

1. **The rationale for seed 2 is strengthened, not weakened.** A coherent,
   size-robust, mechanistically plausible subgroup effect (spatial predicates,
   `+0.020`, CI excluding zero on large cells) is exactly the pattern that
   deserves a replication test rather than abandonment.
2. **The primary endpoint remains the macro `delta_WPRD` over all 20,016 cells.**
   The spatial-subgroup analysis in Section C is **post-hoc** and was not
   registered. It must not be promoted to the endpoint, and a seed-2 spatial
   result must be reported as a pre-specified-by-this-document secondary at
   best.
3. **Seed 2 should be read with the cell-size structure in mind.** 75% of cells
   have a single-row side and the macro metric over-weights them; the same
   `macro` / `weighted` / large-cell breakdown given here should be reported for
   seed 2 so the two seeds are compared on identical footing.

No second seed, tuning, variant, ablation, or rescue experiment has been run or
prepared.

---

## Method note

CPU only; `runs/`, `checkpoints/` and all `.pt` dumps were opened read-only and
none were modified. Analysis scripts were written to a scratch directory outside
the repository and are not committed; every number above is reproducible from
the committed tooling plus the artifacts listed at the top. `docs/PAPER_C_C1_RESULT.md`
is unmodified — where this audit corrects its mechanistic *interpretation* of
the flat gate (Section D), the correction is recorded here rather than by
rewriting the original record.
