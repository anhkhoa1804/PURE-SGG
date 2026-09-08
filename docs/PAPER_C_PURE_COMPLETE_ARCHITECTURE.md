# Paper C — PURE Complete: architecture reconstruction from source and evidence

Analysis-only. **No training, no GPU job, no new experiment, no change to any
registered artifact.** Every claim is traced to source or to an already-generated
result. Scientific state locked at `c8cd81c` (`2c35bba` C0, `a5ce243` C1,
`c8cd81c` mechanism audit).

`C1` remains **POSITIVE DIRECTION, THRESHOLD NOT MET**
(`delta_WPRD = +0.008261032588962776`, CI `[+0.0038311178, +0.0126973772]`,
`P(>0) = 1.000`, threshold `+0.01`). This document does not redefine it.

---

# 1. Research framing

The goal is **not** to beat a published baseline. It is to reconstruct, repair
and validate the PURE implementation so that its intended relational/geometry
pathway is actually functional and internally coherent. `C1a` (coordinate/unit
contract) and `C1b` (Fourier bandwidth) are **architectural repairs**, and the
experiment asks whether repairing them restores functionality the implementation
was failing to realise.

Under that framing the central result of this audit is not the `+0.008`. It is
this: **the geometry *representation* is now repaired, and it is no longer the
binding constraint. The binding constraint has moved to the readout.** Section
11 shows a 20-dimensional MLP on raw boxes still out-discriminates the full
768-dimensional repaired model on the same population.

---

# 2. Current PURE computational graph

`dim = cfg.emb_dim = 768` (confirmed by the dumps' `rel_feat` shape
`(n, 768)`), `img_res = clip_input_res = 336`, `node_layers = 2`,
`edge_layers = 2`, `bilinear_layers = 0`, `geom_fourier_dim = 256`,
`relation_context_layers = 0` (so `relation_context is None`).

| # | stage | file : symbol | shape | geometry? | pixels? | pair id? | prior? | gated? | trainable |
|---|---|---|---|---|---|---|---|---|---|
| 1 | image → CLIP feature map | CLIP ViT (fine-tuned, `freeze_clip=false`) | `(B, C, h, w)` | no | **yes** | no | no | no | yes |
| 2 | token projection | `relational_model.py:356-358 visual_proj` | `(B, hw, 768)` | no | yes | no | no | no | yes |
| 3 | **box → node query** | `:200-204 box_query`, applied `:384-388` | `(n_obj, 768)` | **yes — absolute box** | no | no | no | no | yes |
| 4 | global context bias | `:205 global_proj`, `:379-380` | `(1, 768)` | no | yes | no | no | no | yes |
| 5 | node cross-attention ×2 | `:214-216 node_layers` | `(1, n_obj, 768)` | indirect | yes | no | no | no | yes |
| 6 | **deformable routing** | `:71-124 DeformableObjectRouter`, called `:392-395` | `(n_obj, 768)` | **yes — box centre/size steer sampling** | **yes** | no | no | softmax over 8 points | yes |
| 7 | routing fusion | `:410-414 routing_gate` | `(n_obj, 768)` | indirect | yes | no | no | **yes** (elementwise sigmoid) | yes |
| 8 | object context mixer | `:222-228 context_mixer` (+`context_alpha`) | `(n_obj, 768)` | indirect | yes | no | no | residual `alpha` | yes |
| 9 | pair index select | `:727-728` | `(n_pairs, 768)` ×2 | no | yes | **yes** | no | no | no |
| 10 | **geometry unit branch** | `:735-738` | `(n_obj, 4)` | **yes — THE `C1a` SWITCH** | no | no | no | no | **no** |
| 11 | pair geometry construction | `geometry.py:34-62 geom_feats_torch` | `(n_pairs, 8)` | **yes** | no | yes | no | no | **no** |
| 12 | clamp / nan_to_num | `:444-446` | `(n_pairs, 8)` | yes | no | yes | no | no | no |
| 13 | **Fourier projection** | `:448-450` — **THE `C1b` SCALE** | `(n_pairs, 512)` | yes | no | yes | no | no | `geom_B` **frozen** |
| 14 | geometry MLP | `:255-261 geom_mlp` | `(n_pairs, 768)` | yes | no | yes | no | no | yes |
| 15 | semantic pair feature | `:422-429 _semantic_pair_feature` | `(n_pairs, 768)` | no | yes | yes | no | no | (identity path) |
| 16 | **fusion (gated)** | `:456-461` | `(n_pairs, 768)` | **yes** | yes | yes | no | **yes** | yes (`fusion_gate`) |
| 17 | SPOA branch | `:475-489` | `(n_pairs, 768)` | no | yes | yes | no | no | yes — **OFF at endpoint** |
| 18 | rel seed | `:264-269 rel_seed` | `(n_pairs, 768)` | indirect | yes | yes | no | no | yes |
| 19 | **edge layers ×2 (geometry re-injected)** | `:540-542` → `:48-68` | `(n_pairs, 768)` | **yes — ungated** | **yes** (cross-attn to tokens) | yes | no | internal `gate_proj` on `rel_feat`, **not on geometry** | yes |
| 20 | out norm | `:552 out_norm` | `(n_pairs, 768)` | indirect | yes | yes | no | no | yes |
| 21 | text readout | `:859, :984 score` — cosine | `(n_pairs, 51)` | indirect | yes | yes | no | no | `pred_emb` |
| 22 | classifier readout | `:945-973` | `(n_pairs, 51)` | indirect | yes | yes | optional | no | yes — **unused at `alpha=0`** |
| 23 | composition | `evals.py` + `freq_bias_alpha=3.75` | `(n_pairs, 51)` | no | no | yes | **yes** | no | no |
| 24 | WPRD | `tools/within_pair_discrimination.py::wprd` | 20,016 cells | no | no | yes | **cancels exactly** | no | no |

**Duplication:** geometry appears at stages 3, 6, 10–14, 16, 19 and in pair
pruning (`:1012-1018`). Visual pixels appear at 1–8, 19. The predicate prior
appears **only** at stage 23 and is cancelled by construction at 24.

## 2.1 Two components are trained but disabled at the endpoint

`tools/c0_c1_evaluate.py:118-121` passes `--explicit_spoa_enabled false`,
`--text_conditioned_projection_enabled false`, `--relationness_enabled false`.
The `[ActiveBranches]` line of each endpoint log confirms all three read `false`
live, while each **training** log reads `true`.

Consequences at the endpoint:

- stage 17 is skipped, so `rel_feat = rel_seed(fused_feat)` (`:487`) — the
  trained `spoa_fusion`, `subject_branch`, `object_branch`, `*_attr_branch`,
  `predicate_branch` weights are **present in the checkpoint and unused**;
- `text_relation_features` (`:851-857`) returns `rel_feats` unchanged, so the
  trained `text_space_projection` residual (0.35) is dropped.

This is a **design choice**, not an accident — the docstring states it
reproduces "the SAME evaluator configuration every existing WPRD anchor uses",
which is what makes `C0`/`C1` comparable to `p33`/`p36`/`p60`/`p69`/`p70`. It is
applied identically to both arms, so `delta_WPRD` is unaffected. But it means
**the registered endpoint measures a reduced model, not the trained one**, and
that must not be forgotten when the number is quoted.

Its architectural significance is large: with stage 17 off, `fused_feat` is the
**sole** input to `rel_seed`, and by §4 geometry is half of `fused_feat`.

---

# 3. Geometry pathways — the complete enumeration

There are **six** distinct geometry injections. `C1` touches **two**.

### Path A — absolute box → node query (`box_query`)

```
INPUT           obj_boxes in 336-space, per object
TRANSFORMATION  box_norm = clamp(boxes / img_res, 0, 1)        (:381-383)
                query = box_query(box_norm) + global_bias      (:387-388)
                box_query = Linear(4,768) → GELU → Linear(768,768)   (:200-204)
WHERE USED      every object node, before node attention
GATED?          no — added directly
TRAINABLE?      yes
ACTIVE?         yes, both arms
C0→C1?          **NO — unaffected.** ground_nodes always divides by img_res;
                geom_input_pixel_space is read only at :735, in the pair path
SCIENTIFIC ROLE absolute position/size of each object in the frame
```

### Path B — deformable routing (`DeformableObjectRouter`)

```
INPUT           box_norm centres and sizes + the visual grid
TRANSFORMATION  centers = 0.5*(b[:, :2]+b[:, 2:4]); sizes = (b[:,2:4]-b[:,:2])
                points = centers + tanh(offset_mlp(q)) * sizes * 0.35
                grid_sample(visual_grid, points)                 (:94-123)
WHERE USED      object token construction (stage 6)
GATED?          softmax over 8 sample points; then routing_gate at :410
TRAINABLE?      yes
ACTIVE?         yes (deformable_router = true, both arms)
C0→C1?          **NO — unaffected**
SCIENTIFIC ROLE the one place geometry and pixels genuinely interact — boxes
                steer *where in the image* features are read
```

### Path C — pair-relative geometry → Fourier → `geom_mlp` → **gated fusion**

```
INPUT           the C1a switch (:735-738): C1 passes 336-space boxes,
                C0 passes boxes/img_res
TRANSFORMATION  geom_feats_torch → 8 channels (dx,dy,rw,rh,ar1,ar2,a1,a2)
                clamp(-10,10) → proj = 2*pi*scale*x @ geom_B  (:448-450)
                fourier = cat(sin, cos) → geom_mlp → F.normalize   (:451-456)
                fused = gate*sem_feat + (1-gate)*geom_norm          (:461)
WHERE USED      the fused pair representation, sole input to rel_seed at endpoint
GATED?          **yes** — but see §4; (1-gate) ≈ 0.5
TRAINABLE?      geom_mlp yes; geom_B frozen (requires_grad=False, :254)
ACTIVE?         yes, both arms
C0→C1?          **YES — both flags act here**
SCIENTIFIC ROLE relative offset, size ratio, aspect ratio, area of the pair
```

### Path D — pair-relative geometry → **ungated edge-layer concatenation**

```
INPUT           geom_feat_proj, i.e. geom_mlp output BEFORE F.normalize
TRANSFORMATION  passed unchanged into both edge layers        (:540-542)
                rel_delta = rel_update(cat([sub_upd, obj_upd, rel_feat, geom]))  (:65)
                rel_upd = rel_norm(rel_feat + rel_delta)      (:66)
WHERE USED      twice, inside the relational update
GATED?          **NO.** gate_proj gates rel_feat into the attention queries;
                the geometry concatenation is never multiplied by any gate
TRAINABLE?      yes (rel_update)
ACTIVE?         yes, edge_layers = 2, both arms
C0→C1?          **YES — same upstream repair**
SCIENTIFIC ROLE lets the relational update condition directly on pair geometry
```

### Path E — geometric pair pruning

```
INPUT           raw obj_boxes (336-space)
TRANSFORMATION  geom_score = ||c_s - c_o|| / (w_s+h_s+w_o+h_o); keep smallest k  (:1012-1018)
WHERE USED      candidate pair selection, only when n_pairs > prune_k (64)
GATED?          n/a — a hard top-k selection
TRAINABLE?      no
ACTIVE?         only on images exceeding prune_k; endpoint uses GT pairs
C0→C1?          **NO — reads obj_boxes directly, never the C1a branch**
SCIENTIFIC ROLE a proximity prior on which pairs are considered at all
```

### Path F — vestigial geometry branches

```
geom_alpha (:262)      used ONLY at :468, the vector_fusion_gate == False branch.
                       vector_fusion_gate = True in both arms -> parameter exists
                       in the checkpoint and is NEVER read in the forward pass.
geom_proj (:264)       Linear(12, 768); used ONLY when use_geom_bias == False.
                       use_geom_bias = True in both arms -> dead.
:494-517               an entire parallel non-Fourier geometry branch (tanh +
                       normalize + geom_proj + its own fusion gate), unreachable
                       under use_geom_bias = True.
ACTIVE? no.  C0→C1? no.  SCIENTIFIC ROLE: none in this configuration.
```

**Summary: `C1` repairs the pair-relative geometry (C, D) only.** Absolute box
position (A) and box-steered visual sampling (B) were never broken and were live
in both arms. **`C0` was therefore never geometry-blind** — it was blind
specifically to *pair-relative* geometry. This single fact carries most of §12.

---

# 4. Reconciling the fusion gate

Observed `fusion_gate` mean: `C0` 0.50142, `C1` 0.50081; per-pair mean std
`1.8e-04` / `1.5e-04`; 100% of sampled values in `[0.45, 0.55]` in both arms.

**A. Mathematical role.** `:461`:

```python
fused_feat = gate * sem_feat + (1.0 - gate) * geom_norm
```

An **elementwise convex mixing coefficient** over 768 dimensions between two
**unit-norm** vectors — `sem_feat` from `_semantic_pair_feature` (`F.normalize`,
`:429`) and `geom_norm = F.normalize(geom_feat_proj)` (`:456`). It is not a
relevance detector and not an on/off switch. `gate = sigmoid(fusion_gate(cat([sub_norm, obj_norm, geom_norm])) / 0.7)`.

**B. Why 0.5 is the equilibrium.** The training loss contains an explicit
penalty on departure from 0.5 (`:462-463`, `train.py:2413-2415`):

```python
self.last_gate_reg = ((gate.mean(dim=-1) - 0.5) ** 2).mean()
loss_total = loss_total + (gate_reg_weight * gate_reg)     # weight 0.01
```

`gate_regularizer_weight = 0.01` was active in **both** arms. Any deviation from
0.5 must buy more task loss than it pays in penalty; 0.5 is the designed
attractor. Additionally `fusion_gate_temperature = 0.7` divides the logits,
compressing the sigmoid's usable range further.

**C. What `gate ≈ 0.5` implies.** That the *mixing weights* were the same in
both arms: `sem_feat` and `geom_norm` each contribute ~half of `fused_feat` by
norm. Geometry is admitted at **maximum blend**, in `C0` and `C1` alike.

**D. What it does NOT imply.** It does **not** imply geometry is unused, weakly
used, or that its usage did not change. The gate controls a *weight*; `C0` and
`C1` differ in the *content* of `geom_norm`, which the gate cannot report. The
prior expectation that "the gate should open if geometry became useful" was
**inconsistent with the model's own regulariser** and is withdrawn.

**E. Can geometry be strongly used at `gate ≈ 0.5`?** Yes, necessarily:
`(1 - gate) ≈ 0.5` on a unit vector, plus two ungated injections in Path D that
the gate does not touch at all. Empirically the representation did move —
`rel_feat` cosine similarity between arms is **0.8505** (0.00% of rows above
0.99, mean relative L2 drift 0.52).

**F. Which path carries the gain?** The only path-level causal evidence in the
repository is `p70` (`runs/p70_geometry_causal_ablation/ablation_wprd.json`),
run on the **historical** checkpoint under the **broken** contract:

| arm | WPRD | Δ vs `A_full` |
|---|---|---|
| `A_full` | 0.5542 | — |
| `B_no_fusion_geom` | 0.5506 | **−0.0036** |
| `C_no_edge_geom` | 0.5504 | **−0.0038** |
| `D_no_geom` | 0.5455 | **−0.0087** |

The two paths contributed **almost equally** and roughly additively
(0.0036 + 0.0038 = 0.0074 vs 0.0087 jointly). **This does not attribute the `C1`
gain**, because `p70` ablated a different checkpoint under the C0 contract. No
experiment in this repository separates Path C from Path D under the repaired
contract. **Honest answer to F: undetermined; the best available prior is
"roughly equally".**

---

# 5. Geometry contract defects — bug vs design

| component | observation | type | evidence | scientific consequence |
|---|---|---|---|---|
| `/img_res` in the **pair** path (`:735-738`) | divides 336-space boxes again before `geom_feats_torch` | **BUG / CONTRACT VIOLATION** | `geom_feats_torch` clamps `w,h` at 1.0 (`geometry.py:39-42`); after `/336` every box has `w,h < 1`, so the clamp binds universally | 6 of 8 channels identically constant; verified `rw=rh=ar1=ar2 = −9.54e-07`, `a1=a2 = 0.0` |
| `clamp_min(1.0)` (`geometry.py:39-42`) | forces `w,h ≥ 1` | **DESIGN CHOICE** (correct in pixel units) | prevents division by zero in `dx`, `dy`; harmless when boxes are in pixels | only pathological *in combination* with the `/img_res` bug — the defect is the units, not the clamp |
| `/img_res` in **`ground_nodes`** (`:381-383`) | same division, node path | **WORKING AS INTENDED** | `box_query` consumes a 4-vector directly; `[0,1]` is the natural domain, no clamp downstream | Path A was always correct; `C0` was never geometry-blind |
| Fourier scale `= 1.0` | `2π·1.0·x @ geom_B`, `geom_B` std 9.91 → ~41 rad/unit | **BUG (bandwidth mismatch)** *given* repaired units | `p73`: mean out-of-fold R² `−0.3314` vs shuffled-null `−0.3197` — indistinguishable from a random permutation of itself | at `C1a` units the encoder is a hash; at `C0` units it is *accidentally* survivable |
| Fourier scale `= 0.01` | effective `geom_B` std 0.099 | **DESIGN CHOICE, measured in advance** | `p73` Part C knee: `−0.33 → +0.84 (0.02) → +0.98 (0.01)`; 0.005/0.002 rejected as degenerating the encoder toward linear | all 8 channels ≥ 0.97 recoverable |
| `fusion_gate` | elementwise convex mixer | **WORKING AS INTENDED** | `:456-461` | admits geometry at ~0.5 weight |
| gate regularisation → 0.5 | `((mean_gate−0.5)²)`, weight 0.01 | **DESIGN CHOICE** | `:462-463`, `train.py:2413-2415` | flat gate is enforced; the gate carries **no** diagnostic signal about geometry |
| ungated edge geometry (`:540-542`, `:65`) | `geom_feat_proj` concatenated, never gated | **DESIGN CHOICE** | source | the gate governs only 1 of 3 injections |
| `geom_alpha` (`:262`) | only read at `:468`, the `vector_fusion_gate=False` branch | **DEAD / VESTIGIAL** | `vector_fusion_gate=True` in both arms | none — but it sits in the checkpoint and can mislead an auditor |
| `geom_proj` + branch `:494-517` | unreachable at `use_geom_bias=True` | **DEAD / VESTIGIAL** | source | a whole parallel geometry design is inert |
| `asymmetric_pair_fusion_enabled=False` | `sem_feat = normalize(sub+obj)` is **symmetric** | **DESIGN CHOICE with a real cost** | `:429`; `config.py:52-54` states it "cannot distinguish (a,pred,b) from (b,pred,a) on its own" | **the semantic half of `fused_feat` is order-blind; subject/object order is carried almost entirely by geometry** — see §9 |
| "Saved best … checkpoint" log line | prints when the save is suppressed | **BUG (logging only)** | `train.py:2655-2678` — print is unconditional, `torch.save` is gated | no artifact affected; misleads audit |
| prior term (`freq_bias_alpha=3.75`) | added at composition only | **WORKING AS INTENDED** | stage 23; WPRD prior control exactly `0.5` in both arms | cleanly separated from the model term |
| text readout (cosine) | `score()` = `normalize(rel) @ normalize(emb)ᵀ` | **WORKING AS INTENDED** | `:984-990` | the endpoint's model term |
| classifier head | computed then discarded at `alpha = 0` | **DESIGN CHOICE** | `:891-894` | unused by the endpoint |
| endpoint disables SPOA / text-projection / relationness | trained-on, evaluated-off | **DESIGN CHOICE (anchor consistency)** | `c0_c1_evaluate.py:118-121`; `[ActiveBranches]` | endpoint measures a **reduced** model; identical in both arms so `delta` is valid |

---

# 6. Evidence chain `p68 → p70 → p73 → C0/C1`

| # | arrow | status | evidence |
|---|---|---|---|
| 1 | geometry looked weak → **`p68`: representation is degenerate** | **DIRECTLY SHOWN** | 6 of 8 channels have exactly 1 distinct value; reproduced twice in this cycle and re-derived from the dumps through a passing gate |
| 2 | degenerate representation → **diagnosis: `/img_res` + `clamp_min(1.0)`** | **DIRECTLY SHOWN** | arithmetic is exact and the fix removes the constancy |
| 3 | **`p70`: the pathway is causally live but weak** | **DIRECTLY SHOWN** | `delta_D = −0.0087`; monotone WPRD decline as geometry is removed; `R@50` flat to ±0.0005 across all arms |
| 4 | `p69`/`p71`: **the denied geometry is worth +0.0341 (val) / +0.0366 (test)**, ~90–95% box size | **DIRECTLY SHOWN** (as a probe upper bound) | held-out replication |
| 5 | **`p73`: the frozen Fourier encoder destroys its input** at the repaired units | **DIRECTLY SHOWN** | `I_s1` −0.3314 vs shuffled-null −0.3197, channel by channel |
| 6 | `p73` Part C: **bandwidth 0.01 restores invertibility** | **DIRECTLY SHOWN** | +0.9839 mean R², all channels ≥0.97 |
| 7 | repairs → **all 8 channels non-degenerate in the trained `C1`** | **DIRECTLY SHOWN** | endpoint record, independently reconstructed |
| 8 | repairs → **geometry information reaches the decoder** | **STRONGLY SUPPORTED** | `(1−gate)≈0.5` on a unit vector + two ungated injections (source); `rel_feat` cosine 0.8505 |
| 9 | decoder access → **the model uses it** | **INDIRECT** | gain confined to the registered spatial predicate group (+0.0221; +0.0200 on large cells, CI excludes 0) — but the exploratory per-cell separability association is null (ρ = −0.004) |
| 10 | use → **relational discrimination improves** | **DIRECTLY SHOWN, small** | `+0.008261`, CI `[+0.0039, +0.0126]`, exact pairing, bit-exact reproduction |
| 11 | improvement → **replicates across seeds** | **NOT SHOWN** | n = 1 |
| 12 | improvement → **image-specific reasoning** | **NOT SHOWN** | geometry is a pure function of GT boxes; §8 |
| 13 | `p73` transfer → **measured on each trained model** | **NOT SHOWN** | measured on the frozen `geom_B` (identical, std `9.90808391571045`, in both arms), not re-probed per arm |

---

# 7. What `C1` proves

**Strongest defensible statement:**

> Under the registered estimator and population, repairing PURE's pair-relative
> geometry contract (coordinate units + Fourier bandwidth) restores all eight
> geometry channels and makes the frozen Fourier encoder invertible, and this
> produces a small but statistically positive improvement in prior-free
> within-pair relational discrimination — `delta_WPRD = +0.008261`, paired 95% CI
> `[+0.0039, +0.0126]`, `P(>0) = 1.000` over 20,016 exactly-paired cells — which
> is **below** the pre-registered `+0.010` threshold and is **concentrated in
> spatially defined predicates** (`+0.0221`), being flat-to-negative elsewhere.
> On one seed.

Also established: prior control exactly `0.5` in both arms; estimator and
population identical; exact cell pairing including per-cell row membership;
2-of-101 config difference; and the adaptive-gate hypothesis is disconfirmed as
a *premise* (§4) rather than as a finding.

# 8. What `C1` does **not** prove

1. **Image-specific relational reasoning — NOT ESTABLISHED.** `geom_feats_torch`
   consumes box coordinates and nothing else. Under PredCls the GT boxes are
   *given*. WPRD is prior-free but **not** shortcut-free. The evidence pattern —
   gains on `above`/`behind`/`in front of`/`over`/`across`, losses on
   `has`/`wearing`/`attached to`/`carrying` — is more consistent with **better
   exploitation of supplied box geometry** than with improved visual
   discrimination.
2. **Superiority over any external published model — NOT ESTABLISHED and not
   attempted.** No comparison to external leaderboards is made anywhere here.
3. **A universal PURE improvement — NO.** Non-spatial cells are flat
   (`+0.0012`) and negative on large cells (`−0.0041`).
4. **Benchmark state of the art — NO.** `R@50 +0.00096`, `mR@50 +0.00090`, and
   these move in a *different predicate family* than WPRD (§9).
5. **A reproducible seed-level effect — NO.** n = 1.
6. **Causal superiority of one geometry injection path — NO.** §4F: undetermined.
7. **That the repaired architecture is adequate — NO.** §11 shows the opposite.

---

# 9. Spatial vs non-spatial structure — is the direction mechanistically coherent?

Using only `configs/predicate_metadata_vg150.json` (already consumed by training
via `--predicate_metadata_path`; its own description calls the groups "coarse …
not as ground-truth semantic ontology"):

| stratum | cells | mean `delta` | 95% CI |
|---|---|---|---|
| both predicates spatial | 6,762 | **`+0.022055`** | `[+0.0141, +0.0294]` |
| mixed | 8,273 | `+0.005322` | — |
| neither spatial | 4,981 | `−0.005583` | — |
| both spatial, `na*nb ≥ 25` | 685 | **`+0.019987`** | `[+0.0084, +0.0317]` |
| not-both-spatial, `na*nb ≥ 25` | 2,138 | `−0.004108` | `[−0.0104, +0.0019]` |

**The direction is mechanistically coherent, and for a specific structural
reason.** The eight restored channels are `dx, dy` (relative offset), `rw, rh`
(size ratio), `ar1, ar2` (aspect), `a1, a2` (area). Within a fixed `(subject,
object)` category group — which is exactly what a WPRD cell holds constant — the
quantity that separates `above` from `under`, `in front of` from `behind`, `on`
from `next to`, is precisely a signed relative offset and a size relation. Six
of those eight channels were identically constant in `C0`; only `dx, dy`
survived.

Two further source facts sharpen this:

1. **`sem_feat` is symmetric.** With `asymmetric_pair_fusion_enabled = False`,
   `_semantic_pair_feature` returns `F.normalize(sub_norm + obj_norm)` (`:429`) —
   invariant under swapping subject and object. `config.py:52-54` states this
   explicitly. So the semantic half of `fused_feat` **cannot** tell `(a, pred, b)`
   from `(b, pred, a)`. Order information must come from geometry, whose channels
   are order-dependent (`dx = (c2x − c1x)/w1`, `rw = log(w2/w1)`). In `C0` only
   `dx, dy` carried it; `C1` restores six more. Directional spatial predicates
   are exactly the ones that need it.
2. **Non-spatial predicates gain nothing to lose.** `has`, `wearing`,
   `attached to`, `carrying` are decided by appearance and object semantics; a
   newly informative box-geometry vector occupying half of `fused_feat` is, for
   them, added variance. Their mild decline (`−0.004` on large cells) is the
   expected cost.

This is the most interpretable signature in the cycle. It is **consistent with**
a genuine geometry mechanism. It is **not proof** of one: the association is with
predicate family, not with per-instance geometric separability, which tested null
(ρ = −0.004 on cells where the statistic is meaningful).

## The R/mR dissociation

| | WPRD `delta` | R@50 recall `delta` |
|---|---|---|
| spatial predicates | **`+0.0221`** (cells) | **`−0.00020`** (78,708 GT rows) |
| non-spatial predicates | `+0.0012` | **`+0.00266`** (53,848 GT rows) |
| overall | `+0.008261` | `+0.000958` |

The WPRD gain and the recall gain live in **different, nearly disjoint predicate
families and move oppositely within each**. They are not corroborating views of
one improvement. This reproduces `p70`'s finding that `R@50` is insensitive to
the geometry pathway (flat to ±0.0005 across all four ablation arms) and
`p39`/`p40`'s discrimination-vs-calibration distinction.

---

# 10. Macro vs weighted

```
macro    delta_WPRD  +0.008261      (every cell weight 1)
weighted delta_WPRD  +0.003036      (cell weight = na*nb)
```

| | |
|---|---|
| cells unchanged (exactly 0) | 10,653 / 20,016 (53.22%) |
| 1-vs-1 cells | 6,110 (30.53%) |
| cells with a side of size 1 | 15,062 (**75.25%**) |
| gross positive mass | `+1658.35` over 4,799 cells |
| gross negative mass | `−1492.997` over 4,564 cells |
| net | `+165.35` — cancellation ratio `0.9003` |

| size band (`na*nb`) | % of cells | % of all comparisons | mean `delta` | share of net |
|---|---|---|---|---|
| 1 | 30.53% | 1.10% | `+0.010147` | 37.50% |
| 5–9 | 12.86% | 3.07% | `+0.017981` | 28.00% |
| 25–99 | 9.39% | 16.44% | `+0.001519` | 1.73% |
| 100+ | 4.72% | **69.34%** | `+0.002177` | **1.24%** |

**Statistical reading — all four of the posed propositions hold:**

- **The gain is broad but thin.** ~24% of cells improve, ~23% worsen, 53% do not
  move; the net is the ~10% residual of two opposing masses each ~200× larger.
  Per-cell std is 0.31, so `+0.0083` is a standardised effect of ~0.027. The CI
  excludes zero because n = 20,016, **not** because the effect is large. It is
  not large.
- **The macro endpoint amplifies small cells.** A 1-vs-1 cell counts as much as a
  64×64 cell. The cells holding 69% of all pairwise comparisons contribute 1.2%
  of the macro gain. That is the entire macro/weighted gap.
- **Aggregate practical effect is limited.** Weighted `+0.0030`; `R@50 +0.00096`;
  and the aggregate effect does not survive dropping cells with a single-row side
  (`+0.0028`, CI includes 0).
- **It is concentrated in one predicate family** (§9), and *that* subgroup effect
  **is** robust to cell size.

Both facts are true at once and neither should be suppressed: **the aggregate is
thin and macro-inflated; the spatial subgroup effect is stable.**

---

# 11. Comparison against the repository's own geometry baselines

`runs/p60` (`docs/ESTIMATOR_MATCHED_GEOMETRY_RESULT.md`) is the right reference:
one estimator (AdamW MLP + softmax CE, 5-fold CV), one regime, the **same
132,556 validation rows**, gates 5/5 PASS including `P_prior = 0.500000` exactly.

| arm | dims | WPRD |
|---|---|---|
| `P_prior` (control) | — | **0.5000** |
| `N_shuffled` (null) | — | 0.4992 |
| `A_relfeat` — probe on PURE's `rel_feat` | 769 | 0.5732 |
| `D_geometry_linear` | 20 | 0.5741 |
| `C_fusion` — `rel_feat` + geometry | 789 | 0.5922 |
| **`B_geometry` — MLP on raw box geometry** | **20** | **0.5976** |

Alongside the deployed model term on the same population:

| | WPRD |
|---|---|
| `p33`/`p70` historical PURE (`A_full`) | 0.5542 |
| **`C0`** (registered control) | **0.5667** |
| **`C1`** (both repairs) | **0.5750** |

**Comparability caveat — this is INDIRECT.** The `p60` arms are *probes*
(a classifier fitted on features); `C0`/`C1` are the *deployed model term*
(cosine against predicate embeddings, no fitting). They are not the same
estimator, and the difference could be worth several points either way. What
follows is an architectural reading, not a measurement.

With that caveat, three internal comparisons:

1. **What geometry adds to PURE.** `p70` says the geometry pathway PURE *had*
   (broken contract) was worth `0.0087` WPRD. `C1` says repairing it is worth
   `+0.0083`. **Repairing the contract bought approximately as much as the entire
   pre-existing geometry pathway was worth** — a coherent doubling of a small
   term, not a step change.
2. **What "PURE Complete" adds over raw geometry — currently, nothing.**
   `B_geometry`, a 20-dimensional MLP on two rectangles, reaches **0.5976**.
   `C1`, a 768-dimensional model with a fine-tuned CLIP backbone and repaired
   geometry, reaches **0.5750**. The direction of `p39`/`p40`/`p60` is unchanged
   by the repair: **on prior-free relational discrimination, the full model still
   does not beat boxes.**

   > **CORRECTION (added by `docs/PAPER_C_PURE_READOUT_FORENSIC.md`).** This
   > holds for the **deployed text head** only, and as written it over-generalises
   > to the architecture. Read through `C1`'s own trained classifier head — the
   > same `rel_feat`, the same registered estimator, the same 20,016 cells — the
   > repaired model reads **0.5994**, at or slightly above `B_geometry`'s 0.5976.
   > The deficit is a property of the deployed readout, not of the representation.
   > See §4 and §15 of the readout forensic.
3. **Fusion is lossy in this codebase.** `C_fusion` (0.5922, 789 dims) is
   *below* `B_geometry` (0.5976, 20 dims) — under a matched estimator, adding
   `rel_feat` to geometry **subtracts**. The same signature appears inside the
   model: `p57` found `rel_feat` retains size/overlap (R² 0.68–0.89) but discards
   relative position (`dx_rel` R² **0.052**), and `p69`/`p71` found the denied
   geometry worth `+0.0341`/`+0.0366` while the repair delivered `+0.0083` —
   roughly **a quarter** of the probe-estimated headroom.

**The gap between "the information is present and decodable" and "the model
converts it into discrimination" is the finding of this section.**

---

# 12. The minimal coherent architectural story

One mechanism explains all eleven observations:

> **PURE has always had two working geometry pathways and one broken one.**
> Absolute box position (`box_query`) and box-steered visual sampling
> (`DeformableObjectRouter`) were correct throughout and are untouched by `C1`.
> The *pair-relative* pathway was broken twice over: the pair branch divided
> already-336-space boxes by `img_res` again, so `geom_feats_torch`'s
> `clamp_min(1.0)` bound on every width and height and flattened 6 of 8 channels
> (`p68`); and the frozen Fourier encoder's bandwidth was tuned — accidentally —
> to the *compressed* range that bug produced, so correcting the units alone
> would have pushed the input into the encoder's hash regime (`p73`). `C1`
> repaired both together. The channels came back and the encoder became
> invertible. Because the semantic half of the pair feature is **symmetric** in
> subject and object, pair-relative geometry is the model's principal carrier of
> *directional* information — so the restored signal helped exactly the
> directional spatial predicates and mildly hurt the rest. The fusion gate never
> moved because it is regularised to 0.5 and is only one of three injections;
> it was never the mechanism. The total effect stayed small because the model
> already had absolute position and pixels, because the *readout* — not the
> geometry representation — is what fails to convert geometry into
> discrimination (a 20-dim box MLP still scores higher, and adding `rel_feat` to
> geometry lowers it), and because the composed `R/mR` metric is dominated by a
> prior the model was trained to complement, so it barely moves regardless.

That single story covers: (1) geometry looked weak; (2) `p68` defects;
(3) `p70` live-but-weak pathway; (4) `p73` bandwidth; (5) 8/8 channels restored;
(6) decodability −0.33 → +0.98; (7) WPRD `+0.0083`; (8) gate flat at 0.5;
(9) spatial predicates benefit; (10) total below `+0.01`; (11) `R/mR` barely move.

---

# 13. Remaining architectural bottlenecks

| rank | bottleneck | confidence | evidence |
|---|---|---|---|
| 1 | **Predicate readout / fusion is lossy** — the model cannot convert available geometric information into discrimination | **HIGH** | `B_geometry` 0.5976 (20 dims) > `C1` 0.5750 (768 dims); `C_fusion` 0.5922 < `B_geometry` 0.5976 under a matched estimator; repair delivered `+0.0083` of the `+0.0341`/`+0.0366` probe headroom |
| 2 | **Relative-position information is destroyed downstream of the input** | **HIGH** | `p57`: `rel_feat` retains size/overlap R² 0.68–0.89 but `dx_rel` R² **0.052**, `dy_rel` 0.223 — the encoder discards exactly what spatial predicates need, and this was measured on `rel_feat`, i.e. *after* fusion |
| 3 | **Symmetric semantic pair feature** — order information rests almost entirely on geometry | **HIGH** | `:429` `normalize(sub+obj)`; `config.py:52-54` states it cannot distinguish `(a,p,b)` from `(b,p,a)`; `asymmetric_pair_fusion_enabled` exists but is off |
| 4 | **Prior dominance in the composed metric** | **HIGH** | prior-argmax agreement 0.920/0.919; `p70` `R@50` flat to ±0.0005 across all geometry ablations; `p39`/`p40` discrimination≠calibration |
| 5 | **Train/eval architecture mismatch** — SPOA, text-projection and relationness are trained then disabled at the endpoint | **MEDIUM** | `c0_c1_evaluate.py:118-121` vs training `[ActiveBranches]`; the endpoint scores a reduced model |
| 6 | **Frozen `geom_B` with a hand-set scalar** — bandwidth is a fixed hyper-parameter, not learned | **MEDIUM** | `:254` `requires_grad=False`; `p73` shows outcome swings from −0.33 to +0.98 across a factor of 50 in `scale` |
| 7 | **Gate regulariser may be suppressing useful adaptivity** | **MEDIUM** | `:462-463`, weight 0.01 — the model is *prevented* from learning where geometry matters; §9 shows geometry helps spatial and hurts non-spatial predicates, precisely the contrast an adaptive gate could exploit |
| 8 | **Vestigial branches** (`geom_alpha`, `geom_proj`, `:494-517`) | **MEDIUM** (hygiene, not performance) | unreachable under the active flags |
| 9 | **Weak visual conditioning / feature dilution** | **LOW** | plausible from `C_fusion < B_geometry`, but no experiment isolates the visual contribution |
| 10 | **Pair proposal** | **LOW** | endpoint uses GT pairs, so pruning is largely inert there; no evidence of a bottleneck under PredCls |
| 11 | **Geometry overreliance** | **LOW** | the opposite is indicated — geometry is under-exploited |

**No intervention is proposed here.** Ranks 1–3 are one cluster: the readout
discards relative position and the pair feature cannot encode order without it.

---

# 14. `PURE Complete v1` — specification

A **structural** definition. `PURE Complete` is not "the highest WPRD we
obtained"; it is a system satisfying auditable contracts.

| # | criterion | status | basis |
|---|---|---|---|
| **A** | **Geometry contract** — box coordinates in a single, declared space at every consumer | **EXISTING + VALIDATED for the pair path** (`C1`); **EXISTING + UNCERTAIN globally** — Paths A, B, E consume `boxes/img_res` or raw boxes under separate conventions with no single declared contract | `:381`, `:735`, `:1012` |
| **B** | **Geometry representation** — all 8 intended channels survive construction | **EXISTING + VALIDATED** (`C1`: 0 of 8 constant, reconstructed through a passing gate) | endpoint record |
| **C** | **Fourier contract** — frequency scale matched to the actual coordinate range | **EXISTING + VALIDATED** at `scale = 0.01` (mean R² +0.9839) | `p73` Part C |
| **D** | **Geometry access** — relational layers can read geometry | **EXISTING + VALIDATED** (source: `(1−gate)≈0.5` + 2 ungated injections) | §3 |
| **E** | **Visual access** — image-derived relational features remain active | **EXISTING + VALIDATED** (CLIP unfrozen; deformable routing live) | §2 |
| **F** | **Predicate readout** — the representation actually contributes to discrimination | **EXISTING + DEFECTIVE** — `B_geometry` (20 dims) still exceeds `C1` (768 dims); `C_fusion < B_geometry` | `p60` |
| **G** | **Prior separation** — the frequency prior is not confused with learned signal | **EXISTING + VALIDATED** — prior control exactly `0.5`, prior constant within group to `9.44e-05` | `C0`/`C1` endpoints |
| **H** | **Diagnostic validity** — WPRD behaves as intended | **EXISTING + VALIDATED with a stated limitation** — controls exact; but 75% of cells have a single-row side and macro over-weights them, so macro and weighted must both be reported | §10 |
| **I** | **Reproducibility** — intervention, checkpoint, estimator, population fully auditable | **EXISTING + VALIDATED** — SHA-pinned lineage, 2-of-101 config diff, bit-exact independent reproduction, exact cell pairing | `a5ce243`, `c8cd81c` |
| **J** | **Order contract** *(added; source-supported)* — the pair representation must distinguish `(a,p,b)` from `(b,p,a)` without relying solely on geometry | **EXISTING + DEFECTIVE** | `:429`; `config.py:52-54` |
| **K** | **Train/eval parity** *(added; source-supported)* — the evaluated architecture is the trained architecture | **EXISTING + DEFECTIVE** — 3 trained components disabled at the endpoint | `c0_c1_evaluate.py:118-121` |
| **L** | **No dead pathways** *(added; source-supported)* — every declared component is reachable | **EXISTING + DEFECTIVE** — `geom_alpha`, `geom_proj`, branch `:494-517` unreachable | §3 Path F |

**PROPOSED FUTURE CHANGE — none is validated, none is scheduled, and nothing
below has been run.** Listed only so the specification is complete: a single
declared geometry contract shared by Paths A/B/C/E (criterion A); an order-aware
pair feature (J); train/eval parity or an explicit justification per anchor (K);
removal of dead branches (L); and — addressing bottleneck 1–2 — a readout that
preserves relative position. Each would require its own pre-registration.

---

# 15. The role of seed 2

Three distinct questions must not be conflated:

| | question | what seed 2 can answer |
|---|---|---|
| **A** | **Reproducibility** — is the `C1` repair effect stable across seeds? | **This is what seed 2 is for.** n = 1 cannot separate the intervention from seed variance, and this codebase has no seed-variance estimate for `delta_WPRD` |
| **B** | **Threshold** — does `delta_WPRD` reach `+0.01`? | Seed 2 is **not** a second attempt at the threshold. Running until a seed crosses `+0.01` would be a garden-of-forking-paths failure. The registered rule stands; a second seed changes the *evidence*, not the rule |
| **C** | **Architecture completion** — is PURE now a complete relational model? | **Seed 2 cannot answer this at all.** §11 and §13 show the readout, not the geometry representation, is the binding constraint. No number of seeds addresses that |

**Recommendation: yes, run the registered seed 2 next, unchanged.** Reasons:

1. The subgroup structure is coherent and size-robust (spatial `+0.020`, CI
   excluding zero on large cells alone) — exactly the pattern that warrants a
   replication test rather than abandonment.
2. It is the only remaining registered obligation; everything else in the
   pre-registration is discharged.
3. It is cheap relative to its evidential value, and the analysis pipeline is now
   fully validated end-to-end (bit-exact reproduction, exact identity pairing).

**Conditions.** Do not modify the intervention, threshold, estimator or
population. Report seed 2 with the **same** breakdown given here — macro,
weighted, cell-size bands, and the spatial/non-spatial split — so the seeds are
comparable on identical footing. The spatial-subgroup analysis is **post-hoc**
for seed 1 and must be reported as pre-specified-by-this-document secondary at
best; it must not be promoted to the endpoint.

**Not run. Nothing has been launched or prepared.**

---

# 16. Open scientific questions

1. **Which geometry path carries the `C1` gain — gated fusion (C) or ungated
   edge injection (D)?** Undetermined; `p70`'s equal split was measured on a
   different checkpoint under the broken contract.
2. **Is the effect image-specific or purely GT-box exploitation?** Unresolved;
   would need box-only and box-ablated arms under the identical estimator.
3. **Why does a 20-dim box MLP still out-discriminate the repaired 768-dim
   model?** The central architectural question (§11, §13).
4. **Is the gate regulariser suppressing useful adaptivity?** §9 shows geometry
   helps spatial and hurts non-spatial predicates — the contrast an adaptive gate
   could exploit and is currently penalised for learning.
5. **Would an order-aware `sem_feat` change the spatial concentration?** If
   directional information no longer had to travel through geometry alone, the
   signature in §9 should weaken.
6. **Does macro-WPRD's over-weighting of single-row cells warrant a registered
   companion endpoint?** Both should be reported until resolved.
7. **Does the train/eval architecture mismatch change any anchor?** All existing
   WPRD anchors share it, so it cancels in comparisons but biases absolute values
   by an unmeasured amount.
8. **Does the repair replicate on a second seed?** §15.

---

## Method note

CPU only; `runs/`, `checkpoints/` and all `.pt` dumps opened read-only, none
modified. `docs/PAPER_C_C0_RESULT.md`, `docs/PAPER_C_C1_RESULT.md`,
`docs/PAPER_C_C1_MECHANISM_AUDIT.md` and the pre-registration are unmodified —
this document adds a framing and an architecture map, and does not restate or
revise their conclusions.
