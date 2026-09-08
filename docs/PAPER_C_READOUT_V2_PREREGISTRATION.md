# Paper C — Readout v2 preregistration

Committed **before** any Readout v2 parameter is fit. This document fixes
the hypothesis, the exact architecture, the exact equations, the training
protocol, the endpoints, and the failure/success criteria. Nothing below is
tuned after seeing a result. Geometry is not reopened here — see
`docs/PAPER_C_C1_SEED2_RESULT.md` and `docs/PAPER_C_PURE_COMPLETE_ARCHITECTURE.md`
for why the `C1` geometry contract is now a controlled variable, not a
subject of this experiment.

## 1. Hypothesis (`H_readout`)

> PURE's repaired relational representation (`rel_feat`, under the `C1`
> geometry contract) contains predicate-discriminative information that the
> **frozen** CLIP predicate-prototype readout cannot access. Allowing the
> predicate prototypes to adapt — while initialized from, and semantically
> anchored to, their CLIP embeddings — will recover some of that
> information as measured by WPRD, without materially harming the
> semantic organization the anchoring is meant to preserve.

This is not claimed proven. It is the next hypothesis, motivated by the
forensic decomposition below, not yet tested with a trained prototype.

## 2. Why this hypothesis, not another one (forensic evidence, locked)

On the identical `C1` seed-1234 `rel_feat` (`docs/PAPER_C_PURE_READOUT_FORENSIC.md`,
`docs/PAPER_C_READOUT_PILOT0_DECOMPOSITION.md`):

| readout | WPRD |
|---|---|
| frozen CLIP cosine (deployed `Readout v0`) | ≈0.57499 |
| existing adapter + frozen prototypes (`text_conditioned_projection`, re-enabled offline) | ≈0.57508 (+0.4% of the gap) |
| whitened prototypes | ≈0.57718 (+9% of the gap) |
| learned classifier, GELU removed | ≈0.59659 |
| full learned classifier (`predicate_classifier`, diagnostic reference) | ≈0.59743 |

Pilot 0's decomposition: **≈96.3%** of the `frozen → classifier` gap is
attributable to learned *predicate directions* (where the decision
boundaries point), **≈3.7%** to the classifier's nonlinearity (`GELU`).
Prior contrast, prototype collinearity, and the existing linear adapter
were each tested and rejected as explanations (§16 of the prior task's
context; `docs/PAPER_C_PURE_READOUT_FORENSIC.md`). The dominant, load-bearing,
not-yet-fixed constraint is therefore the *rigidity of the frozen predicate
directions themselves* — not nonlinearity, not the prior, not an
insufficient linear adapter. This is why Readout v2's primary candidate
adapts the **directions** (the prototype matrix) and adds no nonlinear
projection.

## 3. Source-level audit (exact references, this session)

| concern | source location | finding |
|---|---|---|
| predicate embedding creation (`E`) | `openvocab_rel/clip_utils.py:132` `encode_predicate_vocab(clip_model, processor, pred_vocab, device, prompt_fn=pred_prompt_roles, direction="s2o")` | runtime tensor, **not** a model parameter or buffer. One template per predicate: `openvocab_rel/prompts.py:49` `pred_prompt_roles` → `"predicate: {p} \| roles: [SUBJ]->[OBJ]"`. `@torch.no_grad()` in `clip_text_features` (`clip_utils.py:22`). Built fresh from **this checkpoint's own** (possibly fine-tuned) CLIP text encoder at eval-script startup — not stored in the `.pt` file. |
| text-head logits | `openvocab_rel/models/relational_model.py:859-860` `text_predicate_logits(rel_feats, text_feats) = score(text_relation_features(rel_feats), text_feats)` | `text_relation_features` (line 851) applies `text_space_projection` **only if** `text_conditioned_projection_enabled` (default `False`, and explicitly forced `False` by the eval/dump path); otherwise it is the identity on `rel_feats`. |
| classifier head | `relational_model.py:607-613` `self.predicate_classifier = Sequential(LayerNorm(768), Linear(768,768), GELU, Dropout(0.1), Linear(768,51))` | closed-set, fixed output width 51. Diagnostic reference only (§24 of the task instructions — not a target to chase). |
| scoring primitive | `relational_model.py:983-990` `score(a,b) = normalize(a) @ normalize(b).T` | plain cosine, **no learned temperature**, no logit scale. |
| unified readout selector | `relational_model.py:862-894` `predicate_scores(rel_feats, predicate_text_feats, pred_log_prior, mode)` | at `mode="ensemble", ensemble_alpha=0.0` (the registered eval configuration — `tools/c0_c1_evaluate.py:120`), the return value reduces to `LayerNorm_rows(text_logits)`; the classifier branch's weight is exactly zero. **This is `Readout v0` as actually deployed.** |
| existing text-space adapter | `relational_model.py:614-619` `text_space_projection`, `LayerNorm(768)+Linear(768,768)+GELU+Linear(768,768)` | ≈1,182,720 params (`1536 + 590592 + 590592`, checkpoint-verified). Disabled at eval. This is the "≈0.4% recovery" adapter from §2 — a **learned relational projection**, already tried, already known to be near-inert on its own. |
| predicate CE loss (training) | `train.py:173` `_predicate_ce_loss(logits, targets, weights, cfg, pred_sim_matrix, pred_group_matrix)` | mode from `cfg.predicate_ce_loss` (checkpoint default `"focal"`); optional label/group relaxation, both **off** by default; class weights `pred_ce_weights` built once per run (`train.py:1547`). |
| `E` treated as frozen in every existing loss | `train.py:2248,2255,2263` — `out.score(x_pos, pred_emb_s2o.detach())` / `out.text_predicate_logits(x_pos, pred_emb_s2o.detach())` | **confirmed**: every existing training loss explicitly `.detach()`s the predicate embedding before scoring. This is the exact mechanism `H_readout` targets — no existing loss can move a predicate prototype today, by construction. |
| prior composition | `train.py:2249-2250,2256-2257`, `evals.py` freq-bias path | prior enters only as an additive term on **classifier-branch** logits (`logit_adj_tau`/`tail_logit_tau` on `ce_logits`, and separately the deployed `freq_bias_alpha` composition in the evaluator). The text/cosine branch that WPRD scores (`ensemble_alpha=0.0`) never has the prior added to it in the dumped `text_logits` channel — confirmed by `docs/PAPER_C_C1_RESULT.md`/`PAPER_C_C1_SEED2_RESULT.md`'s `prior_control_wprd = 0.5` exactly, in every arm measured so far. |
| checkpoint save/load | `train.py`'s `resume`/`resume_from` path (used identically by `run_c0_c1.sh`, `c0_c1_evaluate.py`) | standard `torch.load` into `model.state_dict()`; new parameters not present in an old checkpoint must have an explicit init path (§7) or `load_state_dict(strict=False)` will silently skip them — **this is a named failure mode to test**, not to assume away. |
| checkpoint parameter dims (ground truth, not config default) | `checkpoints/C1_seed5678.pt::model` state-dict, inspected this session | `predicate_classifier.4 = Linear(768→51)`, `text_space_projection.{1,3} = Linear(768→768)`, `cfg.emb_dim=768`, `cfg.predicate_classifier_classes=51`, `cfg.clip_name="openai/clip-vit-large-patch14-336"`. **`E, P ∈ R^(51×768)`, 39,168 elements** — confirms the target cited in this programme's own framing. |
| eval/dump path | `openvocab_rel/evals.py` (`eval_sgg_standard` region, `text_logits`/`cls_logits`/`pred_emb` written into the pair-logits dump) | additive extension point: a new logits channel can be added alongside `text_logits`/`cls_logits` without touching either. |
| WPRD estimator | `tools/within_pair_discrimination.py::Groups`, `::wprd` | **not touched by this design** — Readout v2 produces a new *input* to the same, unmodified estimator. |

## 4. Open-vocabulary property — traced, not assumed

**What "open vocabulary" means in this repository, as implemented:**

1. The text branch's scoring primitive (`score`, `text_predicate_logits`) is
   **vocabulary-agnostic by construction** — `text_feats`/`predicate_text_feats`
   is a runtime argument of arbitrary row count `N`; nothing in
   `RelationalModel` hard-codes 51. `predicate_scores(mode="text")` will
   happily score against a 5-row or a 500-row `predicate_text_feats` tensor.
2. `encode_predicate_vocab` can encode **any** list of predicate strings
   through the (possibly fine-tuned) CLIP text encoder — there is no
   restriction to the VG150 vocabulary.
3. **The classifier branch cannot do any of this.** `predicate_classifier`'s
   final `Linear` has a fixed output width (`predicate_classifier_classes`,
   51 in every checkpoint this programme has trained). It is architecturally
   closed-set. This is exactly why the classifier reference is a diagnostic
   ceiling, never a deployment target (`docs/PAPER_C_PURE_READOUT_FORENSIC.md`,
   and restated in this task's own §24).
4. **`--eval_zs_predicates` is a reporting harness, not a trained protocol.**
   It lets an operator name a predicate subset for separately-aggregated
   `zR@K` reporting (`evals.py:1780-1782, 2338, 2409-2411`). Grepping
   `train.py`/`config.py` for any mechanism to **exclude** predicate classes
   from supervision during training returns nothing. **No checkpoint in this
   repository has ever been trained with predicates withheld from
   supervision**, so `zR@K` on any existing checkpoint reports recall on
   predicates the model *did* see in training, not genuine zero-shot
   generalization.

**Conclusion, stated at the required precision:** open-vocabulary scoring
is **structurally present** in the text branch (any predicate list can be
encoded and scored) but **empirically unvalidated** in this repository (no
model here has ever been tested on predicates excluded from its own
training). Readout v2 must preserve property (1)–(2) for the frozen path
(`Readout v0` untouched, still scores against a live `encode_predicate_vocab`
call) and must state honestly, not claim more than tested, what happens to
an unseen predicate scored against the **adapted** prototype space `P`
(§9, §14 open-vocabulary check).

## 5. Primary architecture — the exact decision (Phase 6/8/9 of the task)

Candidates considered (task-specified enumeration):

| candidate | form | new params | verdict |
|---|---|---|---|
| **A** | `P` directly trainable, `P_0 = E` | 39,168 | **PRIMARY — selected** |
| B | `P = E + ΔP` | 39,168 | mathematically equivalent to A once `A` carries an anchor loss to `E` (a cosine/L2 penalty toward `E` *is* an implicit `ΔP` parameterization); rejected only because it adds no capability A lacks and obscures the one thing that matters (the anchor loss), not because it is wrong |
| C | `P = E + U V` (low-rank residual) | rank-`r`: `2·51·r` (or `2·768·r` depending on factorization) | rejected at this stage — no evidence yet that full-rank `P` (candidate A) is too free; adding a rank constraint pre-emptively answers a question not yet asked (Phase 9's "do not assume the highest-capacity form"; here the concern runs the other way — do not assume a *constrained* form is needed either, absent evidence) |
| D | shared transform on `E` (e.g. one small MLP applied to all 51 rows) | depends on hidden width, typically ≥ existing adapter's ~1.18M | **rejected — this is a "Design Trap" case** (task §"AVOID A DESIGN TRAP" / §"IMPORTANT ARCHITECTURAL REFRAME"): a shared nonlinear transform is functionally close to the existing `text_space_projection` adapter, already measured at **≈0.4% recovery**. It also cannot give each predicate an independent correction — the forensic evidence (Pilot 0, ≈96% direction / ≈4% nonlinearity) is exactly evidence *against* needing a shared nonlinearity here |
| E | class-specific residual + shared transform | ≥ A + D | rejected, compounds D's problem without new justification |

**Selected: Candidate A — `P` directly trainable, initialized `P_0 = E`,
constrained by a semantic anchoring loss (not a hard-wired residual
parameterization).** This is the minimum function-class change that
targets the exact locus the forensic audit identified (§2): the
*directions* of the frozen prototypes, and nothing else. It does not touch
`rel_feat`, does not add a nonlinear projection, does not touch geometry,
and its parameter count (39,168) is smaller than the existing, already-tried,
already-inert `text_space_projection` adapter (≈1.18M) by a factor of ~30.

This satisfies the task's own stated principle: *"remove the empirically
demonstrated constraint with the minimum additional function class."*

## 6. Exact mathematical specification

Let `d = 768` (CLIP-L/14-336 text width, `= cfg.emb_dim`), `C = 51`
(`predicate_classifier_classes`).

**Frozen predicate embeddings** (unchanged from `Readout v0`):

```
E ∈ R^(C×d)
E = normalize_rows( CLIP_text_encoder( pred_prompt_roles(p, "s2o") )_{p ∈ pred_vocab} )
```

computed once, from the **same frozen `C1` checkpoint's** CLIP text encoder
(see §9 initialization note — this is a specific, checked choice, not
assumed), via the existing `encode_predicate_vocab` call, unmodified.

**Trainable predicate prototypes:**

```
P ∈ R^(C×d),  P_0 := E   (copied, not aliased — see §7)
```

`P` is the **only** new trainable tensor. Every other parameter in the
checkpoint (CLIP vision encoder, CLIP text encoder, `ProgressiveRelationalDecoder`,
`predicate_classifier`, `text_space_projection`, `relationness_head`,
`calibration_gate`, `bias_residual_head`, `object_semantic_proj`) is
**frozen** (`requires_grad = False`) for the duration of Readout v2 training.
This is what makes the geometry pathway a genuinely controlled variable
(§12) — it cannot move, because nothing that produces `rel_feat` receives
a gradient.

**Adaptive logit function**, exactly mirroring the existing `score`
contract (no new normalization convention introduced):

```
r = rel_feat                         # PURE relational representation, C1 geometry, unchanged
z = score(r, P) = normalize(r) @ normalize(P)^T      ∈ R^C
```

This is architecturally identical to the existing
`text_predicate_logits(r, E)` call, with `P` substituted for the runtime
`E` argument and `text_conditioned_projection` left at its registered,
disabled setting (`text_relation_features` is the identity at eval — §12).

**Semantic anchoring — exact form, chosen and justified (not assumed):**

Four forms were compared, per the task's explicit requirement not to
assume `λ‖P-E‖²` is automatically correct:

| form | definition | assessment |
|---|---|---|
| raw L2 | `λ · mean_i ‖P_i − E_i‖²` | **rejected as primary.** `score()` only ever consumes `normalize(P)`, so raw-space L2 partially constrains a functionally-inert degree of freedom (row norm) and under-constrains what actually matters (direction) unless `λ` is tuned to compensate — an extra free parameter this design avoids needing |
| cosine anchoring | `λ · mean_i (1 − cos(P_i, E_i))` | **PRIMARY.** Directly penalizes angular drift, the only thing `score()` is sensitive to. Scale-invariant, matches the metric the prototypes are actually read through |
| normalized-prototype L2 | `λ · mean_i ‖normalize(P_i) − normalize(E_i)‖²` | algebraically `= 2λ · mean_i (1 − cos(P_i,E_i))` — **identical to cosine anchoring up to a constant factor of 2 in `λ`.** Listed separately in the task's own enumeration but not a distinct choice once cosine anchoring is selected; noted here so the equivalence is explicit, not silently assumed |
| residual penalty | `λ · mean_i ‖ΔP_i‖²` under `P = E + ΔP` | equivalent to raw L2 anchoring under candidate A (§5); rejected for the same reason as raw L2 |

**Selected anchor loss:**

```
L_anchor(P, E) = λ_anchor · (1/C) · Σ_i (1 − cos(P_i, E_i))
              = λ_anchor · (1/C) · Σ_i (1 − normalize(P_i)·normalize(E_i))
```

**Predicate classification loss** — reuses the existing, already-tested
`_predicate_ce_loss` (`train.py:173`) **unmodified**, applied to the new
logits instead of the frozen-`E` logits:

```
z_pos = score(r_pos, P)                     # r_pos: rel_feat at GT-positive pair rows
L_readout_v2_ce = _predicate_ce_loss(z_pos, y_pos, pred_ce_weights, cfg)
```

using the checkpoint's own inherited `cfg.predicate_ce_loss` (`"focal"`,
the base run's setting — not re-chosen here) and the same `pred_ce_weights`
class weights the base run used. Label/group relaxation are inherited at
whatever the base checkpoint's `cfg` already sets (**off**, per the base
run's own recorded config — verified, not assumed, in §3).

> **CORRECTION (post-preregistration, pre-GPU audit).** The claim above is
> **factually wrong** and is left in place, struck through in spirit but not
> in text, so the error is visible rather than silently erased. The base
> `C1` run (`scripts/train/run_c0_c1.sh` line 104) actually passes
> `--predicate_label_relaxation_enabled true`. §3's audit table (row
> "predicate CE loss (training)") stated the **`TrainConfig` class default**
> (`False`) and this section then wrongly equated that class default with
> "the base run's own recorded config" — those are two different things,
> and the base run's *actual* recorded config (its own launch command) has
> it `True`. This was caught by a dedicated pre-GPU audit, not by this
> document's own review.
>
> **This does not change the hypothesis.** It changes one training-protocol
> choice: **Readout v2's primary pilot deliberately sets
> `predicate_label_relaxation_enabled=false`**, i.e. it does *not* inherit
> this particular base-run setting. This is an isolation choice, made
> explicitly here rather than by accident: label relaxation is itself an
> intervention on the CE loss (soft-label smoothing toward
> `pred_sim_matrix`-similar predicates), and `H_readout` is a claim about
> prototype adaptation alone. Running Readout v2 with label relaxation on
> would test "prototype adaptation + soft labels," a different, uncontrolled
> question. `_predicate_ce_loss` is still called unmodified, exactly as
> stated above — only the flag governing which of its internal branches
> fire is a Readout-v2-specific choice now stated explicitly, not inherited.
>
> **`explicit_spoa_enabled` and `text_conditioned_projection_enabled` were
> never named in this document at all** — a second, related omission caught
> by the same audit. Both must also be `false` for Readout v2 training, for
> a stronger reason than label relaxation: `explicit_spoa_enabled` changes
> `rel_feat` itself (`ProgressiveRelationalDecoder.forward_pairs`:
> `rel_feat = rel_seed(fused_feat) + spoa_fusion(spoa_state)` when `true`,
> vs. `rel_seed(fused_feat)` alone when `false`), and
> `text_conditioned_projection_enabled` changes what
> `adaptive_predicate_logits` scores (via `text_relation_features`). Every
> existing WPRD number this programme has ever reported (`C0`, `C1` seed
>1234, `C1` seed 5678) was evaluated with **both flags `false`**
> (`tools/c0_c1_evaluate.py` hardcodes them) — so `true` during Readout v2
> training would not even have matched what "the `C1` checkpoint's `rel_feat`"
> means anywhere else in this programme, quite apart from the new train/eval
> mismatch it would introduce within this experiment specifically.
>
> **Resolved value for all three, Readout v2 primary pilot only:**
> `explicit_spoa_enabled=false`, `text_conditioned_projection_enabled=false`,
> `predicate_label_relaxation_enabled=false`. §10's "held fixed" list is
> amended below to name these explicitly by value, not just by module.
>
> **Historical results are unaffected.** `docs/PAPER_C_C1_RESULT.md` and
> `docs/PAPER_C_C1_SEED2_RESULT.md` (seed 1234 / seed 5678 `C1` WPRD) were
> produced by `tools/c0_c1_evaluate.py`, which already hardcodes all three
> flags to their eval-time values independent of what any training script
> passes. Nothing about the locked `WPRD=0.5749881522134409` /
> `WPRD=0.572551887041989` results, their replication classification, or
> the geometry-repair conclusion changes because of this correction.

**Total Readout v2 training objective:**

```
L = L_readout_v2_ce  +  L_anchor(P, E)
```

**No prior term, no calibration term, no geometry term, no other existing
loss (`l_spoa`, `l_ground`, `l_relationness`, `l_calibration_kl`,
`l_calibration_rank`, `l_text_predicate_ce` against `E`, `l_role_swap_rank`,
etc.) is active.** All backbone parameters are frozen, so even if those
loss terms were computed they would contribute zero gradient to anything
trainable; they are simply not computed, to keep the training script
minimal and auditable (§8 "the minimal files that must change").

## 7. Initialization — exact, and the trap named in advance

`P_0 := E`, where `E` is computed via `encode_predicate_vocab` from the
**loaded, resumed checkpoint's own CLIP text encoder** — i.e., the encoder
weights present in `checkpoints/C1_seed1234.pt` (§10, the fixed baseline),
**after** `resume_from` has loaded them, not from a separately-instantiated
fresh CLIP model. This is the same `E` that `Readout v0`'s own WPRD number
was computed against (`tools/c0_c1_evaluate.py` builds `pred_emb` the same
way, from the same loaded checkpoint), so `P`'s starting point is
byte-identical to `Readout v0`'s frozen prototypes, not an approximation of
them.

**Copy, not alias — the named trap.** `P` must be created as
`nn.Parameter(E.detach().clone())`. If `P` is instead created as a view or
an alias of the tensor `encode_predicate_vocab` returns (e.g. by wrapping
that exact tensor in `nn.Parameter` without `.clone()`), in-place
optimizer updates could corrupt the `E` reference other code paths hold
(e.g. if `E` is reused for `Readout v0`'s own logging within the same
process). The offline smoke test (§17) checks this explicitly:
`P` and `E` must be equal at init and *independent* tensors
(`P.data_ptr() != E.data_ptr()`).

**Gradients.** `P.requires_grad_(True)` must be set explicitly at
construction; every other parameter must be set `requires_grad_(False)`
explicitly at the same point (not relied upon implicitly), so a parameter
audit (§17) can assert the exact trainable set is `{predicate_prototypes}`
and nothing else.

## 8. Interaction with prior / calibration / existing text adapter

- **Prior**: not composed into `z = score(r, P)` at all (§6 — no prior term
  in `L`, and the WPRD-scored channel, §11, is `z` alone, exactly mirroring
  how `Readout v0`'s WPRD channel is `text_logits` alone at
  `ensemble_alpha=0.0`, never prior-composed). Prior control must still read
  exactly `0.500000` for the same arithmetic reason it does for every
  existing arm (`docs/within_pair_discrimination.py`'s cancellation
  argument, checkpoint-independent) — this is a gate, not an assumption
  (§17, §19).
- **Calibration** (`calibration_gate`, `bias_residual_head`,
  `adaptive_calibration_enabled`): entirely on the **classifier** branch
  (`calibrated_predicate_logits`), which Readout v2 does not use or modify.
  Frozen, inert, untouched.
- **Existing text adapter** (`text_space_projection`,
  `text_conditioned_projection_enabled`): left at its registered, **disabled**
  setting, exactly as `Readout v0` runs it. Readout v2 does not re-enable it
  — doing so would combine two interventions (a learned relational
  projection *and* adaptive prototypes) in one experiment, which §12
  explicitly forbids.

## 9. Readout baselines (exact, per the task's §11)

- **`R0`**: `C1` repaired geometry (checkpoint `checkpoints/C1_seed1234.pt`,
  unmodified) + deployed frozen-CLIP readout, i.e. **the already-existing
  `runs/eval_C1/result.json`** (`ensemble_alpha=0.0`, `WPRD=0.5749881522134409`).
  Not re-run — reused as-is, because Readout v2 with the flag off must
  reproduce it bit-exactly (§13), which is the actual regression test; a
  fresh `R0` re-run would not test that.
- **`R2`**: identical checkpoint, identical geometry, `predicate_prototypes`
  trained per §6, evaluated via `score(rel_feat, P)` alone (no ensemble).
- **`REF`**: the existing `predicate_classifier` head — diagnostic reference
  only (§2's table). **Not** a target `R2` is tuned toward (task §24,
  binding here without qualification).

**Primary endpoint:** `ΔWPRD_readout = WPRD(R2) − WPRD(R0)`, paired cell
bootstrap, identical procedure to `tools/c0_c1_compare.py` (2,000 resamples,
same 20,016-cell population, same semantic-identity verification as
`docs/PAPER_C_C1_SEED2_RESULT.md` §7 — this is a NEW pairing check, not
inherited, because the population must be re-verified against `R2`'s own
dump the same way it was for `C0`/`C1` seed1/seed2).

## 10. Experimental isolation (task §12/§10 of Phase-transition prompt)

Held fixed, all inherited unmodified from `checkpoints/C1_seed1234.pt`:

- geometry (`geom_input_pixel_space=true`, `geom_fourier_scale=0.01`) —
  **not evaluated for change; frozen weights make this a checked
  invariant, not a promise** (§17 gate: `geom_B_std` and geometry-channel
  stats must match `runs/eval_C1/result.json` exactly)
- backbone (CLIP vision + text encoders, as loaded from the checkpoint)
- `ProgressiveRelationalDecoder` (all of it — node/edge layers, fusion gate,
  Fourier encoder)
- **`explicit_spoa_enabled=false`, `text_conditioned_projection_enabled=false`,
  `predicate_label_relaxation_enabled=false`** — pinned by value, not just by
  module, after the pre-GPU audit correction above. `explicit_spoa_enabled`
  and `text_conditioned_projection_enabled` must match
  `tools/readout_v2_evaluate.py`'s hardcoded eval-time values exactly (both
  `false`) or `rel_feat`/the readout's scoring input differ between train and
  eval; `predicate_label_relaxation_enabled=false` keeps `l_readout_v2_ce`
  a vanilla weighted/focal CE with no second intervention.
- dataset, split (VG150, `datasets_vg150_clean`, local-jsonl)
- GT boxes, pair construction (`eval_sgg_use_gt_pairs=true`, PredCls)
- prior (`frequency_prior_train.json`, `alpha=3.75` where composed at all —
  not composed into the WPRD-scored channel, §8)
- WPRD estimator (`tools/within_pair_discrimination.py`, unmodified)
- evaluation population (10,401 images / 132,556 pairs / 20,016 cells,
  re-verified not assumed, §9)

**Changed:** only `predicate_prototypes` exists and is trained; only the
new `score(rel_feat, P)` channel is added to the dump; everything else is
architecturally incapable of moving because its `requires_grad` is `False`.

Why seed `1234`, not `5678`: `checkpoints/C1_seed1234.pt` is the original,
primary registered `C1` arm (`docs/PAPER_C_C1_RESULT.md`); `5678` served its
purpose as the replication check (`docs/PAPER_C_C1_SEED2_RESULT.md`) and is
not re-used here, to avoid quietly running two experiments (readout
adaptation and seed choice) at once.

## 11. Baseline provenance re-check (before this experiment touches the checkpoint)

Before any training: `sha256sum checkpoints/C1_seed1234.pt` must be
recorded and compared against the untouched file (this document does not
yet run that check — it is a §17 CPU-validation step, not assumed here).
`checkpoints/demo_best/pure_best_adapt_light_mR50.pt` must remain
byte-identical to its already-verified hash (`docs/PAPER_C_C1_SEED2_RESULT.md`
§3) — Readout v2 never writes to either file; a new checkpoint
(`checkpoints/readout_v2_seed1234.pt` or similar) is saved separately.

## 12. Hyperparameters (fixed in advance)

| | value | rationale |
|---|---|---|
| `λ_anchor` | **0.5** | order-of-magnitude choice: at init `cos(P_i,E_i)=1` for all `i` so `L_anchor=0`; as `P` moves, `λ_anchor=0.5` makes the anchor term comparable in scale to a focal CE loss with 51 classes at typical confidence (empirically checked in the offline smoke test, §17, not tuned against WPRD — no WPRD number is computed before this value is fixed) |
| optimizer | AdamW, same as base training (`cfg` inherited) | only `P` has gradients; no reason to depart from the existing optimizer choice |
| learning rate | `2e-3` for `P` only | ~100× the base run's `2e-5` backbone LR — justified because `P` is 39,168 scalars with no backbone noise to protect against, and because the entire signal must move a 51×768 matrix, not fine-tune a large network; **this is a pilot-stage choice, revisited only if §21/§22 pilot diagnostics (prototype movement, loss curve) show instability, not to chase a WPRD number** |
| batch composition | GT-positive pairs only, from `datasets_vg150_clean` train split, identical loader to base training | matches how `l_text_predicate_ce` was already computed against `pos_pred_ids` in the base run (§3) |
| seed | `1234` | matches the frozen baseline checkpoint's own seed, for traceability; `P`'s own init is deterministic (`copy of E`), so seed only affects data shuffling and any dropout in frozen layers (irrelevant — frozen layers run in `eval()` mode, §14) |
| pilot budget | 3,000 GT-positive training pairs (≈1 short epoch-equivalent at this dataset's positive-pair density), batch 64, single pass | matched in spirit to this programme's own pilot-sizing convention (Track B's B1 pilot, `docs/TRACK_B_C_ACTION_QUEUE.md`); cheap enough to run and inspect before committing to a full budget |
| full budget (only after pilot passes) | **not fixed here** — to be set from what the pilot's loss/movement curve actually shows, and stated in an amendment before it is run, per this document's own §13 discipline (no future endpoint result may retroactively edit this registration; a *separate*, dated amendment is required for the full-run budget) |

## 13. Endpoints

**PRIMARY**: `ΔWPRD_readout = WPRD(R2) − WPRD(R0)` on the full VG150
validation split (§9), paired cell bootstrap, 2,000 resamples.

**SECONDARY**: weighted WPRD, R@50/mR@50 (WPRD-side composition, §12 of
`docs/PAPER_C_C1_SEED2_RESULT.md`'s estimator-distinction convention —
not the built-in PredCls evaluator table), prior-argmax agreement.

**MECHANISTIC**:
1. prototype displacement `‖P_i − E_i‖` and `1 − cos(P_i, E_i)`, per
   predicate and aggregate
2. per-predicate `ΔWPRD` (does the gain/loss concentrate the way the
   forensic evidence predicts, or land somewhere unexpected)
3. spatial vs. non-spatial `ΔWPRD`, same registered grouping as
   `docs/PAPER_C_C1_SEED2_RESULT.md` §10 (not redesigned)
4. geometry invariants (§10): `n_constant_channels`, `geom_B_std`,
   per-channel std — must match `R0`'s exactly
5. prior control (`prior_control_wprd`) — must read exactly `0.5`
6. open-vocabulary compatibility check (§14): cosine similarity between `P`
   and a small set of **predicate strings never in the 51-class vocabulary**
   (e.g. VG150 raw-name predicates outside the standard 150/51 restriction,
   or hand-written near-synonyms of existing predicates), scored against
   both `E` (control) and `P` (treatment), to see whether `P`'s neighborhood
   structure around unseen text remains CLIP-like or has collapsed. This is
   descriptive, not a pass/fail gate (§4's honesty requirement — the
   repository cannot mount a true trained zero-shot eval, so this check
   cannot certify open-vocabulary preservation, only report whether the
   cheapest observable symptom of collapse is present or absent).

## 14. Open-vocabulary check — exact protocol, bounded honestly

Given §4's finding, this experiment **cannot** produce a trained,
generalization-tested open-vocabulary result (no held-out-predicate
training run exists to build one from, and building one is out of scope
here — it would be a different, larger experiment). What it **can** do:

1. Take a small fixed list of predicate strings not in the 51-class
   `pred_vocab` (e.g. `"stacked on"`, `"leaning against"`, `"tied to"` —
   real relational language, chosen for being plausible VG-style predicates
   absent from the 51-class list, not adversarial nonsense).
2. Encode them through the same frozen CLIP text encoder as `E`
   (`encode_predicate_vocab`, unchanged).
3. For each unseen predicate embedding `u`, report `max_i cos(u, E_i)` vs.
   `max_i cos(u, P_i)` and the identity of the nearest neighbor in each
   case.
4. Report, do not grade: if `P`'s nearest-neighbor structure for unseen
   text is grossly different from `E`'s (e.g. an unseen spatial predicate's
   nearest neighbor under `P` is a possession predicate, where under `E` it
   was a spatial one), that is evidence of semantic collapse, reported as
   such. If it is similar, that is evidence of preservation — again
   reported, not claimed as proof of open-vocabulary capability, which
   would require an actual held-out training/eval split this repository
   does not have (§4).

## 15. Failure criteria (fixed in advance)

Stop, diagnose, do not tune around, if any of:

- geometry channel stats or `geom_B_std` differ from `R0`'s (§10)
- `prior_control_wprd != 0.5` (beyond float noise, `<1e-6`, matching every
  prior arm's tolerance)
- flag-off (`readout_v2_enabled=false`) does not reproduce `R0`'s
  `result.json` bit-exactly
- any parameter outside `{predicate_prototypes}` has a nonzero gradient
  (checked directly, not inferred from loss behavior)
- `P` initialization does not exactly equal `E` at step 0
  (`torch.equal`, not `allclose`)
- logits are non-finite (`NaN`/`Inf`) at any point
- `‖P_i − E_i‖` diverges (unbounded growth, no plateau) within the pilot
  budget
- `1 − cos(P_i, E_i)` collapses to ~1 for many `i` (prototypes rotating to
  near-orthogonal to their init — semantic collapse) within the pilot
  budget
- training/evaluation population mismatch (population, cell count, or
  semantic cell identity differs from `R0`'s, §9)
- WPRD improvement is fully explained by a prior/calibration shift (not
  structurally possible here since no prior is composed into the scored
  channel, §8 — checked anyway, not assumed)

## 16. Success criteria (fixed in advance, pilot stage — no arbitrary WPRD bar)

Per the task's own instruction (§22/17 of the phase-transition prompt),
**no fixed WPRD threshold is registered for the pilot.** The pilot is
healthy if:

- training is stable (finite, non-divergent loss)
- `P` moves from `E` in a bounded, non-collapsing way
- geometry and prior invariants hold exactly (§10, §15)
- WPRD on `R2` shows a coherent directional change (up or down — a
  negative, stable, mechanistically legible result is still informative
  and is not "rescued")
- the flag-off regression test passes bit-exactly

A full-budget experiment's success bar (if any) will be set in a
**separate**, dated amendment after the pilot, informed by what the pilot's
own numbers show — consistent with §16 of this document and the task's
explicit instruction not to import the classifier reference (`≈0.5974`) as
a target now.

## 17. Regression tests (before any GPU use)

New file `tests/test_readout_v2.py`, covering exactly the task's
enumerated list:

1. flag-off (`readout_v2_enabled=false`) behavior: `predicate_prototypes`
   does not exist as a model attribute at all, or exists but is provably
   never read (`predicate_scores` output identical to pre-change code path
   — implemented as: no new attribute is even created unless the flag is
   on, so this is true by construction, and the test asserts the attribute
   is absent)
2. `P` shape: exactly `(51, 768)`
3. `P` initialization: `torch.equal(P.detach(), E.detach())` at construction,
   **and** `P.data_ptr() != E.data_ptr()` (§7's named trap)
4. predicate indexing: the row order of `P` matches `pred_vocab`'s order
   (the same order `E`, `predicate_classifier`, and the frozen prior all
   already use — checked against `B.classes`/`col_label` ordering used
   throughout this programme's own tooling, §3 of `docs/PAPER_C_C1_SEED2_RESULT.md`)
5. anchor loss: `L_anchor(E, E) == 0` exactly at `P=E`; `L_anchor` increases
   monotonically as a row of `P` is rotated away from its `E` row (synthetic
   check, no real data needed)
6. gradient flow: after one backward pass on synthetic `rel_feat`/labels,
   `P.grad` is non-`None` and finite; every other parameter's `.grad` is
   `None` or exactly zero
7. prior composition: the `R2`-scored channel (`score(r, P)`) contains no
   reference to `pred_log_prior`, `freq_bias`, or any prior tensor —
   checked by call-graph inspection (the new method takes no prior
   argument at all) and by a synthetic test that changing the prior file
   does not change `R2`'s logits for fixed `r, P`
8. checkpoint save/load: save a model with `readout_v2_enabled=true`,
   reload, `P` round-trips exactly; reload the **same** file with
   `readout_v2_enabled=false` does not error (extra key is ignored, not
   fatal) — a named, tested failure mode from §3's audit note
9. deterministic inference: two forward passes on the same frozen `r, P`
   produce bit-identical logits (no dropout/stochasticity on the new path
   — `score()` has none, checked directly)
10. training/evaluation parity: the same `score(r, P)` function is called
    in both the training loss (§6) and the eval/dump path (§9) — a single
    shared method, not two independent implementations that could drift
11. no geometry modification: `geom_feats_torch`, `ProgressiveRelationalDecoder`,
    and every geometry-related config flag are byte-unreferenced by the new
    code (call-graph check, not just "the number didn't change")
12. no accidental prior contamination: covered by (7); restated as its own
    test against the actual training loop's assembled loss dict, asserting
    no prior-derived tensor appears in the `L_readout_v2_ce`/`L_anchor`
    computation graph

**The full existing test suite must be run and must still pass.** No
failure may be ignored or skipped to make this land.

## 18. CPU-only offline validation (before GPU, after tests pass)

1. static source check: grep-level confirmation that
   `openvocab_rel/geometry.py`, the geometry branch inside
   `relational_model.py::_forward_impl`, and `tools/within_pair_discrimination.py`
   have zero diff against the `C1` seed1234-validated commit
2. deterministic forward pass on a handful of real cached `rel_feat` rows
   (reuse `runs/eval_C1/pair_logits.pt`, read-only) through the new
   `score(r, P)` path at `P=P_0=E`: logits must equal `Readout v0`'s stored
   `text_logits` for the same rows, bit-for-bit (the strongest possible
   flag-equivalence proof — not just "flag off skips the code," but "flag
   on at `P=E` reproduces flag off exactly")
3. parameter-count audit: exactly 39,168 trainable parameters when
   `readout_v2_enabled=true`; zero new trainable parameters otherwise
4. `P`-initialization equality check (§17.3, repeated here against the real
   checkpoint, not a synthetic one)
5. logits-finite check on the same cached rows
6. flag-off equivalence: load `checkpoints/C1_seed1234.pt` with
   `readout_v2_enabled=false`, run the existing eval path unmodified,
   diff against `runs/eval_C1/result.json` — must match bit-exactly (this
   is the real regression test; §18.2 is the smoke test that predicts it
   will)
7. prior-control calculation: unaffected, since `prior_control_wprd` is
   computed from the prior tensor alone (`WPD.wprd(Gs, B.prior[...])`,
   §3 of `tools/within_pair_discrimination.py`) and never touches `P`
8. open-vocabulary compatibility check (§14), run once at `P=P_0=E`: by
   construction, `max_i cos(u,E_i) == max_i cos(u,P_i)` at init — a sanity
   check on the check itself before any training moves `P`

No training happens in this section. All of it is read-only against
existing artifacts or a handful of synthetic tensors.

## 19. Implementation plan — files (verified against source, not guessed)

| file | change |
|---|---|
| `openvocab_rel/config.py` | add `readout_v2_enabled: bool = False`, `readout_v2_lambda_anchor: float = 0.5`, `readout_v2_lr: float = 2e-3` |
| `openvocab_rel/train.py` | mirror the three new flags as CLI args (existing pattern, e.g. line 858's style); add the "freeze all but `predicate_prototypes`" block, gated by the flag, executed right after `resume_from` load; add the new loss terms (§6) to the loss dict, gated by the flag |
| `openvocab_rel/models/relational_model.py` | add `predicate_prototypes` (constructed lazily / only when the flag is on — see `_init_readout_v2` below), a `_init_readout_v2(self, E: torch.Tensor)` method (§7's exact copy semantics), and a new `mode="adaptive"` branch in `predicate_scores` (does not touch the existing `"text"`/`"classifier"`/`"ensemble"` branches — pure addition) |
| `openvocab_rel/evals.py` | in the existing dump-writing block (the one that already writes `text_logits`/`cls_logits`/`pred_emb`), add one new key, `adaptive_logits`, written **only** when `readout_v2_enabled` — existing keys untouched |
| `tools/readout_v2_evaluate.py` (new) | mirrors `tools/c0_c1_evaluate.py`'s structure exactly (same monkey-patch-then-`train_mod.main(argv)` pattern), but scores WPRD off the new `adaptive_logits` channel instead of `text_logits`/`fixed_ensemble` |
| `tools/readout_v2_compare.py` (new, or reuse `tools/c0_c1_compare.py` directly) | paired bootstrap `R2` vs `R0`, identical procedure; `c0_c1_compare.py` is generic over its two `result.json` inputs and needs no change — reused as-is with `--c0 runs/eval_C1/result.json --c1 runs/eval_readout_v2/result.json` (naming kept as `--c0`/`--c1` flags for tool reuse; semantically these are `R0`/`R2`, documented in the invocation) |
| `scripts/train/run_readout_v2.sh` (new) | mirrors `run_c0_c1.sh`'s structure; pins `--resume_from checkpoints/C1_seed1234.pt --reset_epoch true --readout_v2_enabled true` and the new hyperparameters (§12) |
| `tests/test_readout_v2.py` (new) | §17 |

**Geometry code (`openvocab_rel/geometry.py`, the geometry branch inside
`_forward_impl`) is not in this table and must not be touched.**

## 20. Branch

`research/pure-complete-readout-v2`, created from the current
`research/architecture-breakthrough` branch (which already carries the
validated `C1` geometry repair and both seed results). No unrelated
refactor is bundled into this branch's commits.

## 21. Stopping rules

- If §15 fires at any stage (tests, CPU validation, or pilot): stop,
  document the failure in a result document, do not silently retry with
  changed hyperparameters.
- If the pilot passes (§16) but a full-budget run has not yet been
  registered: stop and write the amendment (§12) before launching it.
- No rescue architecture (MLP, mixture, low-rank residual) is added
  automatically if candidate A underperforms. A failed `H_readout` pilot
  produces a diagnosis, per the task's explicit instruction, not an
  uncontrolled search over candidates B–E.

## 22. What this document does not decide

- The full-budget training schedule (§12, deferred to a post-pilot
  amendment).
- Whether a positive pilot result justifies a third architecture variant
  (B–E) — out of scope unless the pilot's own diagnostics (§16) point at a
  specific, named failure mode that a specific alternative would fix.
- Whether `PURE Complete Geometry v1` should eventually be re-opened for a
  third seed — explicitly not this document's question (task: "DO NOT run a
  third geometry seed").
