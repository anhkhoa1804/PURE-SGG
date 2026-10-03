# Paper C — Open-vocabulary PURE successor: architecture design (no code, no GPU)

> **Superseded design record (2026-10-03).** This proposal is retained for
> historical context and is not an authorized future experiment. Paper C is
> closed; see [PAPER_C_STATUS.md](research/PAPER_C_STATUS.md).

This is a **design document only**. No model code was modified, no GPU job was
launched, no existing preregistration was rewritten, and the C1 checkpoint /
geometry contract is untouched by this document. Nothing below is measured
evidence until an experiment in §6 is actually run and separately
preregistered.

**Evidence labels used throughout:** `[VERIFIED]` = confirmed directly against
source code or a locked result document this session; `[MEASURED]` = a number
already produced by a real run; `[INFERENCE]` = a reasoned combination of
verified facts, not itself measured; `[HYPOTHESIS]` = an unproven claim this
design exists to test.

## 0. State reconciliation (must be resolved before this design is acted on)

The task that produced this document stated two things that **conflict with
this session's own verified evidence** (drawn directly from this repository's
locked result records, read earlier in this session):

1. **"baseline R0 WPRD ≈ 0.5542"** — this session found no record of this
   number. The verified pre-repair baseline in this program's own locked
   records is **C0 = 0.5667271196244782**. `0.5542` does not match C0, C1, or
   any Readout-v0/v2 number this session observed. It may refer to a genuinely
   different, older baseline this session has not read, or it may be a
   misremembering. **Flagged, not silently corrected or silently trusted.**
2. **"both exceed the preregistered +0.01 threshold"** — this directly
   contradicts the explicit, locked scientific classification already on
   record this session:

   | arm | WPRD | Δ vs C0 | 95% CI | meets Δ≥+0.01? |
   |---|---|---|---|---|
   | C1 seed 1234 | 0.5749881522134409 | +0.008261 | [+0.00383, +0.01270] | **NO** |
   | C1 seed 5678 | 0.572551887041989 | +0.005825 | [+0.00138, +0.01004] | **NO** |

   Registered classification: *"GEOMETRY EFFECT REPLICATED IN DIRECTION,
   EFFECT SIZE IS WEAK/VARIABLE, MATERIAL +0.01 THRESHOLD NOT MET."* Both
   seeds are positive and both fail the registered threshold. This document
   proceeds on the **verified** numbers above, not the task prompt's summary.

This does not change anything about the open-vocabulary design below — the
geometry repair is `CLOSED` either way (weak-positive, replicated,
locked-for-reuse) — but the record should not carry the incorrect framing
forward. **Report this discrepancy to whoever supplied the task context.**

Also worth stating precisely: Readout v2's clean pilot is a separate,
in-progress line of work (this session), evaluating `predicate_prototypes`
(closed-set, 51 fixed rows) against the C1 representation. It is **not** the
subject of this document and its result does not gate anything here — see §8
for exactly how it relates.

---

## 1. Proposed architecture

### Retain from PURE / C1

- **Pair-relative geometry contract** (`geom_input_pixel_space=true`,
  `geom_fourier_scale=0.01`) — `[VERIFIED, LOCKED]`. Twice-replicated
  weak-positive effect. Not reopened by this design. Feeds into `r` exactly
  where it does today.
- **`ProgressiveRelationalDecoder`** (node/edge layers, fusion gate, Fourier
  geometry encoder) producing `r = rel_feat ∈ R^768` — `[VERIFIED]` unchanged.
  No evidence in this program implicates this component; the entire Readout
  v2 forensic chain specifically localized the bottleneck to the **readout**,
  not the representation (`docs/PAPER_C_PURE_READOUT_FORENSIC.md`). Reopening
  it now would undo an already-closed question.
- **Frozen CLIP text encoder** as the source of `e(p)` via the existing
  `encode_predicate_vocab`/`clip_text_features` path — `[VERIFIED,
  clip_utils.py]`. Already vocabulary-agnostic: takes an arbitrary list of
  predicate strings, `@torch.no_grad()`, one call per vocabulary set.
- **`text_space_projection`** (`relational_model.py:614-619`:
  `LayerNorm(768)→Linear(768,768)→GELU→Linear(768,768)`, ≈1.18M params) as the
  starting point for `f(r)` — `[VERIFIED]` this module **already exists**,
  is **already wired** through `text_relation_features`/`text_predicate_logits`
  (`relational_model.py:851-860`), and its consumer, `score()`
  (`relational_model.py:1024-1030`), is **plain cosine similarity with no
  hardcoded class count** — architecturally open-vocabulary-capable today. It
  has only ever been trained against a **frozen, closed 51-column** `E`, with
  a plain (unscaled) CE loss, and has never been evaluated under a
  predicate-disjoint protocol.
- **`configs/predicate_metadata_vg150.json`** group labels (spatial, contact,
  possession, action, pose, attribute, part-whole, textual, other) —
  `[VERIFIED]` already exist, already used for role-swap filtering / long-tail
  diagnostics per the file's own description. Directly reusable for the
  predicate-disjoint split (§3) and hard-negative mining (§2) — no new
  annotation effort.
- **The WPRD estimator** (`tools/within_pair_discrimination.py`) and its cell
  / population / prior-control discipline — extended (seen/unseen
  subsetting, §4), not replaced.

### Discard / demote

- **`predicate_classifier`** (closed-set `Linear(768→51)`) — already named a
  diagnostic-only ceiling in the Readout v2 preregistration. This design goes
  further: it must be **excluded from the open-vocabulary inference path
  entirely**, not merely down-weighted via `ensemble_alpha=0`, because its
  mere presence in a deployed ensemble silently reintroduces a closed-set
  ceiling on any predicate it was never trained on.
- **`predicate_prototypes` (Readout v2's `P`, fixed 51×768) as a deployment
  head** — see §8. Its *finding* is valuable; its *parameterization* (fixed
  cardinality, no text-conditioning at inference) is structurally
  incompatible with "score an unsupervised predicate via its text embedding."
- **`ensemble_alpha`-based classifier/text mixing** — a historical artifact
  from when the text branch alone was not trusted as primary. This design's
  premise is that a correctly-trained text branch *is* primary; the mixing
  machinery adds uncontrolled complexity here.
- **Calibration / bias-residual / adaptive-prior machinery**
  (`calibration_gate`, `bias_residual_head`, `freq_bias_*`) — `[VERIFIED]`
  classifier-branch-only today, and structurally meaningless for a predicate
  with no training frequency at all (no row in `frequency_prior_train.json`).
  Not ported into the new head "because it was always there."

### Exact proposed data flow

```
image, GT boxes
   → CLIP vision encoder (frozen)              [unchanged]
   → pair-relative geometry (C1 contract, LOCKED, 8 channels)
   → Fourier encode (geom_fourier_scale=0.01, pixel-space)  [unchanged]
   → ProgressiveRelationalDecoder (node+edge+fusion gate)
   → r = rel_feat ∈ R^768                       [UNCHANGED FROM C1 -- frozen]

f(r) = text_space_projection(r) ∈ R^768         [existing module, RETRAINED under a new objective]
e(p) = normalize(CLIP_text_encoder(prompt(p)))  ∈ R^768, for ANY predicate string p, seen or unseen
z(r,p) = (1/τ) · cos(f(r), e(p))                [temperature τ is the one new scalar]
```

- **Where geometry enters:** unchanged — before `r` is formed, inside the
  decoder. Locked, not reopened.
- **Where CLIP text enters:** unchanged mechanism (`encode_predicate_vocab`),
  computed from the same frozen text encoder that already produced every
  `E` this program has used.
- **Is CLIP frozen initially:** **yes**, both towers, for the entire
  experiment ladder through Stage B (§6). Unfreezing the CLIP *text* tower
  specifically risks specializing its embedding space to the 51 seen
  predicates' own prompt template — exactly the kind of overfitting that
  would defeat unseen-predicate generalization. Named explicitly as a risk,
  not attempted by default (§5, falsification test 4 partially covers this).
- **Is a projection/adaptor needed:** yes — `f(r)` is required (the relation
  representation and the CLIP text space are not assumed pre-aligned). A
  symmetric text-side projection `g(e)` is **not** part of the minimum
  design; it is an add-only-if-needed option (§6), since the smallest
  defensible architecture keeps `e(p)` as CLIP produces it.

### Scoring function choice

| option | verdict |
|---|---|
| plain cosine (current `score()`) | rejected as the loss's primary scale — no temperature means the softmax's effective margin is whatever raw cosine similarities produce, brittle for a full contrastive objective |
| **temperature-scaled cosine**, `τ` learned (clipped) or fixed by a cheap pilot sweep | **PRIMARY.** Adds at most 1 scalar parameter. Directly matches CLIP's own contrastive-loss convention. Strict superset of Readout v2's un-scaled cosine, so nothing already measured becomes incomparable. |
| bilinear `f(r)ᵀ W e(p)` | rejected at this stage — adds ≈590K new params (comparable to `text_space_projection` itself) with no forensic evidence motivating it specifically; revisit only if temperature-scaled cosine underperforms *and* a diagnosed failure mode (e.g. anisotropic CLIP embedding geometry) points at it |
| MLP compatibility head `MLP([f(r); e(p)])→scalar` | rejected — cannot be scored against a large candidate set in one matrix multiply (no shared embedding space), and reproduces the exact "Design Trap" (a shared nonlinear transform not motivated by evidence) already named and rejected in the Readout v2 preregistration §5 (candidate D) |

### Smallest scientifically defensible architecture (Stage A, §6)

- `r`: unchanged, C1 contract, frozen decoder.
- `f(r)`: the **existing** `text_space_projection` module, retrained.
- `e(p)`: existing `encode_predicate_vocab` output, CLIP frozen.
- `τ`: one learned scalar, initialized ≈0.07 (CLIP convention), clipped to
  e.g. `[0.01, 0.5]` to prevent collapse.
- **New trainable surface: zero new `nn.Module`s.** The module already
  exists in the codebase; what changes is the training objective (§2) and
  the vocabulary used for negatives (§3), not the architecture's parameter
  count. This is a virtue, not an accident: Stage A is a retraining/
  re-evaluation exercise on an existing component, not new code.

---

## 2. Training objective

### Notation

- `r_i` = `rel_feat` for GT-positive pair `i` in a minibatch (positives-only
  sampling, matching this repo's existing, documented behavior —
  `docs/known_issues.md`: `negative_pair_ratio` has no effect while
  `use_all_pairs=false`, so "negative" below always means **negative
  predicate class**, never a negative subject-object pair).
- `y_i ∈ Seen_train` = the ground-truth predicate for pair `i` (§3 defines
  the partition; unseen predicates never appear as any `y_i`).
- `E_seen = {e(p) : p ∈ Seen_train}`, size `S`.

### Primary loss — relation→text direction (vocabulary-wide)

```
L_r2t(r_i, y_i) = -log[ exp(z(r_i,y_i)/τ) / Σ_{p∈Seen_train} exp(z(r_i,p)/τ) ]
```

This is architecturally the **same code path** the repo already runs
(`text_predicate_logits` → `_predicate_ce_loss`-style CE) — the differences
are: (a) explicit temperature scaling, (b) plain inverse-frequency class
weighting instead of focal weighting (focal reweighting distorts the
softmax denominator's role as a proper contrastive normalizer — turn it
off for this objective), (c) computed **only** over `Seen_train`, never
including `Unseen_test` columns even as negatives (§3, leakage requirement).

### Symmetric term — text→relation direction (batch-local retrieval)

The task's own framing (`r → f(r) ↕ e(p)`) asks for a genuinely symmetric
objective, not just the closed-CE direction the repo already has. Because
multiple pairs in a batch can share the same predicate label (unlike
standard image-text CLIP batches with one match per row), the correct
multi-positive form (Khosla et al.-style supervised contrastive, adapted to
text anchors) is:

```
L_t2r(p) = -log[ Σ_{i∈batch: y_i=p} exp(z(r_i,p)/τ) / Σ_{k∈batch} exp(z(r_k,p)/τ) ]
```

summed/averaged over each seen predicate `p` present in the current batch.

```
L_align = L_r2t + L_t2r        (same τ, both directions)
```

This is genuinely new relative to what exists today — the repo currently
only ever computes the `L_r2t` direction.

### Hard-negative strategy

Use the existing `configs/predicate_metadata_vg150.json` `group` field:
for seen predicate `p`, its hard negatives are other **seen** predicates in
the same semantic group (e.g. `on`/`over`/`above`/`under`, all
`group=spatial`). Add a margin hinge:

```
L_hard(r_i) = max(0, m - z(r_i,y_i) + max_{p'∈hardneg(y_i)} z(r_i,p'))
```

`λ_hard` tunable, selected the same way as `τ` (§2's WPRD-free criterion
below) — this directly targets the WPRD-relevant regime, since WPRD itself
measures discrimination among same-pair alternates, hardest exactly within
these semantic groups.

### Relation encoder: frozen or jointly trained

**Frozen**, for Stage A through Stage B (§6). Reasons: (a) `r` is the
C1-locked contract — reopening it recreates the exact confound the program
rules forbid; (b) freezing isolates "can `f`/`e` alignment generalize to
unseen predicates" from "did the representation also change"; (c) it
mirrors the now-reverified Readout v2 protocol (frozen backbone, one
retrained head). Joint fine-tuning of upper encoder layers is explicitly
deferred to a later, separately-preregistered phase, triggered only by
Stage A-B evidence of a frozen-encoder ceiling — not a default next step.

### Avoiding pair-frequency memorization

- Class-balanced / inverse-frequency batch sampling across `Seen_train`
  (reuse `predicate_ce_weight_power`/`predicate_ce_max_weight`, already
  present, `train.py`), plus epoch-level predicate-balanced resampling
  (fixed quota per seen predicate per epoch, oversampling tail, capped).
- **This session's own Readout v2 pilot already demonstrated the risk
  directly** `[MEASURED, this session]`: after one short epoch, the five
  predicates whose learned prototypes moved most were exactly the five
  highest-frequency predicates in the training data (`has`, `on`, `wearing`,
  `of`, `with` — matching the pilot's own printed frequency-ranked list
  verbatim). A naive frequency-weighted objective concentrates almost all
  learning signal on head predicates. The open-vocab redesign must correct
  for this from the start, not rediscover it.
- Evaluation (§4) reports **macro**, frequency-blind unseen-mR regardless of
  training-time balancing, so a model that only "solves" head-adjacent
  unseen predicates gets no undeserved credit.
- The random-text and shuffled-text controls (§4) are direct memorization
  falsification tests, independent of balancing.

### Incorporating WPRD without circularity

- WPRD must **never** appear in the training loss, gradient, early-stopping
  criterion, or hyperparameter-selection criterion (`τ`, `λ_hard`,
  architecture choice) until those are frozen by a WPRD-free criterion —
  reusing the exact discipline the Readout v2 preregistration already
  applied to `λ_anchor` ("no WPRD number is computed before this value is
  fixed").
- Concrete selection criterion instead: held-out `Seen_val` contrastive
  loss / top-1 retrieval accuracy (a slice of pairs disjoint from
  `Seen_train`, same predicate vocabulary — never touching `Unseen_test`).
- WPRD is strictly a post-hoc, held-out **endpoint** metric (§4), computed
  once per registered checkpoint. Trying a second hyperparameter after
  seeing a WPRD number is a new, separately preregistered experiment, not a
  continuation of this one.

---

## 3. TRUE predicate-disjoint protocol

### The "relation" placeholder must be excluded

`[VERIFIED, train.py:1526-1527]`: `global_pred_pool` has a synthetic
`"relation"` token appended when absent — this is a closed-set-pipeline
artifact, not a real VG150 predicate, and must **not** be assigned to
either Seen or Unseen. The true pool for this design is exactly the 50
scanned VG150 predicates.

### Partitioning

Stratify by the existing `group` metadata field so that:
- no group is entirely absent from `Seen_train` (else within-group
  generalization becomes untestable — a harder, different question), and
- no group is entirely absent from `Unseen_test` (else a group's "unseen"
  performance could trivially borrow a same-group seen exemplar).

Preferred target: 8-12 held-out predicates in `Unseen_test`, at least one
but no more than half of each group's members, biased toward **mid-frequency**
predicates (a predicate with only a handful of instances gives a noisy WPRD
read even seen; the highest-frequency predicates like `on`/`has` held out
would both gut usable training supervision and make `Seen_train`'s
distribution unrealistic). The exact predicate list is deferred to the
pilot-registration step, decided from actual per-predicate support counts —
**the stratification rule is fixed here, before looking at which specific
predicates are "convenient," to avoid post-hoc cherry-picking.**

Remaining ~38-42 predicates split `Seen_train` (supervision) /
`Seen_val` (a disjoint slice of *pairs* whose label is in `Seen_train`'s
vocabulary — for checkpoint selection, §2; not a second predicate
partition).

### Hard requirements (failure gates, mirroring the Readout v2
preregistration's own §15 posture)

1. **No unseen label in any training batch** — enforced at the data-loading
   boundary: any GT triplet whose predicate ∈ `Unseen_test` is dropped
   before it reaches the loss. (Design-level requirement; implementation is
   out of scope for this document.)
2. **Unseen text may be encoded at inference/eval only** — architecturally
   free (`@torch.no_grad()`, already the case).
3. **No new classifier row, ever** — by construction: the scoring path is
   `cos(f(r), e(p))`, not `Linear(768→C)`. Adding a predicate at eval time
   is passing one more string to `encode_predicate_vocab`. This is the
   entire architectural point of discarding the closed-set heads (§1, §8).
4. **Synonym / alias / prompt-template leakage** — the subtlest requirement:
   - **Prompt templates**: `pred_prompt_roles` must use the identical
     template for seen and unseen predicates — one fixed template, frozen
     before the split is finalized, no unseen-specific engineering.
   - **Synonym check**: before finalizing `Unseen_test`, compute
     `e(p)`-cosine between every unseen candidate and every seen predicate
     (reusing the Readout v2 preregistration §14 protocol verbatim). If an
     unseen predicate's nearest seen neighbor is a near-duplicate string or
     known synonym, drop or relabel it — scoring it correctly would show
     lexical overlap in CLIP's pretrained space, not generalization.
   - **VG150-specific near-duplicate risk**: the predicate metadata already
     shows candidate near-duplicate pairs (e.g. `laying on` / `lying on`,
     both `group=pose`, both observed in this session's read of
     `configs/predicate_metadata_vg150.json`). Any such pair **must be
     assigned to the same partition** (both seen or both unseen) — never
     split across the boundary. This is a mechanical, predicate-list-in-hand
     check, done before finalizing the split, not a per-predicate judgment
     call made from memory.
5. **Index-bookkeeping leakage**: the training-time vocabulary tensor must
   be **constructed from `Seen_train` alone**, not sliced at loss time from
   a 50-wide tensor that also contains `Unseen_test` columns (even masked).
   This exact class of bug — a config default silently diverging from what
   a script actually resolves to at runtime — is precisely what the Readout
   v2 preregistration's own CORRECTION block caught after the fact; naming
   it here in advance.

### Semantically-close (not synonym) unseen predicates

Report separately, as a named subgroup in §4's evaluation — a model that
only succeeds on unseen predicates near a seen semantic neighbor (e.g.
unseen `leaning against` vs. seen `against`/`attached to`) and fails on
semantically distant ones is a **weaker but still informative** result, not
a failure. The evaluation must preserve this granularity, not collapse it
into one aggregate unseen-mR number.

---

## 4. Evaluation

- **Seen-predicate performance**: `WPRD_seen`, `R@50_seen`, `mR@50_seen` —
  same estimator, cell/population identity logic unchanged from
  `tools/within_pair_discrimination.py`, restricted to GT rows/cells whose
  predicate ∈ `Seen_train`.
- **Unseen-predicate performance**: `WPRD_unseen` analogous, restricted by
  GT-row predicate ∈ `Unseen_test`. Note: the natural adaptation keeps the
  **full scored candidate set** (`Seen ∪ Unseen`, since `e(p)` is free at
  inference) but restricts the **cell population** to unseen-labeled rows —
  narrowing the *candidate columns* to unseen-only would be an easier, less
  meaningful task (it removes the far more realistic confusion with seen
  predicates). Whether WPRD's existing implementation already supports this
  restriction cleanly is an **open implementation question for the pilot**,
  not assumed here.
- **Harmonic mean**: standard generalized zero-shot convention,
  `H = 2·(mR_seen·mR_unseen)/(mR_seen+mR_unseen)` on mR@50 (and, separately,
  on WPRD_macro). Always report seen/unseen individually alongside `H` —
  never `H` alone, since it can hide a collapsed unseen side the way
  accuracy can hide zero recall on a rare class.
- **Novel-predicate mR**: mR@50 macro-averaged over `Unseen_test` predicates
  only — same estimator convention as this program's existing head/mid/tail
  mR breakdowns.
- **Calibration**: bin by predicted confidence
  (`max_p cos(f(r),e(p))/τ` or softmax max-prob), report empirical accuracy
  per bin and Expected Calibration Error, separately for seen and unseen.
- **Prior-only control**: reuse `prior_control_wprd` verbatim
  (`[VERIFIED]` checkpoint-independent, reads exactly `0.5` in every arm to
  date). For `Unseen_test` predicates the frequency prior is **structurally
  undefined** — no row exists in `frequency_prior_train.json` for a
  predicate excluded from supervision. State this as a structural advantage
  of the design: unseen-side performance cannot be inflated by prior
  composition even in principle.
- **Random-text control**: replace `e(p)` with embeddings of a random
  permutation of predicate **strings not in Seen or Unseen** (common English
  verbs/prepositions absent from VG150's predicate list), same frozen
  encoder/template. Above-chance apparent discrimination here would be
  direct evidence the pathway exploits something other than `e(p)`'s
  semantic content (§5, falsification test 2).
- **Shuffled-text control**: keep the real `Unseen_test` embedding set but
  permute the (row, correct-column) assignment across cells. A correctly
  functioning model should collapse to ≈0.5 (§5, falsification test 1).
- **Geometry-only control**: score using only the 8-dim geometry channel
  (bypass CLIP visual features), reusing this program's own established
  geometry-causal-ablation pattern (`[VERIFIED, exists]
  runs/p70_geometry_causal_ablation`). Establishes a floor: for
  geometry-dominated groups (spatial), a nontrivial geometry-only
  `WPRD_unseen` is *expected*, not itself evidence of visual-semantic
  transfer. Only performance **above** this floor — especially for
  non-spatial groups (contact/action/possession) — counts as evidence of
  genuine image-conditioned generalization (§5, falsification test 3).

---

## 5. Scientific falsification tests

Explicit tests that could **prove the mechanism is not using
image-conditioned semantics**:

1. **Shuffled-text control** (§4): if `WPRD_unseen` does not collapse to
   ≈0.5 under shuffling, the model is not reading row-specific
   predicate-text compatibility at all — some other channel (index leakage,
   positional artifact) is responsible. **Falsifies the core mechanism.**
2. **Random-text control** (§4): above-chance WPRD against semantically
   empty text falsifies genuine text-conditioning — some confound in
   CLIP-text-embedding statistics (e.g. norm correlating with something) is
   being exploited instead.
3. **Geometry-only floor** (§4): if `WPRD_unseen` for **non-spatial** groups
   is statistically indistinguishable from the geometry-only ablation's
   number for those same groups, the model has learned nothing beyond what
   pair geometry alone gives away — a geometry shortcut wearing a CLIP-text
   wrapper, not genuine visual semantic transfer (directly targets the
   over-claim risk named in §9).
4. **Untrained-`f` ablation**: freeze `f` at random initialization and
   evaluate `WPRD_unseen`. If this matches the trained-`f` number, training
   taught nothing; any apparent performance comes from CLIP's own pretrained
   alignment alone — still potentially useful, but a different, weaker claim
   than "PURE's relational representation was successfully aligned" (§9).
5. **Image-free nearest-neighbor baseline** (reuses Readout v2
   preregistration §14's protocol): for each unseen predicate, check whether
   scoring by `e(p)`'s raw text-only proximity to the nearest **seen**
   predicate (no per-image `r` at all — e.g. using that seen predicate's
   historical-average `r`) matches the full model's unseen WPRD. If so, the
   "open-vocabulary" result reduces to lexical nearest-neighbor retrieval in
   CLIP's text space, not per-image relational reasoning.
6. **Batch-composition leakage audit**: a static, unit-level, pre-GPU check
   (analogous to Readout v2's own regression suite) that no `Unseen_test`
   predicate string, synonym, or index ever appears in any tensor
   contributing to the training loss. Its **failure** invalidates every
   downstream number regardless of what is measured — the exact lesson this
   session's own Readout v2 invalid-pilot history (two separate invalid
   attempts, both quarantined) already demonstrated for a different
   mechanism.

---

## 6. Minimal experiment ladder

**R0 → C1 → semantic-alignment pilot → predicate-disjoint pilot → full OV
evaluation.**

- **Stage 0 (context, not re-run)**: C0 `[MEASURED, WPRD=0.5667271196244782]`
  → C1 `[MEASURED, WPRD=0.5749881522134409 seed1234 /
  0.572551887041989 seed5678, weak-positive, replicated, CLOSED]`. Not
  reopened (§0's reconciliation note applies — do not carry forward the
  "both exceed threshold" framing).

- **Stage A — Semantic-alignment pilot (closed-set, no held-out predicates
  yet)**: establish that `L_align` (§2) trains stably and does not regress
  closed-set WPRD relative to Readout v0/v2, on the **full 50-predicate**
  vocabulary. Necessary, not sufficient — no point testing open-vocabulary
  generalization of a broken alignment procedure. Scale: same bounded-pilot
  convention as Readout v2 (≈250 images / ≈3,000 GT-positive pairs, 1 short
  epoch), same base checkpoint `checkpoints/C1_seed1234.pt`, same
  frozen-CLIP-but-serialized protocol this session just reverified
  end-to-end. Gate: finite/stable loss, `τ` in a sane range (not pinned to
  either clip bound), no large WPRD_macro regression vs. Readout v0
  (directional check only — no fixed threshold yet, matching this program's
  own stated pilot-stage discipline).

- **Stage B — Predicate-disjoint pilot (the real test, still small)**: apply
  §3's split at the same small pilot scale. Run every §4 metric and every
  §5 falsification test **at this scale first** — cheap, fast, and any
  falsification failure here is a hard stop before any larger GPU budget.
  This is a genuinely new experiment class for this repository — no
  checkpoint here has ever been trained with predicates withheld from
  supervision (`[VERIFIED]`, Readout v2 preregistration §4). Preregister it
  separately, with its own dated document, before running.

- **Stage C — Full OV evaluation (only after Stage B passes every gate)**:
  full VG150 train / full validation-split endpoint, same population
  verification discipline as every existing C0/C1/Readout-v2 result
  (10,401 images / 132,556 pairs — re-verified against the **new**
  seen/unseen partition, not assumed inherited). Only run after a separate,
  dated amendment fixes the full-budget schedule. **This document does not
  pre-authorize Stage C.**

**Explicit non-goals for now**: no full VG150 run at any stage until Stage B
passes every §5 falsification test; no joint unfreezing of the relation
encoder or CLIP text tower at any stage without a separate,
evidence-triggered amendment.

---

## 7. GO / NO-GO criteria

**GO conditions** (all must hold, cheapest first):

1. Static leakage audit (§3.4): zero unresolved flagged
   synonym/near-duplicate pairs in the finalized `Unseen_test` list.
2. Stage A gate: training stable, `τ` in range, `WPRD_macro` regression vs.
   Readout v0 no worse than **-0.01** (reusing this program's own already-
   agreed +0.01 material-effect size as a "no regression" floor, not a new
   arbitrary number).
3. Stage B falsification tests at pilot scale: shuffled-text and
   random-text controls both land in `[0.47, 0.53]` (collapse to the
   prior-control band); geometry-only floor is computed and reported (no
   pass/fail gate here — a diagnostic ceiling, since some geometry-driven
   signal is scientifically expected for spatial predicates).
4. Harmonic mean `H` (mR@50 basis, pilot scale) is computed and both sides
   reported individually — no fixed numeric bar at pilot scale, but a
   **degenerate** result (either side exactly 0, or unseen mR
   indistinguishable from the random-text control) is an explicit NO-GO for
   scaling to Stage C, regardless of the aggregate number.

**NO-GO conditions** (any one triggers stop-and-diagnose, not tune-around,
mirroring the Readout v2 preregistration's §15 posture):

- Any §5 falsification test fails.
- Any §3.4 leakage audit finding is unresolved.
- Training instability persisting after one documented, pre-specified retry
  (e.g. a single `τ`-init change) — a second failure is a stop, not license
  for an open-ended sweep.
- Unseen-side WPRD indistinguishable from its own operative null (the
  random-text control's WPRD, since unseen predicates have no frequency
  prior — `0.5` is not automatically the right null here).

---

## 8. PURE inheritance decision

**Genuinely valuable, keep:**
- The pair-relative geometry contract (C1, locked) — foundational to `r`
  regardless of readout choice.
- `ProgressiveRelationalDecoder` — no evidence implicates it; the readout is
  the localized bottleneck.
- The `score()`/`text_predicate_logits()`/`text_space_projection` scaffolding
  — already architecturally open-vocabulary-capable; this design's core
  recommendation is to finally *train and evaluate it as such*, not build
  something new.
- `encode_predicate_vocab` / frozen CLIP text encoder as `e(p)`'s source —
  the one component already structurally validated as vocabulary-agnostic.
- The predicate metadata group labels — directly reusable, no new
  annotation effort.
- The WPRD estimator and its discipline — this program's single most
  load-bearing shared infrastructure.

**Historical baggage — diagnostic reference only, not deployment:**
- `predicate_classifier` — already named diagnostic-only; must be fully
  excluded from the open-vocab inference path (not just de-weighted).
- `ensemble_alpha` mixing machinery — unnecessary complexity once the text
  branch alone is trusted as primary.
- Calibration / bias-residual / adaptive-prior machinery — classifier-branch
  only, meaningless for a predicate with no training frequency.

**Should NOT be preserved for architectural loyalty:**
- **`predicate_prototypes` (Readout v2's `P`) as a deployed head.** Its
  *finding* — that adapting predicate directions recovers most of the
  frozen-readout gap in the closed-set regime — directly motivates why
  `text_space_projection` needs a better objective than a frozen closed CE.
  But its *parameterization* (fixed-cardinality learned matrix) cannot score
  an unsupervised predicate and must not be carried forward merely because
  it is the most recent work. This is precisely the "architectural loyalty"
  trap this task's own instructions warn against.
- **Readout v2's specific hyperparameters** (`λ_anchor=0.5`, `lr=2e-3`) —
  fit to a different parameterization (a 51×768 matrix copy-initialized from
  `E`), with no a priori claim on `text_space_projection`'s pre-existing
  ~1.18M-parameter MLP under a new contrastive objective. Re-derive via
  Stage A's own WPRD-free selection criterion (§2).
- **`open_vocab_predicate_primary`'s current default wiring**
  (`[VERIFIED, relational_model.py:907]`: `mode="text" if
  open_vocab_predicate_primary else eval_sgg_predicate_score_mode`) — the
  flag/plumbing is useful and should be kept, but its existing default
  behavior has never been exercised under a predicate-disjoint protocol and
  must be re-verified (a cheap code-reading check) before Stage A, not
  assumed correct because it already exists.

---

## 9. Novelty positioning

**No literature audit has been performed in this session.** Everything below
is inference from this repository's own established evidence only, and must
not be read as a literature-grounded novelty claim.

`[VERIFIED, this repo]`:
- The `score()`/`text_predicate_logits` scaffolding is architecturally
  vocabulary-agnostic but has never been (a) trained with a proper symmetric
  contrastive objective, or (b) evaluated under any predicate-disjoint
  protocol. `--eval_zs_predicates` is reporting-only; no checkpoint in this
  repository has ever been trained with predicates withheld from
  supervision.
- The closed-set forensic decomposition (Readout v2 preregistration §2)
  found ≈96% of the frozen-readout performance gap attributable to
  predicate-direction rigidity, not nonlinearity — specific to the
  **closed-set** regime. Its relevance to open-vocabulary generalization is
  `[INFERENCE]`, not measured: it motivates trying a better-trained
  projection, but does not itself predict generalization to text embeddings
  never seen as a supervision target.

`[INFERENCE]` — what this architecture, if it passes Stage A/B, could
support as a scientific claim: that PURE's pair-relative-geometry-repaired
relational representation (`r`, C1 contract) retains enough
CLIP-visual-semantic content, after an appropriately-trained projection and
contrastive objective restricted to seen predicates, to be scored against
the text embedding of a predicate never supervised during training — at a
rate exceeding a text-only nearest-neighbor baseline and a geometry-only
floor (§4-5). This is a claim about **this architecture's relational
representation carrying transferable, open-vocabulary-compatible semantic
content** — not a claim of state-of-the-art open-vocabulary SGG performance
relative to any published method, which has not been attempted.

`[HYPOTHESIS, unproven]` — whether this exact combination (frozen
PURE-Complete-Geometry-v1 representation + retrained `text_space_projection`
+ symmetric temperature-scaled contrastive loss + a rigorous
predicate-disjoint evaluation with shuffled-text/random-text/geometry-only
falsification controls) is **novel** relative to prior open-vocabulary /
zero-shot scene-graph-generation work is **unknown from this repository
alone**. Frozen-text-encoder, learned-visual-projection dual-encoder scoring
is an established pattern in the broader vision-language literature generally
(CLIP-style zero-shot classification). A literature audit would need to
establish, before any external novelty claim: (a) whether this exact
architecture-on-relational-features combination has prior art in scene graph
generation specifically; (b) whether this design's falsification battery
(especially the shuffled-text and geometry-only floor controls) matches or
improves on existing zero-shot SGG evaluation rigor; (c) whether the
PURE-specific geometry contract is a genuine point of differentiation or an
orthogonal detail from the open-vocabulary community's perspective. **Do not
claim novelty in any external-facing document until that audit is done.**
