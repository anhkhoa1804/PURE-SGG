# Paper C — R2 falsification battery

Prepared while the R2 endpoint evaluation was still running.
**No R2 analysis has been run.** `runs/eval_readout_v2_R2_pilot_v3/result.json`
and `.../pair_logits.pt` do not exist yet — every function this document
references has been built and tested only against synthetic data
(`tests/test_readout_v2_offline_analysis.py`, `tests/test_verify_population_identity.py`).
Nothing here required the GPU, and nothing here ran a real-data analysis
concurrently with R2.

**Purpose, stated precisely**: this battery does not exist to maximize a
positive result for Readout v2. It exists to determine whether any
observed R0→R2 improvement is attributable to **adaptive predicate
prototypes specifically**, as opposed to geometry, the pair prior, lexical
nearest-neighbor structure in CLIP's own text-embedding space,
class-frequency effects, calibration/scale artifacts, or an accidental
population mismatch. A battery that only confirms the intended mechanism,
without being capable of returning a result that would disprove it, is not
a falsification battery — every control below states explicitly what
result would count as evidence *against* the intended mechanism, not only
what would support it.

---

## 0. Why WPRD alone is not proof, and the six-way separation this document uses

`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` and every prior result in this
program already treat WPRD as prior-free by construction (§ arithmetic
cancellation in `tools/within_pair_discrimination.py`). Prior-free is not
the same as mechanism-specific. A WPRD delta could, in principle, be fully
explained by any of:

- the pair-relative **geometry** signal alone (a spatial-layout shortcut
  dressed up in a CLIP-text-scored wrapper) — control 4;
- something in the **pair prior** leaking despite the arithmetic
  cancellation argument (checked directly, not merely trusted) — control 3;
- **lexical/text-space structure** in CLIP's own pretrained embeddings,
  independent of whether `rel_feat` carries any real visual signal —
  controls 5, 6;
- **class-frequency** effects (a gain concentrated on whichever predicates
  happened to receive the most gradient signal, already observed directly
  in this program's own recovery-session finding that prototype
  displacement tracked training frequency) — control 9;
- **calibration**/scale artifacts (a temperature or confidence shift that
  changes recall/precision at an operating point without reflecting new
  discriminative information) — control 10;
- an **accidental population mismatch** between the R0 and R2 dumps (the
  exact failure mode `docs/PAPER_C_PROVENANCE_AUDIT.md` §5 and
  `tools/verify_population_identity.py` exist to rule out) — control 8.

This document separates evidence into six categories, **never conflated
into one score**:

| category | what it measures | which controls |
|---|---|---|
| **A. Pair-neutralized discrimination** | WPRD itself — does the score distinguish the true predicate from a same-pair alternate, with the prior arithmetically cancelled | 1, 2, 8 (the paired R0-vs-R2 delta) |
| **B. Retrieval/operating-point performance** | R@50/mR@50, calibration (ECE, confidence-vs-accuracy) — a *different* question from A, sensitive to scale/threshold effects WPRD is specifically designed to be immune to | 10 |
| **C. Prototype movement** | Did `predicate_prototypes` actually move, in a bounded, non-collapsing, non-degenerate way | 7, 11, 12 |
| **D. Geometry dependence** | How much of any discrimination is explainable by box layout alone, no pixels, no rel_feat | 4 |
| **E. Prior dependence** | How much of any discrimination or class-wise gain is explainable by training frequency / the composed pair prior | 3, 9 |
| **F. Lexical/text-space dependence** | How much of any discrimination survives when the *specific* text embedding content is replaced by noise or misassigned | 5, 6 |

**A positive R0-vs-R2 WPRD delta is evidence for the intended mechanism
only if it survives D, E, and F, and is accompanied by non-degenerate C.**
A positive delta that vanishes under geometry-only scoring (D), correlates
strongly with training frequency (E), or survives random/shuffled text
(F) is evidence *against* the intended mechanism, regardless of what the
raw WPRD number says.

---

## 1-12. The twelve controls

Each entry: hypothesis falsified · expected behavior under the null ·
what would count as evidence against the intended mechanism · compute
requirement · exact function.

### 1. R0 baseline

- **Falsifies**: nothing by itself — the reference point every other
  control is read relative to.
- **Expected under the null** (no real mechanism anywhere): not
  applicable — R0 is measured, not tested.
- **Evidence against the intended mechanism**: not applicable to this
  control in isolation.
- **Compute**: **existing dump only** — `runs/eval_C1/result.json` /
  `runs/eval_C1/pair_logits.pt`, already produced, never re-run.
- **Function**: none needed; read directly, or via
  `wprd_with_cell_keys(dump_a="runs/eval_C1/pair_logits.pt", channel="text")`
  for independent recomputation.

### 2. R2 adaptive prototypes

- **Falsifies**: nothing by itself — the arm under test.
- **Expected under the null**: WPRD indistinguishable from R0's (or from
  the prior-control band, §3) if the trained prototypes carry no
  additional discriminative information.
- **Evidence against**: WPRD *below* R0's, outside the paired CI — a
  genuinely harmful result (`docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md`-style
  Case C), which this document treats as informative, not something to
  tune away.
- **Compute**: **existing dump, once it exists** —
  `runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt` (pending).
- **Function**: `wprd_with_cell_keys(dump_a=..., channel="adaptive")`.

### 3. Prior-only / pair-prior control

- **Falsifies**: "the WPRD-scored channel secretly composes the frequency
  prior" — i.e. that any R2 gain is actually a prior effect smuggled
  through the adaptive channel.
- **Expected under the null** (prior correctly excluded, as designed —
  §8 of the preregistration): `prior_control_wprd == 0.500000` exactly,
  to float tolerance, for **both** R0 and R2 (the prior is a function of
  entity class alone and cancels arithmetically within a `Groups` cell
  regardless of which readout is scored — this is a structural guarantee,
  not a probabilistic expectation).
- **Evidence against**: `prior_control_wprd` measurably different from
  `0.5` for R2 specifically (would indicate the adaptive channel is, for
  some reason, not architecturally prior-free the way the text channel is
  — a serious, structural finding, not a tuning issue).
- **Compute**: **existing dump only**, CPU.
- **Function**: `wprd_with_cell_keys(dump_a=..., channel="prior")`
  (already computed and stored as `prior_control_wprd` in every
  evaluator's own `result.json` — this control independently re-derives it
  from the raw dump rather than trusting the stored value alone).

### 4. Geometry-only control

- **Falsifies**: "R2's gain reflects genuine visual/semantic relational
  understanding" — tests whether a model with **no access to pixels, no
  `rel_feat`, no predicate embedding** — only 19 scale-invariant numbers
  derived from the subject/object boxes — already explains the same
  discrimination.
- **Expected under the null** (R2 discriminates via something beyond
  layout): geometry-probe WPRD `<<` R2's WPRD, especially for **non-spatial**
  predicate groups (contact/action/possession/pose) — matching this
  program's own established C1 finding that the geometry repair's effect
  concentrates in spatial predicates specifically
  (`docs/PAPER_C_C1_MECHANISM_AUDIT.md` §C).
- **Evidence against**: geometry-probe WPRD `≈` R2's WPRD (a genuine null
  result would mean R2's apparent gain is a geometry shortcut, not an
  adaptive-prototype effect) — checked via the label-shuffled null
  (`geometry_probe_shuffled_label_null`, must independently read
  `≈0.5`, proving the probe itself is not leaking information through
  some other channel) and, ideally, the train-fitted variant (removes the
  probe's only structural advantage over the model).
- **Compute**: **CPU only, existing dump** — reuses
  `tools/wprd_geometry_control.py` (an existing, already-validated tool in
  this program), never pixels, never GPU.
- **Function**: `geometry_only_control(dump_path, prior_path)` (new wrapper,
  `tools/readout_v2_offline_analysis.py`).

### 5. Random text embeddings

- **Falsifies**: "the scoring pathway exploits something about CLIP's
  text-embedding-space geometry itself (norm scale, anisotropy) rather
  than the semantic content of `e(p)`/`P`."
- **Expected under the null** (scoring genuinely depends on semantic
  content): WPRD against embeddings of real-but-irrelevant English strings
  (never any real VG150 predicate) collapses to the prior-control band,
  `[0.47, 0.53]`.
- **Evidence against**: WPRD measurably above chance against nonsense
  embeddings — would mean *some* structural property of CLIP's
  text-embedding space, not the specific predicate content, drives
  discrimination.
- **Compute**: **CPU only** — a few dozen short strings through the
  checkpoint's own frozen CLIP **text** encoder (seconds on CPU); requires
  `rel_feat` to already be in the dump (`--eval_sgg_dump_rel_feat true`,
  already passed by both evaluators).
- **Function**: `random_text_control(dump_path, ckpt_path, prior_path,
  random_predicate_strings)`. Must be run **twice** — once scoring the
  dumped `rel_feat` (already implemented), the natural extension being to
  run it against R0's dump too, to confirm the frozen text channel is
  equally immune (a control that only checks R2 and not R0 would leave
  open whether the *whole pipeline*, not just Readout v2, is vulnerable).

### 6. Shuffled text embeddings

- **Falsifies**: "the *specific* row-to-predicate assignment of the frozen
  `E` (R0's fixed embedding table) matters" — the R0-side analogue of
  control 7.
- **Expected under the null** (R0's discrimination genuinely depends on
  which embedding is assigned to which predicate): permuting the
  column-to-class mapping of the `text` channel's stored logits collapses
  WPRD to the prior-control band.
- **Evidence against**: shuffled-column WPRD remains elevated — would mean
  R0's own frozen-cosine channel is not actually reading row-specific
  content either, undermining the premise that Readout v2 is improving on
  a *working* baseline mechanism (as opposed to a baseline that was
  already not doing what it appeared to do).
- **Compute**: **CPU only, existing dump** — pure column re-indexing of
  already-dumped logits, no CLIP, no GPU.
- **Function**: `shuffled_prototype_control(dump_path, prior_path,
  channel="text")` — same function as control 7, different `channel`
  argument (documented explicitly in the function's docstring: these are
  the same mechanism applied to the frozen-E and trained-P channels
  respectively, not two different implementations that could silently
  diverge).

### 7. Prototype permutation control

- **Falsifies**: "the trained `predicate_prototypes` matrix `P`'s
  row-to-predicate assignment specifically matters" — i.e. that `P` learned
  something about *which* row is *which* predicate, not merely "some
  predicate-shaped output."
- **Expected under the null** (training genuinely taught row-specific
  content, matching the training objective's design intent): permuting
  `adaptive_logits`' columns collapses WPRD to the prior-control band,
  while the unpermuted (real) value does not.
- **Evidence against**: permuted WPRD remains close to the real value —
  would mean R2's apparent discrimination is column-symmetric (e.g. a
  uniform confidence shift), not genuinely row-specific, directly
  contradicting the training objective's premise (`_predicate_ce_loss`
  targets the *correct* column index).
- **Compute**: **CPU only, existing dump**.
- **Function**: `shuffled_prototype_control(dump_path, prior_path,
  channel="adaptive")`. Already tested end-to-end on synthetic data with a
  known decodable signal (`tests/test_readout_v2_offline_analysis.py::
  test_shuffled_control_collapses_on_decodable_signal`).

### 8. Same-population paired bootstrap

- **Falsifies**: "R0 and R2 were scored on the same population, so a
  positive delta is attributable to the readout change alone, not a
  scoring-population artifact."
- **Expected under the null** (populations genuinely match, as intended by
  design — geometry, dataset, split, and `--eval_sgg_use_gt_pairs` are all
  held fixed between R0 and R2): `verify_population_identity` returns
  `EXACT_MATCH`; `verify_contract_match(expect_identical=True)` passes;
  `verify_paired_dumps` returns `safe_to_pair=True`.
- **Evidence against**: any status other than `EXACT_MATCH`
  (`DIFFERENT_IMAGE_POPULATION`, `DIFFERENT_PAIR_POPULATION`,
  `DIFFERENT_GT_LABELS`, `DIFFERENT_CELL_ASSIGNMENT`,
  `SAME_COUNTS_BUT_DIFFERENT_ROWS`, or `INSUFFICIENT_METADATA`) — **any of
  these must stop the analysis before a single delta number is trusted**,
  per `docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md` §2A's gate table.
- **Compute**: **CPU only, existing dumps**, both required.
- **Functions, in required order**:
  ```python
  ident = verify_population_identity(dump_a="runs/eval_C1/pair_logits.pt",
                                      dump_b="runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt")
  assert ident["status"] == "EXACT_MATCH"
  gate = verify_contract_match(r0_result_json, r2_result_json, expect_identical=True)
  assert gate["pass"]
  pop = verify_paired_dumps(dump_a=..., dump_b=..., channel_a="text", channel_b="adaptive")
  assert pop["safe_to_pair"]
  boot = paired_cell_bootstrap(pop["cell_values_a"], pop["cell_values_b"])
  ```

### 9. Prior correlation of class-wise gains

- **Falsifies**: "any R2 gain is broadly distributed across predicates,
  not concentrated on whichever predicates happened to receive the most
  training signal" — the endpoint-level generalization of this session's
  own recovery-time finding (prototype displacement tracked training
  frequency almost exactly: the 5 predicates whose `P` row moved most were
  the 5 highest-frequency predicates, verbatim).
- **Expected under the null** (gain is genuinely semantic, not
  frequency-shaped): weak or no correlation between per-class recall delta
  (R2 vs prior-only) and log training frequency.
- **Evidence against**: strong positive Pearson correlation — would mean
  the endpoint-level gain is substantially a frequency effect, echoing the
  mechanism-level finding rather than contradicting it.
- **Compute**: **CPU only, existing dump**.
- **Function**: `prior_correlation(dump_path, prior_path, train_freq,
  channel="adaptive")`. `train_freq` should use the exact, already-computed
  real per-predicate training counts from
  `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` §1.1 (all 50 real
  predicates, streamed from `train.jsonl` this session) — not re-derived.

### 10. Calibration comparison

- **Falsifies**: "any apparent R2 improvement in R@50/mR@50-style
  operating-point metrics is a genuine discrimination gain, not a
  confidence-scale/temperature artifact that changes where the
  argmax lands without changing what the score matrix actually knows."
- **Expected under the null** (WPRD gain, if any, is genuinely
  discriminative, not a scale artifact): ECE for R2 comparable to or
  better than R0's; confidence/accuracy relationship not radically
  reshaped.
- **Evidence against**: R2 shows a large ECE change (especially a large
  *overconfidence* increase) with only a marginal or absent WPRD gain —
  would suggest the readout became more confidently wrong or right at the
  same rate, not more discriminative. This control is explicitly why
  category B (retrieval/operating-point) is tracked **separately** from
  category A (WPRD) — a positive B result with a flat A result is not a
  Paper C success (mirroring `tools/c0_c1_compare.py`'s own registered
  `calibration_only_effect` classification for the C0-vs-C1 comparison).
- **Compute**: **CPU only, existing dumps**, both.
- **Function**: `calibration_comparison(dump_a="runs/eval_C1/pair_logits.pt",
  dump_b="runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
  channel_a="text", channel_b="adaptive", prior_path=...)`.

### 11. Prototype angular drift

- **Falsifies**: "`predicate_prototypes` moved in a bounded, coherent way
  during training" (as opposed to not moving at all, or diverging/collapsing).
- **Expected under the null of "training did nothing"**: `1-cos(P_i,E_i)`
  ≈ 0 for every row (no movement) — this was explicitly the failure mode
  of the second invalid pilot attempt
  (`docs/PAPER_C_READOUT_V2_INVALID_PILOT.md`), and this exact function
  already correctly detected it there.
- **Evidence against the *intended* mechanism** (as opposed to "no
  training happened"): movement that is present but **not** structured —
  e.g. uniform across all predicates regardless of training exposure
  (would suggest a generic regularization drift rather than
  label-driven learning), or movement so large it indicates semantic
  collapse (`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §15's own failure
  criterion: "`1-cos` collapses to ~1 for many `i`").
- **Compute**: **CPU only, checkpoint only** — independent of the R2 dump
  entirely; already run once against the actual pilot checkpoint this
  session (recovery-session finding: `mean 1-cos≈0.032`, `max≈0.16`, no
  collapse, top movers = top-5 training-frequency predicates).
- **Function**: `prototype_drift(ckpt_path)`.

### 12. Collapse / anisotropy checks

- **Falsifies**: "the trained `P` matrix uses its representational capacity
  in a healthy way" — as opposed to having collapsed toward a shared
  direction (would make cosine-based discrimination structurally
  impossible regardless of input) or toward a low-rank subspace (would
  indicate the embedding space is not using its full capacity, a
  different, complementary failure mode — see the function's own docstring
  for why these are NOT the same check).
- **Expected under the null** (healthy training): `cos_off_diag_mean` well
  below 1 (rows point in meaningfully different directions);
  `participation_ratio` well above 1 relative to `min(51, 768)` (rows
  occupy a reasonably high-dimensional subspace around their mean).
- **Evidence against**: `high_similarity_flag` or `near_rank_one_flag`
  firing on the trained `P` — especially if `P`'s anisotropy is
  **substantially worse** than `E`'s own (the frozen CLIP baseline's
  anisotropy is a pre-existing, not Readout-v2-induced, property, and
  comparing `P` against `E` on this same check isolates what training
  specifically did).
- **Compute**: **CPU only, checkpoint only** — independent of the R2 dump.
- **Function**: `collapse_anisotropy_check(P)` and
  `collapse_anisotropy_check(E)` (both computable from `prototype_drift`'s
  own reconstruction of `E`, or independently), compared side by side.

---

## Execution order (once R2's dump exists)

1. **Controls 11, 12** — checkpoint-only, can run the instant the pilot
   checkpoint exists (already does; no need to wait for the endpoint eval
   at all for these two).
2. **Control 8** — the population/pairing gate. **Nothing below this line
   may run if control 8 does not pass.**
3. **Controls 1, 2** — the primary WPRD numbers, cross-checked against each
   evaluator's own stored `result.json` values.
4. **Controls 3, 4, 5, 6, 7, 9, 10** — the falsification layer proper, in
   any order (each is independent once control 8 has passed).
5. Only after all of the above: write the actual comparison narrative
   (`docs/PAPER_C_READOUT_V2_PILOT_RESULT.md`, not yet created), explicitly
   organized by categories A-F, never a single headline WPRD number alone.

## What this document does not do

- Does not run any control against real R2 data — `runs/eval_readout_v2_R2_pilot_v3/`
  does not exist yet.
- Does not add a numeric pass/fail threshold for the overall pilot — per
  the preregistration's own §16 ("no arbitrary WPRD bar" at pilot stage),
  this battery reports categorized evidence, not a verdict.
- Does not modify `tools/readout_v2_evaluate.py`, any checkpoint, or any
  existing `result.json`.
- Does not launch the GPU, retrain anything, or run a full real-data
  analysis concurrently with the still-running R2 job.

## New/updated code, all tested on synthetic data only

- `tools/readout_v2_offline_analysis.py`: added `geometry_only_control`
  (control 4, wraps `tools/wprd_geometry_control.py`),
  `collapse_anisotropy_check` (control 12), `calibration_comparison`
  (control 10, thin diff wrapper over the existing `calibration_report`).
- `tests/test_readout_v2_offline_analysis.py`: +9 tests for the three new
  functions (26/26 passing in the file, ~5s), including a corrected
  fixture (`obj_boxes` were previously all-zero, unusable for a real
  geometry probe; `prior_rows` are now constant-within-group by default,
  matching every real dump and the invariant `Groups.prior_is_constant`
  asserts, with an explicit opt-out for the one test that needs
  independent per-image noise instead).
- Controls 1, 2, 3, 5, 6, 7, 8, 9, 11 reuse functions already built and
  tested in the prior sessions (`docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md`,
  `docs/PAPER_C_PROVENANCE_AUDIT.md`) — not rebuilt here.
