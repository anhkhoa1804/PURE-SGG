# Paper C — Readout v2 pilot: the decision gate

**This is a decision protocol, committed before the R2 result is read.**
Written while `tools/readout_v2_evaluate.py --label R2_pilot_v3` was still
running. As of this document: `runs/eval_readout_v2_R2_pilot_v3/result.json`
and `.../pair_logits.pt` do not exist. **No R2 number has been read,
computed, or analyzed while writing this document.** The point of writing
it now, not after the endpoint finishes, is that a gate defined after
seeing the result is not a gate — it is a rationalization. Every threshold
and category below is fixed by the *existing*, already-locked
preregistration and analysis documents, not invented for this document.

**Source of truth**, cited by section throughout:
`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` (§9-16, the registered
hypothesis, isolation controls, and fixed-in-advance failure/success
criteria), `docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md` (the 15-item
checklist and the tooling that implements it),
`docs/PAPER_C_PROVENANCE_AUDIT.md` (the population-identity gap and its
fix), `docs/PAPER_C_R2_FALSIFICATION_BATTERY.md` (the twelve controls and
the A-F evidence categories), `docs/PAPER_C_READOUT_V2_INVALID_PILOT.md`
(the two concrete historical failure modes Gate 1 is written to catch by
name, not by analogy).

**Procedure**: Gates run strictly in order, 1→2→3→4→5. **A failed gate
stops the process at that gate** — later gates are not consulted to
"rescue" an earlier failure, and an earlier gate's pass is not reopened
because a later gate looks favorable or unfavorable. This mirrors
`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §21's own stopping rule:
"stop, document the failure ... do not silently retry with changed
hyperparameters."

---

## Gate 1 — Endpoint integrity

Every sub-check below must independently pass. **Any single failure fails
the whole gate** — these are not weighted or averaged.

| sub-check | pass criterion | function / field |
|---|---|---|
| Checkpoint integrity | `all_pass == True` | `checkpoint_integrity(ckpt_path)["all_pass"]` |
| CLIP presence | `clip` key present, `clip_tensor_count > 0` | `checks["has_clip_key"]`, `checks["clip_tensor_count"]` |
| Exactly one trainable prototype tensor | optimizer state dict has **exactly one** entry (AdamW creates state lazily, only on a tensor's first real step — this is stronger than reading `requires_grad`, which this saved file cannot represent) | `checks["only_one_tensor_ever_stepped"] == True`, equivalently `checks["optimizer_n_state_entries"] == 1` |
| Optimizer state sanity | `predicate_prototypes`' own param group's LR is **not** stuck at `min_lr` (`1e-7`, the exact historical failure mode of the second invalid pilot attempt — `docs/PAPER_C_READOUT_V2_INVALID_PILOT.md`); `step > 0`; `exp_avg` nonzero (proves real gradients flowed, not just a group that was touched once) | `checks["last_group_is_min_lr_stuck"] == False`, `checks["last_group_step"] > 0`, `checks["last_group_exp_avg_abs_mean"] > 0` |
| No NaN/Inf | prototype tensor entirely finite | `checks["predicate_prototypes_finite"] == True` |
| Prototype shape | exactly `(51, 768)` | `checks["predicate_prototypes_shape"] == [51, 768]` |
| Finite angular movement | `P` moved from `E` (not frozen at init — `torch.equal(P,E)` would itself be a preregistration §15 failure) **and** did not diverge | `prototype_drift(ckpt_path)`: `one_minus_cos_mean > 0`, `unbounded_growth_flag == False` (`l2_max <= 10.0`, preregistration §15's own bound) |
| No collapse | prototypes have not rotated to near-orthogonal-to-init for many rows, and have not degenerated into a shared direction or a near-rank-one subspace | `prototype_drift(...)`: `semantic_collapse_flag == False` (preregistration §15: "`1-cos` collapses to ~1 for many `i`"); **additionally**, `collapse_anisotropy_check(P)`: `high_similarity_flag == False` and `near_rank_one_flag == False` — this second check is not in the original preregistration text but was added by `docs/PAPER_C_R2_FALSIFICATION_BATTERY.md` control 12 as a strictly complementary, not redundant, check (a matrix can pass the cosine-to-init check while still being internally degenerate) |

**Also required, independent of the checkpoint**: the flag-off regression
(`readout_v2_enabled=false` reproduces `R0`'s `result.json` bit-exactly,
preregistration §15) must already be on record as passing — it is, per
`runs/smoke_readout_v2_flagoff/result.json` and the regression suite
(`tests/test_readout_v2.py`), and is not re-run here.

**Gate 1 verdict**: `PASS` only if every row above passes. `FAIL` → stop.
Do not proceed to Gate 2. Classification (Gate 5): **INVALID / PROVENANCE
FAILURE**.

---

## Gate 2 — Population identity

**Counts alone are not sufficient** — stated explicitly because this is
the exact gap `docs/PAPER_C_PROVENANCE_AUDIT.md` §5 found: no
`result.json` in this repository persists per-cell identity, and
`tools/c0_c1_compare.py`'s own D1/D2 gates check only that array *lengths*
match, not that `cell_values[i]` in both arms means the same cell.

**Required check, strongest level the dumps permit**:

```python
ident = verify_population_identity(
    dump_a="runs/eval_C1/pair_logits.pt",
    dump_b="runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt")
```

| `ident["status"]` | disposition |
|---|---|
| `EXACT_MATCH` | **Proceed to Gate 3 directly.** Images, pairs, labels, GT predicates, and WPRD cells all match by both content and raw row position — the strongest available claim. |
| `SAME_COUNTS_BUT_DIFFERENT_ROWS` | **Do not proceed on raw `cell_values` arrays.** Content matches by set/multiset comparison, but row order does not. Must additionally run `verify_paired_dumps(...)` and confirm `safe_to_pair == True` (content-based cell-key realignment) before Gate 3 may use the resulting `cell_values_a`/`cell_values_b`. If `verify_paired_dumps` also fails to confirm safety, treat as `FAIL` below. |
| `DIFFERENT_IMAGE_POPULATION`, `DIFFERENT_PAIR_POPULATION`, `DIFFERENT_GT_LABELS`, `DIFFERENT_CELL_ASSIGNMENT` | **FAIL.** Stop. Diagnose which arm's evaluation configuration diverged before doing anything else — do not attempt a partial or corrected comparison. |
| `INSUFFICIENT_METADATA` | **FAIL.** The dump(s) lack a required identity field; do not fall back to a positional or count-based proxy. |

**Additionally required, same gate**: `verify_contract_match(r0_result_json,
r2_result_json, expect_identical=True)["pass"] == True` — R0 and R2 must
share the **identical** geometry contract (`geom_input_pixel_space=true,
geom_fourier_scale=0.01`), the opposite check from
`tools/c0_c1_compare.py`'s gate D4 (which requires contracts to *differ*,
correct only for the C0-vs-C1 geometry comparison, not this one —
`docs/PAPER_C_PROVENANCE_AUDIT.md` §4). A contract mismatch here would mean
geometry silently stopped being a controlled variable, which the
preregistration's own §15 already names as a stop condition ("geometry
channel stats or `geom_B_std` differ from `R0`'s").

**Gate 2 verdict**: `PASS` only if identity resolves to `EXACT_MATCH` (directly,
or via the `SAME_COUNTS_BUT_DIFFERENT_ROWS` → `verify_paired_dumps`
fallback) **and** the contract match passes. `FAIL` → stop. Classification
(Gate 5): **INVALID / PROVENANCE FAILURE**.

---

## Gate 3 — Paired comparison

**Only runs if Gates 1 and 2 both pass.** Compute, on the population
Gate 2 certified:

```python
boot = paired_cell_bootstrap(cell_values_a, cell_values_b, n_boot=2000, seed=11)
```

Report, without judgment yet (interpretation is Gate 5's job, not this
one's):

- **Absolute delta**: `boot["delta"]` (`= mean(cell_values_b - cell_values_a)`)
- **Paired bootstrap 95% CI**: `boot["ci95"]` (2000 resamples, seed 11 —
  the exact convention `tools/c0_c1_compare.py` already established for
  every prior C0/C1 comparison in this program, reused here for
  consistency, not reinvented)
- **`P(delta > 0)`**: `1 - boot["p_delta_negative"]`
- **Cell-level paired effect, if available**: per-cell
  `delta_i = cell_values_b[i] - cell_values_a[i]`, sorted by `|delta_i|`,
  top movers reported **with their cell key**
  (`{group}::{class_a}|{class_b}`, from `wprd_with_cell_keys`) — matching
  `docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md` item 6. Also report the
  spatial-vs-non-spatial breakdown of these deltas (same registered
  grouping as `docs/PAPER_C_C1_SEED2_RESULT.md` §10), since geometry-driven
  gain patterns are exactly what Gate 4's control 4 exists to interpret.

**Gate 3 produces numbers, not a verdict.** It does not classify
MECHANISTIC POSITIVE / NULL / NEGATIVE by itself — that requires Gate 4.
**A positive delta with a CI excluding zero, from Gate 3 alone, is
explicitly NOT sufficient for a positive classification** (see Gate 5's
opening constraint).

---

## Gate 4 — Mechanism falsification

R2's Gate 3 numbers must be interpreted **jointly** with the full battery
in `docs/PAPER_C_R2_FALSIFICATION_BATTERY.md` — every control listed
there is in scope for this gate, with the five named explicitly by this
task singled out below (the battery document defines each one's exact
null/evidence-against/function precisely; this section states only how
each feeds Gate 5's classification):

1. **Random / shuffled text controls** (battery controls 5, 6, 7):
   `random_text_control` and `shuffled_prototype_control` (both `text` and
   `adaptive` channels) must collapse to the prior-control band
   (`[0.47, 0.53]`). A control that does **not** collapse is evidence the
   measured discrimination is not specifically about the trained
   prototypes' semantic content — this pulls the outcome toward
   **PERFORMANCE POSITIVE BUT MECHANISM UNCLEAR** or worse, regardless of
   Gate 3's number.
2. **Prior correlation** (battery control 9): `prior_correlation` between
   per-class recall delta and training frequency. A strong positive
   correlation echoes this session's own recovery-time finding (prototype
   displacement tracked training frequency almost exactly) and is evidence
   the gain is frequency-shaped, not broadly semantic.
3. **Geometry dependence** (battery control 4): `geometry_only_control`.
   If the geometry-only probe's WPRD is close to R2's, **especially for
   non-spatial predicate groups**, any apparent gain is not distinguishable
   from a layout shortcut.
4. **Calibration** (battery control 10): `calibration_comparison`. A large
   ECE/confidence shift with only a marginal WPRD gain is a calibration
   artifact, not a discrimination gain — this is precisely
   `tools/c0_c1_compare.py`'s own registered `calibration_only_effect`
   category, applied here to R0-vs-R2 instead of C0-vs-C1.
5. **Prototype drift** (battery control 11, already gated in part at
   Gate 1): here, drift is read for its **pattern**, not merely its
   existence — does the magnitude and direction of movement plausibly
   track which predicates received real training signal (per
   `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md`-style frequency data),
   consistent with genuine label-driven learning rather than a generic
   regularization drift?

**Gate 4 does not produce a single pass/fail bit.** It produces a
**profile** across the six evidence categories A-F already defined in the
falsification battery (pair-neutralized discrimination, retrieval/
operating-point performance, prototype movement, geometry dependence,
prior dependence, lexical/text-space dependence). That profile, together
with Gate 3's numbers, is Gate 5's input.

---

## Gate 5 — Scientific interpretation

**Constraint, stated first because it is the one most likely to be
violated under pressure to report a result**: *a positive WPRD delta from
Gate 3, by itself, is never sufficient for a positive classification.*
"Any positive WPRD" is explicitly **not** a success criterion — the
preregistration itself registered **no fixed WPRD threshold** for the
pilot (§16: "no arbitrary WPRD bar"), and this document does not invent
one retroactively. What determines the classification is the **joint**
outcome of Gates 1-4, not Gate 3's magnitude.

### The five categories

**1. INVALID / PROVENANCE FAILURE**
Gate 1 or Gate 2 failed. No further interpretation is attempted — Gate 3's
numbers, if they were computed at all, are not reported as a scientific
result. This category takes absolute precedence: it overrides every other
category regardless of what Gates 3-4 would otherwise show.

**2. NEGATIVE**
Gates 1-2 pass. Gate 3's delta is negative with a 95% CI excluding zero —
a genuine, statistically supported harmful result. Per preregistration
§16, this is "still informative and is not 'rescued'" by any subsequent
tuning.

**3. NULL**
Gates 1-2 pass. Either (a) Gate 3's CI includes zero (no statistically
supported directional change), **or** (b) Gate 3 shows a nominally
positive delta but Gate 4's battery shows the entire effect is explained
away — e.g. the geometry-only probe already accounts for it, or the
random/shuffled-text controls fail to collapse in a way that indicates the
gain is not about the trained prototypes specifically, or the gain is a
pure calibration artifact per `calibration_comparison`. A "positive number
on the page" that Gate 4 shows is not attributable to the intended
mechanism is classified **NULL**, not positive-anything.

**4. PERFORMANCE POSITIVE BUT MECHANISM UNCLEAR**
Gates 1-2 pass. Gate 3 shows a positive delta with CI excluding zero.
Gate 4's battery is **mixed**: some controls behave as the null predicts
(e.g. random-text collapses correctly) but at least one does not fully
clear (e.g. prior correlation is moderate-to-strong, or the geometry-only
probe explains a meaningful fraction of the gain without fully accounting
for it, or calibration shifted more than the WPRD gain alone would
suggest). This category exists specifically so a real, CI-supported
positive number is not forced into either MECHANISTIC POSITIVE (overclaim)
or NULL (underclaim) when the evidence is genuinely ambiguous.

**5. MECHANISTIC POSITIVE**
Gates 1-2 pass. Gate 3 shows a positive delta with CI excluding zero.
Gate 4's **full** battery is clean: random- and shuffled-text controls
collapse to the prior-control band, the geometry-only probe does not
account for the gain (especially outside spatial predicates), prior
correlation is weak, calibration change is proportionate to the WPRD gain
(not a bare confidence shift), and prototype drift is bounded,
non-collapsing, and patterned consistently with genuine label-driven
learning. **All five conditions are required jointly** — this category is
deliberately the hardest of the five to reach.

### Explicit exclusions — stated because they are the most likely mislabeling errors

- **Do not call a MECHANISTIC POSITIVE result "open-vocabulary."** R2 is a
  **closed-set, 51-prototype adaptive readout** experiment. `predicate_prototypes`
  has fixed cardinality; there is no mechanism in this architecture to
  score a 52nd predicate without adding and training a new row. This is
  already the explicit, correct framing in
  `docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md` §1/§8 (Readout v2 is
  there described as a *motivating diagnostic* for the open-vocabulary
  successor, precisely *because* its parameterization cannot generalize to
  unsupervised predicates) — this document does not revise that framing
  and a MECHANISTIC POSITIVE result here does not revise it either.
- **Even a MECHANISTIC POSITIVE result cannot establish novelty against
  OV-SGT or prior CLIP-alignment literature.** `docs/PAPER_C_OPEN_VOCAB_LITERATURE_AUDIT.md`
  found that OV-SGT (Vanica & Bors, WACV 2026 Workshop) already describes
  learning relationship embeddings within CLIP's semantic space via a
  contrastive objective for zero-shot predicate transfer — a mechanism
  this audit could not fully compare against (its full text was not
  accessible) but which is, at minimum, in the same architectural family
  as the *adaptive-prototype* idea Readout v2 tests at closed-set scale.
  A positive result here is evidence about **this repository's own
  architecture program** (does adapting predicate directions on this
  specific, geometry-repaired representation help) — it is not, on its
  own, evidence of priority or novelty relative to any external
  publication, and must not be described that way in any external-facing
  document without the literature audit's own recommended follow-up
  (reading OV-SGT's and the OvSGTR family's full text and diffing against
  this program's design) having been done first.
- **A MECHANISTIC POSITIVE result at pilot scale does not authorize a
  full-budget run.** Per preregistration §21/22 and the phase-transition
  task's own repeated instruction across this program's history: any
  full-budget schedule requires a **separate, dated amendment**, written
  after the pilot's own numbers are known, not a default next action.

---

## What this document does not do

- Does not read, compute, or report any number from R2's actual dump or
  `result.json` — neither exists as of this writing.
- Does not modify model code, `tools/readout_v2_evaluate.py`, or any
  checkpoint.
- Does not launch any experiment, GPU or otherwise.
- Does not set a numeric WPRD threshold the preregistration itself
  declined to set — classification here depends on the *joint* Gate 1-4
  outcome, not on Gate 3's magnitude alone.
- Does not pre-decide the outcome. Every branch above is written so that
  it can resolve to any of the five categories depending on what Gates 1-4
  actually show once they are run for real.
