# Paper C — Readout v2 endpoint: offline analysis plan

Prepared while the R2 endpoint evaluation
(`tools/readout_v2_evaluate.py --label R2_pilot_v3
--ckpt checkpoints/readout_v2_pilot_seed1234_v3.pt --readout_v2_enabled true`)
was still running on the GPU. **Nothing in this document required the GPU,
modified the active evaluation, modified model code, or altered any
existing `result.json`.** The one new artifact is a CPU-only,
read-only-of-existing-files analysis library
(`tools/readout_v2_offline_analysis.py`) plus its test suite
(`tests/test_readout_v2_offline_analysis.py`, 19/19 passing on synthetic
data). Per instruction, neither is committed — this plan is for review
first.

---

## 1. Audit of existing infrastructure

### 1.1 `tools/c0_c1_evaluate.py` (produces R0 = `runs/eval_C1/result.json`)

Runs the full validation split through `openvocab_rel.train.main()` in
`--eval_only` mode with the arm's geometry flags set from `--arm` (never
mixable), dumps `pair_logits.pt` (including `rel_feat`, since
`--eval_sgg_dump_rel_feat true` is passed), then computes WPRD (via
`tools/within_pair_discrimination.py`'s `Groups`/`wprd`, `cap=64`) on the
`fixed_ensemble(0.0)` ("text") channel, plus R@50/mR@50, geometry-channel
stats, fusion-gate stats, and `prior_control_wprd`. Writes a single
`result.json` per arm. **R0, for Readout v2's purposes, is
`runs/eval_C1/result.json`** — not re-run; reused as-is, per the
preregistration.

### 1.2 `tools/readout_v2_evaluate.py` (produces R2)

Structurally identical to `c0_c1_evaluate.py` (same `argv` construction
pattern, same monkey-patched geometry/gate instrumentation, same
`result.json` shape), with three differences: (a) geometry is hard-coded to
the C1 contract, no `--arm` flag exists at all; (b) `--readout_v2_enabled`
controls whether `adaptive_logits` is dumped; (c) its own `MechV2` subclass
of `cprime_mechanism.Mech` adds an `adaptive_norm51` channel read from
`d["adaptive_logits"]`, and WPRD is computed on `adaptive_norm51` instead
of `fixed_ensemble(0.0)` when `has_adaptive_channel=True`. **The
`result.json` schema is otherwise byte-for-byte the same shape as
`c0_c1_evaluate.py`'s** — same keys, same `cell_values` (flat float list,
no cell-identity metadata), same `contract` key.

### 1.3 `tools/within_pair_discrimination.py` — the WPRD estimator (cell
definitions, paired-cell logic)

`Groups(B)` groups GT rows by exact `(subject_label, object_label)` string
pair (`self.pair_id`, `self.G` groups), tracks the distinct GT predicate
classes observed per group (`classes_of`), and marks a row `decidable` iff
its group has `>= 2` distinct classes. `wprd(Gs, score, cap=64)` then, for
**every group with >= 2 distinct GT classes**, and **every unordered pair
of those classes `(a, b)`**, forms one **cell**: `AUC({score[i,a]-score[i,b]
: y_i=a} vs {score[j,a]-score[j,b] : y_j=b})`, capped at 64 rows per side
(deterministic `torch.Generator(seed=0)` subsampling above the cap),
weighted by `len(ra)*len(rb)` for the weighted variant. `wprd_macro` is the
unweighted mean over cells; `wprd_weighted` weights by cell size. **This is
the single source of truth for "what is a cell"** — both `c0_c1_evaluate.py`
and `readout_v2_evaluate.py` call it identically, with `cap=64`. The prior
is guaranteed constant within a group by construction (`prior_is_constant`
gate, asserted `< 1e-3` at the top of `within_pair_discrimination.py`'s own
`main()`), which is exactly why the prior cancels in the double-difference
`(score[i,a]-score[i,b]) - (score[j,a]-score[j,b])` and `prior_control_wprd`
reads `0.500000` in every arm to date.

**Gap found**: `wprd()`'s return dict carries `_vals`/`_wts`/`_per_pair`
internally, but every evaluator script (`c0_c1_evaluate.py`,
`readout_v2_evaluate.py`) writes only the flat `cell_values` list into
`result.json`, with **no per-cell identity key** (which group, which
predicate pair) persisted anywhere. Two independently-produced
`cell_values` arrays are therefore only *positionally* comparable if the
two runs' `Groups` objects happened to enumerate cells in exactly the same
order — plausible (deterministic dataset iteration, same image order,
same dict-insertion-order-preserving construction) but **never actually
verified** by any existing tool. This is exactly what item 2/5 below closes.

### 1.4 `tools/c0_c1_compare.py` — the paired bootstrap (**contains a bug for
R0-vs-R2 reuse**)

Reads two `result.json` files, checks four gates (`D1` population match,
`D2` equal cell count, `D3` `prior_control_wprd≈0.5` in both, `D4` contract
sanity), then runs a **paired cell-bootstrap** (resample `d = vB - vA` with
replacement, 2000 draws by default, seeded `manual_seed(11)`, report
`delta`, 95% CI, `P(delta<0)`).

**The bug, found and confirmed this session by reading the exact literal
values both scripts write**: gate D4 is

```python
gates.append({"gate": "D4 each arm ran under its own registered contract",
              "pass": bool(A["contract"]["geom_input_pixel_space"] is False
                           and A["contract"]["geom_fourier_scale"] == 1.0
                           and B["contract"] != A["contract"]),
              ...})
```

This requires **A's contract to be exactly C0's pre-repair contract**
(`geom_input_pixel_space=False, geom_fourier_scale=1.0`) **and** B's
contract to differ from A's — correct for a C0-vs-C1 comparison (the whole
point is that geometry changed), but **R0 = `runs/eval_C1/result.json`**
has contract `{"geom_input_pixel_space": true, "geom_fourier_scale": 0.01}`
(from `c0_c1_evaluate.py`, arm=`C1`) and **R2's `result.json`** hard-codes
the identical `{"geom_input_pixel_space": True, "geom_fourier_scale": 0.01}`
(`readout_v2_evaluate.py` line 230 — never varied, by design, since
geometry is a controlled variable for Readout v2). Plugging R0 as `--c0`
and R2 as `--c1` into `c0_c1_compare.py` unmodified gives
`A["contract"]["geom_input_pixel_space"] is False` → **`False`** →
**gate D4 evaluates to `False`** → the script prints `"*** PAIRING GATE
FAILED ***"` and exits 1, for a comparison that is in fact perfectly valid
(identical contracts are *correct* and *required* here, not a defect).

**The preregistration's claim that `c0_c1_compare.py` "needs no change...
reused as-is" (`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §19) is
incorrect as the code currently stands.** This is exactly the kind of
prior-agent claim this program's own discipline says to challenge, not
inherit. **Fix**: `tools/readout_v2_offline_analysis.py::verify_contract_match`
(built this session, tested) replaces gate D4 with the correct, symmetric
check — `expect_identical=True` for R0-vs-R2 (pass iff contracts are
`==`), `expect_identical=False` reproduces the original C0-vs-C1 semantics
for anyone who later needs it. Gates D1/D2/D3 are unaffected by this bug
and remain correct as written.

### 1.5 `tools/cprime_mechanism.py` (`Mech`/`Bench`) — the shared scoring
substrate

`Bench.__init__` (in `cprime_analysis.py`, extended by `Mech`) flattens the
dump into `model`/`prior`/`text`/`cls` score matrices (each `(n_pair_rows,
n_fg_classes)`, background column dropped), builds `gt_row`/`gt_y` (which
rows are GT-positive and their true class), `col_to_class` (column→class
map, a bijection under `scheme="raw50"`), and `bucket_of` (head/body/tail
by GT frequency in the CURRENT dump — **not** comparable across dumps with
different `N`, a pre-existing, already-documented caveat).
`Mech.fixed_ensemble(ea)` is the evaluator's own normalize-then-slice
order (validated to reproduce the stored `model_logits` to `<1e-4`, an
existing regression-tested gate). `readout_v2_evaluate.py`'s `MechV2`
subclass adds nothing to this substrate except reading one more dump key
(`adaptive_logits`) through the identical `_norm`/`fg_cols` pipeline
already used for `text_logits`/`cls_logits` — so every existing
`Mech`/`Bench` method (`.score`, `.predict`, `.metrics`, `.gt_rank`,
`.per_class_recall_vec`) already works unmodified on the R2 channel once
it is assigned to `.channel_scores` (exactly what `_mech_with_channel` in
the new offline-analysis module does).

### 1.6 Prior / random controls already built into this stack

- `within_pair_discrimination.py`'s own `main()` already runs a
  `"random null"` arm: `torch.randn(text.shape, generator=manual_seed(0))`
  — **structureless Gaussian noise**, not real CLIP text embeddings of
  unrelated words. This is a *weaker* control than item 14's "random-text
  control" below: it cannot test whether CLIP's own embedding-space
  geometry (norm scale, anisotropy) could spuriously produce above-chance
  WPRD, only whether pure noise can. Both are worth running; they are not
  the same control and neither substitutes for the other.
- `prior_control_wprd` (computed identically by both evaluator scripts,
  `WPD.wprd(Gs, B.prior[B.gt_row], cap)["wprd_macro"]`) is the **prior-only**
  control, guaranteed `≈0.500000` by the `Groups` construction's own
  arithmetic (§1.3) — already computed for both R0 and (once it finishes)
  R2, no new code needed.
- **No existing tool implements** a shuffled-column control, a
  CLIP-embedding-space random-text control, or a checkpoint-integrity
  check — these are new, added this session (items 14, 15, 1 below).

---

## 2. The deterministic analysis checklist

Each item states: what it answers, the exact procedure/command, the
pass/fail criterion (drawn from `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md`
§15/§16 where one is registered, or stated fresh where the preregistration
is silent), and whether it can run **now** (before the endpoint finishes)
or needs the finished `result.json`/dump.

### 1. Checkpoint integrity — **can run now**

The training checkpoint (`checkpoints/readout_v2_pilot_seed1234_v3.pt`)
already exists and is untouched by the still-running endpoint evaluation
(the evaluator only *reads* it). Run:

```python
from tools.readout_v2_offline_analysis import checkpoint_integrity
r = checkpoint_integrity("checkpoints/readout_v2_pilot_seed1234_v3.pt")
assert r["all_pass"]
```

Checks: `clip` key present (CLIP was serialized); `predicate_prototypes`
present, shape `(51, 768)`, finite; **the optimizer state dict has exactly
one entry** (the strongest available proof only one tensor was ever
stepped — AdamW creates state lazily, only on a tensor's first real
gradient step, so this is stronger than reading `requires_grad`, which is
a training-time-only property this saved file cannot even represent); the
`predicate_prototypes` param group's LR is not stuck at `1e-7` (the exact
historical failure mode from the second invalid pilot attempt). This
session already ran this exact check by hand against this exact checkpoint
during recovery (see `docs/PAPER_C_READOUT_V2_INVALID_PILOT.md`) and it
passed; this function codifies that check as a reusable, tested regression
rather than a one-off.

**Also run**, since these two files are untouched by anything in this
session and should stay that way:
```python
from tools.readout_v2_offline_analysis import historical_checkpoint_sha256_check
historical_checkpoint_sha256_check(
    "checkpoints/demo_best/pure_best_adapt_light_mR50.pt",
    "8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442")
historical_checkpoint_sha256_check(
    "checkpoints/C1_seed1234.pt",
    "79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854")
```

### 2. Exact population match — **needs R2's dump**

Not "do the counts match" (§1.3's gap) but "are the actual cells the
same." Run, once `runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt` exists:

```python
from tools.readout_v2_offline_analysis import verify_paired_dumps
out = verify_paired_dumps(
    dump_a="runs/eval_C1/pair_logits.pt",
    dump_b="runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
    prior_path="datasets_vg150_clean/frequency_prior_train.json",
    channel_a="text", channel_b="adaptive", cap=64)
assert out["safe_to_pair"]
```

This recomputes both arms' `Groups`/cells directly from the raw dumps and
diffs the **cell identity keys**, not just counts. **Pass**: `safe_to_pair
== True` (population dicts equal AND cell-key lists identical,
element-for-element). **Fail** must be diagnosed, not worked around — a
mismatch here means the paired bootstrap (item 5) cannot be trusted
regardless of what `c0_c1_compare.py`-style count-only gates would report.

### 3. WPRD — **needs R2's `result.json` (or dump)**

Already computed by `readout_v2_evaluate.py` itself
(`res["wprd_macro"]`, on the `adaptive_norm51` channel, `cap=64`, same
estimator as every C0/C1 number). Independently re-verify with:

```python
from tools.readout_v2_offline_analysis import wprd_with_cell_keys
r = wprd_with_cell_keys("runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
                        "datasets_vg150_clean/frequency_prior_train.json",
                        channel="adaptive")
assert abs(r["wprd_macro"] - stored_result_json["wprd_macro"]) < 1e-9
```

**No fixed numeric success threshold is registered for the pilot** (per
the preregistration §16 — "no arbitrary WPRD bar"). Report the number and
its relationship to R0 (item 5), not a pass/fail against a target.

### 4. Weighted WPRD — **needs R2's `result.json`**

Already computed (`res["wprd_weighted"]`). Compare `wprd_macro` vs
`wprd_weighted` for both R0 and R2: a large divergence between them
indicates the WPRD effect (whichever direction) is concentrated in a few
large cells rather than broadly distributed — report both, do not average
them into one number.

### 5. Paired bootstrap against R0 — **needs R2's dump; gated by item 2**

**Only valid if item 2 passes.** Two steps:

```python
from tools.readout_v2_offline_analysis import (verify_contract_match,
    verify_paired_dumps, paired_cell_bootstrap)
import json

r0 = json.loads(open("runs/eval_C1/result.json").read())
r2 = json.loads(open("runs/eval_readout_v2_R2_pilot_v3/result.json").read())

gate = verify_contract_match(r0, r2, expect_identical=True)   # NOT c0_c1_compare.py's D4
assert gate["pass"], "geometry contracts diverged -- STOP, this would confound the comparison"

pop = verify_paired_dumps("runs/eval_C1/pair_logits.pt",
                          "runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
                          "datasets_vg150_clean/frequency_prior_train.json",
                          channel_a="text", channel_b="adaptive")
assert pop["safe_to_pair"]

boot = paired_cell_bootstrap(pop["cell_values_a"], pop["cell_values_b"], n_boot=2000, seed=11)
```

Report `boot["delta"]`, `boot["ci95"]`, `boot["p_delta_negative"]`,
`boot["ci_excludes_zero"]`. **Do not additionally run
`tools/c0_c1_compare.py` unmodified against R0/R2** — per §1.4, its D4 gate
will spuriously fail. If a narrative wants `c0_c1_compare.py`'s exact
console output format, patch its D4 line locally for this one invocation
(or write a 15-line `readout_v2_compare.py` wrapper that imports
`verify_contract_match`/`paired_cell_bootstrap` and reproduces its print
layout) — **not done in this plan**, since the task instructions ask for
analysis-readiness, not a new experiment artifact, and the two functions
above already produce every number `c0_c1_compare.py` would.

No WPRD delta threshold is registered for the pilot (§16), so there is no
"MET/NOT MET" verdict to compute here the way `c0_c1_compare.py` does for
C0-vs-C1 — report the delta, CI, and direction; classification (healthy /
null / negative / invalid) is a judgment call against §16's qualitative
criteria (stable training, bounded prototype movement, coherent directional
WPRD change, flag-off bit-exactness — all already gated separately), not a
single numeric bar.

### 6. Per-cell deltas — **needs R2's dump; gated by item 2**

Using `pop["cell_values_a"]`/`pop["cell_values_b"]` and
`wprd_with_cell_keys`'s `cell_keys` (same order): compute
`delta_i = cell_values_b[i] - cell_values_a[i]` per cell, sort by
`|delta_i|` descending, report the top-20 movers **with their key**
(`{group}::{class_a}|{class_b}`) so a human can see which specific
subject/object pairs and predicate contrasts moved most — this is the
cell-level analogue of `cprime_mechanism.py`'s existing per-predicate
`analysis_DE_buckets_predicates` (§1.5), but at the finer WPRD-cell
granularity rather than the coarser per-class-recall granularity.

### 7. Predicate-tail breakdown — **needs R2's `result.json`**

Already computed at the R@50/mR@50 level (`head_mR`/`body_mR`/`tail_mR`,
both arms, same bucket-of-51 convention — note the caveat in §1.5: buckets
are GT-frequency-defined **per dump**, identical here since R0 and R2 score
the same GT). For the **WPRD-side** tail breakdown (registered as
"mechanistic" evidence in the preregistration §13.3, "spatial vs.
non-spatial ΔWPRD, same registered grouping as
`docs/PAPER_C_C1_SEED2_RESULT.md` §10"): group item 6's per-cell deltas by
whether `class_a`/`class_b` are in the `spatial` group
(`configs/predicate_metadata_vg150.json`, cross-checked against the real
50-predicate vocabulary per `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md`
§0 — do not trust the metadata file's 3 orphaned entries), and report
spatial-vs-non-spatial mean delta separately, matching the C1-seed2
result's own convention rather than inventing a new grouping.

### 8. Zero-training-pair breakdown — **needs R2's dump**

Not a concept any existing tool defines by this name — interpreted here as:
among GT rows/cells, which ones involve a predicate that had **zero**
positive training pairs *in this specific pilot's 250-image / ~3,000-pair
budget* (not zero in the full VG150 training set — every one of the 50 real
predicates appears somewhere in full-VG150 training, per
`docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` §1, but a ~3,000-pair
pilot subsample can easily miss many long-tail predicates entirely by
chance). This is directly checkable from the pilot's own `metrics.jsonl`
(logs the epoch's realized batch composition) or, more precisely, from
`--samples_per_epoch 250`'s realized image sample list combined with
`train.jsonl`'s GT — **not yet built as a function**; the closest existing
proxy is `prior_correlation` (item 12), which will show near-zero or
noisy recall delta for these predicates as a side effect, but does not
explicitly flag "zero training exposure" as its own category. **Recommend**:
before the final write-up, compute this list once (predicates with 0
positive pairs in the realized 250-image pilot sample) and report those
predicates' recall-delta and cell-count separately, since Readout v2's
`_predicate_ce_loss` cannot have moved `predicate_prototypes`' row for a
class with literally zero positive gradient signal — a flat/near-zero
delta for these is the **expected, healthy** result, not evidence against
the mechanism; a *large* movement for one of these would be the surprising
finding worth investigating (possibly explained entirely by the anchor
loss and/or the lattice/confusable-negative pathway identified in
`docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` §2, which — unlike a
predicate-disjoint run — was NOT disabled for this closed-set pilot and
could move a zero-positive-pair predicate's own prototype row purely as
someone else's hard negative).

### 9. Prototype displacement from E — **can run now** (checkpoint only,
independent of the endpoint)

```python
from tools.readout_v2_offline_analysis import prototype_drift
d = prototype_drift("checkpoints/readout_v2_pilot_seed1234_v3.pt")
```

This session already ran this exact procedure by hand during recovery
(reconstructing `E` fresh from the checkpoint's own CLIP text encoder +
the real scanned vocabulary, comparing to the checkpoint's stored `P`) and
found: `cos_mean=0.968`, `one_minus_cos_mean≈0.032`, `one_minus_cos_max≈0.16`,
`l2_mean≈0.234`, no NaN/Inf, top movers exactly the 5 highest-training-
frequency predicates (`has`, `on`, `wearing`, `of`, `with`). `prototype_drift`
codifies this as a reusable function; re-running it against the same,
unchanged checkpoint should reproduce these exact numbers (a useful
self-consistency check that nothing about the checkpoint changed between
recovery and this analysis pass). **Failure criteria** (preregistration
§15): `semantic_collapse_flag` (mean `1-cos > 0.5`) and
`unbounded_growth_flag` (`max L2 > 10.0`) are reported as explicit booleans
— neither fired in the recovery-session run.

### 10. Prototype norm distribution — **can run now** (same function, item 9)

`prototype_drift`'s `row_norm_mean/min/max` (recovery session:
`1.039 / 0.998 / 1.186` — bounded, no collapse toward zero, no blow-up).
Report the full per-predicate `row_norm` alongside `one_minus_cos` in the
`per_predicate_by_displacement` list (already sorted by displacement) so a
reader can see whether norm growth and angular drift co-occur for the same
predicates (they need not — a prototype can rotate without growing, or
grow along its own original direction without rotating).

### 11. Semantic drift from E — **can run now** (same function, item 9)

The `per_predicate_by_displacement` list itself **is** the semantic-drift
report: which predicates moved most/least, in what mix of angular (`cos`)
vs. magnitude (`l2`) terms. Cross-reference against
`configs/predicate_metadata_vg150.json`'s `group` field (join on the real
50-predicate vocabulary, per the split-audit's §0 orphan warning) to check
whether drift concentrates within a semantic group (e.g., do the spatial
predicates all drift together, suggesting a shared geometric confound) or
is scattered — this cross-reference is a manual join over `prototype_drift`'s
output and `configs/predicate_metadata_vg150.json`, not yet a single
function, since it is a one-time reporting step rather than a reusable
check.

### 12. Prior correlation — **needs R2's dump**

```python
from tools.readout_v2_offline_analysis import prior_correlation
train_freq = {...}  # from datasets_vg150_clean/train.jsonl, real counts --
                     # see docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md sec.1.1
                     # for the exact, already-computed table (all 50 predicates)
out = prior_correlation("runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
                        "datasets_vg150_clean/frequency_prior_train.json",
                        train_freq=train_freq, channel="adaptive")
```

Reports the Pearson correlation between each predicate's real training
frequency (log scale) and its recall delta (R2 vs. prior-only) — the
endpoint-level generalization of this session's own manual finding that
prototype *movement* concentrated on the highest-frequency predicates
(item 9/11). A strong positive correlation here would mean R2's WPRD/recall
gain (if any) is itself concentrated on head predicates, not broadly
distributed — directly relevant to whether any headline number reflects a
general readout improvement or a frequency-shaped one, and complementary
to (not a replacement for) `prior_control_wprd` reading exactly `0.5`
(which rules out the prior being *composed into the score*, not the same
claim as "the gain happens to correlate with frequency").

### 13. Calibration — **needs R2's dump**

```python
from tools.readout_v2_offline_analysis import calibration_report
c = calibration_report("runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
                       channel="adaptive",
                       prior_path="datasets_vg150_clean/frequency_prior_train.json")
```

Standard 10-bin Expected Calibration Error on the `adaptive_logits`
channel's softmax-max-confidence vs. top-1 correctness, on GT rows. Run
identically on `channel="text"` for R0 to compare — no calibration
mechanism (`calibration_gate`, `bias_residual_head`) touches the
adaptive/text channel at all (§8 of the preregistration: "entirely on the
classifier branch... Frozen, inert, untouched"), so any calibration
difference between R0 and R2 reflects the readout change itself, not a
confound from calibration machinery.

### 14. Random-text control — **needs R2's dump (uses `rel_feat`)**

```python
from tools.readout_v2_offline_analysis import random_text_control
strings = [...]  # exactly n_classes (51) entries, real English words/phrases
                 # that are NOT any of the 50 real VG150 predicates or "relation"
                 # -- cross-check against datasets_vg150_clean/vocabulary/predicates.json
out = random_text_control("runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
                          "checkpoints/readout_v2_pilot_seed1234_v3.pt",
                          "datasets_vg150_clean/frequency_prior_train.json",
                          random_predicate_strings=strings)
assert out["collapse_to_prior_band"]
```

Uses the dumped `rel_feat` (present because both evaluators pass
`--eval_sgg_dump_rel_feat true`) scored against embeddings of predicate
strings encoded through the checkpoint's own frozen CLIP text encoder —
CPU-only (a few dozen short strings through CLIP's text tower takes
seconds), no GPU, no re-evaluation. **Pass**: `wprd_macro` in `[0.47,
0.53]`. **A failure here (above-chance WPRD against nonsense embeddings)
would be a serious, reportable finding** — evidence that *something other
than the semantic content of `e(p)`* is driving any apparent
discrimination (e.g., a geometric confound in CLIP's own text-embedding
norm/direction statistics interacting with `rel_feat`'s scale) — this is
one of the six falsification tests named in
`docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md` §5, reused here at the
Readout v2 (closed-set) level even though that document's falsification
battery was designed for the *future* open-vocabulary successor — nothing
prevents running it now, on the readout that already exists, and a clean
result here is a small additional piece of assurance before that larger
program invests in it.

### 15. Shuffled-prototype control — **needs R2's dump; scientifically
valid, argued explicitly**

```python
from tools.readout_v2_offline_analysis import shuffled_prototype_control
out = shuffled_prototype_control("runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt",
                                 "datasets_vg150_clean/frequency_prior_train.json")
assert out["collapse_to_prior_band"]
```

**Why this is scientifically valid** (the task explicitly asks to justify
this, not just run it): `predicate_prototypes` was trained with
`_predicate_ce_loss` targeting the **correct** column index for each GT
predicate. Permuting which column is read as "predicate k" at analysis
time — a pure, offline re-indexing of the already-dumped `adaptive_logits`
matrix, no re-scoring, no CLIP, no GPU — tests whether the trained signal
is genuinely **row-identity-specific** (i.e., whether `P`'s row `k`
specifically encodes predicate `k`, which is what the training objective
was supposed to produce) or would look equally good under any relabeling
(which would mean the "signal" is some column-symmetric artifact,
e.g., a uniform overall confidence shift that doesn't care which column is
which). This is a strictly cheaper, complementary check to item 14: item 14
tests whether the *embeddings' content* matters; item 15 tests whether the
*specific row-to-predicate assignment* matters. **Pass**: shuffled
`wprd_macro` collapses to `[0.47, 0.53]` while the unshuffled (real) value
does not — tested exactly this way, on synthetic data with a known
decodable signal, in `tests/test_readout_v2_offline_analysis.py`.

---

## 2A. Population identity — the exact gate required before paired bootstrap

Added after `tools/verify_population_identity.py` was built (see
`docs/PAPER_C_PROVENANCE_AUDIT.md` §5 for the forensic finding that
motivated it: no `result.json` persists per-cell identity, and the
historical C0-vs-C1 pairings were only ever spot-checked for cell identity
by hand, once, not through a reusable tool).

**Item 2's `verify_paired_dumps`** (§1.3, `tools/readout_v2_offline_analysis.py`)
answers "are the WPRD cells identical" by recomputing both arms' `Groups`
and diffing cell *keys*. **`tools/verify_population_identity.py`** is a
stricter, independent, lower-level check that answers a logically prior
question: "is the *population feeding into* those cells identical" — at
six named levels (image identity, pair identity, subject/object labels,
pair index/row order, GT predicate, and the WPRD cell set derived from
all of the above), with no dependency on the score tensors or `Bench`/
`Mech` machinery at all. The two tools are complementary, not redundant:
`verify_population_identity` can and should run **first**, since it can
diagnose *why* two dumps would fail `verify_paired_dumps` in a way that
tool alone cannot (e.g. `DIFFERENT_IMAGE_POPULATION` vs
`DIFFERENT_GT_LABELS` are indistinguishable from `verify_paired_dumps`'s
own output alone — both would simply show `cell_keys_match=False`).

**The required identity level before a paired bootstrap (item 5) may run:**

| status from `verify_population_identity` | is `paired_cell_bootstrap` admissible? |
|---|---|
| `EXACT_MATCH` | **Yes** — the strongest level the dumps permit: content and raw row order both identical. |
| `SAME_COUNTS_BUT_DIFFERENT_ROWS` | **Only after re-deriving `cell_values` via content-based realignment** (i.e. via `wprd_with_cell_keys`/`verify_paired_dumps`, which group by label content, not row position) — **never** by pairing the two dumps' `result.json::cell_values` arrays positionally, since that is exactly the scenario this status exists to catch. |
| `DIFFERENT_IMAGE_POPULATION`, `DIFFERENT_PAIR_POPULATION`, `DIFFERENT_GT_LABELS`, `DIFFERENT_CELL_ASSIGNMENT` | **No.** Stop and diagnose which arm's evaluation configuration diverged (different `--eval_batches`, different dataset split resolution, different `vg150_root`, a code change between the two evaluation runs) — do not proceed with any comparison. |
| `INSUFFICIENT_METADATA` | **No.** The dump(s) must be regenerated with `--eval_sgg_dump_rel_feat true` and the standard identity fields present (this repo's evaluators already write all required fields by default — this status would only fire against a hand-trimmed or corrupted dump). |

**Concretely, the full gate before trusting any R0-vs-R2 number, in order:**

```python
from tools.verify_population_identity import verify_population_identity, EXACT_MATCH
from tools.readout_v2_offline_analysis import (verify_contract_match,
    verify_paired_dumps, paired_cell_bootstrap)

ident = verify_population_identity("runs/eval_C1/pair_logits.pt",
                                    "runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt")
assert ident["status"] == EXACT_MATCH, ident["reason"]   # the strongest available level

gate = verify_contract_match(r0_result_json, r2_result_json, expect_identical=True)
assert gate["pass"]

pop = verify_paired_dumps(...)          # confirms the WPRD-cell view agrees, independently
assert pop["safe_to_pair"]

boot = paired_cell_bootstrap(pop["cell_values_a"], pop["cell_values_b"])
```

**Counts alone — image count, row count, or WPRD cell count matching — are
explicitly insufficient**, per the task's own framing: `runs/eval_C1/
result.json` and a hypothetical R2 run over a *different* validation
subset (e.g. a truncated `--eval_batches` run) could trivially produce the
same `population.cells` integer while scoring an entirely different set
of images — `verify_population_identity` is what turns "the numbers look
the same size" into an actual, checked claim.

---

## 3. Execution order

Not every item is independent; run in this order once R2's `result.json`
and `pair_logits.pt` exist:

1. **Item 1** (checkpoint integrity) — can and should run immediately,
   independent of the endpoint. If this fails, stop; do not trust anything
   downstream (this mirrors the preregistration's own "do not tune around
   a failure" posture).
2. **Items 9-11** (prototype diagnostics) — also independent of the
   endpoint (checkpoint-only); run alongside item 1.
3. **Item 2** (exact population match) — first thing to run once the dump
   exists. Gates item 5 and, implicitly, the trustworthiness of items 3-4,
   6-8, 12-15 (all of which read the same dump).
4. **Items 3-4** (WPRD, weighted WPRD) — cross-check against the endpoint's
   own printed/`result.json` numbers.
5. **Item 5** (paired bootstrap) — only after item 2 passes and
   `verify_contract_match(..., expect_identical=True)` passes.
6. **Items 6-8** (per-cell deltas, tail breakdown, zero-training-pair
   breakdown) — descriptive decomposition of item 5's result.
7. **Items 12-13** (prior correlation, calibration) — independent of items
   2/5, can run in parallel with them.
8. **Items 14-15** (random-text, shuffled-prototype controls) — run last,
   as the falsification layer: if either fails, everything computed in
   items 3-8 needs to be re-read as potentially not measuring what it
   claims to measure, regardless of how clean those numbers look in
   isolation.

## 4. What this plan does not do

- Does not compute a final verdict (`healthy` / `null` / `negative` /
  `invalid`, per the preregistration §16/§28's own categories) — that
  requires the actual numbers, which do not exist until the endpoint
  finishes.
- Does not write `docs/PAPER_C_READOUT_V2_PILOT_RESULT.md` — a separate
  step, after this checklist runs for real.
- Does not fix `tools/c0_c1_compare.py`'s gate D4 in place (§1.4) — flags
  it and provides a corrected, separate function
  (`verify_contract_match`) instead, since patching a script whose own
  header says "reused unmodified" is itself a source-code change the task
  explicitly excludes from this pass.
- Does not implement a dedicated "zero-training-pair" function (item 8) —
  specified precisely enough to implement in ten minutes once the pilot's
  realized sample list is available, but not built speculatively here.

## 5. New artifacts (uncommitted, pending review)

- `tools/readout_v2_offline_analysis.py` — CPU-only library implementing
  items 1, 2, 5 (gate), 9-11, 12, 13, 14, 15, plus the corrected
  `verify_contract_match` replacing `c0_c1_compare.py`'s buggy gate D4 for
  same-contract comparisons.
- `tests/test_readout_v2_offline_analysis.py` — 19 tests, entirely
  synthetic data (same dump-construction convention as the existing
  `tests/test_cprime_mechanism.py`), **19/19 passing**. Two real bugs in
  the first draft of the library were caught and fixed by this suite
  before this document was finalized: a `.max(-1)` tuple-unpacking error
  in `calibration_report`, and a column-count mismatch in
  `random_text_control` that would otherwise have crashed (or silently
  misbehaved) the moment it was run against a real 50-class dump with
  fewer than 50 random strings. `prototype_drift` and
  `random_text_control`'s default (non-injected) CLIP-loading path are not
  covered by mocked unit tests (they need a real CLIP text encoder) —
  `prototype_drift` was instead validated once, this session, against the
  real `checkpoints/readout_v2_pilot_seed1234_v3.pt`, and reproduced the
  exact numbers already on record from the recovery session's manual run.
- `tools/verify_population_identity.py` — CPU-only, dependency-light (no
  `Bench`/`Mech`, no score tensors needed) population-identity checker at
  six levels (§2A above), returning one of seven explicit statuses
  (`EXACT_MATCH`, `SAME_COUNTS_BUT_DIFFERENT_ROWS`,
  `DIFFERENT_IMAGE_POPULATION`, `DIFFERENT_PAIR_POPULATION`,
  `DIFFERENT_GT_LABELS`, `DIFFERENT_CELL_ASSIGNMENT`,
  `INSUFFICIENT_METADATA`) rather than a boolean.
- `tests/test_verify_population_identity.py` — 10 tests, entirely
  synthetic, minimal dumps (no score tensors at all). **10/10 passing.**
  Covers every scenario the task named plus a documented note on why
  `DIFFERENT_CELL_ASSIGNMENT` is structurally unreachable through the
  public API alone (cell identity is a pure function of exactly the
  label/GT data the earlier checks already verify equal, by construction)
  — recorded as a design property, not a coverage gap.
