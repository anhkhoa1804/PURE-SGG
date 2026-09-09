# Paper C — predicate-disjoint leakage hardening: dependency trace + patch

CPU-only. No GPU code run, no training launched, no checkpoint touched.
Written while the R2 endpoint evaluation was running — untouched
throughout. Three source files were modified (minimal, additive, opt-in);
one new test file (19 tests, all passing, ~10s) exercises every leakage
case named in `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` §6, plus two
corrections to that document's own claims, found and fixed while building
these tests (see §4).

---

## 1. The dependency graph — every consumer of `global_pred_pool` traced

The task named three leakage vectors and one choke point
(`global_pred_pool`). Tracing every consumer found **two more, previously
unidentified choke points** — restricting `global_pred_pool` alone, the
"obvious call site," would have closed *none* of the three named vectors,
because none of them actually read from train.py's `global_pred_pool`
variable in isolation; they all read from structures *derived* from it, and
one critical structure (`pos_pred_ids`, the actual training labels) is
derived from a **second, textually distant, independently-constructed
vocabulary** that has no reference to `global_pred_pool` at all.

### 1.1 Choke point A — `openvocab_rel/train.py::global_pred_pool` (line ~1523)

| # | source | transformation | consumer | gradient-affecting? | held-out identity can enter? | held-out frequency can enter? |
|---|---|---|---|---|---|---|
| A1 | `scan_vg150_predicate_vocab()` | lowercase/dedup/append `"relation"` | `global_pred_pool` itself | — | (before restriction) yes | — |
| A2 | `global_pred_pool` | `_iter_predicates_from_dataset_for_weights` scan + dict comprehension (train.py ~1523) | `pred_freq` | No (a lookup table) | **yes, as a dict key**, before restriction | **yes**, before restriction |
| A3 | `global_pred_pool`, `pred_freq` | `split_predicates_head_mid_tail` | `pred_buckets` / `pred_bucket_masks` | No (reporting/bucket masks only — no loss reads `pred_buckets` directly; consumed only by `analysis_DE_buckets_predicates`-style **reporting**, confirmed by grep: no `pred_buckets` reference inside any `l_*` loss term) | yes (name), before restriction | yes (bucket assignment), before restriction |
| A4 | `global_pred_pool` | `encode_predicate_vocab` (CLIP text encoder forward) | `pred_emb_s2o` (= `E`) | **Yes** — every predicate-CE / text-CE / role-swap / triplet-rank loss scores against this tensor | **Yes, directly** — a held-out predicate's own embedding row, if present | — |
| A5 | `global_pred_pool` | `{p: i for i, p in enumerate(...)}` | `pred_to_idx` (train.py's own copy — **not** the one that actually labels training examples, see Choke point B) | Indirectly (used for `_experiment_snapshot`/`predicate_vocab_hash`, a provenance field) | yes (as dict keys), before restriction | — |
| A6 | `global_pred_pool` | `load_predicate_metadata(path, global_pred_pool)` | `pred_metadata` | No directly; feeds A7 | yes (**note**: `load_predicate_metadata` returns a superset including the on-disk file's full content regardless of the `predicates` arg — see §4.2 correction; harmless because A7 only *looks up* names drawn from `global_pred_pool`) | — |
| A7 | `global_pred_pool`, `pred_metadata` | `_build_predicate_group_matrix` | `pred_group_matrix` | **Yes**, when `predicate_group_relaxation_enabled=true` (default **False**) — feeds `_predicate_ce_loss`'s soft-label term | **Yes if enabled** — a held-out predicate's group membership, before restriction | — |
| A8 | `global_pred_pool`, `pred_freq` | `_build_predicate_ce_weights` | `pred_ce_weights` | **Yes** — the class-weight vector `_predicate_ce_loss` and `counterfactual_hard_negative_loss`'s `class_weights=` argument both consume | — | **Yes, directly** — proven in §3, test 7 |
| A9 | `global_pred_pool`, `pred_freq` | `[pred_freq.get(p,1) for p in global_pred_pool]` → `torch.log(.../.sum())` | `pred_log_prior` | **Yes** when `adaptive_prior`/certain calibration paths compose it (classifier branch only — text/WPRD-scored branch never composes the prior, an existing, separately-verified invariant) | — | **Yes, directly**, before restriction |
| A10 | `global_pred_pool` (via `pred_emb_s2o`) | `build_predicate_similarity_matrix(pred_emb_s2o)`, gated `lattice_loss_enabled` (default **True**) | `pred_sim_matrix` | **Yes** — feeds `predicate_label_relaxation` (soft-label smoothing) **and** `lattice_negative_weights` | **Yes, directly** — a held-out predicate's embedding row's similarity to every seen predicate | — |
| A11 | `global_pred_pool` (via `pred_emb_s2o`) | `build_confusable_index(pred_emb_s2o, topm=32)` | `confusable_idx` | **Yes** — `cf_pred_feat[valid] = pred_emb_s2o[conf_ids].detach()` feeds `counterfactual_hard_negative_loss` | **Yes, directly** — the single sharpest vector found in the prior audit session, confirmed here structurally in §3 test 5 | — |
| A12 | `global_pred_pool` | `PairPriorTable.load(pr_path, global_pred_pool, ...)`, gated `prior_residual_enabled` (default **False**) | `prior_residual_table` | Yes, when enabled | — | Yes, when enabled, before restriction |
| A13 | `global_pred_pool` (readout v2 setup block only) | fresh `encode_predicate_vocab` call | Readout v2's `E`/`P0` | **Yes**, when `readout_v2_enabled=true` | **Yes, directly**, before restriction | — |

**Every row in this table is gated by `global_pred_pool`'s content and
shrinks/excludes correctly once `global_pred_pool` is restricted early
enough** — confirmed in §2/§3. The one subtlety that makes "restrict early"
load-bearing rather than a style preference: **A2's `pred_freq` dict
comprehension runs at train.py line ~1523, which is *before* the
pre-existing lowercase/dedup/`"relation"`-append block (lines ~1524-1527)
even finishes.** Restricting `global_pred_pool` *after* that block (the
"obvious," textually-adjacent place to add a one-line filter) would still
leave `pred_freq` — and therefore A3/A8/A9's shapes — built from the
**unrestricted** pool. The patch (§2) restricts immediately after the raw
scan, before any consumer runs.

### 1.2 Choke point B — `VG150DataLoader.__init__`'s own `pred_to_idx` (vg150_loader.py ~line 1030) — **the previously unidentified vector**

| # | source | transformation | consumer | gradient-affecting? | held-out identity can enter? | held-out frequency can enter? |
|---|---|---|---|---|---|---|
| B1 | `_load_vg150_vocab(cfg.vg150_root)` | independent re-scan, **not derived from `global_pred_pool` at all** | `self.pred_to_idx` | — | — | — |
| B2 | `self.pred_to_idx` | `_prepare_rel_payload` line 62: `pred if pred in pred_to_idx else "relation"` | `pred_strs` | Indirectly (determines the label every loss ultimately sees) | **Yes, directly, if unrestricted** — this is what actually determines `pos_pred_ids`, the tensor every training loss reads | — |
| B3 | `pred_strs`, `self.pred_to_idx` | `rel_pred_ids = [pred_to_idx.get(p,-1) for p in pred_strs]` | `pos_pred_ids` (via the batch collate path) | **Yes — the actual training label for every gradient-affecting loss in train.py** (`_predicate_ce_loss`, `counterfactual_hard_negative_loss`, text-CE, role-swap, triplet-rank) | **Yes, directly, if `self.pred_to_idx` is unrestricted** — regardless of what `global_pred_pool` in train.py says | — |

**This is the finding that falsifies "restricting `global_pred_pool` alone
is sufficient."** `VG150DataLoader` is constructed (train.py ~1481/~1514)
*before* `global_pred_pool` is even computed (~1523), from a *completely
independent* call to `_load_vg150_vocab`. In the **unpatched** code, both
call sites happen to read the same underlying vocabulary file in the same
order, so they coincidentally agree — but nothing enforces this, and a
patch that only touches train.py's own `global_pred_pool` variable would
leave `pos_pred_ids` — the tensor every single training loss actually
reads — completely unrestricted. A held-out predicate's GT row would still
resolve to a real index via `self.pred_to_idx` (B1, untouched), and would
either (a) index out of range against the now-shorter `pred_emb_s2o` (a
crash, the *best* possible failure mode), or (b) by index-count
coincidence, silently alias onto a *different seen predicate's* embedding
row — a silent, catastrophic mislabeling, not a clean absence of
supervision.

### 1.3 Choke point C — `evals.py`'s independent `pred_vocab` (lines 1751, 2911) — **traced, deliberately NOT restricted**

`evals.py`'s SGG evaluation forward pass (used by `--eval_every` periodic
in-training monitoring **and** by the standalone endpoint tools
`c0_c1_evaluate.py`/`readout_v2_evaluate.py`) calls
`scan_vg150_predicate_vocab()` **again, independently**, and re-encodes a
**fresh, full, unrestricted** `pred_emb` from the checkpoint's own CLIP
text encoder. This is **not gradient-affecting** (a `@torch.no_grad()`-style
forward pass for metric reporting only — confirmed: no `l_*` loss variable
in train.py's training step reads anything from `evals.py`'s SGG-eval
return value). **This is intentionally left unrestricted** — it is exactly
the mechanism a genuine predicate-disjoint **endpoint** evaluation needs
(scoring Unseen_test predicates via their freshly-encoded text embeddings,
per `docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md`). Restricting it here
would break the one legitimate use of the full vocabulary this whole
research program needs. **Named residual, not a bug**: if a predicate-
disjoint *training* run also enables `--eval_every` for live monitoring,
the periodically-printed R@50/mR@50/head-mid-tail metrics reflect the
*full* vocabulary, not the training-restricted one — a researcher watching
training logs could see Unseen-predicate-inclusive numbers scroll by. This
does not affect what the model learns (gradient-affecting paths are fully
covered by Choke Points A and B), but is a monitoring-hygiene note worth
being aware of, not silently hidden.

---

## 2. The patch — smallest change that makes `global_pred_pool == Seen_train` an enforced property

Three files, additive only, one new opt-in switch
(`predicate_disjoint_seen_predicates`, default `""` = fully disabled,
byte-for-byte unchanged behavior for every existing run):

### `openvocab_rel/datasets/vg150_loader.py`

- **`restrict_predicate_pool(full_pool, seen)`** (new function) — the
  single-source-of-truth transform: `seen=None` is the identity (disabled);
  otherwise filters `full_pool` to `seen ∪ {"relation"}`, **preserving
  the canonical scan's original relative order** — the reason two
  independent call sites (train.py's `global_pred_pool`, the loader's own
  `pred_to_idx`) applying this SAME function to the SAME canonical source
  order land on **identical index assignments** without ever exchanging an
  explicit ordered list, only the `seen` set.
- **`parse_seen_predicates(spec)`** (new function) — parses the
  comma-separated CLI value into a `frozenset`, with `""` → `None` kept
  explicitly distinct from `frozenset()` (empty-but-active, which would
  legitimately restrict training to zero real predicates).
- **`VG150LoaderConfig.predicate_disjoint_seen_predicates: str = ""`**
  (new field).
- **`VG150DataLoader.__init__`**: after building `self.pred_to_idx` exactly
  as before, if the new field is set, applies `restrict_predicate_pool` to
  it — closing Choke Point B.
- **`_build_relation_entries(..., drop_out_of_vocab: bool = False)`** (new
  parameter, default preserves history exactly): when `True`, a
  relationship whose predicate is absent from `pred_to_idx` is **dropped**
  — the entry never reaches `positive_preds`/`_prepare_rel_payload` at all
  — rather than silently remapped to `"relation"` (the pre-existing,
  now-default-preserved fallback for genuinely out-of-vocabulary/malformed
  strings). This is a **relationship-level** drop: a `(subject, object)`
  pair with one seen and one held-out relationship keeps the seen one
  (§3, test 3) — matching the split audit's own requirement, not the
  coarser, incorrect alternative of dropping the whole pair.
- Four call sites updated to pass
  `drop_out_of_vocab=bool(getattr(self.cfg, "predicate_disjoint_seen_predicates", ""))`
  — one flag, threaded through the one config object already available at
  every call site; no new parameter threading required beyond this.

### `openvocab_rel/config.py`

- **`TrainConfig.predicate_disjoint_seen_predicates: str = ""`** (new
  field, mirrors the loader-config field).

### `openvocab_rel/train.py`

- New CLI flag `--predicate_disjoint_seen_predicates`.
- `global_pred_pool` is restricted via `restrict_predicate_pool` **immediately
  after** the raw `scan_vg150_predicate_vocab()` call — before `pred_freq`,
  before the existing lowercase/dedup/`"relation"`-append block, before
  every one of Choke Point A's 13 traced consumers.
- **Both** `vg150_cfg` (train split) and `vg150_val_cfg` (validation split)
  are given the same `predicate_disjoint_seen_predicates` value —
  deliberately, not asymmetrically: leaving the validation loader
  unrestricted while the training-side `pred_emb_s2o`/`global_pred_pool`
  are restricted would make any **in-training** `--eval_every` pass score
  against index positions computed from a *different* (full, 51-wide)
  vocabulary than the one the restricted `pred_emb_s2o`'s columns actually
  correspond to — a real index-space mismatch, not merely an unwanted
  information exposure. The genuine full-vocabulary endpoint evaluation
  (Choke Point C, §1.3) is a **separate tool**, not this in-training val
  loader, and is unaffected by this choice.

**Every default-config code path (empty string) is unchanged.** Verified:
`tests/test_readout_v2.py`'s full 19-test suite (unrelated to this patch)
still passes unmodified after these edits (§5).

---

## 3. Test coverage — the ten leakage cases, adapted to exercise real code

`tests/test_predicate_disjoint_hardening.py`, **19 tests, all passing,
~10 seconds, CPU-only** (only reads small vocabulary JSON files, never
`train.jsonl` or a checkpoint):

1. **Single choke point** — `restrict_predicate_pool` produces exactly
   `Seen_train ∪ {"relation"}`, in canonical order, both when active and
   when disabled (identity).
2. **Choke Point B closure** — the loader's independently-derived
   `pred_to_idx`, restricted via the same function, produces the exact same
   ordered vocabulary as train.py's own `global_pred_pool` — the direct
   proof that the "second choke point" is closed, not merely asserted.
3. **Relationship-level, not pair-level, drop** — a pair with one seen and
   one held-out relationship keeps the seen one; the pair itself survives.
4. **Historical behavior preserved** — `drop_out_of_vocab=False`
   reproduces the exact pre-existing remap-to-`"relation"` fallback.
5. **Zero-unseen-label-in-batch** — across a batch mixing 4 held-out and 3
   seen predicates, `rel_pred_ids` contains no index resolving to a
   held-out name; the held-out rows are simply absent.
6. **`pred_sim_matrix`/`confusable_idx` structural bound** — with a
   restricted, shape-`(n,32)` stand-in embedding table, both derived
   tensors are shape-bounded to `n`, so a held-out predicate's row is not
   merely unlikely to be referenced but **cannot** be, because it was never
   computed. The actual training-loop gather operation
   (`pred_emb_s2o[conf_ids]`) is exercised directly.
7. **`pred_group_matrix` restricted** — shape-bounded to the restricted
   pool; `predicate_group_relaxation_enabled` confirmed `False` by default.
8. **`pred_ce_weights` frequency exclusion** — a held-out predicate's
   frequency is poisoned to `10^9`; the resulting weight vector is
   **bit-identical** to the unpoisoned version, proving the value is never
   read, not merely small.
9. **Frequency-prior lookup pattern** — the exact list-comprehension
   pattern train.py uses for `pred_log_prior` is exercised directly with a
   poisoned held-out value; confirmed never selected.
10. **Metadata orphan exclusion** — the three fictitious predicates found
    in `configs/predicate_metadata_vg150.json`
    (`docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` §0) never survive
    restriction, for any seen set.
11. **Synonym/near-duplicate co-assignment** — a reusable
    `validate_no_synonym_split` helper (new, standalone) confirms the
    Conservative candidate split has no violation, and correctly flags a
    constructed violation (`wearing` seen / `wears` unseen).
12. **`is_symmetric_predicate` audit** — see §4 correction below.

### §3 note on scope: what these tests do NOT cover

They do not run a real `VG150DataLoader`/`VG150JSONLDataset` end-to-end
(would require touching `train.jsonl`, 230 MB — heavier than warranted for
a unit-level trace, and unnecessary since Choke Point B's fix is proven at
the exact transform level, test 2). They do not run an actual forward/
backward pass through `_predicate_ce_loss`/`counterfactual_hard_negative_loss`
with a real model (would require GPU/CLIP, excluded by the task's
constraints) — the gradient-affecting claim for A8/A10/A11/B3 rests on
**structural** proof (shape bounds, bit-identical outputs under
poisoning), which is the correct level of proof for "cannot," not "was not
observed to."

---

## 4. Corrections to the prior audit, found while building these tests

Per this program's own discipline (challenge prior conclusions when
evidence supports doing so) — two claims in
`docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` are corrected here, not
silently:

### 4.1 §2.5's "zero callers" claim was wrong

The prior audit claimed `openvocab_rel/prompts.py::is_symmetric_predicate`
/ `SYMMETRIC_RELATION_BLACKLIST` was dead code. **This is false** —
`vg150_loader.py::_prepare_rel_payload` calls it on every relationship's
predicate string to build `rel_non_sym_mask`, which flows through the
batch collate path into `train.py` as `pos_non_sym_mask` and **does gate a
gradient-affecting branch** (the role-swap counterfactual loss,
`train.py` ~2148-2150). This was missed because the original audit's grep
was scoped to `train.py`/`evals.py` and did not check
`vg150_loader.py`. **The correction does not change this hardening pass's
safety conclusion**: because the relationship-level drop happens strictly
*before* `_prepare_rel_payload` is ever invoked with a held-out
predicate's name, `is_symmetric_predicate` is structurally never evaluated
on one when `drop_out_of_vocab=True` — proven directly in
`test_is_symmetric_predicate_never_called_on_held_out_names_when_dropping`.
The blacklist's own inconsistency with `predicate_metadata.py`'s
`symmetric` field (§2.5's other finding) still stands and is still worth
fixing eventually, but is now correctly filed as "a real, minor,
non-predicate-disjoint-relevant code-quality issue," not "dead code."

### 4.2 §6's implicit assumption about `load_predicate_metadata`'s output

The prior audit's test 8 implied `load_predicate_metadata(path,
pred_vocab)` returns a dict scoped to `pred_vocab`. It does not — it
returns a superset including the full on-disk file's entries (including
the 3 orphans) regardless of the `predicates` argument. This is harmless
for every consumer traced here (`_build_predicate_group_matrix` only
*looks up* names drawn from the already-restricted `pred_vocab`, never
iterates `metadata`'s own keys), but the exact wording of any future test
asserting "`metadata` has no held-out keys" would be testing the wrong
invariant — corrected in `test_predicate_group_matrix_restricted_to_seen`'s
docstring.

---

## 5. Regression check on existing behavior

`tests/test_readout_v2.py` (19 tests, unrelated to this patch, exercises
the real `run_readout_v2.sh`/`run_c0_c1.sh` resolved configs) —
**re-run in full after these edits: 19/19 still pass**, confirming the new
field's default value leaves every existing training recipe unaffected.
The full repository test suite was **not** run, per instruction (R2 was
consuming the machine).

---

## Final report

### A. Exact leakage paths closed

All thirteen Choke-Point-A consumers (§1.1, A1-A13) and both
Choke-Point-B consumers (§1.2, B1-B3) — meaning: **training labels**
(B3/`pos_pred_ids`), **class relaxation** (A7/`pred_group_matrix`,
gated off by default anyway), **negative weights**
(A10/`pred_sim_matrix` → `lattice_negative_weights`), **counterfactual
negatives** (A11/`confusable_idx` → `cf_pred_feat`), **frequency priors**
(A9/`pred_log_prior`, A2/`pred_freq`, A8/`pred_ce_weights`),
**similarity neighbors** (A10/`pred_sim_matrix` itself), **auxiliary
losses** (A12/`prior_residual_table`, A13/Readout v2's own `E`), and
**hard-negative mining** (A11 again) — every one of the task's eight
named categories traces to a specific row in §1's tables, and every row is
either (a) shape-bounded to the restricted pool by construction (proven,
not sampled, in §3 tests 6-9), or (b) never invoked on a held-out name at
all (§3 tests 3, 5, 12).

### B. Residual leakage paths

**One, named and deliberate, not gradient-affecting**: Choke Point C
(§1.3) — `evals.py`'s independently-scanned, always-full `pred_vocab`,
used by periodic in-training `--eval_every` monitoring and by the
standalone endpoint-evaluation tools. This is required, not a bug — a
genuine predicate-disjoint endpoint evaluation needs exactly this
mechanism to score Unseen_test predicates via freshly-encoded text
embeddings. Its only cost is a monitoring-hygiene note: live training logs
for a predicate-disjoint run, if `--eval_every` is left on, will show
metrics computed against the full vocabulary, not the training-restricted
one. **No code path from Choke Point C reaches any loss term** — confirmed
by grep (no `l_*` variable in the training step reads an `evals.py` SGG-eval
return value).

### C. Is the Conservative split now safe for a future CPU-only pipeline validation?

**Yes, for the code paths this hardening pass covers.** With
`--predicate_disjoint_seen_predicates` set to the Conservative split's 46
seen predicates (excluding `carrying, belonging to, playing, painted on`
— `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md` §4.1) and
`drop_out_of_vocab` active via the same flag, a CPU-only dry run (config
resolution + `_build_relation_entries` unit checks, exactly as this
session's tests already do against the real vocabulary) can proceed. This
is **not** the same claim as "safe for GPU training" — that would
additionally require (i) the not-yet-built relationship-level filter to be
exercised against a real `VG150JSONLDataset` batch (§3's scope note), and
(ii) the §5 full regression suite to be re-run once machine time is not
constrained by a concurrent GPU job.

### D. Is the Balanced split ready for preregistration?

**Not yet, and this hardening pass does not change that.** The Balanced
split (11 held-out predicates, `docs/PAPER_C_PREDICATE_DISJOINT_SPLIT_AUDIT.md`
§4.2) is unaffected by anything code-specific in this document — its
readiness gate remains what §6 of that audit already specified: a
dedicated, dated preregistration document, written *after* the
Conservative split's own CPU-only pipeline validation (C, above) passes
cleanly. **This document proves the mechanism works; it does not
constitute that preregistration.**

### E. What "leakage-free" would still require, not claimed here

This document does **not** claim the pipeline is unconditionally
"leakage-free" — it claims, and has shown by structural proof and targeted
test, that the eight named gradient-affecting categories are closed for
the code paths traced in §1, given the patch in §2. Not yet exercised:
(i) a real end-to-end batch through `VG150JSONLDataset` (the one backend
every current script actually uses) with `drop_out_of_vocab=True`, (ii)
an actual forward/backward pass confirming `predicate_prototypes`-style
gradients never touch a held-out row in a live model (excluded here by the
no-GPU constraint), and (iii) the not-yet-built dedicated predicate-disjoint
endpoint evaluator's own correctness (Choke Point C's intended, separate
consumer). These are the concrete next steps, not residual doubts about
what has already been shown.
