# Paper C — Predicate-disjoint split: forensic data/leakage audit

Audit/design only. No dataset file was modified, no code was changed, no
split was created or registered, no GPU job was launched. Every number in
this document was computed this session by streaming the actual
`datasets_vg150_clean/{train,validation,test}.jsonl` files end to end
(83,249 / 10,401 / 10,403 images respectively), and every code claim was
verified by reading the actual source file and line, not recalled from
memory. File:line references are exact as of this session's `HEAD`.

---

## 0. Ground truth about the vocabulary itself (read this first)

Two different "50/51/53-predicate" artifacts exist in this repository and
they are **not** the same set — this matters for every section below:

1. **The real, trained-on vocabulary**: `datasets_vg150_clean/vocabulary/
   predicates.json` (`idx_to_predicate`, indices 1-50) — exactly **50
   predicates**, confirmed to be exactly the set observed in all three
   `.jsonl` splits (train: 50 distinct predicates over 1,046,427
   relationships / 83,249 images; validation: 50 distinct over 132,556
   relationships / 10,401 images; test: 50 distinct over 132,334
   relationships / 10,403 images — **every one of the 50 appears in every
   split**, no split-exclusive predicate exists today). `openvocab_rel/
   datasets/vg150_loader.py:1129` (`scan_vg150_predicate_vocab`) reads this
   same underlying vocab mapping. Training additionally appends a synthetic
   `"relation"` placeholder token (`train.py:1526-1527`), making the
   **model's** working vocabulary 51 — but `"relation"` is not a real
   predicate and must be excluded from any seen/unseen partition (already
   stated in `docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md` §3, reconfirmed
   here against the actual vocabulary file).
2. **`configs/predicate_metadata_vg150.json`**: has **53** entries, not 50.
   Verified this session by set-diffing its keys against the real
   50-predicate vocabulary: it contains **3 orphaned predicates that do not
   exist anywhere in this dataset** — `"around"`, `"growing on"`, `"says"`
   — and has full coverage of all 50 real predicates (no real predicate is
   missing from it). **Any tool that iterates this file's keys assuming it
   is the training vocabulary will silently pick up 3 fictitious
   predicates.** This is a data-hygiene finding, not itself a leakage
   vector (the orphans are never scanned by `scan_vg150_predicate_vocab`,
   which reads the dataset's own vocabulary file, not this metadata file),
   but any split-construction tooling must explicitly intersect this file's
   keys with the real 50 before using its `group`/`symmetric` fields.

---

## 1. Predicate inventory

### 1.1 Full table — all 50 predicates, real counts, group, symmetry

Sorted by training frequency (descending). Group/symmetric columns are
from `configs/predicate_metadata_vg150.json` (orphans excluded, verified
1:1 coverage above).

| rank | predicate | group | symmetric | train | val | test |
|---|---|---|---|---|---|---|
| 1 | on | spatial | no | 380301 | 48102 | 47953 |
| 2 | has | possession | no | 153508 | 19601 | 19766 |
| 3 | in | spatial | no | 124345 | 15566 | 15778 |
| 4 | of | possession | no | 77165 | 9945 | 9875 |
| 5 | wearing | possession | no | 66468 | 8266 | 8223 |
| 6 | with | contact | no | 34458 | 4109 | 4371 |
| 7 | behind | spatial | no | 23534 | 2980 | 3063 |
| 8 | holding | contact | no | 20981 | 2507 | 2655 |
| 9 | next to | spatial | **yes** | 19410 | 2496 | 2431 |
| 10 | near | spatial | **yes** | 16395 | 2055 | 1976 |
| 11 | under | spatial | no | 12725 | 1696 | 1663 |
| 12 | in front of | spatial | no | 10503 | 1350 | 1242 |
| 13 | sitting on | pose | no | 8723 | 1124 | 1120 |
| 14 | above | spatial | no | 8141 | 1156 | 1072 |
| 15 | wears | possession | no | 8002 | 1073 | 1027 |
| 16 | standing on | pose | no | 6903 | 905 | 888 |
| 17 | attached to | contact | no | 5429 | 675 | 652 |
| 18 | for | other | no | 5174 | 708 | 659 |
| 19 | over | spatial | no | 5151 | 680 | 657 |
| 20 | at | spatial | no | 4889 | 612 | 696 |
| 21 | hanging from | contact | no | 4727 | 593 | 612 |
| 22 | wrapped around | spatial | no | 4478 | 617 | 570 |
| 23 | riding | action | no | 3643 | 441 | 425 |
| 24 | carrying | contact | no | 2419 | 301 | 240 |
| 25 | walking on | action | no | 2267 | 223 | 294 |
| 26 | eating | action | no | 2187 | 295 | 240 |
| 27 | and | other | **yes** | 1994 | 247 | 215 |
| 28 | along | spatial | no | 1968 | 245 | 259 |
| 29 | between | spatial | **yes** | 1951 | 215 | 259 |
| 30 | laying on | pose | no | 1948 | 243 | 256 |
| 31 | covering | contact | no | 1878 | 241 | 243 |
| 32 | watching | action | no | 1812 | 229 | 155 |
| 33 | playing | action | no | 1758 | 209 | 252 |
| 34 | looking at | action | no | 1737 | 203 | 196 |
| 35 | against | spatial | no | 1697 | 247 | 221 |
| 36 | belonging to | possession | no | 1599 | 207 | 182 |
| 37 | from | spatial | no | 1566 | 209 | 172 |
| 38 | painted on | attribute | no | 1455 | 248 | 165 |
| 39 | made of | attribute | no | 1367 | 146 | 158 |
| 40 | parked on | action | no | 1353 | 208 | 177 |
| 41 | to | spatial | no | 1275 | 180 | 182 |
| 42 | covered in | contact | no | 1225 | 153 | 154 |
| 43 | mounted on | contact | no | 1090 | 120 | 137 |
| 44 | across | spatial | no | 1085 | 151 | 117 |
| 45 | part of | part-whole | no | 1067 | 106 | 112 |
| 46 | on back of | spatial | no | 1033 | 151 | 132 |
| 47 | using | action | no | 1001 | 127 | 106 |
| 48 | lying on | pose | no | 972 | 136 | 131 |
| 49 | walking in | action | no | 853 | 103 | 87 |
| 50 | flying in | action | no | 817 | 156 | 118 |

Column totals reconcile exactly to the earlier-verified counts: train
1,046,427; validation 132,556 (matching every WPRD number this program has
ever reported); test 132,334.

### 1.2 Long-tail distribution

Ratio of most-to-least frequent (train): `on` (380,301) / `flying in` (817)
≈ **465×**. The top 3 predicates (`on`, `has`, `in`) alone account for
**≈63.7%** of all training relationships (657,154 / 1,046,427). The bottom
20 predicates combined (`riding` through `flying in`) account for **≈2.9%**
of training relationships. This is the same severe imbalance this program's
own `predicate_ce_weight_power`/focal-loss machinery already exists to
manage, and it directly shapes which predicates are *safe* to hold out
(§1.4).

### 1.3 Rare predicates (bottom decile by train count, ≤5 per split floor)

`flying in` (817/156/118), `walking in` (853/103/87), `lying on`
(972/136/131), `using` (1001/127/106), `on back of` (1033/151/132), `part
of` (1067/106/112) — all have **validation and test counts under 160**.
Any held-out evaluation restricted to one of these alone will be
statistically noisy (a handful of cells). Not disqualifying, but any split
using these must be interpreted with that caveat stated up front, not
discovered after the fact.

### 1.4 Semantic clusters (group sizes, from the verified 50-predicate ∩
metadata intersection)

| group | count | members |
|---|---|---|
| spatial | 18 | above, across, against, along, at, behind, between, from, in, in front of, near, next to, on, on back of, over, to, under, wrapped around |
| action | 10 | eating, flying in, looking at, parked on, playing, riding, using, walking in, walking on, watching |
| contact | 8 | attached to, carrying, covered in, covering, hanging from, holding, mounted on, with |
| possession | 5 | belonging to, has, of, wearing, wears |
| pose | 4 | laying on, lying on, sitting on, standing on |
| other | 2 | and, for |
| attribute | 2 | made of, painted on |
| part-whole | 1 | part of |
| textual | 0 | *(only "says" is tagged textual, and it is not a real VG150 predicate here — §0)* |

`spatial` alone is 36% of the vocabulary — any split must not
over-concentrate held-out predicates there, or "unseen" performance
becomes mostly a test of spatial/geometry generalization by construction
(directly relevant to the geometry-only-floor falsification control already
specified in the design doc).

### 1.5 Near-synonyms / aliases requiring co-assignment (hard requirement,
per the design doc's §3.4 leakage rule — verified against the real list,
not assumed)

| pair | risk |
|---|---|
| `wearing` (66,468 train) / `wears` (8,002 train) | same lemma, tense/form variants. **`wearing` is effectively impossible to hold out at all** (§1.6); if `wears` is ever held out, `wearing` MUST also be held out, which is prohibitively expensive — in practice this pair should simply be excluded from consideration entirely. |
| `laying on` (1,948) / `lying on` (972) | near-identical spelling and meaning (both `pose`); genuinely easy to confuse even for a human annotator. Must be co-assigned. |
| `walking in` (853) / `walking on` (2,267) | same verb, different preposition, both `action`; a model could trivially "solve" one via the other's supervision. Must be co-assigned. |
| `near` (16,395) / `next to` (19,410) | both `spatial`, both symmetric, near-synonymous in casual VG annotation. Co-assignment strongly recommended; both are also individually expensive to hold out (§1.6). |
| `attached to` (5,429) / `mounted on` (1,090) | both `contact`, semantically close but not identical (mounting implies a more rigid/structural attachment) — flagged as a **soft** risk (semantically close but plausibly legitimately distinguishable, per the design doc's §3 "semantically-close, not synonym" category) rather than a hard co-assignment requirement. Worth the nearest-neighbor `e(p)`-cosine check (design doc §3.4) before any final decision either way. |
| `on` (380,301) / `on back of` (1,033) | compositionally related (one is a specific case of the other) but not interchangeable; not a hard co-assignment case, but the shared substring makes a prompt-template or tokenization leak worth double-checking explicitly if `on` is ever touched (it will not be — §1.6). |

### 1.6 Predicates that are difficult to hold out, and why

- **`on`, `has`, `in`, `of`, `wearing`** (ranks 1-5): holding out any one
  removes a large, structurally important fraction of training supervision
  (`wearing` alone is 6.4% of all training relationships) and would leave
  `Seen_train`'s distribution unrealistically different from any real
  deployment distribution. Exclude from consideration entirely.
- **`wears`**: individually modest frequency (8,002 train), but tied to
  `wearing` by the co-assignment rule (§1.5) — holding it out alone would
  leak through its extremely common twin; holding out both is prohibitively
  expensive. Exclude.
- **`near` / `next to`**: both individually large enough to matter
  (16,395 / 19,410 train) and tied to each other by the co-assignment rule.
  Holding out both removes meaningful spatial supervision. Treat as
  difficult; avoid unless a design specifically wants to stress-test
  symmetric-predicate generalization (out of scope for the three candidates
  in §4).
- **`part of`**: the *sole* member of the `part-whole` group. Holding it
  out empties an entire semantic group from `Seen_train`, which the design
  doc's §3 partitioning rule explicitly calls a harder, different question
  ("within-group generalization becomes untestable"). Only appropriate as a
  deliberately-labeled stress-test item (§4.3), never in a conservative or
  balanced split.
- **`and`, `for`** (`other` group): these are semantically heterogeneous,
  closer to grammatical connectives than coherent visual predicates ("and"
  often marks a composite/grouped object annotation; "for" covers many
  unrelated purposive senses). Their CLIP text embedding is unlikely to
  occupy a single coherent direction the way "riding" or "eating" does.
  Difficult to hold out **cleanly** — a poor unseen-side result for either
  could reflect semantic incoherence of the predicate itself rather than a
  failure of the alignment mechanism. Usable only with this caveat stated
  explicitly (§4.2 uses `for` this way, flagged).
- **The six rarest predicates** (§1.3): not leakage-difficult, but
  *statistically* difficult — any single one of them, held out alone, gives
  a noisy read. Group them together (§4.3) rather than scattering them
  singly across splits.

---

## 2. Leakage risks — concrete, code-verified findings

This is the most important section of this audit. Three genuine,
gradient-relevant leakage vectors were found in the actual training code
(not hypothetical) — all three currently draw from the **full,
unpartitioned** predicate vocabulary, and all three are **active by
default or active in the actual C1 baseline training recipe**.

### 2.1 `pred_sim_matrix` — feeds two separate active mechanisms

`openvocab_rel/train.py:1638-1642` (and rebuilt per-epoch at `:1921-1926`
after any CLIP fine-tuning step, since CLIP is not frozen in the C1
baseline recipe):

```python
pred_sim_matrix = (
    build_predicate_similarity_matrix(pred_emb_s2o).to(device)
    if bool(getattr(cfg, "lattice_loss_enabled", True))
    else torch.empty((0, 0), device=device)
)
```

`build_predicate_similarity_matrix` (`openvocab_rel/lattice.py:9-27`)
computes a **full 51×51 pairwise cosine-similarity matrix** directly from
`pred_emb_s2o` — the CLIP text embeddings of **every predicate in
`global_pred_pool`**, i.e. the complete, unpartitioned vocabulary
(`scan_vg150_predicate_vocab` has no seen/unseen concept — §3). Two
consumers, both currently reachable in the standard training recipe:

1. **`predicate_label_relaxation`** (`train.py:192-210`, inside
   `_predicate_ce_loss`): when `predicate_label_relaxation_enabled=true`,
   the target distribution for a positive example is **softened toward
   other predicates with high `pred_sim_matrix` similarity to the true
   label**. `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md`'s own correction
   block already established that **the actual C1 baseline recipe
   (`scripts/train/run_c0_c1.sh` line 104) passes
   `--predicate_label_relaxation_enabled true`** — this is not a dormant
   flag, it is active in the exact training recipe any future
   predicate-disjoint run would naturally start from. **If a held-out
   predicate's `e(p)` happens to be CLIP-similar to a seen predicate, its
   embedding identity directly softens that seen predicate's training
   target — a real, gradient-affecting leak of the held-out predicate's
   specific identity, not just its class of the general prior.**
2. **`lattice_negative_weights`** (`openvocab_rel/lattice.py:30-58`, used
   via `--lattice_loss_enabled`, which **defaults to `True`**,
   `config.py:376`): reweights the negative-denominator contribution of a
   candidate predicate based on its `pred_sim_matrix` similarity to the
   positive label — again, a held-out predicate can influence this
   reweighting for a seen predicate purely through its CLIP embedding's
   position relative to the full similarity matrix.

### 2.2 `confusable_idx` → `counterfactual_hard_negative_loss` — the most
concrete leak found

`train.py:1637` / `:1920`:

```python
confusable_idx = build_confusable_index(pred_emb_s2o, topm=min(32, max(1, len(global_pred_pool) - 1)))
```

`build_confusable_index` (`openvocab_rel/clip_utils.py:111-127`) returns,
for **every** predicate in the full vocabulary, the indices of its top-32
nearest CLIP-embedding neighbors (excluding near-duplicates above a 0.95
similarity threshold, which would otherwise trivially return true
synonyms). This is then consumed directly inside the training step
(`train.py:2152-2163`):

```python
cand = confusable_idx.index_select(0, safe_pred_ids)          # safe_pred_ids = the CURRENT batch's SEEN positive labels
...
conf_ids = torch.where(mask_any, conf_ids, safe_pred_ids)
valid_conf = (pos_pred_ids >= 0) & (conf_ids >= 0) & (conf_ids < int(pred_emb_s2o.shape[0]))
if bool(valid_conf.any()):
    cf_pred_feat[valid_conf] = pred_emb_s2o[conf_ids[valid_conf]].detach()
```

`cf_pred_feat` (the **confusable predicate's own text embedding**, sourced
purely by nearest-neighbor lookup over the full vocabulary) is then passed
as `predicate_confusable_y` into `counterfactual_hard_negative_loss`
(`train.py:2166-2175`), gated by `predicate_counterfactual_enabled`
(default `True`) and weighted by `lambda_counterfactual`. **This value is
observed active in this program's own "full" training objective** — the
`R2_pilot_v3` endpoint-evaluation run launched earlier in this session
printed `"lambda_counterfactual_in_spoa": 0.05` and
`"counterfactual_spoa": true` in its resolved `[ActiveBranches]` config
dump (this is the eval-time config; the corresponding C1 *training* recipe
should be checked directly before relying on this exact value, but the
mechanism itself is unconditionally reachable code, not a dead path).

**Concretely**: if a held-out predicate happens to be among a seen
predicate's 32 nearest CLIP-embedding neighbors, its *exact* text embedding
is used as an explicit hard-negative target the relation encoder is
trained to move away from — a direct, non-hypothetical training-time use
of a specific held-out predicate's identity. This is the single most
important finding of this audit: **a predicate-disjoint training run that
does not modify or disable this pathway is not actually predicate-disjoint,
regardless of what the positive-label supervision does.**

### 2.3 `predicate_group_relaxation` — same shape of risk, currently
inactive by default

`train.py:212-229`, gated by `predicate_group_relaxation_enabled` (default
`False`, `train.py:683`) and consuming `pred_group_matrix`
(`_build_predicate_group_matrix`, `train.py:573`, built from
`configs/predicate_metadata_vg150.json`'s `group` field over the full
vocabulary — §0's orphan-entry finding is irrelevant here since
`_build_predicate_group_matrix` is keyed by `global_pred_pool`, the real
50/51, not the metadata file's 53 keys). **Currently off by default**, so
lower *immediate* risk than §2.1-2.2, but it is the same shape of leak
(soft-label mass redistributed toward other members of a held-out
predicate's semantic group) and must be explicitly re-verified off (not
just assumed off) for any predicate-disjoint run, since it is a one-flag
flip away from reintroducing exactly this problem.

### 2.4 `frequency_prior_train.json` — a precomputed artifact that already
contains every predicate's real frequency

`datasets_vg150_clean/frequency_prior_train.json` (280 MB) is a **static,
precomputed** file built once from the full training set, containing
`global_log_probs`, `pair_log_probs`, `subject_log_probs`,
`object_log_probs` keyed by `predicate_vocab` — i.e. it **already bakes in
the exact training frequency of every one of the 50 predicates**,
including whichever ones a future split designates "unseen." This is not a
code bug — it is simply how the file was built, before any notion of a
predicate-disjoint split existed. **Any component that composes this file's
prior into the open-vocabulary scoring channel (`freq_bias_enabled`,
`freq_bias_path`, `freq_bias_alpha`) for an unseen predicate would silently
hand the model information it should never have** (the design doc's §4
already reasons about this correctly at the conceptual level — "unseen
predicates have no frequency prior" — this section grounds that claim in
the actual artifact: the prior *would* have a real entry for them if
looked up, because the file predates the split). The existing WPRD
discipline of never composing the prior into the text/adaptive-scored
channel already sidesteps this for the *scored* channel, but any
diagnostic or logging code that reports "prior agreement" for an unseen
predicate must be checked to confirm it is not silently reading this file's
real entry for that predicate.

### 2.5 `prompts.py`'s `SYMMETRIC_RELATION_BLACKLIST` — inconsistent, but
appears unused (low risk, hygiene finding)

`openvocab_rel/prompts.py:9-19` hardcodes a linguistic blacklist:
`{"near", "next to", "beside", "overlap", "overlapping", "on either side
of", "looking at each other", "adjacent to", "around", "surrounding"}`,
consumed only by `is_symmetric_predicate()` — and a repo-wide grep found
**zero callers of `is_symmetric_predicate()` anywhere else in the
codebase** (the actually-used function is the differently-named
`is_symmetric()` in `predicate_metadata.py`, called from
`evals.py:2144`). Two problems, both low-risk but worth fixing for hygiene:
(a) this blacklist appears to be **dead code**; (b) it **disagrees** with
`predicate_metadata_vg150.json`'s `symmetric` field for the real vocabulary
— e.g. `"between"` is `symmetric: true` in the metadata file but absent
from this blacklist, while `"around"` (not even a real predicate — §0) is
present in the blacklist. Not an active leak since nothing reads it, but a
latent trap for anyone who wires it up later assuming it is authoritative.

### 2.6 `is_symmetric()` (the function actually used) — static linguistic
fact, not data-derived; lower risk

`evals.py:2144`, inside a role-swap diagnostic block, gates whether a
ground-truth relationship is treated as symmetric for role-swap-invariance
bookkeeping. This reads a **fixed, hand-authored linguistic property**
(from `configs/predicate_metadata_vg150.json`) rather than anything learned
from training data — holding a predicate out of label supervision does not
change whether it is linguistically symmetric. Lower risk than §2.1-2.3,
but still worth confirming this code path is not reachable from any loss
term that would matter for a held-out predicate (a quick trace, not done
exhaustively in this audit, since `role_swap_rank`'s own lambda was
observed at `0.0` in every config snapshot seen this session).

### 2.7 Caches and checkpoint metadata — mostly clean, one useful safeguard
found

- `runs/*/pair_logits.pt` dumps store **numeric logits and IDs**, not
  predicate strings — no direct text leakage vector.
- Checkpoint `experiment` snapshots (`train.py:80`,
  `"predicate_vocab_hash": _stable_hash_json(pred_vocab)`) hash the
  **entire vocabulary list** used at training time. This is a **free,
  already-existing safeguard**: a future predicate-disjoint run's
  `global_pred_pool` (Seen_train-only, shorter than 51) would produce a
  **different** `predicate_vocab_hash` than every existing C0/C1/Readout-v2
  checkpoint, and its recorded `num_predicates` would not equal 51. This
  should be turned into an explicit, checked assertion (§6) rather than
  left as an incidental side effect.
- `TextCache`/`_EvalTextCache` (`evals.py`, multiple sites, e.g. `:3029`)
  memoizes CLIP **forward passes** for predicate prompt variants
  (confirmed by this session's own pilot log: `"[TextCache] Pre-computing
  variants for 51 predicates... Cache size: 0.6 MB"`) — this is a pure
  compute cache (recomputes `e(p)` from the same frozen encoder every time
  it is invalidated), not a store of anything supervision-derived. No
  leakage risk identified, provided it is always rebuilt fresh per
  checkpoint/CLIP-state (which it is — keyed by live `clip_model`/
  `processor` instances passed into its constructor, not persisted to
  disk across runs in any of the sites checked).

---

## 3. Current split (train/val/test) construction — where predicate
supervision actually enters the pipeline

- **Image-level split membership** (which images are train/validation/
  test) is fixed by which `.jsonl` file an image's record lives in — this
  is a property of the dataset files themselves (`datasets_vg150_clean/
  {train,validation,test}.jsonl`), decided at dataset-build time, and is
  **orthogonal to the predicate-disjoint question**: a predicate-disjoint
  split does not move images between train/val/test, it filters which
  **relationship records within already-train-split images** are allowed
  to reach the loss.
- **Where a predicate label first becomes a supervision target**:
  `_build_relation_entries` (`openvocab_rel/datasets/vg150_loader.py:
  142-206`) iterates every `relationships` entry for an image
  (`rel.get("predicate", "relation")`, line ~161) and appends it to
  `positive_preds`/`positive_entries` **unconditionally** — there is no
  existing filter, flag, or hook anywhere in this function (or its caller,
  `VG150LocalDataset`/the JSONL-backed loader) that could exclude a
  specific predicate string. **A predicate-disjoint filter must be added
  here** (or in a thin wrapper around this function's output), dropping any
  relationship entry whose `predicate` is in `Unseen_test` **before** it
  reaches `positive_preds`, `pred_to_idx`-based label construction, or
  `pred_freq`/`pred_ce_weights` accumulation.
- **Multi-label pairs — an important implementation nuance**: the loop
  does **not** deduplicate by `(subject_id, object_id)` — if a single image
  annotates the same object pair with two different predicates (VG150 does
  allow this), each becomes an **independent** entry in `positive_entries`/
  `positive_preds`, sharing the same subject/object indices. A correct
  predicate-disjoint filter must operate at the **relationship-entry**
  level, not the **pair** level: a pair with one seen-predicate relation and
  one unseen-predicate relation must keep the seen entry for training and
  drop only the unseen one, not discard the whole pair. Getting this wrong
  either leaks the unseen predicate (keeping it) or silently reduces
  training data more than intended (dropping the whole pair).
- **The predicate vocabulary itself is never split-aware today**:
  `scan_vg150_predicate_vocab` (`vg150_loader.py:1129`) always returns the
  full 50-predicate list from the dataset's own vocabulary file, in a fixed
  index order, with **no parameter or mechanism to restrict it to a
  subset**. `global_pred_pool` (`train.py:1516-1527`) is built directly
  from this function's output, then has `"relation"` appended. Every
  downstream vocabulary-shaped tensor in this session's audit (`pred_emb_s2o`
  / `E`, `pred_sim_matrix`, `confusable_idx`, `pred_group_matrix`,
  `pred_freq`, `pred_ce_weights`) is built from this same
  `global_pred_pool`. **This is the single choke point**: if a
  predicate-disjoint implementation constructs `global_pred_pool` as
  `Seen_train ∪ {"relation"}` instead of the full 50 ∪ `{"relation"}`, every
  one of §2's leakage vectors is fixed simultaneously, because they all
  derive from this one list. This is good news for implementation
  simplicity — but also means a single missed reference to the *original*
  full-vocabulary scan (e.g. an eval-time helper that independently calls
  `scan_vg150_predicate_vocab()` again without going through the
  already-filtered `global_pred_pool`) would silently reintroduce every
  leak at once. `evals.py:1752` (`load_predicate_metadata(...,
  pred_vocab)`) and any other independent call site must be audited to
  confirm they consume the **same, filtered** `pred_vocab`/`global_pred_pool`
  object, not re-derive their own from `scan_vg150_predicate_vocab()`
  directly — **not fully verified in this audit; flagged as a required
  check before implementation (§6)**.

---

## 4. Candidate predicate-disjoint splits

All three candidates below satisfy: no near-synonym pair split across the
seen/unseen boundary (§1.5, with one flagged soft exception), no predicate
from §1.6's "difficult" list used except where explicitly named as a
deliberate stress case, and real counts throughout (no estimates). These
are **proposals for review, not a finalized or implemented split** — per
the task's instruction, nothing here has been registered or built.

### 4.1 Conservative split

| | |
|---|---|
| **Unseen predicates (4)** | `carrying` (contact), `belonging to` (possession), `playing` (action), `painted on` (attribute) |
| **Seen predicates** | the remaining 46 |
| **Train removed** | 2,419 + 1,599 + 1,758 + 1,455 = **7,231** (≈0.69% of 1,046,427) |
| **Val unseen population** | 301 + 207 + 209 + 248 = **965** (≈0.73% of 132,556) |
| **Test unseen population** | 240 + 182 + 252 + 165 = **839** (≈0.63% of 132,334) |
| **Groups touched** | contact, possession, action, attribute (4 of 8 real groups); spatial, pose, part-whole, other fully seen |
| **Semantic difficulty** | Low-moderate. All four are unambiguous, coherent single-sense predicates (unlike `and`/`for`). No synonym-pair entanglement. |
| **Risks** | Smallest signal — 965/839 unseen cells is enough for a first directional read but leaves little room to sub-analyze by group or by spatial-vs-non-spatial (design doc §4's own granularity requirement will be thin here). Best choice if the primary goal is *proving the pipeline and falsification battery work end-to-end* before committing to a larger held-out set. |

### 4.2 Balanced split

| | |
|---|---|
| **Unseen predicates (11)** | `over`, `at` (spatial); `carrying`, `covering` (contact); `belonging to` (possession); `playing`, `eating` (action); `laying on`, `lying on` (pose — co-assigned pair); `painted on` (attribute); `for` (other, flagged §1.6 semantic caveat) |
| **Seen predicates** | the remaining 39 |
| **Train removed** | 5,151+4,889+2,419+1,878+1,599+1,758+2,187+1,948+972+1,455+5,174 = **29,430** (≈2.81% of 1,046,427) |
| **Val unseen population** | 680+612+301+241+207+209+295+243+136+248+708 = **3,880** (≈2.93% of 132,556) |
| **Test unseen population** | 657+696+240+243+182+252+240+256+131+165+659 = **3,721** (≈2.81% of 132,334) |
| **Groups touched** | spatial, contact, possession, action, pose, attribute, other (7 of 8 real groups — only `part-whole`, a single-member group, untouched by design, §1.6) |
| **Semantic difficulty** | Mixed by design: most items are coherent single-sense predicates; `for` is deliberately included as a harder, semantically vaguer case and must be reported separately, not folded into the aggregate unseen-mR without comment (design doc §3's "semantically-close/legitimate difficulty" reporting requirement applies directly here — though `for`'s difficulty is *semantic incoherence*, a different flavor than "near a seen neighbor"). |
| **Risks** | The `laying on`/`lying on` pair, correctly co-assigned, still means the `pose` group loses 2 of its 4 members from supervision — `sitting on`/`standing on` remain the only seen pose exemplars, which is a real but bounded risk to within-group pose generalization specifically. Every other group retains ≥7 of its members seen. **This is the recommended split — see §5.** |

### 4.3 Long-tail stress split

| | |
|---|---|
| **Unseen predicates (7)** | `part of` (part-whole — **sole member, entire group removed from supervision, deliberate**), `flying in` (rarest predicate overall), `on back of`, `using`, `across`, `mounted on`, `covered in` |
| **Seen predicates** | the remaining 43 |
| **Train removed** | 1,067+817+1,033+1,001+1,085+1,090+1,225 = **7,318** (≈0.70%) |
| **Val unseen population** | 106+156+151+127+151+120+153 = **964** (≈0.73%) |
| **Test unseen population** | 112+118+132+106+117+137+154 = **876** (≈0.66%) |
| **Groups touched** | part-whole (fully removed, by design), spatial (2), action (1), contact (2) |
| **Semantic difficulty** | Deliberately elevated: `part of`'s group-emptying violates the design doc's own "no group entirely absent from Seen" rule **on purpose**, to probe the harder, explicitly-different question of whether an entire unseen semantic category can be recovered from CLIP text alone with zero same-group visual supervision. `on back of`/`mounted on`/`covered in`/`across`/`using` are all in §1.3's rare-predicate list — every unseen cell here is small (all val/test counts ≤156), so results will be noisy by construction. |
| **Risks** | This split answers a **different, harder, and explicitly out-of-scope-for-the-primary-question** question (per the design doc §3's own stated distinction). It should never be used as the primary Paper C endpoint, only as a secondary, clearly-labeled stress probe after the primary split (§5) has already produced a healthy result. Small cell counts also make its falsification-test thresholds (design doc §7, the `[0.47, 0.53]` shuffled/random-text band) harder to trust statistically — wider confidence intervals should be expected and reported, not treated as equivalent evidence to the primary split's. |

---

## 5. Recommendation: the Balanced split (§4.2) as Paper C's primary
predicate-disjoint protocol

**Recommended: §4.2, the Balanced split.**

Reasoning:

1. **Coverage without over-concentration**: it touches 7 of the 8 real
   semantic groups (only the single-member `part-whole` group is
   untouched, by necessity — including `part of` here would require
   accepting the group-emptying problem §4.3 exists specifically to probe
   separately), directly satisfying the design doc's own stratification
   requirement (§3: "no group entirely absent from Seen... no group
   entirely absent from Unseen") for every group large enough to make that
   requirement meaningful.
2. **Statistically usable cell counts**: every unseen predicate has
   val/test counts ≥131, avoiding the Conservative split's thinness
   (§4.1: only 4 predicates, harder to sub-analyze by group) and the Stress
   split's noise floor (§4.3: several cells under 160, and one group
   deliberately emptied).
3. **Bounded training cost**: removing ≈2.8% of training relationships is
   a real but non-disruptive cost — `Seen_train`'s distribution remains
   close to the full dataset's, unlike any split that touched a top-10
   predicate.
4. **One clean, correctly-handled synonym pair, not zero and not many**:
   including `laying on`/`lying on` (correctly co-assigned) gives the split
   at least one real test of the leakage-control machinery itself (design
   doc §3.4's synonym/near-duplicate audit) without over-relying on it —
   the other 9 unseen predicates are synonym-clean, isolating most of the
   result from that specific risk.
5. **One deliberately-hard, separately-flagged item (`for`)**: gives an
   honest read on a semantically incoherent predicate without letting it
   silently drag down (or, worse, inflate, if `for`'s vague embedding
   happens to sit centrally in CLIP's text space) the aggregate unseen-mR —
   provided it is reported per-predicate as the design doc's evaluation
   section already requires.

The Conservative split (§4.1) remains the right choice for a **first,
cheap pipeline-validation pass** (fewer moving parts, faster to sanity
check the leakage tests in §6 actually work) before committing to the
Balanced split's larger held-out set — this is a staging recommendation,
not a rejection of §4.1; it can precede §4.2 in the same experiment ladder
the design doc already specifies (Stage A → Stage B), without needing a
fourth candidate.

---

## 6. Exact leakage tests that must pass before any training

These are written as concrete, checkable assertions against the exact
mechanisms found in §2-§3, not restatements of the design doc's more
general principles.

1. **Single-choke-point test**: assert that every one of `pred_emb_s2o`
   (`E`), `pred_sim_matrix`, `confusable_idx`, `pred_group_matrix`,
   `pred_freq`, and `pred_ce_weights` is constructed from the **same**
   `global_pred_pool` object, and that `global_pred_pool` for a
   predicate-disjoint run equals `Seen_train ∪ {"relation"}` — length
   `len(Seen_train) + 1`, not 51. Grep-verify (not just assume) that no
   code path independently re-calls `scan_vg150_predicate_vocab()` without
   routing through this filtered list (§3's flagged, not-yet-verified
   concern — `evals.py:1752` and any other independent call site named
   explicitly and checked).
2. **`predicate_vocab_hash` divergence test**: assert the resulting
   checkpoint's `experiment.predicate_vocab_hash` differs from every
   existing C0/C1/Readout-v2 checkpoint's recorded hash, and that
   `experiment.num_predicates` (or equivalent) equals
   `len(Seen_train) + 1`. This reuses an already-existing, free provenance
   field (§2.7) as an automatic regression check.
3. **Relationship-level (not pair-level) filter test** (§3): construct a
   synthetic image with two relationship entries sharing one
   `(subject_id, object_id)` pair, one seen-predicate and one
   unseen-predicate; assert the built training payload contains the seen
   entry and excludes the unseen one, without dropping the pair entirely.
4. **Zero-unseen-label-in-batch test**: over a full pass of `Seen_train`'s
   training loader, assert no batch's `pos_pred_ids` (or equivalent target
   tensor) ever contains an index resolving to an `Unseen_test` predicate
   string, under the **filtered** `global_pred_pool`'s own index mapping
   (not the original 50-index mapping — an off-by-vocabulary-choice bug
   here would silently pass a naive check against the wrong index space).
5. **`pred_sim_matrix`/`confusable_idx` shape test** (§2.1-2.2): assert
   both are built with shape `(len(Seen_train)+1, ...)`, not `(51, ...)`,
   confirming they cannot reference an unseen predicate's embedding at all
   — this is a **stronger, structural** fix than merely disabling
   `predicate_label_relaxation_enabled`/`lattice_loss_enabled`/
   `predicate_counterfactual_enabled` for the run (disabling the flags is
   also acceptable and simpler, but if any of them are left on for a
   from-baseline-recipe run, this shape assertion is the load-bearing
   guarantee that they cannot leak even if active).
6. **`predicate_group_relaxation` explicit-off test** (§2.3): assert
   `predicate_group_relaxation_enabled=False` is the **resolved runtime
   config**, not just the class default, for any run where §5's shape
   fix is not applied to `pred_group_matrix` specifically.
7. **Frequency-prior exclusion test** (§2.4): assert that no code path
   composes `frequency_prior_train.json`'s entry for an `Unseen_test`
   predicate into the scored/adaptive logits channel or into any reported
   "prior agreement" diagnostic for that predicate — checked by changing
   that predicate's entry in a scratch copy of the prior file and
   confirming zero change in any unseen-side output (a mutation test,
   stronger than a code-reading check alone).
8. **Metadata orphan/coverage test** (§0): assert any tool consuming
   `configs/predicate_metadata_vg150.json` for group/symmetric lookups
   first intersects its keys with the real 50-predicate vocabulary,
   rejecting or ignoring `"around"`, `"growing on"`, `"says"` if they ever
   appear in a derived list.
9. **Synonym/near-duplicate co-assignment test** (§1.5): for the chosen
   split, mechanically re-verify (not from this document's table alone,
   but recomputed from the live vocabulary at implementation time in case
   the vocabulary file ever changes) that no flagged pair
   (`wearing`/`wears`, `laying on`/`lying on`, `walking in`/`walking on`,
   `near`/`next to`) is split across the seen/unseen boundary, plus the
   `e(p)`-nearest-neighbor audit from the design doc §3.4 for every
   `Unseen_test` predicate against every `Seen_train` predicate.
10. **`SYMMETRIC_RELATION_BLACKLIST` dead-code confirmation** (§2.5): a
    one-time grep-based regression test asserting `is_symmetric_predicate`
    still has zero callers — if this ever changes (someone wires it up),
    its disagreement with `predicate_metadata.py`'s `symmetric` field
    (§2.5) must be resolved first, not inherited silently.

---

## Precise recommendation for the next implementation step

**Do not implement the Balanced split (§4.2) directly.** The correct next
step, in order:

1. Implement the **single-choke-point fix** (§6 test 1): parameterize
   `global_pred_pool` construction so it can be restricted to an arbitrary
   `Seen_train` list, and thread that restricted list through every
   consumer named in §2 (`pred_emb_s2o`, `pred_sim_matrix`,
   `confusable_idx`, `pred_group_matrix`, `pred_freq`,
   `pred_ce_weights`) — this is a data-pipeline/config change, not a model
   change, and is the same "minimal function-class change" discipline the
   Readout v2 preregistration already used successfully.
2. Write §6's ten tests as an actual `tests/test_predicate_disjoint_*.py`
   suite, run against the **Conservative split (§4.1)** first (fewest
   moving parts, fastest to debug if a test fails) — this is a CPU-only,
   pre-GPU gate, matching this program's own "static/unit validation before
   any pilot" discipline.
3. Only after all ten tests pass on the Conservative split: register the
   Balanced split (§4.2) as Paper C's actual predicate-disjoint
   preregistration document (a new, dated file, not this audit), re-run
   the same ten tests against it, and only then proceed to
   `docs/PAPER_C_OPEN_VOCAB_SUCCESSOR_DESIGN.md`'s Stage A/Stage B
   experiment ladder.

This audit does not create that preregistration document, and per the
task's instruction, no split has been implemented or registered — §4's
candidates are proposals for review.
