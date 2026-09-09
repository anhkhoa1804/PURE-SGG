# Paper C — cross-arm provenance audit (R0 / C0 / C1 / R2)

Read-only, CPU-only, no GPU code run, no evaluation launched, no
checkpoint/config/model code modified, nothing deleted. Written while the
R2 endpoint evaluation was still running — nothing here depends on it
finishing, and nothing here touches it. One new artifact,
`tools/provenance_audit.py`, is CPU-only and read-only (file I/O + SHA256
hashing only, no `torch.load`), run once with `--no-hash` during this
audit to keep I/O minimal while the GPU job was active; the checkpoint
hashes below were instead taken from `sha256sum` invocations already run,
or freshly run in this session, sequentially, not in parallel with
anything else.

**Evidence labels**: `[VERIFIED]` = read directly from a structured file
this session (`result.json`, `provenance.txt`, or a `.pt` file's raw
bytes); `[LOCKED]` = a `[VERIFIED]` fact that a prior session's document
explicitly declared frozen/registered; `[DERIVED]` = computed from
`[VERIFIED]` facts by a documented, re-run calculation; `[STALE]` = an
artifact that was never wrong but is superseded and must not be read as
current; `[UNDOCUMENTED]` = a number found in a document with no traceable
recomputation path in this session.

---

## 1. The provenance table

Machine-produced by `tools/provenance_audit.py --no-hash`, then hand-annotated
with checkpoint SHA256 (computed separately, see §1.1) and the exact
evaluator invocation reconstructed from `tools/c0_c1_evaluate.py` /
`tools/readout_v2_evaluate.py` source (verified this session, not guessed).

| field | C0 | C1 (== R0) | C1_seed5678 | R2 (`R2_pilot_v3`) |
|---|---|---|---|---|
| **label** | C0 | C1 (also serves as **R0** for the Readout v2 comparison — see §1.2) | C1_seed5678 | R2_pilot_v3 |
| **checkpoint path** | `checkpoints/C0_seed1234.pt` | `checkpoints/C1_seed1234.pt` | `checkpoints/C1_seed5678.pt` | `checkpoints/readout_v2_pilot_seed1234_v3.pt` |
| **checkpoint SHA256** | `2264ced4b5180c151798e395ed52696777b158c7d204342fad1e698139557d2c` | `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` | `91b2f4d5c4db0d412f007bab2cc3a90feed3b71f7a869222d72864bcb4b1203b` | not re-hashed this session; checkpoint integrity (CLIP present, `predicate_prototypes` shape/finiteness, single-trainable-tensor optimizer state) already verified in `docs/PAPER_C_READOUT_V2_INVALID_PILOT.md`'s companion recovery session |
| **training git commit** | `db6aff4ea874f49b8c4871fd49f1124b4c49d9e2` | `2c35bba26baced2b63e5afc41b4693e349e9889f` | `647cec1c3a5444a71fb1a5dbd7aaf94816a8bbee` | `d4fe6876ae5d4c230dcec6ae04ed7e4ec5d65f9a` (current `HEAD`) |
| **seed** | 1234 | 1234 | 5678 | 1234 |
| **training config path (provenance)** | `runs/C0_seed1234/provenance.txt` | `runs/C1_seed1234/provenance.txt` | `runs/C1_seed5678/provenance.txt` | `runs/readout_v2_pilot_seed1234_v3/provenance.txt` |
| **resume_from (base init)** | `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` | same | same | `checkpoints/C1_seed1234.pt` (== the C1/R0 artifact — see §1.2) |
| **base init SHA256 (recorded at launch)** | `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442` | same | same | `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` |
| **eval config/path** | `runs/eval_C0/result.json` | `runs/eval_C1/result.json` | `runs/eval_C1_seed5678/result.json` | `runs/eval_readout_v2_R2_pilot_v3/result.json` — **does not exist yet** |
| **geom_input_pixel_space** | `False` | `True` | `True` | `True` (hard-coded in `readout_v2_evaluate.py`, never varied) |
| **geom_fourier_scale** | `1.0` | `0.01` | `0.01` | `0.01` |
| **text/readout mode** | text head, `ensemble_alpha=0.0` (frozen CLIP cosine) | same | same | `adaptive_logits` channel (`predicate_prototypes`, trained), `readout_v2_enabled=true` |
| **population: images** | 10,401 | 10,401 | 10,401 | unknown — pending |
| **population: pair rows** | 132,556 | 132,556 | 132,556 | unknown — pending |
| **population: GT rows** | 132,556 | 132,556 | 132,556 | unknown — pending |
| **population: WPRD cells** | 20,016 | 20,016 | 20,016 | unknown — pending |
| **WPRD macro** | `0.5667271196244782` | `0.5749881522134409` | `0.572551887041989` | **not computed — endpoint still running** |
| **WPRD weighted** | `0.5339868001264658` | `0.5370226991837204` | `0.5349712218326051` | pending |
| **WPRD CI** | n/a alone (paired CI is vs. another arm, §2) | vs C0: `[+0.00383, +0.01270]` | vs C0: `[+0.00138, +0.01004]` | pending |
| **R@50 / mR@50** | `0.6731 / 0.2377` | `0.6741 / 0.2386` | `0.6737 / 0.2386` | pending |
| **evaluator command (reconstructed)** | `python3 tools/c0_c1_evaluate.py --arm C0 --ckpt checkpoints/C0_seed1234.pt` | `python3 tools/c0_c1_evaluate.py --arm C1 --ckpt checkpoints/C1_seed1234.pt` | `python3 tools/c0_c1_evaluate.py --arm C1 --ckpt checkpoints/C1_seed5678.pt --out_dir runs/eval_C1_seed5678` | `python3 tools/readout_v2_evaluate.py --label R2_pilot_v3 --ckpt checkpoints/readout_v2_pilot_seed1234_v3.pt --readout_v2_enabled true` (this exact command was launched by this session and is still running) |
| **artifact timestamp (training finished)** | `2026-09-07T07:35:41Z` | `2026-09-07T14:48:26Z` | `2026-09-08T04:26:52Z` | training finished `2026-09-09T~01:46Z` (this session); evaluation launched `2026-09-09T01:46Z`, still running as of this audit |
| **provenance confidence** | **VERIFIED** | **VERIFIED** | **VERIFIED** | **PENDING** (training-side artifacts complete and internally consistent; evaluation-side artifacts do not exist yet — not `CONFLICTED`, just incomplete) |

### 1.1 Checkpoint hash provenance

Computed this session, sequentially, via `sha256sum` directly (not through
`tools/provenance_audit.py`, to avoid redundant multi-GB re-hashing while
the GPU job runs):

```
sha256sum checkpoints/demo_best/pure_best_adapt_light_mR50.pt   # matches recorded 8845c3af...
sha256sum checkpoints/C1_seed1234.pt                            # matches recorded 79ca1565...
sha256sum checkpoints/C0_seed1234.pt                            # 2264ced4...  (new this session)
sha256sum checkpoints/C1_seed5678.pt                            # 91b2f4d5...  (new this session)
```

All four match their respective `provenance.txt`'s own `ckpt_sha256`
field where one is recorded — `[VERIFIED]`, no tampering, no silent
substitution.

### 1.2 R0 and C1(seed1234) are the same artifact — stated explicitly, not
just implied

`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §9 already says this in
prose ("**`R0`**: ... the already-existing `runs/eval_C1/result.json` ...
Not re-run — reused as-is"). This audit confirms it at the file level:
**`R0` is not a separate evaluation run at all** — every field in the "R0"
row anyone would want (checkpoint, contract, population, WPRD) is
literally `runs/eval_C1/result.json`, `checkpoints/C1_seed1234.pt`. There
is no `runs/eval_R0/` directory and there should not be one. Any future
table, script, or document that treats "R0" as if it had its own
independent checkpoint/eval-log is introducing a distinction that does not
exist in this repository and must be corrected on sight.

---

## 2. The C0/C1 "+0.02 vs +0.008" discrepancy — fully traced, not chosen

**The registered, locked, thrice-independently-reproduced macro delta is
`[LOCKED]`:**

| | C1 seed1234 vs C0 | C1 seed5678 vs C0 |
|---|---|---|
| `delta_WPRD` (macro, all 20,016 cells) | `+0.008261032588962776` | `+0.0058247674175108` |
| paired 95% CI | `[+0.0038311178, +0.0126973772]` | `[+0.0013811488, +0.0100445352]` |
| registered threshold | `delta >= +0.01` | same |
| met? | **NO** | **NO** |
| source | `runs/paper_c_c0_c1_comparison.json` (`tools/c0_c1_compare.py` output, gates D1-D4 all pass) | `runs/paper_c_c0_c1_seed2_comparison.json` (same tool, same gates, all pass) |

Both files' own `gates` array shows `D1`/`D2`/`D3`/`D4` all `True` for
these specific comparisons — **note this is the C0-vs-C1 case, where D4's
"contracts must differ" logic is the CORRECT check** (C0's contract really
is `{pixel_space: False, fourier_scale: 1.0}`, C1's really is
`{pixel_space: True, fourier_scale: 0.01}`) — this is not the buggy-reuse
scenario (§4 below), which only arises for R0-vs-R2.

### 2.1 Where the "+0.02" figures actually come from — two independent,
non-conflicting sources, neither of which is the macro delta above

**Source A — the spatial-subgroup-restricted delta, `[DERIVED, documented]`.**
`docs/PAPER_C_C1_MECHANISM_AUDIT.md` (analysis-only addendum to the C1
result, no new training, working tree confirmed clean at commit `a5ce243`
before and after) explicitly re-derives `cell_values` from the raw dumps
and confirms they match `result.json` **element-for-element** for both C0
and C1 seed1234, then partitions the SAME 20,016 cells by whether **both**
scored predicates are in the `spatial` semantic group:

```
both spatial:      6,762 cells   mean delta +0.022055   95% CI [+0.0141, +0.0294]   (90.19% of net gain)
mixed:              8,273 cells   mean delta +0.005322
neither spatial:    4,981 cells   mean delta -0.005583   (net negative)
```

`docs/PAPER_C_C1_SEED2_RESULT.md` §C (line 251) reports the analogous
seed5678 number: **`+0.018012`** on the same both-spatial stratum. The
document's own words are exact and load-bearing: *"the registered macro
endpoint reports the net"* — the spatial effect (`≈+0.02`) is real,
large, and CI-excludes-zero even restricted to large cells
(`+0.019987` on `na·nb>=25` cells, seed1234), but it is **diluted by
non-spatial cells trending flat-to-negative** when averaged over the full,
registered 20,016-cell population. **This is a population-scope
difference, not an error**: `+0.02` and `+0.008` are both correct, for
different, explicitly-named populations (6,762 spatial-only cells vs.
20,016 all cells).

**Source B — the readout-gap delta, a different comparison axis entirely,
`[LOCKED, but a DIFFERENT quantity]`.** `docs/PAPER_C_PURE_READOUT_FORENSIC.md`
and `docs/PAPER_C_READOUT_PILOT0_DECOMPOSITION.md` report, on the **same**
C1 checkpoint:

```
macro gap = 0.5994290264 (classifier/cls head) - 0.5749881522 (text head) = +0.0244408742
```

and the analogous number on the C0 checkpoint, `+0.020694`
(`docs/PAPER_C_READOUT_PILOT0_DECOMPOSITION.md` line 92). **This is not a
C0-vs-C1 comparison at all** — both numbers hold the checkpoint fixed and
compare two different *readout heads* (the deployed text/cosine head vs.
the discarded classifier head) on that one checkpoint. It only resembles
the geometry-repair delta numerically (`≈+0.02`) and lexically (both
documents discuss "C0" and "C1" prominently), which is exactly the shape
of confusion that produces an "informal report of ≈+0.02" if someone
recalls "there was a +0.02-ish number near C0/C1 in the readout work"
without recalling which of two entirely different comparisons it was.

### 2.2 Classification of every source located

| source document | number | what it actually measures | classification |
|---|---|---|---|
| `runs/paper_c_c0_c1_comparison.json` | `+0.008261` | C1(seed1234) macro WPRD − C0 macro WPRD, all 20,016 cells, paired bootstrap | **LOCKED** — registered, primary Paper C result |
| `runs/paper_c_c0_c1_seed2_comparison.json` | `+0.005825` | C1(seed5678) macro WPRD − C0 macro WPRD, all 20,016 cells | **LOCKED** — registered replication |
| `docs/PAPER_C_C1_MECHANISM_AUDIT.md` | `+0.022055` | C1(seed1234) − C0, **restricted to the 6,762 both-spatial cells** | **DERIVED**, documented, non-conflicting — a legitimate subgroup statistic, not the registered endpoint |
| `docs/PAPER_C_C1_SEED2_RESULT.md` §C | `+0.018012` | C1(seed5678) − C0, same both-spatial restriction | **DERIVED**, documented, non-conflicting |
| `docs/PAPER_C_PURE_READOUT_FORENSIC.md`, `docs/PAPER_C_READOUT_PILOT0_DECOMPOSITION.md` | `+0.0244` (C1), `+0.0207` (C0) | classifier-head WPRD − text-head WPRD, **same checkpoint**, not a C0-vs-C1 comparison | **LOCKED** (as a readout-gap fact) but **a different comparison axis** — must never be quoted as a geometry-repair number |
| `docs/PAPER_C_PURE_READOUT_V2_DESIGN.md` line 217 | "up to ~`+0.023` on WPRD by analogy with row 5" | a **hypothetical, explicitly-labeled-as-speculative** projection for a *candidate* Readout v2 architecture (candidate C, not the one actually built), reasoning "by analogy" from Source B's readout gap | **UNDOCUMENTED as a measured number** — it is a forecast in a design document, not a result, and was superseded by the actually-registered Readout v2 preregistration's own §16 ("no arbitrary WPRD bar") |

**No source located in this audit reports a macro, all-20,016-cell,
C0-vs-C1 WPRD delta anywhere near `+0.02`.** Every "+0.02"-shaped number
traces to either a smaller, explicitly-named subpopulation (Source A) or a
different comparison entirely (Source B). **This is not a discrepancy
requiring reconciliation — it is several different, individually correct
numbers that answer different questions, none of which contradicts the
registered `+0.008261` / `+0.005825` macro deltas.**

### 2.3 What is actually proven (§2 summary)

- The registered macro C0-vs-C1 WPRD deltas (`+0.008261`, `+0.005825`) are
  `[LOCKED]`, independently re-derived from raw dumps at least once each
  (`PAPER_C_C1_MECHANISM_AUDIT.md`, `PAPER_C_C1_SEED2_RESULT.md`), and
  **stand as the only defensible answer to "what is the C0→C1 geometry
  repair's WPRD effect."**
- The spatial-subgroup effect (`≈+0.02`) is real and separately `[LOCKED]`
  within its own stated population, and is the correct citation for
  "how large is the geometry repair's effect *where geometry is actually
  relevant*" — a different, also-legitimate question.
- The readout-gap numbers (`+0.0244`/`+0.0207`) must never be cited as a
  geometry effect; they measure something else and happen to be near-
  identical to Source A's numbers, which is exactly what makes them a
  hazard for future citation errors, not what makes them equivalent.

---

## 3. Repository history sanity check

`git log` confirms the three training provenance commits (`db6aff4`,
`2c35bba`, `647cec1`) and the four analysis/result documents cited above
are all real, present commits on the current branch history — no
force-push, no rewritten history, no orphaned commit references. `git
status --short` was clean before this audit began (only this audit's own
three new files are untracked after it).

---

## 4. `c0_c1_compare.py`'s gate D4 — confirmed scoped correctly for C0-vs-C1,
confirmed broken for R0-vs-R2 (carried over from the prior session's
analysis-plan audit, re-verified here against the literal files)

```python
# tools/c0_c1_compare.py, gate D4:
"pass": bool(A["contract"]["geom_input_pixel_space"] is False
             and A["contract"]["geom_fourier_scale"] == 1.0
             and B["contract"] != A["contract"])
```

- **C0 vs C1** (`runs/paper_c_c0_c1_comparison.json`, §2 above): `A`=C0
  (`pixel_space=False, fourier=1.0` — literally matches the hard-coded
  check), `B`=C1 (`pixel_space=True, fourier=0.01` — literally differs).
  **D4 correctly requires and finds a contract DIFFERENCE.** `[VERIFIED]`
  gate reads `True` in the stored file.
- **R0 vs R2** (would-be comparison, not yet run): `A`=R0=C1
  (`pixel_space=True, fourier=0.01`), `B`=R2
  (`pixel_space=True, fourier=0.01`, hard-coded identically in
  `readout_v2_evaluate.py` line 230, by design — geometry is a controlled
  variable for Readout v2, per `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md`
  §10). `A["contract"]["geom_input_pixel_space"] is False` evaluates to
  **`False`** (it is `True`) → the whole D4 expression is **`False`**
  regardless of anything else → **gate D4 would fail** for a comparison
  that is in fact exactly correctly configured. **`[VERIFIED]` by direct
  evaluation of the literal dict values both scripts write** — not
  inferred, not assumed.

**Confirmed, per the task's explicit framing**: *C0 vs C1 requires
contract DIFFERENCE (and gets it); R0 vs R2 requires contract MATCH
(because both intentionally use the identical repaired geometry contract)
and `c0_c1_compare.py`'s D4 has no code path that accepts a match.*

**Resolution, per instruction to prefer a separate assertion over
modifying existing code**: `tools/readout_v2_offline_analysis.py::
verify_contract_match(result_a, result_b, expect_identical: bool)`
(built in the prior session's analysis-plan task, not this one) is exactly
this separate, comparison-specific assertion — `expect_identical=True` for
R0-vs-R2, `expect_identical=False` reproduces `c0_c1_compare.py`'s
original D4 semantics for any future C0-vs-C1-shaped comparison. `
tools/c0_c1_compare.py` itself is untouched (per this task's own
constraint against modifying comparison code unless necessary — it is not
necessary; the new function is a complete substitute for the one call
site that matters).

---

## 5. Row/cell identity manifest — audited, does not exist as a reusable
artifact

**No `result.json` in this repository persists per-cell identity.**
Verified by reading `tools/c0_c1_evaluate.py`, `tools/readout_v2_evaluate.py`,
and `tools/within_pair_discrimination.py::wprd()` directly: `wprd()`
computes `_vals`/`_wts`/`_per_pair` internally, but every evaluator script
writes only `"cell_values": r["_vals"]` — a flat list of floats, no
`(group, class_a, class_b)` key alongside each value. **Counts alone
(`population.cells == 20016`) are what `c0_c1_compare.py`'s gates D1/D2
check — they do not prove `cell_values[i]` means the same cell in both
arms**, only that both arrays have the same length.

**However**, this gap has already been closed **manually, once, for the
C0-vs-C1 pairing specifically** — not as a persisted manifest file, but as
documented prose with a specific, quotable claim: `docs/
PAPER_C_C1_MECHANISM_AUDIT.md` states *"Both arms' `cell_values` were
re-derived from the raw dumps and again matched the stored `result.json`
element-for-element; cell identity and per-cell row membership are
identical across arms"* — and `docs/PAPER_C_C1_SEED2_RESULT.md` makes the
equivalent claim for the seed5678 pairing (§7 of that document, referenced
in this session's earlier reading as "same semantic-identity verification
as `docs/PAPER_C_C1_SEED2_RESULT.md` §7"). **These are one-off, ad hoc
verifications, done by re-running the WPRD loop by hand against the raw
dumps** (not a saved manifest, not a reusable tool) — `[VERIFIED]` as
having been done, for these two specific pairings only.

**For R0 vs R2, no such check — manual or automated — has been performed,
because R2's dump does not exist yet.** The prior session's
analysis-plan task built exactly the reusable tool this gap calls for:
`tools/readout_v2_offline_analysis.py::verify_paired_dumps` recomputes
both arms' `Groups`/cells directly from their raw dumps and diffs the
**cell identity keys**, not just counts — tested on synthetic data
(`tests/test_readout_v2_offline_analysis.py`, includes a dedicated
population-mismatch-detection test). This is the correct, general-purpose
replacement for the ad hoc manual check the C0-vs-C1 pairings relied on,
and **must be run against R0 vs R2 once R2's dump exists, before any
paired bootstrap is trusted.**

---

## 6. Scientific consequences

1. Nothing in this audit changes the registered Paper C geometry-repair
   result: `+0.008261` (seed1234) and `+0.005825` (seed5678), both
   weak-positive, both below the `+0.01` material threshold, both
   `[LOCKED]`. The "informal +0.02" reports are explained, not overridden
   — they were never wrong, only about a different, narrower population or
   a different comparison axis, and should not be cited interchangeably
   with the macro number going forward.
2. Any future document reporting a Paper C "geometry effect size" must
   state **which population** (macro-20,016, or both-spatial-6,762)
   explicitly in the same sentence as the number — the two numbers differ
   by nearly 3× and both are correct.
3. `c0_c1_compare.py` must not be pointed at R0/R2 unmodified. Use
   `verify_contract_match(..., expect_identical=True)` +
   `verify_paired_dumps` + `paired_cell_bootstrap` (all three already
   built and tested) instead.
4. R0-vs-R2 paired inference is **not yet admissible** — not because of
   any flaw in the method, but because R2 has no `result.json`/dump yet.
   The moment it does, `verify_paired_dumps` must run and pass
   (`safe_to_pair=True`) **before** `paired_cell_bootstrap` is trusted —
   this is now a mechanically checkable gate, not a matter of hoping the
   two independently-produced `cell_values` arrays happen to align.

---

## A/B/C/D — final report

**A. What is proven:**
- The registered C0-vs-C1 macro WPRD deltas (`+0.008261032588962776`
  seed1234, `+0.0058247674175108` seed5678) are `[LOCKED]`, reproduced
  independently from raw dumps at least once each, computed under gates
  that all pass, and are the correct answer to "what is the geometry
  repair's macro effect."
- Every checkpoint's SHA256 in the table above is `[VERIFIED]` this
  session and matches its own recorded provenance where one exists — no
  checkpoint substitution, no silent mutation.
- `c0_c1_compare.py`'s gate D4 is correctly scoped for C0-vs-C1 (contract
  difference required and present) and would incorrectly fail R0-vs-R2
  (contract match required and correctly present, but D4 has no code path
  for that) — `[VERIFIED]` by direct evaluation of the literal contract
  dicts both scripts write.
- No persisted, reusable cell-identity manifest exists in this
  repository's `result.json` schema; the C0-vs-C1 pairings were manually,
  one-off verified for cell identity in two prior documents; R0-vs-R2 has
  had no such verification because R2's dump does not exist.

**B. What is still conflicting:** Nothing, once traced. Every number this
audit located is internally consistent with its own stated population and
comparison axis. What looked like a conflict (`+0.02` vs `+0.008`) is
fully explained as two different populations of the same comparison
(macro vs. spatial-subgroup) plus one unrelated comparison (readout gap)
that happens to produce a similar-looking number. **No unresolved
numerical conflict remains** — the remaining risk is citation discipline
(quoting the wrong one of these correct-but-distinct numbers without
naming its population), not an unresolved fact.

**C. Which exact artifact is canonical for Paper C:**
- Geometry repair, primary result: `runs/paper_c_c0_c1_comparison.json`
  (seed1234) and `runs/paper_c_c0_c1_seed2_comparison.json` (seed5678),
  both already `[LOCKED]`.
- The "C1" checkpoint for all downstream Readout v2 work, including its
  reuse as "R0": `checkpoints/C1_seed1234.pt`
  (`sha256=79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854`),
  its frozen-readout evaluation `runs/eval_C1/result.json`.
- R2's canonical artifact does not exist yet — `checkpoints/
  readout_v2_pilot_seed1234_v3.pt` is the canonical **checkpoint**
  (already integrity-verified, §1), but there is no canonical **result**
  until the running endpoint finishes and `verify_paired_dumps` confirms
  it is pairable with R0.

**D. Whether R0/R2 paired analysis is currently scientifically
admissible:** **No, not yet — and this is a completeness gap, not a
defect.** R2 has no `result.json` or dump as of this audit (endpoint
still running). Once it does: (1) run `checkpoint_integrity` on the final
checkpoint if it changed since the version already verified; (2) run
`verify_contract_match(r0, r2, expect_identical=True)`; (3) run
`verify_paired_dumps` and require `safe_to_pair=True`; only then (4) run
`paired_cell_bootstrap`. All three functions already exist and are
tested. Nothing about this gate is new work — it is exactly
`docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md`'s items 2 and 5, restated here
because this task asked the provenance question independently.
