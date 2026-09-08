# Paper C `C1` — the geometry contract is repaired, and WPRD moves **+0.0083**: positive direction, **registered threshold NOT met**

Pre-registered in `docs/PAPER_C_C0_C1_PREREGISTRATION.md` (commit `2d69543`,
amended `db6aff4`), committed **before** the run. `C0` is the locked control
recorded in `docs/PAPER_C_C0_RESULT.md`.

> **Headline.** Both registered interventions worked mechanically and exactly as
> `p73` predicted: all 8 geometry channels are non-constant (`C0`: 6 of 8 dead)
> and the Fourier encoder is invertible at the selected bandwidth. The primary
> endpoint moved in the predicted direction with a paired 95% CI excluding zero
> — `delta_WPRD = +0.008261`, CI `[+0.00383, +0.01270]`, `P(delta > 0) = 1.000`
> over 20,016 exactly-paired cells. It is **below the +0.010 threshold fixed in
> advance**. Under the registered decision rule this is
> **POSITIVE DIRECTION, THRESHOLD NOT MET** — a `WEAK-POSITIVE`, not a Paper C
> success. Nothing has been tuned, rescued, or re-run.

**Verdict: `POSITIVE DIRECTION, THRESHOLD NOT MET` (first seed).**
A second registered seed is still required and has **not** been run.

---

## Registration

| | |
|---|---|
| pre-registration | `docs/PAPER_C_C0_C1_PREREGISTRATION.md` |
| primary endpoint | WPRD (macro over cells), `tools/within_pair_discrimination.py::wprd` |
| success threshold | `delta_WPRD >= +0.010`, fixed in advance |
| this arm | `C1` — both geometry fixes together |
| control | `C0`, locked at `0.5667271196244782` |
| decision | threshold **not met** |

The threshold was **not** rounded before being applied. In full precision:

```
C0 WPRD              0.5667271196244782
threshold  C0+0.01   0.5767271196244782
C1 WPRD              0.5749881522134409
C1 - threshold      -0.0017389674110372955     -> BELOW THRESHOLD
delta_WPRD           +0.008261032588962776
```

## SSH recovery note

The SSH session dropped during the registered `C1` run. On reconnection, before
any action was taken:

1. `ps aux | grep -E '37166|C1|train'` — **no matching process**. PID 37166 is
   gone; nothing was left running.
2. `nvidia-smi` — L4 idle, **0 MiB / 23,034 MiB, 0% utilisation, no processes**.
   Nothing was competing for the GPU and nothing of ours was still on it.
3. `git status --short` — empty; working tree clean at `2c35bba`.
4. `runs/C1_seed1234/provenance.txt` carries `finished_utc=2026-09-07T14:48:26Z`,
   appended by `scripts/train/run_c0_c1.sh` **only after** the training process
   exits 0 under `set -Eeuo pipefail`. Its presence is exit evidence, not an
   inference from a checkpoint existing.
5. `logs/C1_seed1234.log` contains three `[Epoch Summary]` blocks (`ep=0,1,2`)
   and no traceback.
6. `runs/eval_C1.log` ends with the evaluator's full summary block including
   `wrote runs/eval_C1/result.json`. In `tools/c0_c1_evaluate.py` that line is
   printed **after** `result.json` is written, which is itself after all
   post-processing — so reaching it proves the whole endpoint ran to completion.

**Classification: (B) `C1` training finished successfully — and the registered
full endpoint had also already completed** (launched `14:50:05Z`, finished
`18:21:40Z`) before the disconnect was noticed. Not (A) still running, not (C)
stopped/failed, not (D) validation-did-not-run, not (E) inconsistent.

**No GPU job was launched, relaunched or resumed during recovery. No experiment
code, config or checkpoint was modified.** All work below is CPU-only,
read-only analysis of artifacts that already existed on disk.

## Experiment identity

| | |
|---|---|
| arm | `C1` |
| seed | `1234` |
| epochs | `3` (`ep=0,1,2`, three `[Epoch Summary]` blocks) |
| samples | `12,000` per epoch x 3 = **36,000** |
| optimiser budget | batch 6 x accum 4 = effective 24; 2,000 steps/epoch |
| launcher | `scripts/train/run_c0_c1.sh` with `ARM=C1 SEED=1234` |
| started / finished (UTC) | `2026-09-07T12:23:47Z` / `2026-09-07T14:48:26Z` |
| git commit | `2c35bba26baced2b63e5afc41b4693e349e9889f` |
| git dirty | `0` (clean) |
| GPU | NVIDIA L4, 23,034 MiB, driver 580.173.02 |
| environment | Python 3.10.12, torch 2.9.1+cu129, CUDA 12.9 |

### The intervention — exactly two flags

```
geom_input_pixel_space = true      (C0: false)
geom_fourier_scale     = 0.01      (C0: 1.0)
```

Both arms are the **same script** with `ARM` changed. Data, split, seed,
optimiser, LR, schedule, batch, accumulation, budget, losses, sampler and
evaluator are pinned identically in `run_c0_c1.sh`; the two flags above are the
only ones the `case` statement touches.

Observed **inside the live model** during the endpoint (not read back from
config): `decoder.geom_fourier_scale = 0.01`, `decoder.img_res = 336`,
`geom_B_std = 9.90808391571045`. Pre-run preflight
(`runs/preflight_C1/preflight.json`) additionally recorded
`geom_B_requires_grad = false`.

## Checkpoint lineage

`C1` initialised from the **registered frozen historical checkpoint**, not from
`C0` and not from any `best_*` checkpoint:

| | |
|---|---|
| init checkpoint | `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` |
| init SHA256 | `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442` |
| registered SHA256 | `8845c3af...` — **matches** |
| final checkpoint | `checkpoints/C1_seed1234.pt` |
| final SHA256 | `79ca156524539ea1cdaaa61fde5420ad9f21145d21703a42e8c290a6c06d6854` |
| evaluated checkpoint | `checkpoints/C1_seed1234.pt` (recorded in `result.json::ckpt`) |

**Explicit negative checks:**

- The `C1` run did **not** start from `checkpoints/C0_seed1234.pt`. That path
  appears nowhere in `run_c0_c1.sh`, nowhere in `logs/C1_seed1234.log`, and
  `resume_from` is hard-coded to the `demo_best` path. `C0_seed1234.pt`
  (SHA `2264ced4...`) is a distinct file from both the init and the final
  checkpoint.
- No `best_*` checkpoint was substituted, because **no `best_*` checkpoint
  exists**. `run_c0_c1.sh` passes `--save_best_checkpoints false` for both arms
  (the arms are compared at a fixed budget, never at a per-arm selected best
  epoch, which would contaminate the pairing). `find` over the home tree returns
  no `C1_seed1234_best*` file.
- **Caveat, logged for the record:** `openvocab_rel/train.py:2655-2678` prints
  `[System] Saved best ... checkpoint to <path>` *unconditionally* while gating
  the actual `torch.save` on `save_best_checkpoints`. `logs/C1_seed1234.log`
  therefore contains four "Saved best" lines for files that were never written.
  This is a misleading log message, not a missing or deleted artifact, and it
  affects both arms identically. It is the only inconsistency found in the run.

## No source or config drift

`C0` trained at `db6aff4`, `C1` at `2c35bba`. Everything that changed between
those two commits:

```
docs/PAPER_C_C0_RESULT.md   +276    (documentation)
tools/c0_c1_compare.py      +149    (post-hoc comparison tool, added)
```

`git diff --stat db6aff4 2c35bba -- openvocab_rel/ scripts/ configs/` is
**empty**. No model, training, evaluation or config source differs between the
two arms. `tools/c0_c1_compare.py` runs after both endpoints and is not imported
by either.

Config drift at evaluation time was checked directly by diffing the
`[ActiveBranches]` JSON emitted by each endpoint into its own log:

```
101 configuration keys compared,  2 differ:
  architecture.geom_fourier_scale        C0 = 1.0    C1 = 0.01
  architecture.geom_input_pixel_space    C0 = False  C1 = True
```

Exactly the two registered flags, and nothing else.

## Endpoint

The **full registered endpoint** was run — not the in-training evaluator, not
the 60-batch evaluator, not a subset:

| | `C0` | `C1` |
|---|---|---|
| images | 10,401 | 10,401 |
| pairs | 132,556 | 132,556 |
| GT rows | 132,556 | 132,556 |
| cells | 20,016 | 20,016 |
| estimator | `within_pair_discrimination::wprd`, `cap=64`, macro | identical |
| composition | model term only, `ensemble_alpha = 0.0` | identical |
| prior | `frequency_prior_train.json`, `alpha 3.75` | identical |

`tools/c0_c1_evaluate.py` sets each arm's two geometry flags **from `--arm`**, so
an arm cannot be evaluated under the other arm's contract. The raw endpoint dump
`runs/eval_C1/pair_logits.pt` (353,725,891 bytes) is preserved for independent
reproduction.

## Independent reproduction

The stored endpoint was re-derived offline on CPU from the raw dump by a
**separate script that does not import `c0_c1_evaluate.py`** — it rebuilds
`Mech`, `Groups` and the WPRD loop from the dumps directly and compares against
`result.json`. Both arms were re-derived, so `C0`'s locked value is re-proven
here rather than assumed.

```
                              C0                      C1
model-term identity gate      3.576e-06  OK           2.861e-06  OK
prior-constant-in-group gate  9.441e-05  OK           9.441e-05  OK
population img/pair/gt/cell   10401/132556/132556/20016  (both, OK)
wprd_macro                    0.5667271196244782      0.5749881522134409   EXACT
wprd_weighted                 0.5339868001264658      0.5370226991837204   EXACT
R@50                          0.6731042265892029      0.6740623116493225   EXACT
mR@50                         0.23768034534654386     0.23858144135192583  EXACT
top1_equals_prior_argmax      0.9203506708145142      0.9193774461746216   EXACT
prior_control_wprd            0.5                     0.5                  EXACT
cell_values (n=20,016)        element-for-element exact_match = True (both)
```

Every stored value reproduced **bit-exactly**, including both full 20,016-element
`cell_values` vectors. **Prior control reads exactly `0.5` in both arms**, which
is the arithmetic guarantee the metric rests on: the train-derived prior is
constant within an `(s,o)` group (max deviation `9.441e-05`), so it cancels in
WPRD's double difference.

`C0` reproduces the locked `0.5667271196244782` over 20,016 cells with prior
control `0.5` exactly — identical to the value recorded in
`docs/PAPER_C_C0_RESULT.md`.

## Cell pairing by semantic identity

Pairing was **not** taken on trust from array position. Each cell was given the
semantic key the evaluator itself defines — `"{subject_label}||{object_label}||#{a}|{b}"`,
where `(a,b)` is the ordered predicate pair — and the two arms' key lists were
compared as multisets, as sets, and as sequences.

| | |
|---|---|
| `C0` cell count | **20,016** |
| `C1` cell count | **20,016** |
| `C0` distinct identities | 20,016 |
| `C1` distinct identities | 20,016 |
| duplicate IDs (`C0` / `C1`) | **0 / 0** |
| identities missing from `C0` | **0** |
| identities missing from `C1` | **0** |
| extra identities | **0** |
| identity **set** equality | **True** |
| identity **sequence** equality | **True** (same order, element-for-element) |

Three stronger checks passed as well:

- `gt_row` and `gt_y` vectors are **identical** between the two dumps.
- subject and object label lists are **identical**.
- **per-cell row membership is identical** — after the `cap=64` subsampling, each
  cell compares the *same GT rows* in both arms, so the pairing holds at row
  level and not merely at cell level.

**Exact pairing: PROVEN.** The paired test was therefore run.

## PRIMARY — registered paired cell bootstrap

2,000 paired resamples over the 20,016 identical cells (`tools/c0_c1_compare.py`,
reproduced independently):

| | |
|---|---|
| `C0` WPRD | **0.5667271196244782** |
| `C1` WPRD | **0.5749881522134409** |
| **`delta_WPRD`** | **+0.008261032588962776** |
| paired 95% CI | **[+0.0038311178, +0.0126973772]** |
| `P(delta > 0)` | **1.000** |
| `P(delta < 0)` | 0.000 |
| paired cells | **20,016** |
| exclusions | **0** |
| CI excludes zero | **yes** |
| `delta >= +0.010` | **NO** |

Distribution of the effect across cells:

```
improved   4,799      worsened   4,564      unchanged  10,653
median delta  0.0
```

The mean gain is real and its sign is unambiguous, but it is **thin and
broadly distributed**: 53% of cells do not move at all, and the improved and
worsened counts are close. The effect is a small net shift, not a
concentrated one.

### Registered decision rule

```
delta_WPRD = +0.008261   ->   0 < delta < +0.010
```

**`POSITIVE DIRECTION, THRESHOLD NOT MET`** — the pre-registered
`WEAK-POSITIVE` band ("reportable, but not a Paper C success"). The registered
`GO` conditions are not all met: the CI does exclude 0 and the effect is not
calibration-only, but the magnitude threshold fails and the second seed has not
been run. Nothing was tuned and the intervention was not altered after seeing
the result.

## SECONDARY metrics

Project metrics only. These are **not** the primary endpoint and are **not**
compared to external literature.

| | `C0` | `C1` | delta |
|---|---|---|---|
| R@50 | 0.6731042265892029 | 0.6740623116493225 | **+0.00096** |
| mR@50 | 0.23768034534654386 | 0.23858144135192583 | **+0.00090** |
| head / body / tail mR@50 | 0.4327 / 0.2183 / 0.0685 | 0.4343 / 0.2178 / 0.0706 | +0.0016 / −0.0005 / +0.0021 |
| weighted WPRD | 0.5339868001264658 | 0.5370226991837204 | **+0.00304** |
| prior-argmax agreement | 0.9203506708145142 | 0.9193774461746216 | **−0.00097** |
| `model_term_std` | 0.9936900138854980 | 0.9937604665756226 | +0.00007 |
| prior control WPRD | 0.5 | 0.5 | 0.0 |

Keeping the registered distinction explicit:

- **WPRD is the primary discrimination endpoint.** It is prior-free by
  construction.
- **R@50 / mR@50 are secondary project metrics.** They are composed with the
  frequency prior and are not evidence of visual discrimination. Their +0.001
  movement is **not** interpreted as improved discrimination here, and the
  registration explicitly classifies a recall-positive / WPRD-flat outcome as a
  calibration effect rather than a success.
- **Prior-argmax agreement is calibration/prior behaviour**, reported for the
  alternative-explanation audit, not as a result.

Note the ordering: WPRD moved ~9x more than R@50/mR@50. The registered
`calibration_only_effect` branch (WPRD flat while recall moves) is **not**
triggered.

## MECHANISM

### Geometry channels — the units fix worked completely

| channel | `C0` std | `C0` distinct | `C1` std | `C1` distinct |
|---|---|---|---|---|
| `dx` | 0.2073 | 11,247 | 2.7952 | 17,572 |
| `dy` | 0.1866 | 10,443 | 2.0124 | 18,145 |
| `rw` | **3.41e-13** | **1** | 1.6374 | 17,661 |
| `rh` | **3.41e-13** | **1** | 1.5018 | 17,577 |
| `ar1` | **3.41e-13** | **1** | 1.2716 | 13,001 |
| `ar2` | **3.41e-13** | **1** | 1.1085 | 12,532 |
| `a1` | **0.0** | **1** | 2.1801 | 16,381 |
| `a2` | **0.0** | **1** | 2.1192 | 15,083 |
| **constant channels** | **6 of 8** | | **0 of 8** | |

`C0` reproduces `p68`'s degeneracy exactly, as a control arm must. `C1`
restores all six dead channels. **This link is directly demonstrated.**

### Fourier decodability

`geom_B` is `nn.Parameter(..., requires_grad=False)` — frozen. Its std is
`9.90808391571045` in **both** arms after three epochs of training, i.e. bit-identical
and equal to the historical matrix `p73` probed. The `p73` Part B/C label-free
invertibility measurement therefore transfers directly to these two arms
(`docs/FOURIER_NULL_AND_INVERTIBILITY_RESULT.md`, `runs/p73c_fourier_bandwidth_curve`):

| configuration | mean out-of-fold R² | reading |
|---|---|---|
| `I_s1_prod` — the `C0` contract | **+0.0920** | only `dx` 0.39 / `dy` 0.33 survive; other six are `p68`-constant |
| `I_shuffled_rows` — null control | −0.3197 | floor |
| **`scale = 0.01` — the `C1` contract** | **+0.9839** | all 8 channels ≥ 0.97 |
| `I_raw` — identity control | +0.9997 | probe ceiling |

The bandwidth fix worked as measured in advance: at the `C1` contract the
encoder is invertible rather than a hash. **This link is directly demonstrated
for the encoder map** — with the stated caveat that it was measured on the
frozen `geom_B` in the `p73` runs, and the probe was **not** re-run on the two
trained models. Because `geom_B` is frozen and observed identical in both, the
map is the same object; but this is transfer, not a fresh per-arm measurement,
and the registration's wording ("on each trained model") is only satisfied in
that sense.

### Fusion gate — essentially unmoved

| | `C0` | `C1` |
|---|---|---|
| mean | 0.5014164 | 0.5008108 |
| std | 0.0056717 | 0.0059241 |
| min / max | 0.4727 / 0.5312 | 0.4707 / 0.5312 |
| q25 / median / q75 | 0.4980 / 0.5000 / 0.5039 | 0.4961 / 0.5000 / 0.5039 |
| per-pair mean std | 0.00017896 | 0.00014794 |
| frac in [0.45, 0.55] | 1.000 | 1.000 |
| elements | 5,615,930,880 | 5,615,930,880 |

**This is the most important negative result in the run.** The gate sits at
0.500 with a per-pair spread of `1.5e-04` in both arms. It is *not*
input-dependent: it does not open for pairs where geometry is informative and
close where it is not. Restoring the geometry did **not** cause the fusion gate
to start using it differentially. Whatever produced `+0.0083` did so through a
gate that is, to four decimal places, a constant 0.5 blend — and the gate moved
*down* slightly (−0.0006) rather than up.

### Prior / calibration

Prior-argmax agreement fell 0.9204 -> 0.9194 (−0.00097) and `model_term_std` is
flat (0.99369 -> 0.99376). The model agrees with the prior's argmax on 92% of GT
rows in both arms. There is no calibration or prior shift of a size that could
carry the primary result, and WPRD cancels class-level calibration by
construction in any case.

### The scientific chain, link by link

```
units fix + Fourier bandwidth fix
   -> geometry representation restored
      -> geometry available to decoder
         -> relational discrimination changes?
```

| link | status | evidence |
|---|---|---|
| flags applied as registered | **directly demonstrated** | live-model observation in both preflight and endpoint; 2-of-101 config diff |
| units fix restores geometry at the input | **directly demonstrated** | 6-of-8 constant -> 0-of-8; per-channel std and distinct-value counts |
| bandwidth fix makes the encoder invertible | **directly demonstrated (transferred)** | `p73` Part C, mean R² −0.33 -> +0.98, on the same frozen `geom_B` (std identical in both arms); not re-probed per trained model |
| geometry becomes **available to the decoder** | **indirectly supported at best** | the information provably survives the encoder, but nothing here shows the decoder *reads* it. The fusion gate — the one mechanism instrumented for this — is flat at 0.500 and did not become input-dependent |
| geometry is **used** by the decoder | **not demonstrated** | no probe in this run measures decoder use; `p73` states explicitly that "decodability is not use" |
| relational discrimination changes | **directly demonstrated, small** | `delta_WPRD = +0.0083`, CI excludes 0, exact pairing, bit-exact reproduction — but below threshold |
| the change is **caused by** the restored geometry | **indirectly supported** | the two flags are the only difference between the arms, so the *causal attribution to the intervention* is clean; the *mechanistic route* from restored geometry to the WPRD gain is **not** established, and the flat gate is evidence against the route that was hypothesised |

The honest summary: the intervention is demonstrably responsible for the
`+0.0083`, but the mechanism by which it produces that gain is **not** the one
the chain predicted, because the fusion gate never opened.

## Alternative-explanation audit

| | explanation | verdict | evidence |
|---|---|---|---|
| **A** | prior change | **ABSENT** | identical prior file and `alpha 3.75` in both arms; prior control WPRD `0.5` exactly in both; prior constant within group to `9.441e-05`, so it cancels in WPRD's double difference |
| **B** | calibration change | **ABSENT** | WPRD cancels any per-class additive term, temperature, tau or logit adjustment by construction; prior-argmax agreement moved −0.00097 and `model_term_std` +0.00007; the registered calibration-only branch (WPRD flat, recall up) is the opposite of what was observed — WPRD moved ~9x more than R@50 |
| **C** | estimator change | **ABSENT** | same function, same `cap=64`, same seed, same `ensemble_alpha=0.0`; both arms independently recomputed from raw dumps by a script that does not import the evaluator, reproducing every stored value bit-exactly including both 20,016-element `cell_values` vectors |
| **D** | population mismatch | **ABSENT** | 10,401 / 132,556 / 132,556 / 20,016 identical; `gt_row`, `gt_y`, subject and object label vectors identical element-for-element |
| **E** | checkpoint mismatch | **ABSENT** | init SHA `8845c3af...` matches registration; `result.json::ckpt` records `checkpoints/C1_seed1234.pt` (SHA `79ca1565...`); `C0_seed1234.pt` is a distinct file never referenced by the `C1` launcher; no `best_*` checkpoint exists to substitute |
| **F** | config drift | **ABSENT** | 101 `[ActiveBranches]` keys compared between the two endpoint logs; exactly 2 differ and both are the registered flags |
| **G** | invalid cell pairing | **ABSENT** | semantic-identity pairing proven: 0 duplicates, 0 missing, 0 extra, set **and** sequence equality, plus identical per-cell row membership after subsampling |
| **H** | numerical instability | **ABSENT** | model-term identity gates `3.6e-06` / `2.9e-06`; exact tie-corrected AUC; bit-exact reproduction on a second independent implementation; the effect (`8.3e-03`) is three orders of magnitude above any observed numerical residual |
| **I** | geometry leakage / accidental shortcut | **UNCERTAIN** | no *new* leakage path is introduced — both arms are PredCls with GT boxes, and both receive the same boxes; `C1` only changes the units and bandwidth of an already-present input. But the intervention by design makes GT-box geometry usable, and this run contains **no probe separating "genuine relational discrimination" from "better exploitation of the given boxes"**. WPRD is prior-free, not shortcut-free. This is a real interpretive limitation and is left open rather than dismissed |
| **J** | accidental change outside the two registered flags | **ABSENT** | `git diff db6aff4..2c35bba -- openvocab_rel/ scripts/ configs/` is empty; both arms clean at their commits (`git_dirty=0`); the only inter-arm delta in the launcher is the `case "${ARM}"` branch |

## Limitations

1. **The registered threshold is not met.** `+0.0083 < +0.010`. This is a
   `WEAK-POSITIVE` by a rule fixed before the run, and it is reported as such.
2. **One seed.** The registration requires a second seed; `n=1` here. A single
   seed cannot separate the intervention from seed-level training variance, and
   no seed-variance estimate exists for this pipeline.
3. **The effect is thin.** 10,653 of 20,016 cells are unchanged and the median
   per-cell delta is exactly 0; 4,799 improve against 4,564 that worsen. The
   paired CI excludes zero, but the lower bound is `+0.0038`.
4. **The hypothesised mechanism is not confirmed.** The fusion gate is flat at
   0.500 with per-pair std `1.5e-04` in both arms. The route "geometry restored
   -> gate opens -> decoder uses geometry" is **contradicted** at the gate step.
5. **Fourier decodability was transferred, not re-measured.** It rests on `p73`
   probing the same frozen `geom_B` (std identical in both arms), not on a fresh
   per-arm probe as the registration's wording implies.
6. **Decodability is not use.** Nothing here shows the decoder reads the
   restored channels.
7. **Shortcut exploitation is untested** (audit item I).
8. **No external comparison.** R@50 / mR@50 are project-internal and are not
   compared to any published VG150 number.
9. **A log message in `train.py` is misleading** — "Saved best ... checkpoint"
   prints even when the save is suppressed. It affected both arms identically
   and no artifact is missing, but the line should not be trusted as evidence
   that a file exists.

## Artifact provenance

| artifact | bytes | mtime (UTC) |
|---|---|---|
| `checkpoints/C1_seed1234.pt` | 5,293,177,801 | 2026-09-07 14:40:17 |
| `runs/eval_C1/pair_logits.pt` | 353,725,891 | 2026-09-07 17:31:52 |
| `runs/eval_C1/result.json` | 260,593 | 2026-09-07 18:21:40 |
| `runs/eval_C1/metrics.jsonl` | 66,410 | 2026-09-07 18:21:07 |
| `runs/eval_C1/latest_metrics.json` | 91,073 | 2026-09-07 18:21:07 |
| `runs/eval_C1/epoch_metrics/epoch_000.json` | 91,073 | 2026-09-07 18:21:07 |
| `runs/eval_C1.log` | 6,007 | 2026-09-07 18:21:40 |
| `runs/C1_seed1234/provenance.txt` | 399 | 2026-09-07 14:48:26 |
| `logs/C1_seed1234.log` | 68,311 | 2026-09-07 14:48:24 (training, 3 epochs) |
| `runs/preflight_C1/preflight.json` | 3,005 | 2026-09-07 04:29:35 |
| `runs/paper_c_c0_c1_comparison.json` | 5,397 | 2026-09-08 01:07:06 |

`runs/` and `checkpoints/` are gitignored; the `.pt` dumps and the 5.3 GB
checkpoint are **not** committed and must not be. This document is the committed
record.

## Status

**`POSITIVE DIRECTION, THRESHOLD NOT MET` — first seed.**

The geometry contract repair is mechanically confirmed and the primary endpoint
moved in the predicted direction with a paired CI excluding zero, but below the
threshold fixed in advance. This is **not** a Paper C success and **not** `GO`.
The result is recorded as measured; it has not been rescued, re-thresholded, or
followed by any change to the intervention.

**The second registered seed is still required and has not been run.** No
further experiment — second seed, tuning, variant, ablation or rescue — has been
launched, pending review of this evidence.
