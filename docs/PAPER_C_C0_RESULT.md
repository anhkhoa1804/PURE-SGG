# Paper C — `C0` (control arm) endpoint result

**This document reports `C0` only. It is NOT a Paper C success/failure decision.**
The registered primary endpoint is `delta_WPRD = WPRD(C1) − WPRD(C0)`, which is
not evaluable until `C1` exists. See [Scope](#what-this-document-does-not-decide).

> **Recovery note.** The SSH session was lost during the `C0` endpoint
> evaluation. On reconnection the endpoint was found **already complete**: the
> evaluator had written `runs/eval_C0/result.json` and printed its final summary
> line before the disconnect. **No GPU work was repeated.** The numbers below
> were re-derived offline, on CPU, from the original `pair_logits.pt` dump and
> reproduce the stored `result.json` **exactly** (see
> [Recovery and integrity verification](#recovery-and-integrity-verification)).

## Experiment identity

| | |
|---|---|
| arm | **`C0`** — the control arm (historical geometry contract) |
| registration | `docs/PAPER_C_C0_C1_PREREGISTRATION.md` |
| contract | `geom_input_pixel_space = false`, `geom_fourier_scale = 1.0` |
| training script | `scripts/train/run_c0_c1.sh` (`ARM=C0 SEED=1234`) |
| evaluator | `tools/c0_c1_evaluate.py --arm C0` |
| seed | 1234 |
| budget | 3 epochs × 12,000 samples = 36,000 samples, batch 6 × accum 4 (effective 24) |
| initialised from | `checkpoints/demo_best/pure_best_adapt_light_mR50.pt`, sha256 `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442`, `--reset_epoch true` |
| training window | `2026-09-07T05:00:22Z` → `2026-09-07T07:35:41Z` (`runs/C0_seed1234/provenance.txt`) |
| endpoint eval window | ended `2026-09-07T12:02:25Z` (`runs/eval_C0.log`) |
| git commit at training | `db6aff4ea874f49b8c4871fd49f1124b4c49d9e2`, `git_dirty=0` |
| git commit at recovery | `716cdbeee142199983ad4c61448ddcc4a1caad8f` (branch `research/architecture-breakthrough`), working tree **clean** |

### Evaluated checkpoint

| | |
|---|---|
| path | `checkpoints/C0_seed1234.pt` |
| sha256 | `2264ced4b5180c151798e395ed52696777b158c7d204342fad1e698139557d2c` |
| size | 5,293,177,801 bytes |
| mtime | `2026-09-07T07:27:26Z` |

This is the **final-epoch** checkpoint, as registered — never a per-arm "best"
epoch. `logs/C0_seed1234.log` shows `Saved lightweight checkpoint to
checkpoints/C0_seed1234.pt` at each of the three epoch boundaries (lines 99,
206, 312); the file on disk is the epoch-3 write.

> **Log-message caveat, recorded rather than left to be rediscovered.** The same
> log also prints four `Saved best … checkpoint` lines per epoch. Those files
> were **not** written: in `openvocab_rel/train.py` the `torch.save` calls are
> guarded by `_save_best` (line 2655) but the `print` statements (lines 2661,
> 2667, 2673, 2679) sit **outside** that guard. `--save_best_checkpoints false`
> behaved exactly as registered — `checkpoints/` contains no `C0_seed1234_best_*`
> file. The log line is cosmetic, not a protocol deviation.

## Endpoint protocol (as registered, unchanged)

Full VG150 **validation** split, GT pairs, `ensemble_alpha = 0.0` (text head =
the evaluated model term), frequency prior
`datasets_vg150_clean/frequency_prior_train.json` at `alpha = 3.75`,
`smoothing = 1.0`, `clip_input_res 336`, `cap = 64` rows per (group, class)
cell, macro over cells, via `tools/within_pair_discrimination.py::wprd` — the
same function every existing WPRD anchor uses.

**No estimator, population, checkpoint, threshold, or aggregation was changed
during recovery.**

## Population

| quantity | value |
|---|---|
| images | **10,401** |
| pairs | **132,556** |
| GT rows | **132,556** |
| (group, predicate-pair) cells | **20,016** |
| pairwise comparisons | 556,672 |
| (subject, object) groups | 45,607 |
| distinct predicate pairs covered | 772 |

This is the `p33`/`p36`/`p60`/`p69`/`p70` population **exactly**, and the cell
count is identical to `p70`'s 20,016 — so `C1` will be pairable against `C0`
cell-for-cell when it exists.

### Predicate-cell coverage

| stratum | rows | share |
|---|---|---|
| decidable (group has ≥2 distinct GT predicates) | 75,366 | 56.9% |
| singleton-group rows (no contrast possible) | 30,918 | 23.3% |
| multi-row groups with constant GT (contrast trivial) | 26,272 | 19.8% |

## PRIMARY endpoint — `C0` WPRD

| | |
|---|---|
| **WPRD (macro over cells)** | **0.5667** (0.5667271196244782) |
| WPRD (weighted) | 0.5340 (0.5339868001264658) |
| cells above 0.5 | 51.18% |
| descriptive 95% CI (cell bootstrap, 2,000 resamples, seed 1) | **[0.5616, 0.5716]** |
| descriptive 95% CI (200 resamples, seed 1) | [0.5620, 0.5709] |

> The CI is **descriptive only**. It is a one-arm cell bootstrap using the
> project's existing `tools/within_pair_discrimination.py` procedure. It is
> **not** the registered inferential test, which is a *paired* bootstrap over
> identical cells against `C1` (`tools/c0_c1_compare.py`, 2,000 resamples).

## Estimator identity gates

| gate | value | verdict |
|---|---|---|
| model-term identity, `\|fixed_ensemble(0.0) − model\|` max | 3.576e-06 (< 1e-4) | **PASS** |
| GATE W1 — prior constant within (s,o) group, max deviation | 9.441e-05 (< 1e-3) | **PASS** |
| prior control WPRD (must be exactly 0.5 — prior is arithmetically incapable of contributing) | **0.500000** | **PASS** |
| `model_term_std` | 0.9937 | — |

## SECONDARY endpoints

Under the checkpoint's own composition, `model_term + 3.75 · prior`, background
masked (`tools/cprime_analysis.py::score` / `::metrics`):

| metric | `C0` |
|---|---|
| **R@50** | **0.6731** |
| **mR@50** | **0.2377** |
| head mR | 0.4327 |
| body mR | 0.2183 |
| tail mR | 0.0685 |
| top-1 == prior argmax | **0.9204** |

> **These are project-internal estimators and must not be compared to
> literature leaderboard numbers.** In this codepath `R` is top-1 accuracy over
> GT rows and `mR` is its per-class macro average — not the standard image-level
> ranked SGG recall. They are directly comparable to this project's own anchors
> (which use the identical estimator) and to `C1`, and to nothing else.

### Three different numbers, deliberately kept apart

| source | R@50 | mR@50 | what it is |
|---|---|---|---|
| **registered secondary endpoint** (dump composition, full split) | **0.6731** | **0.2377** | the number that enters the `C1` comparison |
| built-in PredCls evaluator, full split (`runs/eval_C0/epoch_metrics/epoch_000.json`) | 0.6848 | 0.2466 | a *different* estimator, same checkpoint and split; not the registered endpoint |
| in-training eval, **60 batches only**, epoch 3 (`logs/C0_seed1234.log`) | 0.6835 | 0.2459 | a cheap stability probe on a small subset — **not** an endpoint |

## MECHANISTIC endpoints

Geometry channels observed in the live model during the endpoint pass:

| channel | std | distinct values |
|---|---|---|
| `dx` | 0.2073 | 11,247 |
| `dy` | 0.1866 | 10,443 |
| `rw` | 3.41e-13 | **1** |
| `rh` | 3.41e-13 | **1** |
| `ar1` | 3.41e-13 | **1** |
| `ar2` | 3.41e-13 | **1** |
| `a1` | 0.0 | **1** |
| `a2` | 0.0 | **1** |

**6 of 8 geometry channels are constant** — `C0` reproduces `p68`'s degeneracy,
which is exactly what a control arm must do.

Live contract observed inside the model: `decoder.geom_fourier_scale = 1.0`,
`img_res = 336`, `geom_B` std 9.9081 (the frozen historical matrix). `C0` did
**not** inherit any `C1` flag.

`fusion_gate` (5,615,930,880 elements): mean **0.50142**, std 0.00567, range
[0.4727, 0.5313], q25/median/q75 0.4980 / 0.5000 / 0.5039, per-pair mean std
0.000179, fraction in [0.45, 0.55] = 1.000.

The `C0` preflight (`runs/preflight_C0/preflight.json`, 33,698 real validation
pairs) passed **5/5** gates and independently recorded 6-of-8 constant channels
and gate mean 0.50155.

## Historical `p70` anchor — sanity check only

`p70` arm `A_full` (`runs/p70_geometry_causal_ablation/ablation_wprd.json`),
**same population, same 20,016 cells, same estimator**:

| metric | `p70` `A_full` (frozen historical ckpt) | `C0` (this arm) | difference |
|---|---|---|---|
| WPRD | 0.5542 | **0.5667** | +0.0125 |
| weighted | 0.5266 | 0.5340 | +0.0074 |
| R@50 | 0.6717 | 0.6731 | +0.0014 |
| mR@50 | 0.2321 | 0.2377 | +0.0056 |
| prior-argmax agreement | 0.9244 | 0.9204 | −0.0040 |
| cells | 20,016 | 20,016 | 0 |

**This deviation is expected and is not a problem.** The registration states it
in advance (§"Stated in advance: what this cannot show", item 1): `C0` is *not*
the historical checkpoint — it is that checkpoint fine-tuned for 36,000 further
samples, which is precisely why `C0` exists as a control. All Paper C
comparisons are `C1` vs `C0`, **never** `C1` vs 0.5542.

The anchor's role here is only to confirm that the population, the cell
construction, the estimator and the prior control are the same machinery
producing the same shape of number — which they are.

## What this document does NOT decide

- **`C0` alone cannot provide the registered paired `C0`-vs-`C1` inferential
  test.** No Paper C GO/NO-GO is claimed, implied, or available here.
- The registered success threshold is **`delta_WPRD ≥ +0.010`**, and
  `delta_WPRD` is **only evaluable once `C1` exists**.
- `GO` further requires the paired 95% CI to exclude 0, the gain not to be a
  prior/calibration effect, and replication on a second seed.
- Nothing here compares to external leaderboards.

## Recovery and integrity verification

Established on reconnection, before any action was taken:

1. **`nvidia-smi`** — L4 idle, 0 MiB / 23,034 MiB used, 0% utilisation, **no
   running processes**. Nothing was competing for the GPU.
2. **`ps aux`** — no `python`, `evaluate`, `eval` or `C0` process alive.
3. **`git status --short`** — empty; working tree clean at `716cdbe`.
4. No `STOPPED.md` or recovery marker existed for this run.
5. **`runs/eval_C0.log` ends with the evaluator's final summary block**,
   including `wrote runs/eval_C0/result.json`. In `tools/c0_c1_evaluate.py`
   `result.json` is written *after* all analysis, and that print follows it — so
   reaching this line proves the whole endpoint, including post-processing, ran
   to completion.

**Classification: (A) the process completed successfully before the SSH
disconnect.** Not (B) killed/interrupted, not (C) completed-but-unanalysed, not
(D) incomplete/corrupt output.

Completion was **not** inferred from file size. The endpoint was re-derived
offline on CPU from the original `runs/eval_C0/pair_logits.pt` (353,725,891
bytes, `2026-09-07T11:12:12Z`), reloading the dump and recomputing WPRD, the
secondary metrics, both gates and the prior control from scratch. Every stored
value reproduced **bit-exactly**, including the full 20,016-element
`cell_values` vector element-for-element:

```
population.images/pairs/gt_rows/cells   OK   10401 / 132556 / 132556 / 20016
wprd_macro        0.5667271196244782    OK
wprd_weighted     0.5339868001264658    OK
R / mR            0.6731042265892029 / 0.23768034534654386   OK
top1_equals_prior_argmax  0.9203506708145142   OK
prior_control_wprd        0.5                  OK
cell_values vector  n=20016  exact_match=True  OK
```

**No rerun was necessary and none was performed. No GPU work was repeated. No
experiment code, config, or checkpoint was modified.**

## Artifact provenance

| artifact | bytes | mtime (UTC) |
|---|---|---|
| `checkpoints/C0_seed1234.pt` | 5,293,177,801 | 2026-09-07 07:27:26 |
| `runs/eval_C0/pair_logits.pt` | 353,725,891 | 2026-09-07 11:12:12 |
| `runs/eval_C0/result.json` | 260,423 | 2026-09-07 12:02:25 |
| `runs/eval_C0/metrics.jsonl` | 66,363 | 2026-09-07 12:01:53 |
| `runs/eval_C0/latest_metrics.json` | 91,026 | 2026-09-07 12:01:53 |
| `runs/eval_C0/epoch_metrics/epoch_000.json` | 91,026 | 2026-09-07 12:01:53 |
| `runs/eval_C0.log` | 6,007 | 2026-09-07 12:02:25 |
| `runs/C0_seed1234/provenance.txt` | 399 | 2026-09-07 07:35:41 |
| `logs/C0_seed1234.log` | 68,311 | 2026-09-07 07:35:39 (training log, 3 epochs) |
| `runs/preflight_C0/preflight.json` | 3,048 | 2026-09-07 04:27:22 |

`runs/` and `checkpoints/` are gitignored; the `.pt` dumps and the 5.3 GB
checkpoint are **not** committed and must not be. This document is the committed
record.

The exact shell invocation of the endpoint evaluation is not present in
`~/.bash_history` (it was launched outside an interactive shell). Its parameters
are recoverable from `runs/eval_C0/result.json` (`tool`, `arm`, `ckpt`,
`contract`, `observed_in_live_model`) and the `[Config]` line of
`runs/eval_C0.log`, which agree with the registration.

## Status

`C0` is **complete and validated**. The workspace is ready for `C1`
(`ARM=C1 SEED=1234 bash scripts/train/run_c0_c1.sh`, then
`tools/c0_c1_evaluate.py --arm C1`, then `tools/c0_c1_compare.py`).

**`C1` was not launched.**
