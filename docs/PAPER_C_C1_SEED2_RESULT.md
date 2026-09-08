# Paper C — `C1` seed-2 replication result

Post-endpoint validation only. No training, no GPU, no endpoint re-run was
performed to produce this document — the `C1` seed-2 training and evaluation
endpoints had already finished (training `finished_utc=2026-09-08T04:26:52Z`,
evaluation completed `2026-09-08T09:27Z`) before this analysis began. This
document independently re-derives every number from the raw dumps and states
where the finding sits against the pre-registered decision rule in
`docs/PAPER_C_C0_C1_PREREGISTRATION.md`, without moving that rule.

## 1. Objective

`docs/PAPER_C_C0_C1_PREREGISTRATION.md` requires, as a precondition for any
`GO` verdict, "replication on a second seed with the identical `C1`
intervention." `docs/PAPER_C_C1_RESULT.md` (seed 1234) recorded
`POSITIVE DIRECTION, THRESHOLD NOT MET` and explicitly stated the second seed
"is still required and has not been run." This document runs that check.

## 2. Registration

The preregistration pins seed `1234` for the original paired `C0`/`C1`
comparison and states only that a second seed is required, without pinning
its value. **`seed = 5678` was operator-specified** because the registration
required an independent second seed but did not fix which one. `C0` is not
re-run per seed: it is the single registered control arm, and the seed-2
question is whether a second, independently-trained `C1` still beats that
same fixed `C0` baseline. This matches how seed 1 itself was evaluated and is
the only design under which "replicates against `C0`" is a meaningful
statement (re-running `C0` per seed would turn a paired single-intervention
test into an unpaired four-arm comparison the registration never specified).

## 3. Seed-2 provenance

`runs/C1_seed5678/provenance.txt`:

```
arm=C1
seed=5678
epochs=3
samples_per_epoch=12000
geom_input_pixel_space=true
geom_fourier_scale=0.01
git_commit=647cec1c3a5444a71fb1a5dbd7aaf94816a8bbee
git_dirty=0
resume_from=checkpoints/demo_best/pure_best_adapt_light_mR50.pt
ckpt_sha256=8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442
started_utc=2026-09-08T02:12:36Z
gpu=NVIDIA L4, 23034 MiB
finished_utc=2026-09-08T04:26:52Z
```

Verified independently in this session:

- `sha256sum checkpoints/demo_best/pure_best_adapt_light_mR50.pt` →
  `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442` — **matches**
  the provenance record and the historical checkpoint used to seed both `C1`
  arms. The historical checkpoint file was not modified (read-only check).
- `git log --since "2026-09-07 14:00" -- config.py relational_model.py
  scripts/train/run_c0_c1.sh tools/c0_c1_evaluate.py` → **empty**. No source
  file that could affect training or evaluation semantics changed between
  seed-1's launch (`2026-09-07T12:23Z`) and seed-2's evaluation
  (`2026-09-08T09:27Z`). Seed 1 and seed 2 ran the identical code.
- `checkpoints/C1_seed1234.pt` and `checkpoints/C1_seed5678.pt` are both
  exactly `5,293,177,801` bytes — same architecture, same saved-state shape.
- Budget: 3 epochs × 12,000 samples = 36,000 samples, identical to seed 1.
  `git_dirty=0` at launch.

**One flagged-and-resolved observation.** The `[ActiveBranches]` config
banner printed at *evaluation* time (`explicit_spoa=false`,
`relationness_head=false`, `text_conditioned_projection=false`,
`predicate_label_relaxation=false`, `relationness=false`,
`open_vocab_predicates.enabled=false`) differs from the banner printed at
*training* time for the same seed (all of those `true`). This looks like a
train/eval mismatch on first read. It is not one specific to seed 2: the
identical reduced set of flags appears in seed 1's evaluation log
(`runs/eval_C1.log`) byte-for-byte, confirmed by diff. `tools/c0_c1_evaluate.py`
explicitly hard-codes `--explicit_spoa_enabled false`,
`--text_conditioned_projection_enabled false`, `--relationness_enabled false`
on its own `argv` (lines 115–118) — these are training-time auxiliary
heads/losses (SPOA alignment, a text-conditioning branch, a relationness
head) that the eval-only, `gt_pairs=True`, `ensemble_alpha=0.0` WPRD path does
not exercise, by the registered evaluator's own design, for both arms and both
seeds alike. This is the standard, already-accepted evaluation protocol, not
a new defect.

## 4. Training integrity

`runs/C1_seed5678_launch.log`: three `[Epoch Summary]` blocks (`ep=0,1,2`),
no `Traceback`, no `Error`, no `Exception`, no OOM, no kill signal (grepped
case-insensitively across the full log). `finished_utc` recorded. Loss curve
is visually consistent with seed 1's run (same objective terms present,
comparable magnitudes).

## 5. Endpoint integrity

| check | status |
|---|---|
| `runs/eval_C1_seed5678/result.json` exists | **yes** |
| `runs/eval_C1_seed5678/pair_logits.pt` exists | **yes**, 353,725,891 bytes (identical size to `C0`/seed-1 dumps) |
| endpoint process exited | **yes** — `nvidia-smi`: 0 MiB used, 0% util, no process; `ps aux` shows no training/eval process |
| log ends with successful result write | **yes** — `wrote runs/eval_C1_seed5678/result.json`, no trailing traceback |
| population correct | **yes** — 10,401 images / 132,556 pairs / 20,016 cells, matches `C0` and seed 1 exactly |
| checkpoint evaluated | `checkpoints/C1_seed5678.pt` (`result.json::ckpt`), not `C0`'s or seed-1's checkpoint |

## 6. Independent WPRD reproduction

A separate script (not `tools/c0_c1_evaluate.py`) rebuilt `Mech`, `Groups` and
the WPRD loop directly from all three raw dumps (`C0`, `C1` seed 1234, `C1`
seed 5678) and recomputed every stored quantity from scratch, on CPU, using
the checkpoint's own dump — the same design as `docs/PAPER_C_C1_RESULT.md`'s
independent-reproduction section, extended to the new arm.

```
                              C0                    C1 seed1              C1 seed2
wprd_macro       0.5667271196244782    0.5749881522134409    0.572551887041989      EXACT (all 3)
wprd_weighted    0.5339868001264658    0.5370226991837204    0.5349712218326051     EXACT (all 3)
R (WPRD-side)    0.6731042265892029    0.6740623116493225    0.6737152338027954     EXACT (all 3)
mR (WPRD-side)   0.23768034534654386   0.23858144135192583   0.2386334041263234     EXACT (all 3)
prior_control    0.5                   0.5                   0.5                    EXACT (all 3)
top1==prior      0.9203506708145142    0.9193774461746216    0.917800784111023      EXACT (all 3)
model_term_std   0.993690013885498     0.9937604665756226    0.9936521649360657     EXACT (all 3)
n_cells          20016                 20016                 20016                  EXACT (all 3)
cell_values (20,016-vector)                                                          EXACT (all 3, element-for-element)
```

Every value the stored `result.json` reports for the new seed-2 arm — the
full 20,016-element `cell_values` vector included — reproduces **bit-exactly**
from an independently-written offline recomputation. Seed 1's and `C0`'s
already-locked numbers were re-derived in the same pass, as a live check that
the independent script's methodology is correct (they must reproduce
`docs/PAPER_C_C1_RESULT.md`'s locked values, and did).

## 7. Cell identity validation

Semantic cell keys `(group_key, a, b)` — where `group_key` is
`"{subject_label}||{object_label}"` and `(a, b)` is the ordered predicate-class
pair — were built independently for all three arms (not read from any stored
array position) and compared as multisets, sets, and ordered sequences.

| | `C0` | `C1` seed1 | `C1` seed2 |
|---|---|---|---|
| cell count | 20,016 | 20,016 | 20,016 |
| distinct identities | 20,016 | 20,016 | 20,016 |
| duplicate identities | 0 | 0 | 0 |

| pairwise check | result |
|---|---|
| set equality, `C0` == seed1 | **True** |
| set equality, `C0` == seed2 | **True** |
| set equality, seed1 == seed2 | **True** |
| identities missing from `C0` | **0** |
| identities missing from seed1 | **0** |
| identities missing from seed2 | **0** |
| extra identities (either direction) | **0** |
| sequence equality, `C0` == seed2 (same order) | **True** |
| sequence equality, seed1 == seed2 (same order) | **True** |
| `gt_row` identical, `C0` vs seed2 | **True** |
| `gt_y` identical, `C0` vs seed2 | **True** |
| subject-label list identical, `C0` vs seed2 | **True** |
| object-label list identical, `C0` vs seed2 | **True** |

**Exact pairing: PROVEN**, for the three-way comparison, not just `C0` vs one
arm. The 20,016 cells scored under `C1` seed 2 are the same 20,016 cells,
same predicate pairs, same underlying GT rows, in the same order, as `C0` and
seed 1. The paired test below is valid.

## 8. Paired `C0`-vs-seed2 statistics (PRIMARY)

Registered paired cell-bootstrap, 2,000 resamples, identical procedure to
`tools/c0_c1_compare.py` (seed 11, resample-with-replacement over the 20,016
paired cells), run twice — once via the registered tool itself
(`runs/paper_c_c0_c1_seed2_comparison.json`) and once via the independent
reproduction script — with identical results:

| | seed 1 (locked) | **seed 2 (this document)** |
|---|---|---|
| `C0` WPRD | 0.5667271196244782 | 0.5667271196244782 |
| `C1` WPRD | 0.5749881522134409 | **0.572551887041989** |
| `delta_WPRD` | +0.008261032588962776 | **+0.0058247674175108** |
| paired 95% CI | [+0.0038311178, +0.0126973772] | **[+0.0013811488, +0.0100445352]** |
| `P(delta > 0)` | 1.000 | **0.9965** |
| `P(delta < 0)` | 0.000 | **0.0035** |
| paired cells | 20,016 | 20,016 |
| CI excludes zero | yes | **yes** |
| cells improved / worsened / unchanged | 4,799 / 4,564 / 10,653 | **4,827 / 4,586 / 10,603** |
| registered gates D1–D4 | 4/4 PASS | **4/4 PASS** |
| `delta >= +0.010` (MATERIAL) | **NO** | **NO** |
| registered band | WEAK-POSITIVE (+0.005…+0.010) | **WEAK-POSITIVE (+0.005…+0.010)** |
| calibration-only effect | No (`|delta|>=0.005`) | **No (`|delta|>=0.005`)** |

Secondary metrics (WPRD-side composition, not the built-in PredCls
evaluator — see §12):

| | `C0` | seed2 | delta |
|---|---|---|---|
| R@50 | 0.6731042265892029 | 0.6737152338027954 | +0.00061 |
| mR@50 | 0.23768034534654386 | 0.2386334041263234 | +0.00095 |
| prior-argmax agreement | 0.9203506708145142 | 0.917800784111023 | −0.00255 |
| `model_term_std` | 0.993690013885498 | 0.9936521649360657 | −0.00004 |

**Both seed-1 and seed-2 land in the identical registered band**
(WEAK-POSITIVE: reportable, CI excludes 0, but below the pre-registered
`+0.010` MATERIAL bar). Neither run's `meets_registered_success_threshold`
flag is `true`. This document does not move that threshold.

## 9. Seed1-vs-seed2 reproducibility

Direct paired comparison, seed2 vs seed1, same 20,016 cells, same bootstrap
procedure:

```
delta(seed2 - seed1) = -0.0024362652
95% CI                = [-0.0058552155, +0.0013021483]     CI INCLUDES ZERO
```

The seed-to-seed difference in `delta_WPRD` is **not statistically
distinguishable from zero** at this sample size. The point estimate for seed
2 (+0.0058) is nominally ~30% smaller than seed 1's (+0.0083), but that gap is
not resolvable from noise with n=1 training run per seed — this programme has
no seed-variance estimate for its own pipeline (a limitation already flagged
in `docs/PAPER_C_C1_RESULT.md` §"what this cannot show," item 3, and still
true here with n=2).

**Replication classification: A — REPLICATED**, on the following grounds,
stated so the classification can be checked rather than taken on faith:

- **Direction**: positive in both seeds, CI excludes zero in both (seed1
  `P(d>0)=1.000`, seed2 `P(d>0)=0.9965`).
- **Registered band**: both seeds fall in the identical pre-specified
  WEAK-POSITIVE band; neither reaches MATERIAL, and neither is NULL or
  HARMFUL.
- **Mechanism**: geometry restoration, Fourier basis, gate behavior, and the
  spatial-predicate signature all replicate closely (§10–11).
- **Magnitude**: the seed1-vs-seed2 point-estimate gap (~30% relative) is
  visible but not statistically resolved (§9's CI includes zero); it is
  flagged, not hidden, and a reader who weighs the point estimate more
  heavily than the CI could reasonably call this **B — direction replicated,
  magnitude variable** instead. Both readings agree on the load-bearing
  fact: **the geometry-repair effect is real, positive, and reproducible,
  and neither seed clears the pre-registered MATERIAL bar.**

## 10. Spatial predicate signature

Same registered grouping as `docs/PAPER_C_C1_MECHANISM_AUDIT.md` §C.1
(`configs/predicate_metadata_vg150.json`, unmodified, not redesigned after
seeing seed 2), partitioning all 20,016 cells by whether zero, one, or both
of the cell's two predicates are in the `spatial` group:

| stratum | cells | mean WPRD, `C0` | mean WPRD, seed1 | mean WPRD, seed2 | delta (`C0`→seed1) | delta (`C0`→seed2) |
|---|---|---|---|---|---|---|
| both spatial | 6,762 | 0.569155 | 0.591210 | 0.587167 | **+0.022055** | **+0.018012** |
| mixed | 8,273 | 0.565144 | 0.570465 | 0.567734 | +0.005322 | +0.002590 |
| neither spatial | 4,981 | 0.566062 | 0.560478 | 0.560714 | −0.005583 | −0.005348 |

Share of each seed's own net gain attributable to each stratum
(`n_stratum × mean_delta_stratum ÷ (delta_WPRD × 20,016)`):

| stratum | seed1 share of net gain | seed2 share of net gain |
|---|---|---|
| both spatial | 90.19% | **104.47%** |
| mixed | 26.63% | 18.38% |
| neither spatial | −16.82% | −22.85% |

The seed1 numbers above reproduce `docs/PAPER_C_C1_MECHANISM_AUDIT.md`'s
Section C.1 table exactly (`+0.022055` / `+0.005322` / `−0.005583`,
`90.19%` share), confirming this document's spatial-signature methodology is
the same one already used, not a new post-hoc redesign.

**The pattern replicates**: in both seeds, cells where both predicates are
spatial move the most positively, mixed cells move weakly positively, and
cells with neither predicate spatial move slightly negatively. In seed 2 the
attribution is if anything *more* concentrated in the spatial stratum — the
both-spatial stratum alone accounts for slightly more than the entire net
gain (104%), with the other two strata netting mildly negative. This is the
same mechanistic story as seed 1, at a smaller absolute magnitude, which is
consistent with (not contradicted by) the geometry-repair-improves-spatial-
predicates hypothesis.

## 11. Geometry mechanism

| | `C0` | `C1` seed1 | `C1` seed2 |
|---|---|---|---|
| constant geometry channels | **6 of 8** | **0 of 8** | **0 of 8** |
| live `decoder.geom_fourier_scale` | 1.0 | 0.01 | 0.01 |
| `geom_B_std` (Fourier basis) | 9.90808391571045 | 9.90808391571045 | **9.90808391571045** |
| `dx` std | 0.207 | 2.795 | **2.795** (bit-identical to seed1) |
| `dy` std | 0.187 | 2.012 | **2.012** (bit-identical to seed1) |
| `rw`/`rh`/`ar1`/`ar2` std | ~0 (clamp-collapsed) | 1.50–1.64 | **1.50–1.64** (bit-identical to seed1) |

The raw geometry-channel statistics (`dx`, `dy`, `rw`, `rh`, `ar1`, `ar2`,
`a1`, `a2`) and the Fourier basis `geom_B` are **bit-identical between seed1
and seed2**. This is expected and is a genuine cross-check, not a
coincidence: `geom_feats_torch` is a deterministic function of the GT box
pairs and the `(geom_input_pixel_space, geom_fourier_scale)` contract only —
it does not depend on model weights or the training seed. Both seeds
evaluate the identical validation split under the identical `C1` contract, so
this pathway *must* match exactly if the contract truly reached the live
model in both runs, and it does. **Geometry restoration replicates
byte-for-byte across seeds** — this is the strongest possible replication
result for the geometry mechanism specifically (it is not a statistical
question at all).

### Fusion gate

| | `C0` | `C1` seed1 | `C1` seed2 |
|---|---|---|---|
| mean | 0.50142 | 0.50081 | 0.50075 |
| std | 0.00567 | 0.00592 | 0.00601 |
| per-pair std | 0.000179 | 0.000148 | 0.000144 |
| range | [0.4727, 0.5312] | [0.4727, 0.5312] | [0.4727, 0.5312] |
| frac in [0.45, 0.55] | 1.0 | 1.0 | 1.0 |

The gate stays within a few thousandths of 0.5 in every arm, seed-1 and
seed-2 alike. Per the standing interpretation
(`docs/PAPER_C_PURE_COMPLETE_ARCHITECTURE.md` / the readout forensic docs):
this does **not** imply geometry is unused. The gate is explicitly
regularised toward 0.5 (`lambda_relationness`-adjacent regularizer,
`gate_regularizer_weight=0.01`), it is a mixing coefficient rather than an
on/off switch, and geometry also enters through ungated edge-layer pathways.
The near-identical, near-0.5 gate statistics across both `C1` seeds is
consistent with that reading, not evidence against it.

## 12. Secondary metrics — estimator distinction preserved

Two different R/mR numbers exist for this endpoint and must not be
conflated:

- **WPRD-side R@50/mR@50** (`result.json::R`/`mR`, reported in §8): the
  checkpoint's own composition, `model_term + 3.75·log_prior`,
  background-masked — 0.6737 / 0.2386 for seed 2.
- **Built-in PredCls evaluator** (`runs/eval_C1_seed5678.log`'s
  `[Evaluation Summary]` table): PredCls R@50 = 0.6854, mR@50 = 0.2475,
  R@20 = 0.6567, mR@20 = 0.2265, ngR@50 = 0.8425. SGCls: R@50 = 0.0246,
  mR@50 = 0.0045, ngR@50 = 0.0291 (SGCls is not the registered population
  for this comparison and is reported only because the log contains it —
  PredCls is the object of every number elsewhere in this document).

These come from different evaluator code paths and compositions and are kept
separate above (§8 uses only the WPRD-side numbers, matching seed 1's
result document).

## 13. Failure-mode audit

| failure mode | status | evidence |
|---|---|---|
| endpoint did not finish / crashed | **ABSENT** | `finished_utc` recorded; no traceback in either log; `result.json`/`pair_logits.pt` both present and correctly sized |
| checkpoint mismatch (wrong file evaluated) | **ABSENT** | `result.json::ckpt = checkpoints/C1_seed5678.pt`; distinct from `C0_seed1234.pt` and `C1_seed1234.pt`; sizes match the `C1` architecture (5,293,177,801 B), not `C0`'s |
| initialization drift | **ABSENT** | `ckpt_sha256` in provenance matches independently recomputed SHA-256 of `checkpoints/demo_best/pure_best_adapt_light_mR50.pt`; identical init used for seed1 |
| source drift between seed1 and seed2 | **ABSENT** | `git log --since` over the relevant files between the two launches is empty |
| geometry contract not reaching the live model | **ABSENT** | `observed_in_live_model` (a live hook on `forward_pairs`, not a config echo) shows `geom_fourier_scale=0.01`, `img_res=336` for seed2, matching the registered `C1` contract, matching seed1 |
| estimator change | **ABSENT** | same `within_pair_discrimination::wprd`, `cap=64`, seed 0, `ensemble_alpha=0.0`, `alpha=3.75`; independently re-derived from the raw dump by a script that does not import `c0_c1_evaluate.py`, reproducing every stored value bit-exactly including the full 20,016-element `cell_values` vector |
| population mismatch | **ABSENT** | 10,401 images / 132,556 pairs / 20,016 cells, identical to `C0` and seed1; semantic cell-identity check (§7) proves set, sequence, and row-membership equality, not merely a matching count |
| cell identity / pairing error | **ABSENT** | §7: 0 duplicates, 0 missing, 0 extra, across all three arms |
| numerical instability | **ABSENT** | exact tie-corrected AUC; bit-exact reproduction on an independent implementation; effect magnitude (5.8e-03) is orders of magnitude above any observed numerical residual |
| train/eval config mismatch (flagged, resolved) | **RESOLVED, not a defect** | the `[ActiveBranches]` train-vs-eval diff (§3) is identical between seed1 and seed2 and is the registered evaluator's intended behavior (auxiliary training-only branches disabled at eval), not new drift |
| provenance/endpoint mismatch (category E) | **ABSENT** | all of the above |

No failure mode fired. Category **E (INVALID)** does not apply.

## 14. Reproducibility classification

Per §9: **A — REPLICATED**, with the explicit caveat that a stricter reading
of the ~30%-smaller point estimate (not statistically resolved at n=2 seeds)
could reasonably be read as **B — direction replicated, magnitude
variable**. Both readings agree that direction and mechanism replicate and
that neither seed reaches the pre-registered `+0.010` MATERIAL threshold.

## 15. Scientific interpretation

The `C1` geometry repair (`geom_input_pixel_space=true`,
`geom_fourier_scale=0.01`) produces a small, positive, CI-excludes-zero shift
in within-pair relational discrimination in **two independently trained
models** (different random seeds, identical data/split/prior/estimator/
budget), with a **mechanistically coherent, near-bit-identical geometry
pathway** and a spatial-predicate signature (both-spatial cells move most,
non-spatial cells move slightly negative) that reproduces in both seeds. This
is now the strongest evidence this programme has produced that the geometry
repair is a real, reproducible effect rather than a one-seed artifact of
training variance.

It is simultaneously true, and not in tension with the above, that **the
registered `+0.010` MATERIAL bar for a full Paper C "GO" is not met by either
seed** (seed1 +0.0083, seed2 +0.0058, both WEAK-POSITIVE). Per
`tools/c0_c1_compare.py`'s own hard-coded decision logic,
`meets_registered_success_threshold` is `false` for both. This document does
not, and per its own instructions must not, invent a looser threshold to
call this a `GO`.

## 16. Limitations

1. **n = 2 seeds.** This rules out "seed-1 was a fluke in the sense of a sign
   flip," but two points cannot estimate a seed-variance distribution. A
   third seed would materially strengthen the magnitude question (§9) but is
   out of scope for this document (explicitly not authorized here).
2. **The seed1-vs-seed2 magnitude gap is not resolved.** §9's CI on the
   direct seed1-vs-seed2 comparison includes zero; this document reports
   that honestly rather than picking the reading that looks better.
3. **Spatial grouping is coarse**, by the metadata file's own stated caveat,
   reused here unmodified from the mechanism audit, not re-validated.
4. **WPRD measures within-pair, prior-cancelled predicate discrimination
   under GT boxes.** A positive result here is evidence about geometry
   usability given GT boxes, not a general claim about image-specific visual
   reasoning (`docs/PAPER_C_C0_C1_PREREGISTRATION.md`'s own scope statement,
   unchanged).
5. Neither `C0` nor the historical checkpoint was re-run per seed; the
   design compares two independently-seeded `C1` runs against one fixed `C0`
   baseline (§2). This is the registered design, not a new choice made here,
   but it means seed-level variance in the baseline itself is unmeasured.

## 17. Implications for PURE Complete

Per the standing framework (`docs/PAPER_C_PURE_COMPLETE_ARCHITECTURE.md`,
`docs/PAPER_C_PURE_READOUT_FORENSIC.md`):

- **Geometry status**: the `C1` repair is now validated across two seeds
  with a coherent, byte-identical geometry mechanism. It is reasonable to
  treat it as the fixed representation baseline ("PURE Complete Geometry
  v1") for downstream readout work, while being explicit in any such label
  that the *effect size* is WEAK-POSITIVE, not MATERIAL, by this
  programme's own pre-registered bar.
- **Readout status**: unaffected by this document. The existing readout
  forensic evidence (frozen CLIP cosine ≈0.575, trained classifier ≈0.597,
  ≈96% prototype / ≈4% nonlinearity decomposition) stands as previously
  reported and is not re-derived here.
- **Current bottleneck**: unchanged from the standing diagnosis — the frozen
  CLIP predicate-prototype readout, not the (now twice-validated) geometry
  pathway.

## 18. Next architectural step

If the geometry repair is accepted as sufficiently validated at its measured
(WEAK-POSITIVE, twice-replicated) magnitude, the next step per the standing
design documents is the Readout v2 hypothesis (CLIP-initialized trainable
predicate prototypes with semantic anchoring) — design work only, source
implementation not authorized by this document. That decision belongs to the
next task, not this one.

---

## Provenance

- Result document: this file, `docs/PAPER_C_C1_SEED2_RESULT.md`.
- Inputs: `runs/eval_C1_seed5678/result.json`, `runs/eval_C1_seed5678/pair_logits.pt`,
  `runs/C1_seed5678/provenance.txt`, `runs/eval_C0/result.json`,
  `runs/eval_C1/result.json`, `runs/paper_c_c0_c1_seed2_comparison.json`.
- Analysis: independent offline reproduction script (CPU-only, not committed —
  scratch artifact), plus `tools/c0_c1_compare.py` run against
  `runs/eval_C0/result.json` and `runs/eval_C1_seed5678/result.json`
  (output: `runs/paper_c_c0_c1_seed2_comparison.json`).
- No source file was modified to produce this document. No GPU was used. No
  training or endpoint was (re-)launched.
