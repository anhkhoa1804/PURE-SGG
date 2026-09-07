# Post-quota forensic recovery — 2026-09-07

Written after the previous session hit its quota and the VM was powered off and
restarted. **Nothing in this document is reconstructed from a prior session's
narrative.** Every number is read from a machine-generated artifact on disk, or
was produced this session by a CPU-only tool run against artifacts that already
existed. No completed experiment was rerun.

---

## 1. VM / GPU state

| item | value |
|---|---|
| host | `research-no-1`, Linux 6.8.0-1066-gcp |
| GPU | NVIDIA L4, 23,034 MiB |
| GPU utilisation at recovery | **0 MiB used, 0 %, no compute processes** |
| driver / CUDA | 580.173.02 / 13.0 |
| CPU / RAM / disk | 8 cores, 31 GB (29 GB free), 32 GB free on `/` |
| interpreter | `.venv/bin/python3` (bare `python` is not on PATH) |

Every detached process from before the power-off is dead, as assumed. **No GPU
job was launched this session.** All work below is CPU-only.

## 2. Git state

- branch `research/architecture-breakthrough`, HEAD `9bcbda1`
- working tree **clean** at recovery; no stashes; reflog intact
- `origin` is 6 commits behind HEAD at `255429b` — **the previous session's
  last six commits were never pushed**
- ignored-but-present: `runs/`, `checkpoints/`, `datasets_vg150_clean/` — the
  large artifacts survived the reboot intact

The last commit, `9bcbda1`, is *implementation* (`C1a`/`C1b` behind flags, plus
`tests/test_geometry_contract_flags.py`). **There is no result commit for any
`C` experiment, and no `C` run directory exists on disk.**

## 3. What actually happened before the quota

Reconstructed from file mtimes, which form an unambiguous sequence:

| time (UTC, 2026-09-06) | event |
|---|---|
| 19:47–19:51 | `p68` smoke + pilot (GPU, ~4 min) — geometry degeneracy measured |
| 19:59–20:01 | `p69` (CPU) — missing geometry worth +0.0341 |
| 20:15–20:21 | `p70` smoke, smoke2 (GPU, bounded) |
| 20:32 | `p71` (CPU) — `p69` replicates on TEST, +0.0366 |
| 20:36 | `p70` smoke2 offline WPRD |
| 20:39–20:43 | `C1a`/`C1b` implemented behind flags; 418 tests pass; commit `9bcbda1` |
| 20:45 | `p72` run — **failed its own null gate**, preserved as `runs/p72_VOID_null_gate_failed/` |
| ~20:46 | **`p70` full causal ablation launched on the L4** |
| **23:16** | **`p70` finished** — 10,401 images, 132,556 pairs, 10,403 s, all artifacts written |
| — | **quota exhausted. The CPU analysis of `p70` never ran.** |

**The single most important recovery finding: a completed 2 h 53 m GPU
experiment was sitting on disk, fully valid, entirely unanalysed.** Its
`ablation_diagnostics.json`, `pair_logits_armA.pt` (354 MB) and
`arm_rel_feats.pt` (1.24 GB) were all present and intact; only
`ablation_wprd.json` was missing. Running the already-committed CPU tool
`tools/geometry_ablation_wprd.py` (commit `cadb22e`) against them produced the
result below in ~7 minutes.

## 4. Experiment recovery table

| exp | status | git SHA | artifact | split | population | primary result | gates |
|---|---|---|---|---|---|---|---|
| `p68` | **VERIFIED COMPLETE** | `e37860b` | `runs/p68_geom_gate_pilot/gate_diagnostics.json` | val (17,196 pairs) | pilot | 6/8 geometry channels bit-exactly constant | n/a (audit) |
| `p69` | **VERIFIED COMPLETE** | `bce22f1` | `runs/p69_pure_visible_geometry/est.json` | val, 132,556 | full | `delta_missing = +0.0341` MATERIAL | 6/6 PASS |
| `p71` | **VERIFIED COMPLETE** | `751c1cd` | `runs/p71_pure_visible_geometry_test/est.json` | **test**, 132,334 | full | `delta_missing = +0.0366` REPLICATES | 6/6 PASS |
| `p70` | **RECOVERED COMPLETE (this session)** | run at `9bcbda1`, analysed by `cadb22e` | `runs/p70_geometry_causal_ablation/ablation_wprd.json` | val, 132,556 | full | `delta_D = −0.0087` WEAK | **6/6 PASS** |
| `p72` | **VOID — ran, failed gate G4** | `bb33d00` | `runs/p72_VOID_null_gate_failed/est.json` | val | full | *(void, uncited)* | 4/5, G4 FAIL |
| `p73` | **NEW (this session, CPU)** | this commit | `runs/p73_fourier_null_and_invertibility/est.json` | val, 132,556 | full | Fourier map DESTRUCTIVE-CONFIRMED | 4/4 PASS |
| **`C0`** | **MISSING — never launched** | flags exist at `9bcbda1` | — | — | — | — | — |
| **`C1a`** | **MISSING — never launched** | impl exists, untested on GPU | — | — | — | — | — |
| **`C1b`** | **MISSING — never launched** | impl exists, untested on GPU | — | — | — | — | — |
| **`C1`** | **MISSING — never launched** | — | — | — | — | — | — |

Seeds/checkpoints: every row above evaluates the frozen historical checkpoint
`checkpoints/demo_best/pure_best_adapt_light_mR50.pt`
(sha256 `8845c3af…`). **No training was performed by any of them** — `p70` is an
evaluation-only forward pass with input interventions. There is therefore no
training seed, no new checkpoint, and no training duration to report for any
`C` arm, because **no `C` arm was ever trained**.

**Programme state: STATE D — no successor training completed.**

## 5–6. `p68` / `p69` / `p71` verification

**VERIFIED.** `p68`'s pilot finding (6 of 8 channels constant; `fusion_gate`
mean 0.50155, per-pair std 0.00013) **replicates on the full validation split**
in `p70`'s own diagnostics, over 40,201,728 gate elements — see §8. `p69` and
`p71` were re-read from their `est.json` files: 6/6 gates each, `A_relfeat` and
`B_geometry` reproducing `p60`'s anchors to ±0.0000, prior control exactly
0.5000. No contradiction found; nothing needed rerunning.

## 7. `p70` result (recovered)

Full detail in `docs/GEOMETRY_CAUSAL_ABLATION_RESULT.md`.

```
A_full            WPRD 0.5542   (reproduces p33's 0.5542 to +-0.0000)
D_no_geom         WPRD 0.5455
PRIMARY  delta_D = -0.0087   95% CI [-0.0127, -0.0049]   P(d<0) = 1.000  -> WEAK
PATHS    fusion -0.0036   edge -0.0038   additivity residual -0.0013
CHANNELS removing dy costs -0.0071 ; removing dx costs -0.0023
gates    6/6 PASS
```

The geometry pathway **is causally live** (the `INERT` band is excluded with
P = 1.000) but **WEAK** — it misses the registered `LIVE` threshold of −0.010
by 0.0013. Both injection points contribute about equally and additively.

## 8. Gate behaviour (full population, VERIFIED)

`fusion_gate` over **40,201,728** elements (132,556 pairs × 768 dims):

| mean | std | min | max | q25 | median | q75 | frac in [0.45,0.55] |
|---|---|---|---|---|---|---|---|
| 0.50153 | 0.00485 | 0.47852 | 0.52734 | 0.49805 | 0.50000 | 0.50391 | **1.0000** |

Per-pair mean: std **0.00013** across pairs. **Not one element of 40 million
lies outside [0.478, 0.528].** `p68`'s pilot number replicates to four
decimals at 2,300× the sample size. The gate is architecturally
input-dependent and empirically a constant ½.

This is the historical `C0` baseline for the Phase-8 comparison. **The
corrected-model comparison does not exist**, because no corrected model was
ever trained.

## 9. Geometry behaviour and the new mechanism (`p73`)

Full detail in `docs/FOURIER_NULL_AND_INVERTIBILITY_RESULT.md`.

Label-free invertibility probe — can the 8 raw channels be decoded back out of
their own Fourier encoding?

```
I_raw            (identity control)      mean R2  +0.9997
I_s1             (checkpoint's geom_B)   mean R2  -0.3314
I_shuffled_rows  (null control)          mean R2  -0.3197
I_s1_prod        (production today)      mean R2  +0.0920   dx +0.39  dy +0.33
I_s0.25 / I_s0.1 / I_s0.05               -0.3354 / -0.2631 / +0.1064
gates 4/4 PASS
```

**`I_s1` equals its own row-shuffled null, channel by channel.** The frozen
Fourier encoder at the checkpoint's bandwidth carries exactly as much
information about the geometry as a random permutation of itself.

And the finding that changes the plan:

> **The `p68` units bug is load-bearing.** Production's `/336` normalisation
> compresses `dx` to std 0.177, which is narrow enough that the frozen encoder
> stays locally invertible — production recovers `dx`/`dy` at R² 0.39/0.33.
> The units-fixed contract expands `dx` to std 2.09 against a `clamp(±10)`
> domain, ~85 phase wraps, and the encoder destroys **all eight** channels
> (R² −0.33, on the null floor). Fixing the units alone restores six channels
> at the input and then hashes away all eight at the encoder.

**`C1a` alone is predicted null or harmful, for a reason unrelated to the units
hypothesis.** Running it first would have produced a misattributed null.

## 10. `C0` / `C1a` / `C1b` / `C1` results

**None exist.** No run directory, no checkpoint, no metrics file, no log. The
`C1a`/`C1b` implementation exists behind default-off config flags
(`geom_input_pixel_space = False`, `geom_fourier_scale = 1.0`) with five tests
asserting the defaults are bit-exact with the historical behaviour, so `C0`
would be a valid control — but it was never launched.

## 11. Prior / calibration analysis

Not applicable to `C` (nothing to analyse), and **resolved for `p70`**:

- R@50 is flat to ±0.0005 across all six arms and moves the *wrong way*
  (`D_no_geom` scores +0.0005 **higher** than `A_full`).
- WPRD falls monotonically 0.5542 → 0.5455 as geometry is removed.
- Agreement with the frequency prior's argmax rises monotonically
  0.9244 → 0.9302.
- `model_term_std` is constant to 4 s.f. across arms — no arm is rescaled.

Geometry is what pushes this model *off* the prior. The effect is a genuine
within-pair discrimination effect that R@50 is structurally blind to — not a
calibration artifact. Read off R@50 alone, one would conclude geometry does
nothing.

## 12. Second-seed status

**N/A.** Replication is a question about a `C` result, and there is no `C`
result. `p69` *is* replicated out-of-sample by `p71` (val +0.0341 → test
+0.0366), but that is a probe, not an intervention.

## 13. Paper C verdict

### **NO-GO — not because the mechanism failed, but because the intervention was never run.**

Per the Phase-13 rule, `GO` requires a material WPRD improvement,
mechanistically explained, replicated on a second seed. **Zero of the three
conditions can be evaluated**: no `C` experiment exists. Declaring anything
else would be manufacturing a result.

The honest classification of Paper C today:

| claim | label |
|---|---|
| 6 geometry channels are bit-exactly constant in the production path | **VERIFIED** (`p68`, replicated at full scale by `p70`) |
| the missing geometry contains +0.0341 (val) / +0.0366 (test) WPRD of information | **VERIFIED** (`p69`, `p71`, 6/6 gates each) |
| ~90–95% of that information is box size | **VERIFIED** (`p69`, `p71`) |
| the geometry pathway is causally live in the deployed model, worth 0.0087 WPRD | **VERIFIED** (`p70`, 6/6 gates, CI excludes 0) |
| the `fusion_gate` is empirically a constant ½ over 40M elements | **VERIFIED** (`p70`) |
| the frozen Fourier encoder destroys its input entirely at the checkpoint's bandwidth | **VERIFIED** (`p73`, 4/4 gates, equals its own null) |
| the units bug is the only reason any geometry survives to the model | **VERIFIED** (`p73`, `I_s1_prod` +0.09 vs `I_s1` −0.33) |
| the geometry pathway is broken in two independent places | **VERIFIED** (`p68` + `p73`) |
| `C1a` alone would be null or harmful | **INFERRED** (mechanistically derived from `p73`; not yet tested) |
| restoring valid geometry would raise WPRD | **NOT YET VERIFIED — the central claim, untested** |
| the constant gate is *caused* by the degenerate input | **HYPOTHESIS** |

**Paper C already has a real, gated, mechanistic diagnosis** — arguably a
stronger one than it had before the quota, since `p73` converts "the encoder
looks like a random hash" into a gated measurement against its own null, and
finds that the two defects interact in a way nobody had predicted. What it does
not have is the intervention.

## 14. Paper B status

- **PURE** — done, the programme's own model.
- **IMP+** — done (`docs/CROSS_MODEL_IMP_PLUS_RESULT.md`), checkpoint present
  at `~/external_models/checkpoints/imp_plus.pth` (2.5 GB).
- **VCTree/TDE** — environment **fixed and committed** (`b6125a6`, torch
  2.9/CUDA 12.9 compat patch); `~/external_models/checkpoints/vctree/` is
  **empty**. Blocked on one manual browser download of
  `causal_motifs_predcls.pth` from a OneDrive anonymous link that refuses
  programmatic redemption. Not an engineering problem and **not chased this
  session**, per the standing instruction.

## 15. Missing work

1. `C0` (control), `C1a`, `C1b`, `C1` — none launched. GPU work, not started.
2. A bandwidth curve below `geom_fourier_scale = 0.05` — CPU, cheap, and now
   clearly necessary before choosing `C1b`'s value.
3. The `p73` Part-A null question is closed but left `p72` void; `p72`'s
   `+0.0892` is discarded and uncited.
4. Paper B's third model — one human browser download.

## 16. Recommended next action

**Do not launch `C1a`. Do not launch `C0` yet.**

`p73` establishes that the registered `C1b` scale grid is mostly on the null
floor and that the informative region starts at 0.05 and goes *down*. Launching
a multi-hour GPU factorial whose `C1b` arm is pinned to an uninformative
bandwidth would waste the L4 and produce an uninterpretable null — the exact
failure mode `p72` was written to prevent.

The correct next step is the **cheapest decisive one**, and it is CPU-only:
extend `p73`'s invertibility probe into a proper bandwidth curve over
`geom_fourier_scale ∈ {0.05, 0.02, 0.01, 0.005, 0.002}`, plus a
`fourier-disabled` arm (raw 8-vector straight into `geom_mlp`), and pick the
scale that maximises recovered R² while keeping the encoding smooth. **Only
then** launch the `C0`/`C1` pair on the L4 with that scale pre-registered.

This is ~15 CPU-minutes and it converts `C1b` from a guess into a measured
setting before any GPU is spent.
