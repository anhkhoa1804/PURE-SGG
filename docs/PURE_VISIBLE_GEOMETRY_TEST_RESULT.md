# `p71` — `p69` REPLICATES on the held-out TEST split: +0.0366

Pre-registered in `docs/PURE_VISIBLE_GEOMETRY_TEST_PREREGISTRATION.md`, commit
`66d9d6a`, before the run. CPU only, no GPU. Run:
`runs/p71_pure_visible_geometry_test`. Tool: `tools/pure_visible_geometry.py`
(the `p69` tool, unmodified in behaviour — the four new flags default to `p69`'s
values). Cache: `runs/p54_test_relfeat_cache` (VG150 **test**, 10,403 images /
132,334 pairs), same checkpoint `8845c3af…`, same eval configuration as the
`p36` validation cache.

## Result

| arm | TEST WPRD | 95% CI | validation (`p69`) | Δ test−val |
|---|---|---|---|---|
| `A_relfeat` (768-d) | 0.5702 | [0.5645, 0.5747] | 0.5732 | −0.0030 |
| `B_geometry` (19 numbers) | **0.5929** | [0.5870, 0.5976] | 0.5976 | −0.0047 |
| `P_pure_visible` (**the 2 channels PURE receives**) | **0.5563** | [0.5517, 0.5613] | 0.5635 | −0.0072 |
| `Q_visible_plus_sizes` | 0.5909 | [0.5866, 0.5960] | 0.5940 | −0.0031 |
| `N_shuffled` (null) | 0.5063 | [0.5019, 0.5117] | 0.4992 | +0.0071 |
| `P_prior` (control) | 0.5000 | [0.5000, 0.5000] | 0.5000 | 0.0000 |

```
PRIMARY   delta_missing = B_geometry - P_pure_visible = +0.0366   (val +0.0341)
                                                       -> MATERIAL, REPLICATES
SECONDARY delta_size    = Q_visible_plus_sizes - P    = +0.0346   (val +0.0305)
```

**Gates: 6/6 PASS.** Prior control exactly 0.5000; shuffled null 0.5063, inside
the registered [0.49, 0.51]; every arm scores all 132,334 rows. `G1`/`G2` were
registered in advance as **SKIPPED** — TEST has no `p60`-equivalent anchor, so
this run *establishes* the TEST anchors rather than reproducing them. TEST fold
sizes, recorded for any future TEST run to gate against:
`[27213, 27040, 27210, 25614, 25257]`.

## What this establishes

1. **`p69` is no longer validation-only.** The geometry PURE is structurally
   denied is worth **+0.0366 WPRD on held-out TEST**, above the registered
   MATERIAL threshold of +0.03 and slightly *larger* than the validation
   estimate. `C1a`'s motivating measurement survives out of sample.
2. **The size attribution replicates and strengthens.** `delta_size` = +0.0346
   of the +0.0366 total — **95%** of the effect is the four box-size channels
   alone, against 90% on validation. IoU, containment, aspect and log-area
   ratio contribute ~+0.002.
3. **The `p69` reframing replicates.** `A_relfeat` (0.5702) sits **above**
   `P_pure_visible` (0.5563) by **+0.0139** on TEST, against +0.0097 on
   validation. The encoder remains *ahead of its own geometry inputs*. The
   "input starvation, not representational destruction" reading of H6 is now
   supported on both splits.

## Honest reading of the differences

Every arm reads slightly lower on TEST than on validation (−0.003 to −0.007),
and the shuffled null reads slightly higher (+0.0071). Both are consistent with
TEST being a marginally harder split for this estimator, and the null's drift is
within its registered tolerance but is the largest single discrepancy in the
table — it should be read as a reminder that ±0.007 is roughly this estimator's
split-to-split noise floor, not as evidence of a problem. Crucially, the
**primary contrast is a within-split difference between two arms fitted on
identical rows and folds**, so a uniform split-level shift cancels out of it;
that is why `delta_missing` moves by only +0.0025 while the arms move by up to
−0.0072.

## What it still does not establish

Unchanged from `p69`, and worth restating because replication makes the
temptation stronger: this bounds the **information content** of the missing
channels under one estimator on held-out data. It does **not** show that a
retrained PURE would convert them into discrimination. `p67` remains the
standing warning — a directly supervised, information-theoretically real signal
still failed to move WPRD. `p70` (does the geometry pathway do anything *now*?)
and `p72` (does the frozen Fourier map destroy this information on the way in?)
are the two questions that stand between this result and the `C1` pilot.
