# Paper C — representation decoder ladder: preregistration

**Written before the experiment is implemented or run.** No number from this
ladder exists at the time of writing. Every architecture, budget, seed,
partition, gate, and interpretation rule below is fixed here and may not be
changed after seeing a result. This document is a commitment device, not a
results section.

---

## 1. The question

Paper C's Readout v2 pilot closed as a **NULL** (`docs/PAPER_C_READOUT_V2_DECISION_GATE.md`):
R0 = 0.574988, R2 = 0.574998, Δ = +1.02e-05, CI [0, 3.04e-05], 20,013 of
20,016 cells bit-unchanged. In the same battery, a cross-fitted linear probe
on **19 box-geometry numbers** scored **0.588251**, exceeding both.

That leaves one unresolved question, and this experiment exists only to
address it:

> Is the relational information genuinely absent from `rel_feat`, or is it
> present but poorly decoded by the current readout?

**This is a diagnosis, not an optimization.** No arm here is a candidate
deployment readout. The output is a classification, not a better model.

### Hypotheses

- **H1 — representation bottleneck**: `rel_feat` does not contain enough
  usable relational information for *any* modest decoder to extract.
- **H2 — decoder bottleneck**: `rel_feat` contains usable relational
  information that the deployed cosine readout cannot decode.
- **H3 — geometry-specific shortcut**: the geometry probe's advantage
  reflects a signal that is unusually easy for WPRD's within-pair
  construction, and is not necessarily expected to be equivalently encoded
  in `rel_feat`.

H3 is not directly falsifiable by this ladder alone and is carried as an
interpretive caveat, not an arm.

---

## 2. Prior work this replicates, and what is new

**`runs/p37_readout_vs_representation` already ran this ladder once**
(`docs/READOUT_VS_REPRESENTATION_RESULT.md`), returning
`REPRESENTATION-LIMITED` and `BELOW GEOMETRY`, with the conclusion "H1
(readout bottleneck) is DEAD."

| p37 arm | WPRD |
|---|---|
| R1_text (evaluated head) | 0.5542 |
| R2_cls (discarded classifier head) | 0.5728 |
| R3_linear on `rel_feat` | 0.5601 |
| R4_mlp on `rel_feat` | 0.5600 |
| R5_residual (group-centred `rel_feat`) | 0.5807 |
| R8_geom (train-fitted) | 0.5961 |
| R9_`rel_feat`+geometry | 0.5735 |

**What is new here**: p37 ran on the **pre-C1 geometry-repair**
representation. Its `R1_text = 0.5542` is that checkpoint's text-head WPRD.
The **C1-repaired** representation that R0/R2 evaluate (`R0 = 0.574988`) has
never been through this ladder. Since the C1 repair changed exactly the
geometry pathway feeding `rel_feat`, whether p37's verdict survives on the
repaired representation is an open, non-duplicate question.

**Registered expectation, stated in advance**: p37's verdict is the prior.
If this ladder reproduces it on the repaired representation, that is a
replication, not a new discovery, and must be reported as such.

---

## 3. Arms — exact architectures, fixed now

All arms decode the **same frozen** `rel_feat` (768-d, fp16, as cached in
`runs/eval_C1/pair_logits.pt`), on the **same** population, scored by the
**same** WPRD estimator.

| id | arm | exact form | params | fitted? |
|---|---|---|---|---|
| **A1** | frozen baseline readout | stored `text_logits` channel, `Mech.fixed_ensemble(0.0)` | 0 | no |
| **A2** | linear | `Linear(768 → 51)` on standardized `rel_feat` + bias column | 39,219 | yes, cross-fitted |
| **A3** | MLP | `Linear(768→768) → GELU → Linear(768→51)` | 629,811 | yes, cross-fitted |
| **A4** | cosine / text-space | `normalize(rel_feat) @ normalize(pred_emb)ᵀ`, `pred_emb` frozen from the dump | 0 | no |
| **A5a** | geometry, cross-fitted | existing validated 19-feature probe, cross-fitted on validation | 1,020 | yes |
| **A5b** | geometry, train-fitted | same probe, fitted on `train.jsonl` (p37's `R8` protocol) | 1,020 | yes |
| **A6** | fusion | `Linear(768+19+1 → 51)` on concatenated standardized `[rel_feat, geometry]` | 40,188 | yes, cross-fitted |
| **N1** | shuffled-label null | A2's exact form, fitted on **permuted** labels | 39,219 | yes |
| **N2** | prior control | `Mech.prior` scored directly | 0 | no |

**A4 is not an independent capacity probe.** With `pred_emb` frozen, the
cosine readout *is* the deployed baseline; A4 recomputes it from cached
`rel_feat` and is expected to match A1 to fp16 storage tolerance. It is
included as the **V2 identity gate** (proving the cached feature is the
tensor the deployed head actually read), not as a distinct decoder. This is
stated now so a near-zero A4−A1 gap is not later presented as a finding.

**Deliberate exclusion, flagged for review**: p37's `R5_residual`
(group-centred `rel_feat`, 0.5807 — its single positive finding) is **not**
in this ladder, because the task specification enumerated the arms and
adding one would be an architecture choice made after seeing p37's result.
If it should be included, that decision must be made **before** this
experiment runs, and recorded here.

---

## 4. Training budget — fixed now, not tunable

| setting | value | rationale |
|---|---|---|
| seed | **0**, single | no seed sweep |
| partition | 5 folds, **image-level**, `candidate_scorer_probe.fold_of_image(image_id, 5, salt=0)` | the identical partition every other probe in this repo uses (p22/p25/p26/p29/p32/p37); rows from one image never straddle a fold boundary |
| linear arms (A2, A6, N1) | closed-form ridge: `(AᵀA + λnI)⁻¹AᵀY`, one-hot `Y`, **λ = 1e-4** | deterministic, seed-independent, no epochs — p37's established `xfit_linear` |
| MLP (A3) | AdamW, **lr = 2e-3**, **weight_decay = 1e-4**, **batch = 4096**, **epochs = 25** | optimizer settings copied verbatim from p37; only the architecture differs (per specification) |
| geometry (A5a/A5b) | LBFGS, `max_iter=200`, `strong_wolfe`, **λ = 1e-4** | the existing validated `wprd_geometry_control` protocol, unmodified |
| standardization | per-feature z-score over **all** validation rows, then a bias column | matches p37 exactly; see the conservatism note below |
| WPRD | `within_pair_discrimination.wprd`, **cap = 64**, macro + weighted | identical to every other Paper C number |
| paired bootstrap | 2000 resamples, **seed 11** | identical to `tools/c0_c1_compare.py` |

**Two deliberate asymmetries that favour `rel_feat`, making a negative
result conservative** (both inherited from p37, stated so they are not
mistaken for oversights):

1. The `rel_feat` probes are **cross-fitted on validation**; the deployed
   head never saw validation at all. The probes get an advantage the
   baseline does not.
2. Standardization statistics are computed over all validation rows
   (mildly transductive). This is a feature-scaling leak only — no label
   information crosses folds — and it also advantages the fitted probes.

If `rel_feat` probes still fail to clear geometry *despite* both
advantages, the representation-limited reading is strengthened, not
weakened.

**No tuning is permitted.** No hyperparameter is selected on validation
WPRD, no seed is resampled, no architecture is swept. If a run fails
mechanically (crash, non-finite loss), it may be re-run **only** after the
cause is documented, and the fix must not be a hyperparameter change.

---

## 5. Validity gates — all must pass before any arm is interpreted

| gate | requirement |
|---|---|
| **G-A** | prior is constant within every (subject, object) group: `Groups.prior_is_constant < 1e-3`. Without this, WPRD is not prior-free and no arm is interpretable. |
| **G-B** | `rel_feat` present, complete, finite: shape (132556, 768), no NaN/Inf |
| **G-C** | **A4 ≈ A1**: `max\|recomputed − stored text_logits\| < 1e-2` (fp16 tolerance) — proves the cached `rel_feat` is the tensor the deployed head read |
| **G-D** | `rel_feat` is **bit-identical between the R0 and R2 dumps** — Readout v2 froze the entire backbone, so the representation must not have moved. This is a new gate (p37 had no R2 to compare against) and directly tests the "frozen representation" premise. |
| **G-E** | N2 (prior control) reads **exactly 0.5** |
| **G-F** | N1 (shuffled-label null) reads within **[0.47, 0.53]** |
| **G-G** | every arm is scored on the **same 20,016 cells** from the same `Groups` object |

Any gate failing → the run is INVALID and no interpretation is offered.

---

## 6. Reported quantities — per arm

- WPRD macro, WPRD weighted, number of cells
- paired delta vs **A1** (baseline) with 95% CI and P(delta > 0)
- paired delta vs **A5b** (train-fitted geometry) with 95% CI
- parameter count, training steps/epochs, seed, fold sizes
- **cell-level distribution**: fraction of cells changed, count positive /
  negative / exactly zero, gross positive and negative mass, and the top-15
  movers by |delta| with their cell keys

**Binding rule on cell concentration**: a WPRD gain concentrated in a tiny
number of cells is **not** evidence of broad representation quality. If any
arm's advantage over A1 is dominated by <1% of cells, that must be stated
in the same sentence as the number, exactly as the Readout v2 null was
reported (3 cells of 20,016).

---

## 7. Interpretation rules — fixed before the result exists

Let `P* = max(A2, A3)` (best fitted `rel_feat` decoder) and
`G = A5b` (train-fitted geometry).

**Outcome A — decoder ceiling remains near A1.** If A2 and A3 both remain
near ~0.575 while geometry is ~0.588: evidence favouring **H1 /
representation bottleneck**. **Not causal proof.**

**Outcome B — stronger decoder materially exceeds A1.** If a fixed
stronger decoder substantially improves WPRD on the same frozen `rel_feat`:
evidence favouring **H2 / decoder bottleneck**. Readout v2's null would
then be a **decoder-specific null, not a representation-level null**.

**Outcome C — fusion materially improves.** If A6 materially exceeds both
`P*` and `G`: evidence that geometry carries information complementary to,
and weakly represented in, `rel_feat`. Motivates a representation redesign
explicitly testing geometry/relational complementarity.

**Outcome D — geometry remains strongest.** If `G` exceeds every
frozen-`rel_feat` decoder: strong **descriptive** evidence that the current
relational representation under-expresses geometry-relevant relational
structure. **Still not causal proof.**

**Cross-reference (p37's registered thresholds, reused not reinvented)**:
`BEYOND GEOMETRY` if `P* ≥ G + 0.02`; `GEOMETRY-EQUIVALENT` if
`|P* − G| < 0.02`; `BELOW GEOMETRY` if `P* ≤ G − 0.02`.

"Materially" is operationalized as the same **0.02** used in p37 — chosen
there as slightly larger than the classifier-vs-text gap and roughly four
CI half-widths. It is not re-derived here after the fact.

---

## 8. What this experiment explicitly is not

- **Not open-vocabulary.** It uses the existing closed-set 51-predicate
  vocabulary and the existing supervised training split. No held-out
  predicate information is used anywhere. No result from this ladder may
  be described as an open-vocabulary finding.
- **Not a literature comparison.** No arm is compared to any external
  published number.
- **Not a Readout v2 rescue.** Readout v2 remains a closed, registered
  NULL. Nothing here reopens, retunes, re-seeds, or re-runs it.
- **Not a deployment candidate search.** The best-scoring arm is not
  thereby proposed as a readout.
- **Not causal.** Every outcome above is descriptive evidence about
  decodability, not proof about what the encoder "learned."

---

## 9. Artifacts

| artifact | path |
|---|---|
| implementation | `tools/representation_decoder_ladder.py` |
| tests | `tests/test_representation_decoder_ladder.py` |
| machine-readable result | `runs/paper_c_representation_decoder_ladder.json` |
| console log | `runs/paper_c_representation_decoder_ladder.log` |

**Historical artifacts that must not be modified**: `runs/eval_C1/*`,
`runs/eval_readout_v2_R2_pilot_v3/*`, `runs/p37_readout_vs_representation/*`,
every checkpoint, and every preregistration.

## 10. Compute

**CPU only.** Both dumps already cache `rel_feat` on disk, so no forward
pass and therefore no GPU is required. `nvidia-smi` will still be checked
before execution per standing policy, but this experiment does not launch
GPU work. Estimated runtime ~10-20 minutes CPU (p37's comparable ladder
took 504 s with a smaller MLP; the geometry LBFGS took 69 s when run
during the Readout v2 falsification battery).
