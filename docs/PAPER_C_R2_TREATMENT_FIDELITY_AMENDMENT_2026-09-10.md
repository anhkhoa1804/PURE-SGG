# Paper C — Readout v2: treatment-fidelity amendment and execution contract

**Status: PROPOSED / UNCOMMITTED DRAFT.** This document is written *before*
any corrected R2 run exists. No corrected-P endpoint has been computed,
read, or analyzed at the time of writing. Nothing below is chosen after
seeing a corrected number, because no corrected number exists.

This is an **additive** registration document. It does not rewrite, edit,
relabel, or delete `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md`,
`docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md`,
`docs/PAPER_C_READOUT_V2_DECISION_GATE.md`,
`docs/PAPER_C_GD_FORENSIC_AUDIT.md`, any checkpoint, or any historical run
artifact. Where it changes the *disposition* of an existing result, it says
so explicitly and leaves the original record in place.

---

## A. Registration timestamp

```
Document created (UTC):   2026-09-10T05:44:09Z
Authoring session:        read-only protocol audit + registration session
                          (no GPU, no training, no evaluation executed)
Amendment scope:          R2 treatment fidelity (§C-§H)
                          representation-ladder G-D gate input (§I-§J)
                          compute policy for the future run (§K)
Preceding audit:          same session, read-only; findings frozen in §B
```

No experiment was executed between the audit findings in §B and the
creation of this document. Git HEAD is unchanged across both (§B).

---

## B. Pre-amendment locked state

### B.1 Repository state at registration

```
branch:                        research/pure-complete-readout-v2
HEAD:                          ad3e73217a8782b52ac788c3f045390da6887fc8
HEAD subject:                  chore: add portable migration and research audits
tracked tree:                  CLEAN (staged = 0 files, unstaged = 0 files)
untracked artifacts:           144 entries, all under runs/, all intentional
                               (.gitignore policy: /runs/** excluded,
                                !/runs/**/*.{json,jsonl,md,txt} re-included)
remote:                        origin https://github.com/anhkhoa1804/Research-No.1.git
upstream for this branch:      NONE
git branch -r --contains HEAD: EMPTY  -> branch is LOCAL-ONLY, HEAD is unpushed
position:                      172 ahead / 0 behind origin/main
                               30 ahead / 0 behind origin/research/architecture-breakthrough
```

### B.2 G-D: RESOLVED (representation identity)

Frozen as established. Not reopened by this amendment.

```
G-D literal gate:                         SATISFIED by the fresh matched R2 dump
New matched R2 == historical R0:          YES, bitwise, over the full population
Historical R2 != historical R0:           YES, on exactly the final 9 images
No post-hoc tolerance introduced:         YES (criterion remains exact bit-identity)
G-D representation identity:              RESOLVED
```

Bitwise evidence (full-tensor SHA-256 over concatenated per-image
channels; `runs/eval_C1/pair_logits.pt` vs
`runs/eval_readout_v2_R2_matched_20260909T154739Z/pair_logits.pt`):

| channel | shape | R0 == matched R2 |
|---|---|---|
| `rel_feat` | (132556, 768) fp16 | IDENTICAL (`4cffe855…`) |
| `text_logits` | (132556, 51) fp32 | IDENTICAL (`e166fa2a…`) |
| `cls_logits` | (132556, 51) fp32 | IDENTICAL (`5137e51b…`) |
| `model_logits` | (132556, 51) fp32 | IDENTICAL (`b01ff4cd…`) |
| `pred_emb` | (51, 768) fp32 | IDENTICAL (`a9f8b8e7…`) |
| `prior_rows` | (132556, 51) fp32 | IDENTICAL (`4acd170e…`) |

Historical R2 divergence, localized:

```
differing images:        9 of 10,401  -> indices 10392..10400 (contiguous, terminal)
differing pair rows:     65 of 132,556  (0.049%)
max |delta rel_feat|:    1.689e-01
max |delta text_logits|: 6.365e-03
mean|delta text_logits|: 3.584e-04
batch arithmetic:        10,401 = 866 x 12 + 9  -> the final ragged batch
image_id ordering:       identical across dumps
per-image row counts:    identical across dumps
```

**Mechanistic interpretation, kept deliberately narrow.** The historical
odd-final-batch discrepancy is **transient / process-dependent**. It was
**not** caused by R0-vs-R2 representation movement: the G-D GPU matrix
(`runs/paper_c_gd_gpu_20260909T151736Z.json`, 36 pairwise `rel_feat`
contrasts) shows every 9-image condition agreeing at
`changed_fraction = 0.0` regardless of checkpoint or prototype provenance,
including the condition with the trained checkpoint P installed.

**This amendment does not claim that "padding caused it."** The padded
conditions are a *diagnostic contrast*, not an attribution. The immediate
numerical mechanism remains unidentified. What is resolved is the narrower
registered question G-D actually asks: whether the representation moved
between arms. It did not.

#### B.2.1 Legacy machine status vs human scientific disposition

These are two different objects and this amendment keeps them separate.

```
Legacy machine artifact:
    GD-UNRESOLVED remains unchanged in
    runs/paper_c_gd_gpu_20260909T151736Z.json

Human scientific disposition after forensic confirmation:
    G-D representation-identity question = RESOLVED
```

Explicitly:

- **The machine JSON is immutable.** `runs/paper_c_gd_gpu_20260909T151736Z.json`
  is not edited, regenerated, or re-run by this amendment. Its
  `status: "GD-UNRESOLVED"`, `automatic_promotion_enabled: false`,
  `human_interpretation_required: true`, and
  `decision_flags_are_unassessed_placeholders: true` fields stand exactly as
  written.
- **Its legacy status is not rewritten.** `GD-UNRESOLVED` was the correct
  output of that runner's own conservative decision function, which requires
  a demonstrated cause before promoting. No cause was demonstrated, so the
  machine verdict is correct and remains correct. This amendment does not
  contradict it.
- **The fresh matched R2 artifact satisfies the literal bit-identity
  criterion.** `runs/eval_readout_v2_R2_matched_20260909T154739Z/pair_logits.pt`
  has `rel_feat` bit-identical to `runs/eval_C1/pair_logits.pt` over the full
  population (§B.2). That is the entire predicate ladder gate G-D states.
- **The two are compatible because they answer different questions.** The
  runner's `GD-UNRESOLVED` is about *why* the historical final batch diverged
  numerically — still open. The ladder's G-D is about *whether the
  representation moved between arms* — closed, negatively, by a full-population
  bit-identity match plus the 36-contrast invariance matrix.
- **This amendment provides the registered human disposition for future
  ladder rerun bookkeeping**, and nothing more. It is the authority a future
  ladder run cites when it records G-D as passing; it is not a promotion of
  the machine record, and the machine record must not be described as having
  been promoted.

### B.3 R2: established status

```
Observed R2 endpoint:      NULL
Treatment fidelity:        FAILED
Trained-P effectiveness:   NOT MEASURED
```

Reasoning chain, as established:

1. `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §9 defines R2 as the
   identical checkpoint with `predicate_prototypes` **trained per §6**,
   evaluated via `score(rel_feat, P)` alone.
2. `checkpoints/readout_v2_pilot_seed1234_v3.pt` **does** contain a trained
   `predicate_prototypes` tensor.
3. The evaluation **drops** the checkpoint P before the parameter is
   created (§C).
4. The evaluation then **creates** `P := E` (§C).
5. `adaptive_logits == text_logits` **bitwise** over the full population in
   both existing R2 dumps (§D).
6. Therefore the recorded endpoint did not exercise trained P.

**Explicit prohibitions on how this may be described.** This is **not**
"R2 failed," and it is **not** evidence that an adaptive trained readout is
ineffective. No statement in this document, or in any document derived from
it, may characterize the historical R2 record as bearing on the truth of
`H_readout`. The hypothesis registered in
`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §1 remains **untested**.

### B.4 Representation ladder: established status

```
CONDITIONAL
REQUIRES REGISTERED RERUN
```

Recorded run: `runs/paper_c_representation_decoder_ladder.json`,
`all_gates_pass: false`, failing gate **G-D**
(`max|RF_R0 - RF_R2| = 1.689e-01`), with
`dump_r2 = runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt`.

The ladder **arm scores** are computed from the frozen R0 dump
(`dump_r0 = runs/eval_C1/pair_logits.pt`) and **do not consume R2 adaptive
logits or `predicate_prototypes` in any arm** (ladder prereg §3: "All arms
decode the **same frozen** `rel_feat` … as cached in
`runs/eval_C1/pair_logits.pt`"; A1 is the *stored* `text_logits` channel,
A4 uses `pred_emb` "frozen from the dump").

The remaining issue is therefore **only** the preregistered G-D gate and
whether the fresh matched R2 dump is an admissible gate input — resolved in
§I.

The ladder's conditional status and the R2 treatment-fidelity defect are
**causally disjoint** and must not be conflated: prototype provenance has
no path into `rel_feat` (§D.5), so the defect in §C cannot have caused the
G-D failure.

---

## C. Discovered defect

The defect is a **load-order defect in the evaluation resume path**. It is
stated at source-line precision below and is deliberately not paraphrased
as a "loading issue."

### C.1 Where the trained P is dropped

`openvocab_rel/train.py:1689–1693`

```python
compatible_state = {
    key: value
    for key, value in model_state.items()
    if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
}
```

At line **1688**, `current_state = mdl.state_dict()`. At that moment the
model has **no** `predicate_prototypes` attribute, because the only code
that creates it (`init_readout_v2`) has not yet run. The
`key in current_state` predicate therefore evaluates **False** for
`predicate_prototypes`, and the trained tensor is excluded from
`compatible_state`.

At line **1694** it lands in `skipped_state`. At line **1695**
`mdl.load_state_dict(compatible_state, strict=False)` loads everything
*except* it. At lines **1696–1697** the only diagnostic emitted is:

```
[System] Skipped 1 incompatible model tensors while resuming.
```

— a **count with no key name**. This is the exact `strict=False` silent-skip
hazard that `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §3 named in
advance ("new parameters not present in an old checkpoint must have an
explicit init path (§7) or `load_state_dict(strict=False)` will silently
skip them — **this is a named failure mode to test**, not to assume away").

**Precise statement of what the defect is, and is not.** The skip at this
filter is a *structural consequence* of lazy parameter creation and is not
by itself the defect — under the corrected design (§F.2) the very same skip
still occurs and is expected. The defect is the **absence of any subsequent
restoration**: nothing between line 1695 and the first forward pass ever
puts the trained tensor back. §F.2 fixes the missing restoration step; it
does not, and must not, try to prevent the skip.

### C.2 Where a new `P = E` is created

`openvocab_rel/train.py:1713–1728` (gated on `cfg.readout_v2_enabled`),
which rebuilds `E` live from the loaded CLIP text encoder and calls:

`openvocab_rel/train.py:1728`

```python
mdl.init_readout_v2(readout_v2_e)
```

`openvocab_rel/models/relational_model.py:875–879`

```python
if hasattr(self, "predicate_prototypes"):
    raise RuntimeError("init_readout_v2() called twice on the same model.")
p0 = E.detach().clone()
self.predicate_prototypes = nn.Parameter(p0, requires_grad=True)
self.register_buffer("_readout_v2_anchor", E.detach().clone(), persistent=False)
```

Line **878** unconditionally assigns `P := E`. Lines 875–876 *raise* on a
second call, so there is no code path in which a checkpoint P could
subsequently be installed. Line **1695** is the only `mdl.load_state_dict`
in the resume path, and it precedes construction — the trained tensor is
never reloaded.

### C.3 Why the defect was invisible

The failure is doubly silent:

1. The skip diagnostic (C.1) reports a count, not a name.
2. The very next log line, `openvocab_rel/train.py:1743`, prints what reads
   as a success message:

```
[ReadoutV2] enabled: P=predicate_prototypes (51, 768) init=E (copy, not alias) lambda_anchor=0.5 lr=2.00e-03 trainable_tensors=1 (must be exactly 1)
```

`init=E` is **correct** for a training launch resuming from
`checkpoints/C1_seed1234.pt` (§7's registered initialization) and **wrong**
for an evaluation resuming from the trained R2 checkpoint. The same message
is emitted in both cases.

Both lines appear verbatim in
`runs/paper_c_gd_gpu_20260909T151736Z/Z_R2_P_registered_9.log:10,14`. The
two R2 evaluation runs persisted **no stdout log at all**, so even the
count was never durably recorded.

### C.4 Execution order, as traced

```
tools/readout_v2_evaluate.py:151        train_mod.main(argv)  [--resume true, --readout_v2_enabled true]
  |
  |-- train.py:1685                     ckpt = torch.load(resume_path, map_location="cpu")
  |-- train.py:1687                     model_state   = ckpt["model"]     # CONTAINS predicate_prototypes
  |-- train.py:1688                     current_state = mdl.state_dict()  # DOES NOT (attribute not created yet)
  |-- train.py:1689-1693                compatible_state = {...}          # trained P DROPPED here
  |-- train.py:1694                     skipped_state == ['predicate_prototypes']
  |-- train.py:1695                     mdl.load_state_dict(compatible_state, strict=False)
  |-- train.py:1696-1697                print("[System] Skipped 1 incompatible model tensors ...")   # count only
  |-- train.py:1698-1700                clip_model.load_state_dict(ckpt["clip"], strict=False)
  |
  |-- train.py:1713                     if cfg.readout_v2_enabled:
  |-- train.py:1720-1723                    _, readout_v2_e = encode_predicate_vocab(...)   # E rebuilt live
  |-- train.py:1724-1727                    freeze CLIP + all model params
  |-- train.py:1728                         mdl.init_readout_v2(readout_v2_e)
  |     `-- relational_model.py:878             self.predicate_prototypes = nn.Parameter(E.clone())  # P := E
  |-- train.py:1743                         print("[ReadoutV2] enabled: ... init=E ...")
  |
  |-- (eval_only, epochs=0 -> zero optimizer steps -> P never moves)
  |
  |-- evals.py:1491                     text_logits     = out.text_predicate_logits(rel_feat, pred_emb)  # score(r, E)
  |-- evals.py:1517-1519                adaptive_logits = out.adaptive_predicate_logits(rel_feat)
  |     `-- relational_model.py:888         score(text_relation_features(r), predicate_prototypes)       # score(r, E)
  |                                         text_relation_features == identity
  |                                         (text_conditioned_projection_enabled=false, rel_model.py:855)
  |-- evals.py:1257-1263                dump["adaptive_logits"].append(...)
  `-- tools/readout_v2_evaluate.py:183-198   WPRD scored off adaptive_norm51
```

---

## D. Evidence

All values below were verified read-only. Tensor hashes use the scheme
already established by `tools/run_final_batch_gd_cuda.py:96` (`_tensor_hash`):
SHA-256 over `[str(dtype).encode(), json.dumps(list(shape)).encode(), raw
uint8 tensor bytes]`. This scheme is named explicitly so future runs
compare like with like.

### D.1 Checkpoint P exists and is trained

```
checkpoint:                  checkpoints/readout_v2_pilot_seed1234_v3.pt
top-level keys:              ['cfg', 'clip', 'epoch', 'experiment', 'model', 'optim', 'scaler']
model tensors:               252
'predicate_prototypes' in ckpt['model']:   TRUE
P shape:                     (51, 768)
P dtype:                     torch.float32
P finite:                    TRUE
P hash (runner scheme):      294d01cccb7bc78233d52de2c5b532e261b1782f92a77cf72f9b768954943e5c
P row-norm mean:             1.0393872261047363
ckpt cfg:                    readout_v2_enabled=True, readout_v2_lambda_anchor=0.5,
                             readout_v2_lr=0.002, seed=1234, epoch=0,
                             resume_from=checkpoints/C1_seed1234.pt,
                             explicit_spoa_enabled=False,
                             text_conditioned_projection_enabled=False,
                             predicate_label_relaxation_enabled=False
```

The recorded P hash was independently recomputed in the audit session and
matches the `checkpoint_P_hash` recorded by the G-D runner exactly.

### D.2 P != E

Compared against `E` as taken from the dumps' own `pred_emb` (identical
across all three dumps, hash `a9f8b8e7…`):

```
torch.equal(P, E):                   FALSE
torch.allclose(P, E, atol=1e-6):     FALSE
max |P - E|:                         3.5398e-02
mean|P - E|:                         7.0886e-03
||P_i - E_i||        mean:           2.3376e-01     max: 6.0984e-01
1 - cos(P_i, E_i)    mean:           3.1951e-02     max: 1.5948e-01   min: 7.4425e-03
rows with 1 - cos > 1e-6:            51 / 51
E row norms:                         exactly 1.0 (row-normalized); P's mean norm 1.0394
```

P moved on **every** row, boundedly (max `1-cos` = 0.159, far from 1), with
no semantic collapse — consistent with what decision-gate Gate 1 already
certified from the checkpoint file.

### D.3 `adaptive_logits == text_logits` in the existing R2 endpoint

```
runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt
    adaptive_logits sha256 == text_logits sha256   (1e696948…)   max|delta| = 0.0
runs/eval_readout_v2_R2_matched_20260909T154739Z/pair_logits.pt
    adaptive_logits sha256 == text_logits sha256   (e166fa2a…)   max|delta| = 0.0
```

Both over the full (132556, 51) population. The matched dump's adaptive
channel is additionally bit-identical to historical R0's `text_logits`.

### D.4 The tensor the evaluator actually installed

```
E hash (runner scheme, recomputed from dump pred_emb):
    5b8091c5de40942c02dbe5225ec1afe69f5955b3105c9926c4839f5276a5eaa6

G-D condition Z_R2_P_registered_9  (the current R2 construction):
    prototype_source                                = "registered_from_E_after_resume"
    setup.installed_P.sha256                        = 5b8091c5de40942c02dbe5225ec1afe69f5955b3105c9926c4839f5276a5eaa6
    checkpoint_P_hash.sha256                        = 294d01cccb7bc78233d52de2c5b532e261b1782f92a77cf72f9b768954943e5c
    checkpoint_state_audit.skipped_keys             = ['predicate_prototypes']
    checkpoint_state_audit.prototype_existed_at_resume = False
```

`installed_P == E`, byte-for-byte. `installed_P != checkpoint P`.

**Historical R2 therefore scored `E`, not trained P.**

### D.5 Positive evidence that trained P is not inert

G-D condition `Zp_R2_checkpointP_registered_9` restored the checkpoint P
inside the registration hook and evaluated the same nine final-batch
images:

| condition | installed P | `rel_feat` vs `X_R0_noP_9` | text WPRD | adaptive WPRD |
|---|---|---|---|---|
| `Z_R2_P_registered_9` | E (`5b8091c5…`) | `changed_fraction 0.0` | 0.6800 | **0.6800** |
| `Zp_R2_checkpointP_registered_9` | checkpoint P (`294d01cc…`) | `changed_fraction 0.0` | 0.6800 | **0.7200** |

`rel_feat` is unchanged; the adaptive endpoint moves. This is a **5-cell /
65-row local probe and is not an endpoint estimate** — the G-D audit's own
caveat applies verbatim: "An unchanged local-65-row WPRD is **not** proof of
full registered endpoint inertness." It establishes exactly two things, and
no more: the parameter is live, and prototype provenance has no path into
`rel_feat`.

**This table may not be used as a prior, an expectation, or a target for
the corrected run's endpoint.** See §H.4.

### D.6 Recorded numerical endpoints (archived, unchanged)

```
R0            runs/eval_C1/result.json                                  WPRD = 0.5749881522134409
historical R2 runs/eval_readout_v2_R2_pilot_v3/result.json              WPRD = 0.5749983647489196
matched R2    runs/eval_readout_v2_R2_matched_.../result.json           WPRD = 0.5749881522134409
delta (historical R2 - R0) = +1.02e-05, CI [0, 3.04e-05], 20,013 of 20,016 cells bit-unchanged
population (all arms): 10,401 images / 132,556 pairs / 20,016 cells
prior_control_wprd (all arms): exactly 0.5      n_constant_channels (all arms): 0
```

---

## E. Disposition of historical R2

Pre-registered classification:

```
Historical R2 treatment-fidelity status:
INVALID / PROVENANCE FAILURE AS EVIDENCE OF TRAINED-P EFFECT
```

This classification is **scoped**, and the scope is the point of the
wording. It says the historical R2 record is invalid **as evidence about
the trained-P effect**. It does not say the run was arithmetically wrong,
and it does not retroactively invalidate anything the run's own passing
gates established.

### E.1 What remains valid in the historical record

- The arithmetic. `WPRD = 0.5749983647489196` is correctly computed from
  the dump it was computed from.
- Population identity (`EXACT_MATCH`; 10,401 / 132,556 / 20,016).
- The prior control (exactly `0.5`) and the geometry invariants
  (`n_constant_channels = 0`, contract match).
- Decision-gate **Gate 1**'s certification of the **checkpoint**: P moved
  from E, bounded, non-collapsing, non-degenerate, optimizer stepped, the
  prototype param group's LR was not stuck at `min_lr`. Gate 1 read the
  checkpoint file and is unaffected by the evaluation defect.
- The `NULL` classification **as a description of the endpoint that was
  computed**.

### E.2 What is reclassified

- The historical R2 endpoint is **no longer treated as a measurement of
  trained-P effectiveness**, because §C–§D establish it measured
  `score(rel_feat, E)`.
- Decision-gate Gate 5 category **1. INVALID / PROVENANCE FAILURE** is the
  governing category for that interpretive use, and by its own text "takes
  absolute precedence: it overrides every other category regardless of what
  Gates 3-4 would otherwise show."
- `H_readout` (`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §1) reverts to
  **untested**.

### E.3 Immutability

```
Observed numerical endpoint remains archived.
It is not treated as a measurement of trained-P effectiveness.
```

No historical file is deleted, overwritten, or relabeled. The following
remain byte-immutable and read-only for all future work:

```
runs/eval_C1/*
runs/eval_readout_v2_R2_pilot_v3/*
runs/eval_readout_v2_R2_matched_20260909T154739Z/*
runs/paper_c_representation_decoder_ladder.json
runs/paper_c_representation_decoder_ladder.log
runs/paper_c_gd_gpu_20260909T151736Z*
runs/paper_c_gd_checkpoint_forensic.json
runs/paper_c_gd_forensic.json
runs/smoke_readout_v2_flagoff/*
checkpoints/readout_v2_pilot_seed1234_v3.pt
checkpoints/C1_seed1234.pt
checkpoints/C1_seed5678.pt
checkpoints/C0_seed1234.pt
checkpoints/demo_best/*
docs/PAPER_C_READOUT_V2_PREREGISTRATION.md
docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md
docs/PAPER_C_READOUT_V2_DECISION_GATE.md
docs/PAPER_C_R2_FALSIFICATION_BATTERY.md
docs/PAPER_C_READOUT_V2_ANALYSIS_PLAN.md
docs/PAPER_C_GD_FORENSIC_AUDIT.md
```

Corrections are recorded **additively**, in this document and in the future
result document, never by editing the above.

---

## F. Corrected R2 treatment

The corrected arm is named **`R2c`** throughout, so it can never be
confused with the archived `R2_pilot_v3` or `R2_matched_*` records.

### F.1 Definition

`R2c` is the treatment the preregistration §9 already defines, instantiated
for the first time: the identical checkpoint, identical geometry contract,
`predicate_prototypes` **as trained**, evaluated via `score(rel_feat, P)`
alone, no ensemble, no prior on the scored channel.

```
checkpoint:            checkpoints/readout_v2_pilot_seed1234_v3.pt
expected P hash:       294d01cccb7bc78233d52de2c5b532e261b1782f92a77cf72f9b768954943e5c
expected P shape:      (51, 768)   dtype torch.float32
baseline:              runs/eval_C1/result.json  (UNCHANGED, reused as-is)
```

**This is a load-path correction, not an architecture change.** The
function class is bit-for-bit the one already registered: P remains a
single `(51, 768)` `nn.Parameter`, the scoring primitive remains
`score(a,b) = normalize(a) @ normalize(b).T` with no learned temperature,
`text_relation_features` remains the identity
(`text_conditioned_projection_enabled=false`), no parameter is added, and
no parameter is removed. Only the *provenance of P's values* changes.

### F.2 Loading strategy — chosen and fixed BEFORE execution

Two strategies were considered. **Strategy S1 is selected and registered.**
The other is recorded so the choice is auditable rather than improvised.

| id | strategy | verdict |
|---|---|---|
| **S1** | **Post-init restoration.** After `init_readout_v2(E)` at `train.py:1728`, copy the checkpoint's `predicate_prototypes` into the created parameter under `torch.no_grad()` via `Parameter.copy_()`, gated behind an explicit flag, with hash assertions before and after. | **SELECTED** |
| S2 | Reorder: create the parameter *before* the resume block so `load_state_dict` picks it up. | Rejected. |

**Why S1.**

1. It leaves `train.py:1689–1695` — the resume path shared by **every**
   other experiment in this repository, including the C0/C1 results the
   whole programme rests on — completely untouched. S2 would modify that
   shared path and put unrelated registered results at regression risk.
2. It preserves the anchor semantics the preregistration requires. §6's
   `L_anchor` is defined against `E`, and
   `relational_model.py:879` registers `_readout_v2_anchor := E`. S1 keeps
   that buffer equal to `E`; S2 risks it becoming `P`, which would silently
   redefine the registered anchor loss.
3. It respects §7's explicit ordering constraint. `E` must be built from
   the **post-resume** CLIP text encoder — the code comment at
   `train.py:1715–1717` names this: "not the pre-resume `pred_emb_s2o`
   computed at line ~1529, which predates the '[System] Loaded fine-tuned
   CLIP weights.' load." S2 would need `E`, or the parameter, to exist
   before that load.
4. It is additive and flag-gated, so the existing `P := E` behaviour
   remains reachable and testable as a control (§F.4).

**Registered S1 contract.** A new provenance flag is added with a
fail-closed default:

```
--readout_v2_prototype_source {reinit_E | checkpoint}
      default: reinit_E        (exactly today's behaviour, unchanged)
      R2c requires:            checkpoint
```

Under `checkpoint`, and only under `checkpoint`, the implementation must:

1. read `ckpt['model']['predicate_prototypes']`;
2. assert presence, shape `(51, 768)`, dtype `torch.float32`, all-finite;
3. assert its hash equals the expected value in §F.1;
4. call `init_readout_v2(E)` unchanged, leaving `_readout_v2_anchor == E`;
5. copy the checkpoint tensor into the created parameter with **safe
   no-grad copy semantics**, not by rebinding or mutating `.data`:

   ```python
   with torch.no_grad():
       mdl.predicate_prototypes.copy_(checkpoint_p)
   ```

   `copy_` under `no_grad` preserves the `nn.Parameter` identity, its
   `requires_grad` flag, its device, and — critically — the optimizer param
   group that `train.py:1730` already registered against that exact object.
   Rebinding the attribute or assigning to `.data` would silently detach the
   optimizer's reference. There is no repository-specific reason to prefer
   `.data` here;
6. re-hash the installed parameter and assert it equals the expected value;
7. record everything §G requires;
8. `raise` — not warn, not fall back — on any failed assertion.

The flag name, its default, and the seven steps above are fixed by this
document and may not be varied at execution time. Nothing else in
`train.py`, `relational_model.py`, `evals.py`, or
`tools/readout_v2_evaluate.py` may be modified for this run beyond what
steps 1–8 and §G's logging require.

### F.3 Treatment fidelity is established from parameter provenance

**The governing principle, stated before the conditions.** Treatment fidelity
is a property of **which tensor the model scored with**, not of **what the
resulting numbers looked like**. It is established from parameter provenance
alone. No endpoint value, and no relationship between the endpoint and the
baseline, may be used to decide whether the treatment was instantiated.

This distinction is load-bearing and is stated as two separate registers:

```
parameter/treatment identity:   provenance gate      (§F.3.1 — decides validity)
endpoint equality:              scientific result    (§F.3.3 — decides nothing about validity)
```

#### F.3.1 Required treatment-fidelity assertions (the provenance gate)

All five must hold. Any failure is fail-closed: the run aborts, and is never
salvaged or reported with a caveat.

```
1. checkpoint P exists                     ckpt['model']['predicate_prototypes'] present
2. checkpoint P hash == registered hash    294d01cccb7bc78233d52de2c5b532e261b1782f92a77cf72f9b768954943e5c
3. installed P hash == checkpoint P hash   asserted after restoration, before the first forward
4. installed P != E                        torch.equal(installed_P, E) must be False
5. prototype_source == "explicit_checkpoint_P"
```

Assertion **3** is the **decisive** post-restore provenance assertion. It, and
it alone, certifies that the tensor the model will score with is the tensor
the checkpoint holds. Assertion 4 is a redundant guard against a
same-hash-different-tensor pathology and against a checkpoint whose training
never moved P (which would independently be a
`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §15 failure).

#### F.3.2 Additional fail-closed structural conditions

```
6. checkpoint P shape != (51, 768), dtype != torch.float32, or any non-finite element
7. skipped_keys_by_name contains any model key OTHER than 'predicate_prototypes' (see §F.3.4)
8. checkpoint provenance is not explicitly recorded per §G
9. optimizer performed any step (this is eval_only, epochs=0: steps must be 0)
10. trainable tensor set != {predicate_prototypes}
11. population, cell count, or semantic cell identity differs from R0's
12. prior_control_wprd != 0.5 (tolerance < 1e-6, the standing convention)
13. geometry contract or geom channel stats differ from R0's
14. any logit is non-finite
```

Conditions 11–14 restate `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §15
unchanged.

#### F.3.3 `adaptive_logits == text_logits` — required diagnostic, NOT an invalidation

`adaptive_equals_text` (bitwise, over the full population) is a **REQUIRED
RECORDED DIAGNOSTIC** under §G. It is **not** a fail-closed condition and
**not** an invalidation criterion.

Reason, registered explicitly so it cannot be re-litigated after the number
is seen: **a correctly instantiated trained P could, in principle, still
produce an observationally null endpoint.** If provenance is certified by
§F.3.1 and the endpoint nonetheless coincides with the baseline, that is a
**scientifically valid null result** about `H_readout` — not a provenance
failure. Declaring it invalid merely because the output matched the baseline
would be the mirror image of the defect this amendment exists to correct: it
would let the *result* decide whether the *treatment* happened.

Recording obligations, which are strict even though the diagnostic is not a
gate:

- `adaptive_equals_text` must be computed bitwise over the full 132,556-row
  population and written into the run's provenance record and `result.json`.
- If it is `true`, the result document must state so prominently, alongside
  the §F.3.1 provenance certification, and must classify the outcome through
  the unchanged decision-gate Gate 5 categories.
- Its value may never be used to accept or reject the run.

#### F.3.4 `skipped_keys` semantics under S1 — the expected skip

Under the selected S1 strategy the trained prototype tensor **is expected to
be skipped** by the original resume filter. This is structural, not a fault:

```
resume
  -> checkpoint P is necessarily absent from mdl.state_dict() at resume time
  -> predicate_prototypes is expected to be skipped
  -> init_readout_v2(E)
  -> explicit checkpoint-P restoration
```

Therefore:

```
expected skipped_keys_by_name at the pre-init resume stage:
["predicate_prototypes"]

NO OTHER model key may be skipped.
```

Registered consequences:

- `predicate_prototypes` appearing in `skipped_keys_by_name` at the original
  resume filter is **EXPECTED** under S1 and must be recorded as expected,
  not as a warning or an error.
- It is **subsequently restored explicitly from checkpoint P**, and that
  restoration is what §F.3.1 assertion 3 verifies.
- **Any other skipped model key is a fail-closed condition** (§F.3.2.7) —
  it would mean the backbone did not fully load, which no registered
  experiment tolerates.
- The expected skip must **not be hidden**. Suppressing it, filtering it out
  of the record, or reporting a bare count in its place is itself a §G
  violation. The record shows the full sorted list of names, with the
  expected entry labelled as expected.

### F.4 Registered control arms — no new run required

```
R2(P=E) control:  runs/eval_readout_v2_R2_matched_20260909T154739Z/   ALREADY EXISTS
```

The matched dump is exactly the `P := E` construction under the current
code, verified bit-identical to R0. It serves as the corrected run's
same-code control and **must not be re-run**. Any difference between `R2c`
and this control is attributable to P's values alone, since §D.5 and the
G-D matrix establish `rel_feat` is invariant to prototype provenance.

Registered expectation, stated in advance: `R2c`'s `rel_feat`,
`text_logits`, `cls_logits`, `pred_emb`, and `prior_rows` should be
**bit-identical** to R0's, and only `adaptive_logits` is free to differ. If
any channel other than `adaptive_logits` differs, that is an unregistered
change in the evaluation path and the run is INVALID under §F.3.2.

Note the asymmetry deliberately: `adaptive_logits` is *free* to differ, not
*required* to. Whether it differs is the scientific question (§F.3.3), and
it is recorded as a diagnostic rather than gated.

---

## G. Exact provenance assertions

Every `R2c` run must write a **persistent** provenance record, in the run's
own output directory, containing at minimum:

```
checkpoint_path
checkpoint_P_hash
installed_P_hash
prototype_source
skipped_keys_by_name
P_vs_E summary
stdout/stderr
```

Fully specified:

| field | requirement |
|---|---|
| `checkpoint_path` | absolute or repo-relative path actually opened |
| `checkpoint_sha256` | SHA-256 of the checkpoint **file** |
| `checkpoint_P_hash` | `_tensor_hash` of `ckpt['model']['predicate_prototypes']`, with `dtype` and `shape` |
| `installed_P_hash` | `_tensor_hash` of `mdl.predicate_prototypes` immediately **before** the first forward |
| `installed_P_matches_checkpoint` | boolean; must be `true`. This is the decisive post-restore provenance assertion (§F.3.1.3) |
| `installed_P_equals_E` | boolean; must be `false` (§F.3.1.4) |
| `prototype_source` | literal `"explicit_checkpoint_P"` for `R2c` |
| `skipped_keys_by_name` | the **full sorted list of key names**, never a count. Under S1 this is expected to be exactly `["predicate_prototypes"]` (§F.3.4); any other entry is fail-closed |
| `skipped_keys_expected` | the registered expectation, literal `["predicate_prototypes"]`, recorded alongside the observed list so the expected skip is visible rather than inferred |
| `skipped_keys_unexpected` | observed minus expected; must be `[]` |
| `prototype_skip_was_expected` | boolean; must be `true` under S1, and must be recorded as an expected condition rather than a warning |
| `loaded_keys_count` | count, in addition to (not instead of) the names |
| `E_hash` | `_tensor_hash` of the live-encoded `E`, for comparison |
| `P_vs_E` | `max|P-E|`, `mean|P-E|`, per-row `‖P_i-E_i‖` (mean/max), `1-cos(P_i,E_i)` (mean/min/max), count of rows with `1-cos > 1e-6` |
| `adaptive_equals_text` | boolean, bitwise over the full population. **Required diagnostic, not a gate** (§F.3.3) — recorded whatever its value; never used to accept or reject the run |
| `optimizer_steps` | must be `0` |
| `trainable_tensor_names` | exact set; must be `{predicate_prototypes}` |
| `flags` | `readout_v2_enabled`, `readout_v2_prototype_source`, `explicit_spoa_enabled`, `text_conditioned_projection_enabled`, geometry contract |
| `stdout` / `stderr` | captured to files under the run directory, retained |
| `env` | `nvidia-smi` output, CUDA/cuDNN/TF32/autocast flags, torch version, git HEAD, `git status --porcelain` |

**No silent count-only logging.** The `train.py:1696–1697` message must be
supplemented (not replaced, to avoid touching the shared path's behaviour
for other experiments) by a named-key record emitted from the `R2c`
provenance path. The absence of any required field above is fail-closed
condition §F.3.2.8.

**The expected skip is disclosed, not suppressed.** The record carries both
the observed `skipped_keys_by_name` and the registered
`skipped_keys_expected`, so a reader sees that `predicate_prototypes` was
skipped *and* that this was anticipated *and* that it was restored and
hash-verified. Hiding the expected skip to make the record look cleaner is
prohibited (§F.3.4).

The two archived R2 runs persisted no stdout at all. That gap is the
proximate reason this defect survived a five-gate decision protocol, and
closing it is part of the registered contract, not an optional nicety.

---

## H. Endpoint and gates

### H.1 Unchanged — explicitly frozen

The following are **not** modified by this amendment:

```
WPRD definition            tools/within_pair_discrimination.py, unmodified
WPRD cap                   64
WPRD thresholds            NONE registered for the pilot, and none introduced here
bootstrap method           paired cell bootstrap, 2,000 resamples
bootstrap seed             11
population                 10,401 images / 132,556 pairs / 20,016 cells
predicate vocabulary       51-class pred_vocab, same row order
model architecture         unchanged (see §F.1)
optimizer / training budget unchanged; NO training occurs in R2c
geometry configuration     geom_input_pixel_space=true, geom_fourier_scale=0.01
evaluator configuration    ensemble_alpha=0.0, GT pairs, freq prior alpha=3.75,
                           smoothing=1.0, batch_size=12, full split
```

### H.2 Endpoint

```
Delta_WPRD_readout = WPRD(R2c) - WPRD(R0)
```

against the unchanged R0 baseline `runs/eval_C1/result.json`
(`WPRD = 0.5749881522134409`). Paired cell bootstrap, 2,000 resamples,
seed 11, identical procedure to `tools/c0_c1_compare.py`.

Secondary and mechanistic endpoints are inherited verbatim from
`docs/PAPER_C_READOUT_V2_PREREGISTRATION.md` §13 and are not restated or
altered here.

### H.3 Gate inheritance — stated in advance

| gate | disposition | reason |
|---|---|---|
| §17 regression tests (`tests/test_readout_v2.py`) | **INHERITED**, must still pass in full | unchanged code paths; the full suite must pass, no skips |
| §18 CPU offline validation | **INHERITED** | unchanged |
| §15 flag-off regression (`runs/smoke_readout_v2_flagoff/`) | **INHERITED** | tests `readout_v2_enabled=false`, untouched by S1 |
| Decision-gate **Gate 1** (endpoint integrity) | **INHERITED** | reads the *checkpoint file*, which is unchanged and already certified: P moved, bounded, non-collapsing, non-degenerate, `only_one_tensor_ever_stepped`, LR not stuck at `min_lr` |
| Decision-gate **Gate 1F** (new, provenance) | **NEW, non-numeric** | the §F.3.1 treatment-fidelity assertions plus the §F.3.2 structural conditions; asserts the evaluated P *is* the certified checkpoint P. This closes the loop Gate 1 never closed. It reads **parameter provenance only** — never the endpoint (§F.3.3) |
| Decision-gate **Gate 2** (population identity) | **MUST RERUN** | consumes the new dump; `verify_population_identity(eval_C1, R2c)` must return `EXACT_MATCH`, plus `verify_contract_match(..., expect_identical=True)` |
| Decision-gate **Gate 3** (paired comparison) | **MUST RERUN** | consumes the endpoint |
| Decision-gate **Gate 4** (mechanism falsification) | **MUST RERUN** | consumes the endpoint; full battery `docs/PAPER_C_R2_FALSIFICATION_BATTERY.md`, all controls, A–F profile |
| Decision-gate **Gate 5** (interpretation) | applies to the **joint** Gate 1–4 outcome of `R2c` | five categories unchanged, verbatim |

Gate ordering remains strictly 1 → 1F → 2 → 3 → 4 → 5, with a failed gate
stopping the process at that gate. Later gates are not consulted to rescue
an earlier failure.

New tests to be added alongside the S1 implementation (registered now, so
they are not invented after the fact):

1. under `--readout_v2_prototype_source checkpoint`, a synthetic checkpoint
   with a known non-`E` P round-trips into `predicate_prototypes` exactly;
2. a hash mismatch **raises**, and does not warn-and-continue;
3. after restoration, `installed_P == checkpoint_P` **and**
   `_readout_v2_anchor == E`;
4. under the default `reinit_E`, behaviour is bit-identical to today's;
5. an **unexpected** skipped model key fails closed;
6. the **expected** skipped key `predicate_prototypes` is accepted under S1,
   provided it is subsequently restored and hash-verified (§F.3.4);
7. `adaptive_logits == text_logits` is **not** itself a treatment-fidelity
   failure — provenance, not endpoint equality, determines validity
   (§F.3.3);
8. the existing `tests/test_readout_v2.py` suite remains green.

### H.4 No new performance threshold

**No performance threshold is introduced.** `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md`
§16 registered "no arbitrary WPRD bar" for the pilot, and this document does
not invent one retroactively. Classification depends on the joint Gate
1–4 outcome, not on Gate 3's magnitude.

Explicitly prohibited as expectations or targets:

- the `Zp` local `0.72` (§D.5) — a 5-cell / 65-row probe, not an endpoint
  estimate;
- the classifier reference `≈0.5974` (§2 of the preregistration, and §22's
  standing prohibition);
- any ladder arm score (§B.4);
- the archived `+1.02e-05` delta.

A negative, stable, mechanistically legible `R2c` result is a valid
registered outcome and will not be rescued (preregistration §16).

### H.5 Post-run discipline

A `MECHANISTIC POSITIVE` result at pilot scale **does not** authorize a
full-budget run. Preregistration §12/§21/§22 and the decision gate's own
exclusion require a **separate, dated amendment** written after `R2c`'s
numbers are known. This document does not grant that authorization and
does not pre-decide `R2c`'s outcome.

---

## I. G-D amendment (representation ladder gate input)

This section resolves an ambiguity that was **never registered either way**:
ladder preregistration §5's G-D text says "`rel_feat` is bit-identical
between the R0 and R2 dumps" without naming which artifact is "the R2
dump," and §9 lists `runs/eval_readout_v2_R2_pilot_v3/*` among artifacts
that must not be modified without stating that this pins the gate's input.

### I.1 The criterion is unchanged

```
G-D criterion:  rel_feat bit-identical between the R0 dump and the R2 dump.
                EXACT bit-identity. No tolerance. No epsilon. No fp16 slack.
```

This amendment introduces **no tolerance whatsoever**. The criterion is the
same predicate it always was.

### I.2 Authorized gate input

This amendment **explicitly authorizes** the following substitution, and it
is authorized only because this document says so:

```
G-D gate input, authorized:
    dump_r0 = runs/eval_C1/pair_logits.pt                                  (unchanged)
    dump_r2 = runs/eval_readout_v2_R2_matched_20260909T154739Z/pair_logits.pt
```

Grounds, stated so the authorization is auditable rather than convenient:

1. The matched dump is a **complete, full-population** R2-configuration
   evaluation (10,401 / 132,556 / 20,016), produced by the same
   `tools/readout_v2_evaluate.py` from the same checkpoint under the same
   registered evaluator configuration.
2. It satisfies the criterion in §I.1 **exactly**, on `rel_feat` and on
   every other shared channel (§B.2).
3. G-D's registered *purpose* is stated in its own gate text: "Readout v2
   froze the entire backbone, so the representation must not have moved."
   The matched dump answers precisely that question, affirmatively.
4. The G-D GPU matrix establishes that `rel_feat` is invariant to
   checkpoint identity and prototype provenance, and varies only with batch
   composition. The historical failure was therefore not a representation
   movement — which is the only thing G-D is registered to detect.

### I.3 The historical record is unchanged

```
runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt        IMMUTABLE
runs/paper_c_representation_decoder_ladder.json        IMMUTABLE
    -> continues to record, permanently:
       gate G-D: pass = false
       detail:   "max|RF_R0 - RF_R2| = 1.689e-01"
       dump_r2:  runs/eval_readout_v2_R2_pilot_v3/pair_logits.pt
       all_gates_pass: false
```

The original failed comparison stands in the record. It is not deleted, not
edited, not reinterpreted as a pass. The future rerun's record will name its
own `dump_r2` and cite this amendment.

### I.4 Why this is a registration correction, not a threshold relaxation

The distinction is the whole substance of §I, so it is stated plainly.

A **threshold relaxation** would change the predicate the gate evaluates —
replacing exact bit-identity with a tolerance, so that the *same* pair of
artifacts that previously failed would now pass. That is not what happens
here. The `1.689e-01` discrepancy between R0 and the historical R2 dump
still fails G-D, and will always fail G-D, under this amendment. Nothing
about that comparison is forgiven.

A **registration correction** supplies a specification the original
document omitted. Ladder §5 never named the artifact path its gate reads;
the original run chose the only R2 dump that existed at the time. A second,
equally valid full-population R2 dump now exists and satisfies the
unmodified criterion exactly. Naming which artifact the gate reads — in
advance of the rerun, in a dated document, with the original failure left
standing in the record — is filling a registration gap, not moving a bar.

The honest residual, recorded rather than argued away: the original ladder
prereg did not authorize *any* substitution, so this authorization is new,
and it is why the ladder rerun requires this amendment instead of
proceeding on its own. Had the matched dump not satisfied the criterion
exactly, no amendment of this kind would have been available.

### I.5 Registered confirmatory prediction

Stated in advance, as a check rather than a precondition: `R2c` (§F) will
produce its own dump, and that dump's `rel_feat` is predicted to be
**bit-identical to R0's** as well (grounds: §D.5 and the G-D matrix — P has
no path into `rel_feat`). If `R2c` satisfies G-D, it becomes an additional
admissible gate input under §I.2. If it does **not**, that is a new and
unexplained finding which must be documented before any ladder rerun
proceeds, and this amendment does not pre-authorize interpreting it.

The ladder rerun does **not** depend on `R2c` and must not be blocked
waiting for it (§J.5).

---

## J. Representation ladder rerun policy

Fixed in advance. The rerun's purpose is to satisfy the registered
gate/status contract, **not** to optimize scores.

1. **No new arm.** The arm set is exactly ladder §3's:
   `A1, A2, A3, A4, A5a, A5b, A6, N1, N2`.
2. **No `R5_residual`.** p37's group-centred `rel_feat` arm (0.5807)
   remains excluded, for the reason ladder §3 already registered: adding it
   after seeing p37's result would be an architecture choice made after
   seeing a number. This amendment does not reopen that exclusion.
3. **No hyperparameter change.** Ladder §4 is inherited verbatim: seed 0
   single; 5 image-level folds via
   `candidate_scorer_probe.fold_of_image(image_id, 5, salt=0)`; ridge
   `λ = 1e-4` closed form; MLP AdamW `lr = 2e-3`, `wd = 1e-4`,
   `batch = 4096`, `epochs = 25`; geometry LBFGS `max_iter = 200`,
   `strong_wolfe`, `λ = 1e-4`; z-score standardization over all validation
   rows plus bias column; WPRD `cap = 64`; bootstrap 2,000 resamples,
   seed 11. The `0.02` materiality threshold and the
   `BEYOND / EQUIVALENT / BELOW GEOMETRY` classification bands are
   inherited unchanged.
4. **No change to the frozen R0 input.** `dump_r0` remains
   `runs/eval_C1/pair_logits.pt`, unmodified. Only the **G-D gate input**
   changes, per §I.2.
5. **Independence.** The ladder rerun does not depend on `R2c` and may be
   registered and executed independently. Neither blocks the other. They
   address disjoint defects (§B.4) and must never be conflated in the
   record.
6. **Expected arm scores reproduce.** Because every arm consumes only
   `runs/eval_C1/pair_logits.pt`, the rerun is expected to reproduce the
   recorded numbers:

   ```
   A1_frozen_baseline      0.574988      A5a_geometry_xfit       0.588251
   A4_cosine_recomputed    0.575834      A5b_geometry_trainfit   0.596055
   A2_linear               0.580122      A6_fusion               0.585529
   A3_mlp                  0.585034      N1_shuffled_label_null  0.505459
                                         N2_prior_control        0.500000
   P* = A3_mlp = 0.585034     G = A5b = 0.596055     vs geometry: GEOMETRY_EQUIVALENT
   ```

   `A1, A4, A5a, A5b, A2, A6, N2` are deterministic (stored channels,
   closed-form ridge, LBFGS, prior). `A3_mlp` uses seeded AdamW on CPU and
   may differ in the last bits; a deviation there is a reproducibility
   observation to record, not a finding, and does not by itself invalidate
   the rerun. Any *material* deviation (> `0.02`, ladder §7's own
   operationalization) in any arm is an unregistered change in the input or
   the harness and must be diagnosed before the rerun is interpreted.
7. **Status on success.** If all gates including G-D pass under §I.2, the
   ladder's status moves from `CONDITIONAL / REQUIRES REGISTERED RERUN` to
   `CLEAN`, and its already-registered §7 interpretation rules apply
   unchanged. The recorded `GEOMETRY_EQUIVALENT` classification is **not**
   pre-blessed by this document; it stands or falls on the rerun's own gate
   outcome.
8. **Interpretation rules unchanged.** Ladder §6's binding rule on cell
   concentration and §8's five exclusions ("Not open-vocabulary", "Not a
   literature comparison", "Not a Readout v2 rescue", "Not a deployment
   candidate search", "Not causal") apply verbatim.

---

## K. Compute policy

Recorded for the future `R2c` run and the future ladder rerun. These are
the operator's standing execution rules for this programme, stated here so
the run's conditions are registered rather than ad hoc.

```
execution host:              direct SSH to the VM only
GPU check:                   nvidia-smi immediately before model construction,
                             output captured into the run's provenance record (§G)
fail-closed:                 unknown or busy GPU state aborts the run
Codex sandbox CUDA:          PROHIBITED
competing L4 workload:       PROHIBITED
parallel GPU workload:       PROHIBITED
concurrency:                 exactly one GPU job at a time
```

Additional registered conditions:

- `R2c` is a **single evaluation pass**, `eval_only=true`, `epochs=0`,
  `optimizer_steps` must be `0`. **No training occurs.**
- `R2c` writes to a **new** output directory
  (`runs/eval_readout_v2_R2c_<UTC timestamp>/`). It overwrites nothing.
- The ladder rerun is **CPU only** (ladder §10: both dumps cache `rel_feat`
  on disk, so no forward pass and no GPU is required). `nvidia-smi` is
  still checked per standing policy, but the ladder launches no GPU work.
- Every checkpoint is opened read-only. No checkpoint is written, moved,
  renamed, or re-saved by either run.
- Git HEAD and `git status --porcelain` are captured into each run's
  provenance record before execution.

---

## L. What this amendment does not do

- Does not execute, launch, or schedule any experiment.
- Does not read, compute, or report any corrected-P endpoint — none exists.
- Does not modify `docs/PAPER_C_READOUT_V2_PREREGISTRATION.md`,
  `docs/PAPER_C_REPRESENTATION_DECODER_LADDER_PREREGISTRATION.md`,
  `docs/PAPER_C_READOUT_V2_DECISION_GATE.md`,
  `docs/PAPER_C_R2_FALSIFICATION_BATTERY.md`, or
  `docs/PAPER_C_GD_FORENSIC_AUDIT.md`.
- Does not modify source code, the G-D runner, or any historical artifact.
- Does not weaken, rewrite, or promote the G-D gate or its
  `GD-UNRESOLVED` machine verdict.
- Does not relax any threshold, and does not introduce a new one.
- Does not make endpoint equality a validity criterion, in either direction
  (§F.3.3).
- Does not suppress, filter, or reinterpret the expected `predicate_prototypes`
  resume skip (§F.3.4).
- Does not edit `runs/paper_c_gd_gpu_20260909T151736Z.json` or promote its
  `GD-UNRESOLVED` machine status (§B.2.1).
- Does not authorize a full-budget Readout v2 run (§H.5).
- Does not authorize a third geometry seed, a seed-2 rerun, or any
  re-opening of the C1 geometry contract.
- Does not pre-decide `R2c`'s outcome or the ladder rerun's classification.

---

## M. Authorization required before execution

This document is a **draft**. It is uncommitted at the time of writing.
Before either run may execute, the following must be explicitly approved:

1. this amendment, as the registration of record for `R2c` and the ladder
   G-D gate input;
2. the §E reclassification of the historical R2 record's interpretive
   status;
3. the §F.2 S1 loading strategy and its flag contract;
4. the §I.2 G-D gate-input substitution;
5. the commit of this document (and, separately, of the S1 implementation
   plus the §H.3 tests) **before** any GPU work begins.

Until all five are approved, no corrected R2 run and no ladder rerun is
authorized by this document.

---

## N. Implementation as landed

Recorded here so this document is the verifiable registration of record. The
implementation is committed **together with** this amendment, before any GPU
work, exactly as section M.5 requires.

### N.1 Files

| file | change |
|---|---|
| `openvocab_rel/readout_v2_provenance.py` | **new.** S1 mechanism and section G record: `tensor_hash` (identical digest scheme to `tools/run_final_batch_gd_cuda.py::_tensor_hash`), `file_sha256`, `audit_skipped_keys`, `assert_no_unexpected_skips`, `extract_checkpoint_prototypes`, `restore_checkpoint_prototypes`, `p_vs_e_summary`, `write_provenance_record`, `ReadoutV2ProvenanceError`, and the registered constants |
| `openvocab_rel/config.py` | **+2 fields.** `readout_v2_prototype_source: str = "reinit_E"`, `readout_v2_expected_p_sha256: str = ""` |
| `openvocab_rel/train.py` | **+2 CLI args**; captures the resume block's `model_state` / `skipped_state` / loaded count into function scope (the resume path itself is byte-unmodified); S1 restoration after `init_readout_v2(E)`; writes `readout_v2_provenance.json`; replaces the misleading unconditional `init=E` log line with the actual provenance plus a named-key skip line |
| `tools/readout_v2_evaluate.py` | **+2 CLI args** (`--prototype_source`, `--expected_p_sha256`, the latter defaulting to the registered hash in checkpoint mode); passes both through to `train.main`; computes and records the `adaptive_equals_text` diagnostic; adds `prototype_source`, `expected_p_sha256`, `provenance_record`, `adaptive_equals_text` to `result.json` |
| `tests/test_readout_v2_treatment_fidelity.py` | **new.** The section H.3 registered tests |

`openvocab_rel/models/relational_model.py` is **unmodified** — no architecture
change was needed, which is itself the check that this is a load-path
correction (section F.1).

### N.2 Behaviour preservation

- The resume filter at `train.py:1689–1695` is **byte-unmodified**. Nothing was
  reordered. C0/C1 and every other experiment take exactly the path they took
  before.
- `reinit_E` is the default everywhere (`TrainConfig`, the argparser, and the
  evaluator), and on that path no restoration function is called at all: `P := E`
  as before.
- The one raise that could abort a run — the trainable-set assertion — fires
  **only** in `checkpoint` mode. The pre-amendment `reinit_E` path merely printed
  a count and still merely records; making it fatal there would have changed
  semantics for existing experiments, which this amendment forbids.
- The `reinit_E` path gains exactly one new artifact, `readout_v2_provenance.json`
  in the run's own `out_dir`. It is additive provenance and touches no
  computation, no existing output file, and no historical artifact.

### N.3 Validation performed (no CUDA)

```
python3 -m py_compile  openvocab_rel/train.py
                       openvocab_rel/models/relational_model.py
                       tools/readout_v2_evaluate.py
                       openvocab_rel/readout_v2_provenance.py
                       openvocab_rel/config.py
                       tests/test_readout_v2_treatment_fidelity.py     -> rc 0

pytest tests/test_readout_v2_treatment_fidelity.py                     -> 17 passed
pytest tests/test_readout_v2.py
       tests/test_readout_v2_offline_analysis.py
       tests/test_gd_runner.py
       tests/test_representation_decoder_ladder.py
       tests/test_readout_v2_treatment_fidelity.py                     -> see commit message

CUDA_VISIBLE_DEVICES=""  throughout. No GPU work of any kind.
```

The `adaptive_equals_text` diagnostic was additionally exercised read-only
against the archived `runs/eval_readout_v2_R2_matched_20260909T154739Z/pair_logits.pt`
and returned `True` — the correct answer for that `P := E` control, and a
direct demonstration that the diagnostic detects the historical defect's
signature while (per section F.3.3) never gating on it.

### N.4 What is still NOT done

No corrected R2 run exists. No ladder rerun exists. No GPU has been used. The
authorizations in section M remain outstanding.
