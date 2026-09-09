# Paper C — G-D forensic audit

This is an additive audit note. It does not rewrite the R0/R2 checkpoints,
dumps, preregistration, ladder result, or the locked R2 decision.

## Verdict

`GD-UNRESOLVED`

The serialized R0 and R2 backbones are identical, and the prototype tensor
has no source-level path into `rel_feat`. Nevertheless, the two completed
evaluation dumps are not bit-identical: the discrepancy is real, localized
to the final incomplete evaluation batch, and its immediate mechanism has
not been reproduced from source alone.

## Evidence

Existing artifacts were read-only. The new machine-readable records are:

- `runs/paper_c_gd_checkpoint_forensic.json`
- `runs/paper_c_gd_forensic.json`

Checkpoint comparison:

- R0 model: 251 tensors; R2 model: the same 251 tensors plus
  `predicate_prototypes`.
- All 251 common model tensors match by streaming SHA-256 digest.
- All 590 CLIP tensors match by streaming SHA-256 digest.
- No model or CLIP buffer/parameter difference exists outside the new
  prototype tensor.

Dump comparison:

- Population identity: `EXACT_MATCH` (10,401 images, identical order;
  identical pair rows, labels, GT predicates, and WPRD cell structure).
- `rel_feat`: 9 of 10,401 image entries changed; 65 pair rows in total;
  maximum absolute difference `0.1689453125`.
- All nine changed images are indices 10,392–10,400, the final batch
  (batch index 866 at batch size 12).
- `text_logits`, `model_logits`, and `cls_logits` change only on those same
  nine entries; `prior_rows` are unchanged.
- `pred_emb` is exactly equal between dumps.

## Source dependency audit

The R2 setup is after checkpoint loading in `openvocab_rel/train.py:1684–1730`.
It freezes existing parameters, creates P as a clone of E, and adds P to the
optimizer. Evaluation then explicitly calls `model.eval()` and
`clip_model.eval()` at `openvocab_rel/train.py:1820–1823`.

The representation path indexes object features, computes geometry, and calls
the decoder at `openvocab_rel/models/relational_model.py:727–763`. The
prototype is not read there. The only P-dependent operation is
`adaptive_predicate_logits()` at `openvocab_rel/models/relational_model.py:881–888`,
which is called after `rel_feat` has already been produced by
`openvocab_rel/evals.py:2090–2095`.

The CPU regression `test_forward_from_featmap_unaffected_by_readout_v2` also
passes with identical fixed inputs and common weights. This proves the
intended dependency separation in the tested CPU execution path, but it does
not prove that the historical CUDA evaluations were numerically identical.

## Ranked hypotheses

1. CUDA execution or incomplete-batch numerical/runtime behavior is currently
   the leading hypothesis. The discrepancy is confined to the final batch,
   while model state, CLIP state, configuration, population, and prior rows
   match. The recorded evaluations used CUDA autocast and did not record a
   deterministic-algorithm guarantee.
2. An evaluation-time side effect associated with the R2-only setup remains
   possible, including the extra CLIP text encoding or allocator/runtime
   state. The source-level P-to-`rel_feat` graph does not support a direct
   causal path, and the CPU isolation test argues against one, but the
   historical CUDA path has not been replayed under instrumentation.
3. Checkpoint mutation, backbone training, row misalignment, or a broad
   preprocessing/configuration mismatch are not supported: tensor hashes,
   exact population identity, recorded configs, and unchanged prior rows rule
   out these explanations at the available evidence level. Pixel tensors were
   not serialized, so preprocessing cannot be ruled out absolutely.

## Scientific disposition

The preregistered representation decoder ladder has G-D marked FAIL. Under
its own stopping rule, the ladder is not a valid all-gates-passing registered
experiment; its arm scores should be retained as conditional diagnostics, not
as a clean decoder-ladder conclusion.

The R2 endpoint NULL remains valid as a result of the separately locked R2
decision protocol: R2 did not produce a statistically meaningful WPRD gain,
and its integrity/population/control gates passed. The stronger statement
that R2 was a strictly same-`rel_feat`, pure-readout intervention is not
currently defensible without resolving G-D.

## Required next action

Do not rerun or retrain R2 automatically. The next authorized forensic action
should be a tiny, separately logged reproduction of the final batch under the
same two evaluator constructions, capturing input pixel hashes, RNG state,
module train/eval flags, parameter/buffer fingerprints immediately before
forward, and deterministic-vs-original CUDA settings. Only after that audit
identifies the source should the ladder be amended or rerun under a new
registration.

## Future bounded GPU protocol clarification

The original four-condition sketch is not by itself sufficient to separate
all remaining explanations. It has no same-condition repeat for
nondeterminism and no comparison of the nine images as a 9-image batch against
the same images in a 12-slot padded batch. The additive protocol is defined
in `tools/gd_forensic_protocol.py`.

The minimum added controls are:

- repeat the R0/no-prototype baseline and the R2/prototype-registered path;
- run the R0 baseline and R2 registered path with the same nine examples
  padded to a 12-image batch by duplicating three already-selected examples;
- record an explicit R2 condition in which the checkpoint's
  `predicate_prototypes` tensor is restored after registration.

The last control is a provenance guard, not a new scientific arm. The current
`openvocab_rel/train.py` load order resumes the checkpoint before calling
`init_readout_v2()`. Consequently, a checkpoint key named
`predicate_prototypes` is not present in the model state dict at resume time
and can be reported as skipped; the subsequent registration initializes P
from the freshly encoded E. The future run must record this fact rather than
silently treating checkpoint P as loaded.

Each condition must record input pixel hashes, parameter and buffer hashes,
module modes, prototype-read events, RNG states, CUDA/autocast/TF32/cuDNN
settings, full error summaries (changed fraction, absolute/relative max and
mean, RMS), downstream logits, and WPRD computed on the final 65 rows. The
locked decision function is conservative: incomplete reproduction remains
`GD-UNRESOLVED`; `GD-EXPLAINED-INERT` additionally requires a demonstrated
cause and an unchanged scientific endpoint.

## Source-level runner audit (2026-09-09; no CUDA execution)

Status remains `GD-UNRESOLVED`. This note supersedes the earlier claim that
the first runner implementation already implemented all advertised semantics.
No R2 conclusion, preregistration, checkpoint, or existing result is edited.

The audited runner is `tools/run_final_batch_gd_cuda.py`. The nine condition
names are retained, with these explicit prototype semantics:

| Condition family | Checkpoint | P before forward | Restored checkpoint P? | P trainable? |
| --- | --- | --- | --- | --- |
| X (including repeat/padding) | C1 | absent | no | n/a |
| Y | C1 | registration only, clone of cached R0 E | no | yes |
| Q | R2 | absent; saved P is skipped on resume | no | n/a |
| Z (including repeat/padding) | R2 | initialized from freshly encoded E after resume | no | yes |
| Zp | R2 | initialized from E, then overwritten from saved P | yes, exact shape/dtype/hash required | yes |

Y previously enabled the whole R2 setup, conflating registration with extra
text encoding, freezing, and the nonpersistent anchor buffer. It now runs
the R0 construction and registers only P after load; existing parameter
trainability is unchanged, no anchor is added, and no new text forward is
introduced. Cached E is a documented input to this registration control.
Z retains the actual current R2 construction; Zp restores saved P inside the
registration hook before evaluation. `checkpoint_P_hash`, `setup.installed_P`,
and before/after parameter fingerprints record the actual tensors. The load
hook records the exact loaded/skipped keys and rejects unexpected omissions.

All conditions execute zero optimizer steps (AdamW.step is blocked). Here
"adaptation bypassed" means no parameter optimization; adaptive **scoring**
is allowed after the captured relational forward. A temporary attribute
guard raises on P/anchor access within that forward, and the adaptive method
logs its ordering after `rel_feat_returned`. Source inspection of
`RelationalModel._forward_impl`, `forward_from_featmap`, and
`evals._predicate_logit_components` supplies the graph context. The guard
does not claim to detect arbitrary future access via `_parameters` or a
previously retained alias; such source changes require re-audit.

Each condition now execs a separate Python interpreter. Repeats independently
construct and resume the model/CLIP, reset seed 1234, and execute one new
relational forward; CUDA allocator/autotuning state from another condition
cannot carry across the process boundary. Only output paths/run labels are
excluded from repeat-control equality checks. Every child rechecks
`nvidia-smi` before model construction; unknown/busy GPU state fails closed.

Input selection is asserted against these source indices and IDs:

| Index | Image ID | Positive rows |
| --- | --- | --- |
| 10392 | 2385790 | 1 |
| 10393 | 2353744 | 2 |
| 10394 | 2390749 | 7 |
| 10395 | 2343247 | 12 |
| 10396 | 2405812 | 9 |
| 10397 | 2390381 | 6 |
| 10398 | 2335326 | 10 |
| 10399 | 2406736 | 11 |
| 10400 | 2361448 | 7 |

The validation dataset is sliced at original indices 10392:10401 before
iteration. With drop_last=False and loader capacity 12, its nine records
produce one batch of 9. Padded conditions append copies of slots 0, 1, 2
(indices 10392, 10393, 10394), producing one batch of 12. Identity, counts,
duplicate pixels, source indices, box tensors and pair indices are recorded
or asserted at preparation. Raw GT annotations are also checked against both
historical dumps. Whole-batch and per-image hashes retain dtype/shape metadata.
Historical pixel identity is explicitly `UNPROVEN_FROM_HISTORICAL_DUMPS`.

The old blanket `v[:9]` operation truncated global lists including pred_vocab
from 51 to 9. It is replaced by an explicit image-field whitelist; vocabulary,
background indices, alias map and predicate embeddings remain intact, and
trimmed counts are recomputed as 9 images/65 rows. Padding never enters WPRD.
WPRD reuses `Mech` normalization across all 51 columns **before** foreground
selection and the existing `Groups/wprd` estimator (cap=64, seed=0), reporting
cell values/weights as well as macros. Zero-cell cases are undefined/null.
An unchanged local-65-row WPRD is **not** proof of full registered endpoint
inertness: those rows can contrast with earlier rows in the full population.

The old fast-eval path both altered settings and ran an extra grounding pass.
The runner now preserves the corresponding R0/R2 wrapper diagnostic patches,
uses full SGG settings, calls only `eval_sgg_standard(max_batches=1)`, and
exits train.main at the core-evaluation boundary. A second relational forward
raises. This is a bounded replay of current source with num_workers=0, not a
recreation of the historical preceding 866-batch allocator/runtime history.

Before/after snapshots include all named parameters (including requires_grad),
all named buffers including nonpersistent buffers, every module mode, and
unregistered tensor/list/dict/scalar attributes. Model source has LayerNorm
and dropout; eval mode disables dropout. Mutable diagnostic attributes include
`last_routing_attention`, `last_deformable_points`, `last_vector_gate`,
`last_gate_reg`, `last_gate_val`, and `_last_calibration_reg`; these are
captured separately from buffers. The source writes the decoder diagnostics
during forward; it does not feed their cached values into the next rel_feat.
Unknown Python attribute types are marked uninspected, not silently called
equal in scientific interpretation. Backend/allocator state remains outside
module fingerprints. Instrumentation itself introduces synchronization/host
copies and therefore cannot certify historical runtime-state identity.

Tensor hashes now support scalar counters and bfloat16 without casting.
NumPy RNG records contain the entire state array instead of truncated repr;
Python, CPU Torch and CUDA Torch states are recorded before/after, alongside
runtime/autocast/TF32/cuDNN flags. Native rel_feat is retained in each worker's
record and in `rel_feat_native.pt`; all 36 pairwise contrasts and both repeat
contrasts use those actual tensors. Error reports include changed rows/elements,
fraction, relative error with an explicit denominator floor, absolute max/mean,
RMS and quantiles. Historical fp16 dump conversion is kept separately.

The output always states `automatic_promotion_enabled=false`,
`human_interpretation_required=true`, and that the false decision flags are
unassessed placeholders. These flags are not a data-driven classification.
Local equality, batch-size sensitivity, or stable repeats alone cannot
promote the scientific status. The existing forensic JSON files are untouched.
