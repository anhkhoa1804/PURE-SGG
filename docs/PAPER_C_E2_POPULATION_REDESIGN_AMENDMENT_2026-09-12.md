# Paper C E2 population redesign amendment

**Status:** additive development amendment; final E2 scoring is not authorized by
this document alone.

## 1. Reason for amendment

The original E2 population required exact ordered noun pairs, same coarse
relation family, and a tight caliper on all 19 box-geometry features. Cached
metadata feasibility auditing showed that this intersection is too sparse for
the intended non-spatial question. The support is also concentrated in
predicate pairs that are aliases, vague labels, or relations that can
co-exist. The original population is therefore rejected as the primary E2
population. Its feasibility results remain diagnostic and are not rewritten.

## 2. Scientific question and claim boundary

E2 asks whether a model follows a human-verified image-supported relation when
the ordered object labels are held fixed and measured box geometry is treated
as an explicit competing nuisance explanation. The target claim is controlled
image-conditioned relational discrimination on the audited population. E2 does
not establish general semantic understanding, causal grounding, geometry
invariance, open-vocabulary understanding, or globally unseen predicate
knowledge.

## 3. Primary population

The primary population consists of two-image contrast blocks constructed from
the cached C1 validation population. Candidate generation uses only image IDs,
object labels, predicate annotations, boxes, the frozen relation-pair whitelist,
and declared random seeds. It cannot import or inspect model scores, decoder
errors, WPRD outputs, or VLM outputs.

Each candidate block has identical ordered subject/object labels in two distinct
images. The images carry distinct predicates from one frozen relation-pair
stratum. Same-family membership is not required; relation-pair strata are
reported separately. `wearing`/`wears`, `has`/`wearing`, `has`/`wears`, pairs
involving vague `with`, and other alias-like pairs are excluded unless a later
additive amendment explicitly admits them after development annotation.

The current development whitelist is:

* `holding` vs `looking at`;
* `holding` vs `using`;
* `holding` vs `riding`;
* `holding` vs `playing`;
* `carrying` vs `riding`;
* `riding` vs `using`.

This whitelist is a candidate-generation policy, not semantic ground truth.
Every candidate requires three independent human annotations and adjudication.
Candidates with multi-valid, neither-visible, uncertain, directionally unclear,
or otherwise ambiguous truth are excluded from the primary population.

## 4. Geometry role

The full registered 19-feature geometry vector is retained for every candidate
and every accepted block. It is not a hard primary inclusion gate. A frozen
geometry-only baseline is fit on non-final images and scored on the blocks.
The primary claim is therefore conditional on measured geometry being modeled
as a nuisance baseline; the protocol must not say that geometry was removed.

A tight geometry-matched subset may be reported as a secondary sensitivity
stratum only if its support and annotation gates pass. A secondary
geometry-conflict stratum may be selected using a geometry baseline frozen
before focal-model scoring. Neither stratum may be selected using focal-model
errors.

## 5. Estimand and endpoint

The primary unit is an independently frozen two-image block. For model `m`,
`S_m(b)` is one when both image-description assignments are correct and zero
otherwise. The primary endpoint is mean non-spatial both-correct block accuracy.
Primary effects compare the focal visual model with the frozen geometry-only
and object-only baselines using paired block-level inference. Language/frequency
and cached C1 readout are diagnostic baselines.

## 6. Development and final separation

The first run is a development instrument check of approximately 50 candidate
blocks. It estimates annotation agreement, human solvability, relation-pair
yield, geometry residuals, baseline interface correctness, and paired
discordance for final sample-size planning. Development results do not authorize
final test scoring and may not be used to select final examples by model
performance.

The final block count is frozen only after development-only precision and power
calculations. Final blocks are selected, annotated, adjudicated, hashed, and
frozen before focal-model inference.

## 7. Unchanged controls

The following remain mandatory: exact ordered object labels, distinct images,
human truth validation, direction handling, multi-label exclusion, no image
reuse in the final primary set, model-blind selection, full geometry reporting,
object-only and geometry-only baselines, a blind language/frequency baseline,
at least one independent visual positive control, block-level inference, and
complete artifact provenance.

## 8. M1 and M2 lineage

M1 may use the stored C1 text logits in the accepted pair dump when the block
uses the stored predicate vocabulary. It must be labeled `cached C1 readout`;
this is not fresh C1 image inference. M2 must be a fresh held-out decoder
probe. Reusing A3 predictions is prohibited. If cached `rel_feat` is used,
the decoder is fit on non-final image IDs and the lineage is recorded as a
validation-population held-out probe, not as a canonical train-derived model.

## 9. Amendment scope

This amendment changes only the prospective E2 population, geometry role,
relation-pair policy, and development/final separation. It does not modify the
historical C1/R0, R2c, ladder, exploratory results, predicate bytes, or their
manifests.
