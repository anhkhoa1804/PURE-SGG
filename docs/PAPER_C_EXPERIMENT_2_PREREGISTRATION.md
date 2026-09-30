# Paper C Experiment 2 — controlled image-conditioned relational discrimination

**Status: DRAFT — NOT REGISTERED.** This document specifies the development and
final-test protocol. It authorizes no final model scoring until the development
gates are passed, the final block manifest is frozen and hashed, and this file
is amended with the frozen development outputs. It does not modify historical
ladder, R2c, C1, R0, or exploratory evidence.

## 1. Question and claim boundary

The question is whether a model follows the visually supported relation when
the ordered object names and measured box-geometry cues are approximately
matched, with the primary test restricted to human-valid non-spatial relation
contrasts.

The target claim is **controlled image-conditioned relational discrimination
beyond specified object and measured box-geometry baselines**. Success does not
establish general semantic understanding, geometry invariance, globally unseen
concept knowledge, or open-vocabulary understanding.

## 2. Estimand and population

The unit of inference is an independently frozen two-image contrast block. No
image may occur in more than one final block. Let `S_m(b)` be one when model `m`
correctly assigns both descriptions to the two images in block `b`, and zero
otherwise. For a predeclared final population `P` and nuisance baseline `q`,

`Delta(m,q) = mean_{b in P} [S_m(b) - S_q(b)]`.

The primary population is the fixed set of human-valid, non-spatial,
geometry-matched blocks selected without model inference. The primary focal
model is M2, a fresh task-seen closed-set decoder fitted without final-test
images. The two confirmatory comparisons are M2 versus M3 (geometry-only) and
M2 versus M4 (object-only), with Holm correction. M1 and M6 are prespecified
diagnostic and positive-control models. `Delta` is a paired predictive
contrast, not a causal treatment effect and not an absolute semanticity score.

## 3. Relation families

The candidate list is fixed before final selection.

* **Primary non-spatial:** contact/possession/action relations such as
  `holding`, `carrying`, `touching` if present and reliable, `wearing`/`wears`,
  `looking at`/`watching`, `riding`, `attached to`, and `mounted on`. A label is
  admitted only if its direction and visible truth are reliable in development.
* **Positive spatial/topological control:** `above`, `below`/`under`, `left` or
  `right` when present, `in`, `overlap`/`intersect` when present. These are not
  pooled into the primary non-spatial endpoint.
* **Excluded from the primary analysis:** vague or highly polysemous labels
  such as `near`, `next to`, `of`, `and`, `at`, `with`, `for`, and any label
  failing the agreement, support, or direction gates. Mixed relations may be
  reported descriptively only.

The final family list, alias map, and direction map are frozen before final
model inference. A family cannot be admitted because of model performance.

## 4. Candidate-pool construction

Candidate enumeration uses only the transferred image/annotation source,
canonical object labels, predicate annotations, box metadata, declared random
seeds, and the rules in this document. It must not read C1, A3, geometry,
VLM, WPRD, or any candidate-model score. Candidate records include image ID,
source ID, ordered subject/object instance IDs, canonical ordered object labels,
predicate label, relation family, direction, boxes, image dimensions, and
source provenance.

An admissible candidate image has a visible ordered object pair, one candidate
relation label, and no unresolved duplicate/near-duplicate or source-leakage
flag. A pair of candidates can form a block only when the ordered canonical
object labels are identical, the images are distinct, the relation labels are
distinct members of the same predeclared family, and independent annotation
finds exactly one valid relation per image.

## 5. Contrast block

Each block contains images `I_A`, `I_B`, ordered object labels `(N_s,N_o)`,
relations `R_A != R_B`, and descriptions `D_A,D_B`. The canonical model text
template is fixed as:

`The relation of the highlighted subject to the highlighted object is: {R}.`

The only intended text difference is `{R}`. Nouns, order, punctuation, and
template are identical. A model receives the same subject/object order for
both images. The block score matrix is:

```text
                 D_A       D_B
I_A              score_AA  score_AB
I_B              score_BA  score_BB
```

The diagonal is correct. `S_m=1` iff `score_AA > score_AB` and
`score_BB > score_BA`; exact ties are incorrect. The image order and relation
assignment are randomized by a frozen seed. A two-description block has chance
both-correct accuracy `0.25` under independent random binary choices.

The primary block must have unanimous single-label human truth, no
both-valid/neither-visible/uncertain response, verified direction, and pass the
geometry caliper. A block with two genuinely valid relations is retained in an
exclusion log and may be reported as a multi-label sensitivity set; it is not
forced into the binary primary endpoint.

## 6. Human annotation

Three independent annotators inspect each candidate image-relation item. They
receive the image, highlighted subject and object, their identifiers, the
candidate relation, and the choices `valid`, `invalid`, `both valid`,
`neither visible`, and `uncertain`, plus relation direction and confidence on a
1–5 scale. Annotators do not see model scores, model names, or geometry-model
outputs.

The primary truth rule is unanimous `valid` for one candidate relation and
unanimous rejection of the competing relation, with no uncertainty and
direction agreement. Disagreements are adjudicated by a senior annotator for
classification into primary, multi-valid, ambiguous, or excluded. Adjudication
cannot promote a disagreement into the primary set unless the raw annotations
meet the predeclared rule; otherwise it remains excluded.

Agreement is reported with Krippendorff's nominal alpha and the unanimous-item
fraction. The agreement gate is alpha at least 0.80 overall, at least 0.70 in
each admitted primary family, and at least 90% unanimous primary truth items.

A separate human solvability panel of three annotators performs the final 2x2
matching task on a blinded sample after blocks are frozen. Human both-correct
accuracy must be at least 0.90 and its two-sided 95% block-bootstrap lower
bound at least 0.80 overall; each primary family must have a point estimate at
least 0.80. This validates task solvability, not machine capability.

## 7. Geometry matching

The registered 19-feature representation from `tools/wprd_geometry_control.py`
is used without alteration: normalized subject/object centers, relative
offsets, image-scale offsets, normalized widths/heights, log area ratio,
log-aspect ratios, IoU, subject/object containment, and normalized center
distance. The implementation's scene-box normalization is retained and
recorded. These features control measured box geometry only; they do not
control depth, pose, texture, visible parts, viewpoint, context, or occlusion.

Feature means and standard deviations are fit on the development candidate
pool only, with standard deviations clamped at `1e-6`, then frozen. For a
candidate pair, let `z` be the standardized 19-vector difference. The default
final caliper is `max(abs(z)) <= 0.50` and
`sqrt(mean(z*z)) <= 0.35`. These thresholds may be rejected for insufficient
development support, but they may not be loosened after final model scores;
any replacement thresholds must be frozen in an additive amendment before
final block construction.

Within each `(ordered nouns, family, relation pair)` stratum, matching is
maximum-cardinality one-to-one bipartite matching subject to the caliper, with
minimum total standardized distance as the tie-break. Image IDs provide the
final deterministic tie-break. No image is reused across final blocks. The
matcher does not inspect model outputs.

For every accepted block, save raw 19-feature distance, standardized RMS
distance, largest absolute standardized residual, caliper status, and cosine
similarity of standardized feature vectors. Report median, IQR, 90th, 95th
percentile, and maximum. Poor matches are excluded only by the frozen caliper
before model inference and appear in the exclusion log.

A geometry-conflict secondary stratum may be constructed only with a baseline
frozen before final block selection, or by model-independent rules. It may not
be selected because a geometry model fails on it and is not part of the
primary endpoint.

## 8. Random cohort

For every final family/noun/relation-pair quota, sample a same-source-pool
comparison block with identical ordered noun and family quotas but without the
geometry match constraint. Sampling is without replacement where feasible and
uses a declared seed. No model scores enter sampling. The random cohort is not
pooled with the primary endpoint. It is used to report selection effects in
object labels, relation families, geometry distances, source IDs, object size,
occlusion/context proxies, and human confidence. Absolute standardized mean
differences above 0.10 are flagged; failure of this audit narrows the claim to
the curated conditional population.

## 9. Models and lineage

* **M1:** exact C1 text readout, task-seen, frozen checkpoint and prompt.
* **M2:** fresh 51-output supervised closed-set decoder fit on declared
  training/development images only; final images are never used for fitting,
  scaling, checkpoint choice, or prompt choice. It must be freshly fit for E2;
  A3 predictions cannot be reused.
* **M3:** 19-feature geometry-only model using the same training-only fitting
  rule and frozen standardization.
* **M4:** object-only empirical relation model fitted on training labels with a
  fixed backoff hierarchy chosen before scoring: ordered noun pair, subject
  noun, object noun, global predicate frequency.
* **M5:** blind language/relation-frequency baseline that scores descriptions
  without image input using only frozen training predicate frequency and fixed
  template/length features. A text-only language model is optional and
  secondary.
* **M6:** one frozen independent visual relation model or VLM with an exact
  checkpoint, prompt, preprocessing, and scoring interface fixed before test
  inference. It is a positive model control, not an oracle.

For every model record foundation checkpoint and hash, pretraining source when
known, task-training data, predicate and alias exposure, image exposure,
checkpoint ancestry, text encoder, prompt, preprocessing, seed, optimizer and
epochs where applicable. Terminology is `task-seen`, `task-unseen`, or
`globally unknown`; E2 is task-seen and must not be called unseen-predicate or
open-vocabulary evaluation.

## 10. Endpoints and analysis

The single primary endpoint is pooled non-spatial both-correct block accuracy.
Secondary endpoints are: (1) per-image 2AFC accuracy, (2) mean signed diagonal
margin, (3) family-macro block accuracy including the spatial positive control,
(4) paraphrase consistency on a frozen secondary phrase bank, and (5) matched
versus random-cohort accuracy. Full-bank retrieval metrics are exploratory.

For each primary comparison report the paired block difference, exact block
counts, two-sided 95% confidence interval, and paired randomization p-value.
The primary null is `Delta <= 0`. Use 5,000 stratified block-bootstrap
resamples with seed 11, preserving the registered family composition. Use
paired label-swapping/randomization within blocks for p-values. Apply Holm
correction to the two M2 nuisance comparisons at familywise alpha `0.05`.

The practical margin is an absolute five percentage points, fixed before final
scoring. A meaningful focal result requires point estimate at least `0.05` and
the multiplicity-adjusted confidence interval to exclude zero. An upper bound
below `0.05` is evidence against a material gain; intervals spanning the margin
are inconclusive, never evidence of equivalence.

The final sample size is selected from development estimates using a frozen
paired-Bernoulli simulation: 100,000 simulations, seed 20260911, smallest
multiple of 25 that gives at least 80% power to detect a 0.05 paired gain and
expected 95% CI half-width at most 0.05, subject to at least 100 blocks in
each admitted non-spatial family. If no design within the predeclared
annotation budget satisfies this, the claim is narrowed or the experiment is
stopped; alpha and the margin are not weakened.

## 11. Nulls and permutations

| Operation | Question | Does not prove |
|---|---|---|
| swap descriptions within a block | does the score use image-description association? | semantic understanding |
| swap images within a block | does the score follow image identity? | causal visual grounding |
| permute relation phrases across blocks | does performance depend on the relation query rather than frequency? | compositional transfer |
| permute image features across blocks | is the visual input necessary? | absence of all visual shortcuts |
| randomize image/description orientation | does position/order create bias? | semanticity |
| geometry shuffle or masking | sensitivity to measured geometry | geometry causality or removal |
| blind language baseline | can text artifacts solve the task? | visual failure mechanism |

All permutation schemes and seeds are frozen before final scoring. They are
diagnostics for the named nulls, not generic causal tests.

## 12. Go/no-go gates

1. **G1 Human solvability:** human panel thresholds above pass overall and by
   family.
2. **G2 Agreement:** alpha, unanimous fraction, and direction rules pass.
3. **G3 Match validity:** every primary block passes the frozen caliper and no
   image is reused.
4. **G4 Cohort audit:** random cohort is generated and its deviations are
   reported; broad claims require SMD <= 0.10 on predeclared covariates.
5. **G5 Positive model:** M6 exceeds chance 0.25 with a 95% lower bound above
   0.25 in at least one primary family; otherwise the instrument is not
   machine-validated.
6. **G6 Leakage:** no final image or source duplicate enters fitting, tuning,
   prompts, or checkpoint selection.
7. **G7 Support:** at least 100 final blocks in each of at least three admitted
   non-spatial families, unless the claim is explicitly narrowed.
8. **G8 Precision:** the frozen sample-size simulation passes.
9. **G9 Freeze:** final blocks, prompts, checkpoints, and configs are hashed
   before candidate-model inference.
10. **G10 Lineage:** all model and data provenance fields are complete.

Failure of G1, G2, G3, G5, G6, G8, G9, or G10 is a final-scoring no-go.
G4 or G7 failure permits only a narrower, explicitly conditional report after
review; it does not license a broad semantic claim.

## 13. Freeze and provenance package

The run must save `candidate_pool.jsonl`, raw and adjudicated annotations,
`final_blocks.jsonl`, `random_cohort.jsonl`, `geometry_match_report.json`,
`selection_flow.json`, `exclusion_log.jsonl`, `model_lineage.json`, frozen
prompt/alias maps, `model_scores.parquet`, `primary_results.json`,
`secondary_results.json`, `bootstrap_results.json`, `permutation_results.json`,
`environment.json`, `analysis_config.json`, `manifest_sha256.txt`, and a
README. Raw annotation records are retained under the applicable privacy
policy. Every file is hashed. The final manifest is generated, reviewed, and
hashed before any test-model inference.

The environment record includes git HEAD/status, branch, command, timestamps,
Python/PyTorch, CPU, CUDA availability, GPU `nvidia-smi` output if any, all
input/checkpoint hashes, seeds, and software versions. Any GPU use requires a
fresh `nvidia-smi` immediately before launch and records active processes and
memory.

## 14. Registration and amendments

This file becomes registered only after the development set fixes the admitted
family list, matching feasibility, final N, and any permitted threshold
amendment. The final test manifest is then frozen. No model score may alter
selection, matching, annotation, family assignment, or block membership. Any
change after freeze requires a dated additive amendment and preserves all prior
files.

