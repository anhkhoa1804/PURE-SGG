# Paper C E2 M2 lineage amendment — 2026-09-15

Status: additive prospective amendment. This document does not alter the
historical C1, R2/R2c, representation-ladder, exploratory, or predicate
artifacts.

## 1. Reason for amendment

The earlier E2 audit correctly established that no canonical train-split
`rel_feat` artifact, C1 checkpoint, or serialized train-derived decoder is
available locally. It therefore classified a train-derived M2 as unavailable.
That conclusion was too broad for the cached validation dump. The accepted C1
dump itself contains image IDs, ordered pair slots, `rel_feat`, GT predicate
labels, and the 51-column vocabulary for all 10,401 cached validation images.

The frozen E2 packet contains 170 unique development and random-cohort image
IDs. All 170 occur in the cached dump. Excluding those IDs leaves 10,231
images and 129,826 labeled 768-dimensional representation rows. Excluding
the random cohort as well ensures that the same artifact remains out-of-sample
if that sensitivity population is scored later. No human annotation, accepted
status, E2 gold, nuisance score, or focal-model score is used in constructing
this fit partition.

## 2. Corrected lineage classification

| M2 interpretation | Status | Allowed use |
|---|---|---|
| canonical train-derived M2 | `UNAVAILABLE` | cannot be claimed or reported |
| validation-population held-out M2 | `AVAILABLE_CLEAN` | prospective E2 focal probe after gold freeze |
| reuse of accepted A3 predictions | `PROHIBITED` | never use |

The available model is named **`M2_valheldout`**. It is fit from the cached
C1 validation representation after excluding every image in the frozen E2
packet, including the random cohort, regardless of whether that image later
survives human adjudication. This exclusion is fixed before human truth and
prevents selection feedback.

## 3. Fitting protocol

The decoder is the predeclared A3-capacity network:

`Linear(768 -> 768) -> GELU -> Linear(768 -> 51)`

with seed `0`, AdamW learning rate `0.002`, weight decay `1e-4`, batch size
`4096`, and `25` epochs. Feature mean and standard deviation are fit only on
the 129,826 non-E2 rows. The 51st background column remains an output column;
the cached GT targets are the 50 foreground predicates. No architecture
search or result-dependent tuning is permitted.

Duplicate ordered pair slots are resolved only when all available cached
channels are exactly identical. A non-identical duplicate is a hard error.
The audit found 4,014 duplicate pair keys and verified identical cached
`rel_feat`, text, model, prior, and classifier channels across them. All 90
frozen development candidate sides resolve under this rule.

## 4. Confirmatory comparison

The primary confirmatory comparisons are amended to:

* `M2_valheldout` versus `M3_geometry`;
* `M2_valheldout` versus `M4_object`.

They retain the block-level both-correct endpoint, paired block inference, and
Holm correction over these two comparisons. M1 remains `M1_cached_C1`, a
descriptive cached-readout diagnostic; M6 remains an independent visual
positive control.

The estimand is conditional on the cached validation population and on the
frozen E2 block construction. It asks whether the cached C1 relational
representation supports a fresh held-out closed-set decoder that exceeds the
declared nuisance baselines on human-accepted blocks. It does not estimate
train-to-validation generalization, semantic transfer, causal grounding,
open-vocabulary understanding, or task-unseen performance.

## 5. Required artifacts and safeguards

The fit artifact must record the exact C1 dump hash, candidate-packet hash,
excluded image-ID hash, feature standardization, decoder hyperparameters,
fit-row count, and the fact that human gold and model scores were unused. It
must be frozen before scoring accepted E2 blocks. The current CPU-only fit is
stored under `runs/e2_preflight_20260915/` as a pre-scoring artifact; it does
not constitute an E2 result.

## 6. Unchanged rules

The E2 candidate population, relation whitelist, image identity, model-blind
selection, human truth requirement, no-image-reuse rule, M3/M4/M5 train-jsonl
lineage, primary endpoint, historical artifacts, and no-GPU policy are
unchanged.
