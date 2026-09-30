# Paper C E2 Preflight Report

## 1. Executive decision

**READY_FOR_HUMAN_GOLD**. The frozen 45-block packet passes the CPU-only identity, leakage, independence, and image-integrity checks. Human-dependent gates remain blocked until the three independent annotation files exist. The canonical train-derived M2 remains unavailable; the amended validation-held-out M2 is reconstructible and must be named `M2_valheldout`.

## 2. Frozen packet identity

```json
{
  "candidate_count": 45,
  "candidate_order_sha256": "aac94e821dd3a9c9770a5f89e62a18b2553cf66a71d189b5a86eecaf6aa0334e",
  "candidate_sha256": "ca53f3c0496e5323f408a6bc84d08c7991947ecae37d7d5aec2f7b2964523f80",
  "development_duplicate_images": [],
  "development_image_count": 90,
  "development_random_overlap": [],
  "frozen_candidate_sha256": "ca53f3c0496e5323f408a6bc84d08c7991947ecae37d7d5aec2f7b2964523f80",
  "frozen_image_manifest_sha256": "d17b1ec99841da0959f5a46c4c9cee5516b497144a06a12cd72c656a0aee819b",
  "frozen_order_sha256": "aac94e821dd3a9c9770a5f89e62a18b2553cf66a71d189b5a86eecaf6aa0334e",
  "image_decode_errors": [],
  "image_manifest_count": 170,
  "image_manifest_missing": 0,
  "image_manifest_sha256": "d17b1ec99841da0959f5a46c4c9cee5516b497144a06a12cd72c656a0aee819b",
  "random_candidate_count": 40,
  "random_image_count": 80,
  "status": "PASS"
}
```

## 3. Gate table

| Gate | Status | Evidence |
|---|---|---|
| G01 frozen population identity | **PASS** | candidate and image-manifest hashes, counts, image decode |
| G02 candidate generation model-blind | **PASS** | candidate fields, selection flow, generator AST |
| G03 image independence | **PASS** | block and cohort image IDs |
| G04 annotation independence | **PASS** | single-annotator server state and browser payload isolation audited |
| G05 schema consistency | **PASS** | validator compatibility rules present |
| G06 adjudication safety | **PASS** | adjudicator accepts only single-valid, visible, direction-verified truth |
| G07 human statistics | **BLOCKED_HUMAN_INPUT** | three annotation files absent |
| G08 nuisance training lineage | **PASS** | M3/M4/M5 scorer requires explicit train source |
| G09 M1 mapping | **PASS** | exact pair-slot and predicate-vocabulary lookup |
| G10 geometry leakage | **PASS** | M3 path is train artifact → frozen baseline → blocks |
| G11 random cohort model-blind | **PASS** | random candidate fields and selection metadata |
| G12 M2 forensic status known | **PASS** | AVAILABLE_CLEAN_VALIDATION_HELDOUT |
| G13 M6 options | **PASS** | no M6 run; interface-only preparation |
| G14 sample-size calculation | **PASS** | paired simulation artifact |
| G15 environment | **PASS** | environment.json |
| G16 claim matrix | **PASS** | claim-evidence matrix document |
| G17 failure modes | **PASS** | failure-mode matrix document |
| G18 provenance consistency | **PASS** | provenance reconciliation |

## 4. Model and lineage decisions

- M2 forensic status: `AVAILABLE_CLEAN_VALIDATION_HELDOUT`.
- M2 confirmatory name: `M2_valheldout` when the frozen fit artifact is supplied; this is not canonical train-derived generalization.
- M3/M4/M5 must fit from the explicit `datasets_vg150_clean/train.jsonl` source; the scorer has been repaired to require `--train`.
- M1 remains `M1_cached_C1`; fresh C1 inference is not required for current development.
- M6 is not run in this CPU-only preflight.

## 5. Human dependency

Annotation files are currently absent. The validator must report `ANNOTATION_PENDING`; no human truth, accepted gold, or scientific model interpretation has been fabricated.

## 6. Historical immutability

Accepted historical dumps and ladder results were read-only during this audit.

## 7. Required changes before scoring

1. Complete three independent annotation files and run validation.
2. Adjudicate only after validation is `VALID`; preserve exclusions.
3. Run human agreement and solvability gates.
4. Fit M3/M4/M5 from the declared train artifact and record its hash.
5. Supply the frozen `M2_valheldout` artifact to the scorer; do not call it train-derived.

## 8. Intentionally unchanged

The frozen candidate population, relation whitelist, historical results, historical predicate bytes, and no-GPU policy were not changed.
