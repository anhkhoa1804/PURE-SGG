# M2 reconstruction audit

## Decision

`M2_valheldout` is reconstructible cleanly from the accepted C1 cache. A
canonical train-derived M2 is not available because no train-split `rel_feat`
artifact or exact C1 checkpoint exists locally.

## Evidence

| Check | Result |
|---|---:|
| Cached C1 images | 10,401 |
| Frozen E2 images excluded before fitting | 170 |
| Non-E2 fitting images | 10,231 |
| Non-E2 fitting rows | 129,826 |
| Feature width | 768 |
| Non-finite feature rows | 0 |
| Missing target vocabulary labels | 0 |
| Candidate blocks checked | 45 |
| Candidate image sides resolved | 90 / 90 |
| Duplicate pair keys | 4,014 |
| Duplicate channels verified identical | yes |
| Human gold used | no |
| Model scores used for selection | no |

The source dump is
`runs/eval_C1/pair_logits.pt`, SHA256
`2543c87512ba6b85e4bc8a629f4617ffd0fe57fccafce91325f636e0a34a4453`.
The frozen development candidate SHA256 is
`ca53f3c0496e5323f408a6bc84d08c7991947ecae37d7d5aec2f7b2964523f80`.
The frozen random-cohort SHA256 is
`3869aa1ed8f3c7bd7896645fef98cc5276d089f2002f69f2dbe0010c5259a090`.

## Interpretation

The fitted decoder is a held-out probe within the cached validation
population. It tests whether a fresh closed-set decoder can use `rel_feat` on
human-accepted E2 blocks beyond the declared geometry and object nuisance
baselines. It does not establish train-to-validation generalization, semantic
understanding, causal grounding, open-vocabulary transfer, or task-unseen
generalization.

The frozen fit artifact is
`runs/e2_preflight_20260915/m2_validation_heldout_decoder.pt`, SHA256
`73bea39e5e6146af22a45f52610cd24bfa8e498bec70e67bfdc685caeb065efd`.

## Safeguards

The artifact was fit with the registered A3-capacity network and fixed seed,
learning rate, weight decay, batch size, and epoch count. The standardization
was fit only on the 129,826 non-E2 rows. Duplicate pair slots are accepted
only because all cached pair-level channels are exactly identical; a
non-identical duplicate raises an error. The artifact is frozen under
`runs/e2_preflight_20260915/m2_validation_heldout_decoder.pt` and must not be
refit after human gold or model results are observed.
